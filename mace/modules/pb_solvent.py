"""Nonlinear Poisson-Boltzmann implicit-solvent charge model.

Replaces the planar truncated-Gaussian compensation layer with the laterally
averaged ionic charge profile rho_ion(z) of a full 3D nonlinear PB solve
(pure-Python solver from the validated cep-dip-python-pb repository, which
reproduces VASPsol/CEP-DIP PHI/RHOB/RHOION from a frozen CHGCAR).

Conventions bridged here (verified against the 2-NiN dataset and cal18):
- MACE radial coefficients model the *net* charge density in physics sign
  (electrons negative); a net grid integrates to ``total_charge`` (~-1 e).
- The PB solver uses the VASP electron-positive convention: CHGCAR-like
  "values" arrays with sum/N = electron count, ``q_sol = -total_charge``.
- Electron density fed to the cavity: neutral-atom Gaussian baseline
  (Z_val at the model smearing width) minus the net density.
- Solute potential: Hartree of that electron density plus Gaussian nuclei of
  charge -Z_val (electron-positive sign); the Gaussian form factor replaces
  the POTCAR local pseudopotential, whose Z/G^2 tail it matches exactly
  (EDEPS/(TPI^2 g^2 V) == 4 pi FELECT / g_abs^2 / V). Differences are
  confined inside the cavity where the solvent response vanishes.
- Returned profile is in physics sign: integral(rho_ion_z * area dz) ~=
  -total_charge, matching what the planar layer machinery expects.

No gradients flow through the solve; callers must treat outputs as detached.
"""

from __future__ import annotations

import json
import math
import os
import sys
from time import perf_counter
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

_FFT_FRIENDLY_CACHE: Dict[int, int] = {}


class _PBSolveFn(torch.autograd.Function):
    """Differentiable wrapper around a converged nonlinear-PB solve via the
    implicit function theorem (IFT). The forward RETURNS the already-converged
    phi_solv_g* (no re-solve -> forward is byte-identical to the detached path);
    the backward solves the adjoint linear system with the solver's own Newton
    operator (_minimize_l, which handles the G=0 charge-neutrality mode and
    preconditioning) and gets the input sensitivities via autograd VJPs of the
    residual in the solver's dprod_rc inner product. FD-verified (ratio ~1.0)
    against central differences on phi_sol and the cavity fields.
    """

    @staticmethod
    def forward(ctx, phi_sol, s_ion, s_diel, phi_solv_g_star, meta):
        tp, grid, params, w_b = meta["tp"], meta["grid"], meta["params"], meta["w_b"]
        with torch.no_grad():
            phi_total = grid.ifft_real(phi_solv_g_star) + phi_sol
            fields = tp._field_quantities(phi_total, s_ion, s_diel, grid, params, w_b)
            response, ekappa2 = tp._response_from_fields(fields, s_ion, s_diel, grid, params)
        ctx.save_for_backward(phi_solv_g_star, phi_sol, s_ion, s_diel)
        ctx.meta, ctx._response, ctx._ekappa2 = meta, response, ekappa2
        return phi_solv_g_star

    @staticmethod
    def backward(ctx, grad_phisolv_g):
        phi_solv_g, phi_sol, s_ion, s_diel = ctx.saved_tensors
        m = ctx.meta
        tp, grid, params, w_b = m["tp"], m["grid"], m["params"], m["w_b"]
        q_sol, tol, cg_max = m["q_sol"], m["tol"], m["cg_max"]
        wgt = grid.spectral_weight
        rhs = grad_phisolv_g / wgt if wgt is not None else grad_phisolv_g
        lam, _, _ = tp._minimize_l(rhs, ctx._response, ctx._ekappa2, w_b, grid, tol, cg_max)
        with torch.enable_grad():
            ps = phi_sol.detach().requires_grad_(True)
            si = s_ion.detach().requires_grad_(True)
            sd = s_diel.detach().requires_grad_(True)
            phi_total = grid.ifft_real(phi_solv_g.detach()) + ps
            f = tp._field_quantities(phi_total, si, sd, grid, params, w_b)
            resid, _ = tp._residual_g(phi_solv_g.detach(), f["n_b"], f["n_ion"], q_sol, grid)
            s = grid.dprod_rc(lam.detach(), resid)
            gps, gsi, gsd = torch.autograd.grad(s, [ps, si, sd])
        return -gps, -gsi, -gsd, None, None


def _fft_friendly_even(n: int) -> int:
    """Smallest even integer >= n whose prime factors are all in {2,3,5,7}."""
    if n in _FFT_FRIENDLY_CACHE:
        return _FFT_FRIENDLY_CACHE[n]
    m = max(int(n), 2)
    if m % 2:
        m += 1
    while True:
        k = m
        for p in (2, 3, 5, 7):
            while k % p == 0:
                k //= p
        if k == 1:
            _FFT_FRIENDLY_CACHE[n] = m
            return m
        m += 2


class PBPlanarSolvent:
    """Per-structure nonlinear PB solve returning rho_ion(z), lazily imported.

    Parameters mirror the validated driver (solve_from_chgcar_newton) with
    the POTCAR-dependent pieces replaced by Gaussian surrogates built from
    the model's own density representation.
    """

    def __init__(
        self,
        config_path: str,
        repo_path: Optional[str] = None,
        grid_spacing: float = 0.15,
        fixsol_steps: int = 2,
        tol: float = 1.0e-3,
        max_outer: int = 20,
        cg_max_iter: int = 200,
        coarse_init: bool = True,
        nuclear_sigma: float = 0.4,
        axis: int = 2,
    ) -> None:
        self._init_kwargs = {
            "config_path": config_path,
            "repo_path": repo_path,
            "grid_spacing": grid_spacing,
            "fixsol_steps": fixsol_steps,
            "tol": tol,
            "max_outer": max_outer,
            "cg_max_iter": cg_max_iter,
            "coarse_init": coarse_init,
            "nuclear_sigma": nuclear_sigma,
            "axis": axis,
        }
        if int(axis) != 2:
            raise NotImplementedError("PB solvent model is defined for z-axis slabs")
        if repo_path:
            repo_path = os.path.expandvars(os.path.expanduser(repo_path))
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
        try:
            from pure_python.grid import Grid, normalized_gaussian_kernel_g  # noqa: F401
            from pure_python.pb import create_cavity, derived_params
            from pure_python.dipole_correction import (
                EwaldDipoleMixer,
                cdipol_indmin_from_center,
                cdipol_potential_1d,
                solvent_moments,
                valence_ion_dipole_cart,
            )
            from pure_python.solute_potential import (
                hartree_potential_g,
                structure_factor_for_positions,
            )
            from pure_python.solve_from_chgcar_newton import (
                prolong_double,
                restrict_half,
                solve_nlpb_for_phi_sol,
            )
        except ImportError as exc:
            raise ImportError(
                "solvent_model='pb' requires the cep-dip-python-pb package; "
                "pass --solvent_pb_repo or add it to PYTHONPATH"
            ) from exc
        self._Grid = Grid
        self._create_cavity = create_cavity
        self._solve = solve_nlpb_for_phi_sol
        self._restrict_half = restrict_half
        self._prolong_double = prolong_double
        self._hartree_potential_g = hartree_potential_g
        self._structure_factors = structure_factor_for_positions
        self._EwaldDipoleMixer = EwaldDipoleMixer
        self._cdipol_indmin_from_center = cdipol_indmin_from_center
        self._cdipol_potential_1d = cdipol_potential_1d
        self._solvent_moments = solvent_moments
        self._valence_ion_dipole_cart = valence_ion_dipole_cart

        config_path = os.path.expandvars(os.path.expanduser(config_path))
        with open(config_path) as f:
            cfg = json.load(f)
        solvation = cfg.get("solvation", cfg)
        self.params = derived_params(solvation)
        self.grid_spacing = float(grid_spacing)
        self.fixsol_steps = int(fixsol_steps)
        self.tol = float(tol)
        self.max_outer = int(max_outer)
        self.cg_max_iter = int(cg_max_iter)
        self.coarse_init = bool(coarse_init)
        self.nuclear_sigma = float(nuclear_sigma)
        self.axis = int(axis)
        self.last_diagnostics: Dict[str, float] = {}

    def __getstate__(self) -> Dict:
        # Checkpoints pickle the owning model; rebuild from init kwargs so
        # no imported solver functions end up in the pickle.
        return dict(self._init_kwargs)

    def __setstate__(self, state: Dict) -> None:
        self.__init__(**state)

    def _grid_shape(self, cell: np.ndarray) -> tuple:
        lengths = np.linalg.norm(cell, axis=1)
        return tuple(
            _fft_friendly_even(int(math.ceil(l / self.grid_spacing)))
            for l in lengths
        )

    def solve_rho_ion_z(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        z_valence: np.ndarray,
        total_charge: float,
        neutral_sigma: float,
        eval_net_density: Callable[[np.ndarray], np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Run the PB solve for one slab structure.

        positions: [n_atoms, 3] cartesian (A); cell: [3, 3] (A, rows);
        z_valence: [n_atoms] valence electron counts; total_charge: physics
        net charge (e); neutral_sigma: Gaussian width of the neutral-atom
        electron baseline (the model's atomic_multipoles_smearing_width);
        eval_net_density: maps cartesian points [m, 3] -> net charge density
        [m] in e/A^3 (physics sign), used to sample the model density.

        Returns dict with z (nz,), rho_ion_z (nz,) physics e/A^3,
        q_ion (float, ~ -total_charge... in physics sign it is the layer
        charge, i.e. -total_charge), layer_mean (float, A).
        """
        t0 = perf_counter()
        cell = np.asarray(cell, dtype=float).reshape(3, 3)
        positions = np.asarray(positions, dtype=float)
        shape = self._grid_shape(cell)
        grid = self._Grid(cell, shape)
        volume = float(grid.volume)
        nx, ny, nz = shape

        # Fractional coordinates (wrapped) and grid points.
        inv_cell = np.linalg.inv(cell)
        pos_frac = (positions @ inv_cell) % 1.0

        fx = (np.arange(nx) / nx)
        fy = (np.arange(ny) / ny)
        fz = (np.arange(nz) / nz)
        frac_pts = np.stack(
            np.meshgrid(fx, fy, fz, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        cart_pts = frac_pts @ cell

        # Net charge density from the model (physics sign, e/A^3).
        net = eval_net_density(cart_pts).reshape(shape)
        t_net = perf_counter()

        # Neutral-atom electron baseline via structure factors (exact
        # periodic Gaussians in the "values"-amplitude convention),
        # grouped by valence count so only a few full-grid factors exist.
        _, _, _, gsq = grid.reciprocal_mesh_full()
        g_abs2 = gsq * (2.0 * math.pi) ** 2
        z_val = np.asarray(z_valence, dtype=float)
        damp_neutral = np.exp(-0.5 * g_abs2 * float(neutral_sigma) ** 2)
        damp_nuclear = np.exp(-0.5 * g_abs2 * float(self.nuclear_sigma) ** 2)
        neutral_g = np.zeros(shape, dtype=complex)
        nuclei_g = np.zeros(shape, dtype=complex)
        for zv in np.unique(z_val):
            sf = self._structure_factors(shape, pos_frac[z_val == zv])
            neutral_g += zv * damp_neutral * sf
            nuclei_g += -zv * damp_nuclear * sf
        neutral_values = grid.ifft_real_full(neutral_g)

        # Electron density (electron-positive, values units then e/A^3).
        n_e_values = neutral_values - net * volume
        n_e_density = np.clip(n_e_values / volume, 0.0, None)
        t_dens = perf_counter()

        s_ion, s_diel, _ = self._create_cavity(n_e_density, grid, self.params)
        t_cav = perf_counter()

        # Solute potential: Hartree of electrons + Gaussian nuclei.
        charge_g = grid.fft_full(n_e_values) + nuclei_g
        cvhar = grid.ifft_real_full(self._hartree_potential_g(charge_g, grid))

        q_sol = float(-total_charge)  # electron-count convention

        # Solute (valence electrons + nuclei) dipole for the z dipole
        # correction; same routine and branch conventions as the driver,
        # with one singleton "type" per atom.
        val_ion_dipole = self._valence_ion_dipole_cart(
            n_e_values,
            pos_frac,
            [float(z) for z in z_val],
            [1] * len(pos_frac),
            cell,
        )
        val_ion_dipole[0:2] = 0.0

        mixer = self._EwaldDipoleMixer.fresh()
        qsol_cache = 0.0
        dsol_cache = np.zeros(3, dtype=float)
        phi_total = np.zeros(shape, dtype=float)
        n_ion = np.zeros(shape, dtype=float)
        n_b = np.zeros(shape, dtype=float)
        indmin_z = self._cdipol_indmin_from_center(nz, 0.5)
        length_z = float(np.linalg.norm(cell[2]))
        center_abs = 0.5 * (cell[0] + cell[1] + cell[2])

        for step in range(self.fixsol_steps):
            dip_for_field = val_ion_dipole.copy()
            dip_for_field[2] += dsol_cache[2] - qsol_cache * center_abs[2]
            _, ef_direct = mixer.ewald_dipol(dip_for_field, cell, 3)
            cvdip_z = self._cdipol_potential_1d(
                nz, length_z, ef_direct[2], indmin_z
            )
            phi_sol = cvhar + cvdip_z[None, None, :]
            if (
                step == 0
                and self.coarse_init
                and all(n % 2 == 0 for n in shape)
            ):
                grid_c = self._Grid(cell, tuple(n // 2 for n in shape))
                phi_c, _, _, _, _ = self._solve(
                    np.zeros(grid_c.shape),
                    self._restrict_half(phi_sol),
                    self._restrict_half(s_ion),
                    self._restrict_half(s_diel),
                    grid_c,
                    self.params,
                    q_sol,
                    self.tol,
                    self.max_outer,
                    self.cg_max_iter,
                )
                phi_total = self._prolong_double(phi_c, grid_c, grid)
            phi_total, n_b, n_ion, _, history = self._solve(
                phi_total,
                phi_sol,
                s_ion,
                s_diel,
                grid,
                self.params,
                q_sol,
                self.tol,
                self.max_outer,
                self.cg_max_iter,
            )
            qsol_cache, dsol_cache = self._solvent_moments(n_b + n_ion, cell)
        t_solve = perf_counter()

        # Physics-sign profiles: rho_ion_z integrates (times area) to about
        # -total_charge; rho_bound_z (dielectric polarization, -div P) to ~0
        # but carries the implicit-region screening dipole.
        rho_ion_z = -(n_ion / volume).mean(axis=(0, 1))
        rho_bound_z = -(n_b / volume).mean(axis=(0, 1))
        z = np.arange(nz) * (length_z / nz)
        dz = length_z / nz
        area = volume / length_z
        q_ion = float(rho_ion_z.sum() * dz * area)
        q_bound = float(rho_bound_z.sum() * dz * area)
        denom = q_ion if abs(q_ion) > 1.0e-12 else 1.0e-12
        layer_mean = float((rho_ion_z * z).sum() * dz * area / denom)
        mu_bound = float((rho_bound_z * z).sum() * dz * area)

        self.last_diagnostics = {
            "t_net_density": t_net - t0,
            "t_baseline": t_dens - t_net,
            "t_cavity": t_cav - t_dens,
            "t_solve": t_solve - t_cav,
            "t_total": t_solve - t0,
            "rms_last": float(history[-1][1]) if history else float("nan"),
            "q_ion": q_ion,
            "q_bound": q_bound,
            "layer_mean": layer_mean,
            "mu_bound": mu_bound,
        }
        return {
            "z": z,
            "rho_ion_z": rho_ion_z,
            "rho_bound_z": rho_bound_z,
            "q_ion": q_ion,
            "q_bound": q_bound,
            "layer_mean": layer_mean,
            "mu_bound": mu_bound,
        }


class PBTorchBackend:
    """Device-resident PB solve: density sampling, cavity, and the Newton-PCG
    solve all stay on the GPU (pure_python.torch_pb); only O(nz) profiles and
    O(1) dipole bookkeeping touch the CPU.

    Speed features, both result-preserving:
    - per-sample warm start: the converged phi (float32, CPU-cached, keyed by
      sample_id) seeds the next epoch's fixstep-0 Newton solve. The dipole
      fix-point loop still runs the FULL cold protocol from a zero state, so
      the only difference vs a cold solve is the Newton initial guess —
      convergence is driven to the same residual tolerance either way
      (tolerance-identical results; verified in test_torch_backend.py).
      Caching the dipole state with a single warm fix-step was tried and
      rejected: the undamped fix-point update oscillates (~0.03 A in the
      layer mean, ~0.07 V).
    - torch coarse-grid warm start for cold solves (same construction as the
      numpy driver).
    ``warm_fixsol_steps=0`` (default) means "same as fixsol_steps".
    """

    def __init__(
        self,
        config_path: str,
        repo_path: Optional[str] = None,
        grid_spacing: float = 0.15,
        fixsol_steps: int = 2,
        tol: float = 1.0e-3,
        max_outer: int = 20,
        cg_max_iter: int = 200,
        coarse_init: bool = True,
        nuclear_sigma: float = 0.4,
        axis: int = 2,
        warm_start: bool = True,
        warm_fixsol_steps: int = 0,
        baseline_cache: Optional[str] = None,
        phi_cache_dir: Optional[str] = None,
    ) -> None:
        self._init_kwargs = {
            "config_path": config_path,
            "repo_path": repo_path,
            "grid_spacing": grid_spacing,
            "fixsol_steps": fixsol_steps,
            "tol": tol,
            "max_outer": max_outer,
            "cg_max_iter": cg_max_iter,
            "coarse_init": coarse_init,
            "nuclear_sigma": nuclear_sigma,
            "axis": axis,
            "warm_start": warm_start,
            "warm_fixsol_steps": warm_fixsol_steps,
            "baseline_cache": baseline_cache,
            "phi_cache_dir": phi_cache_dir,
        }
        if int(axis) != 2:
            raise NotImplementedError("PB solvent model is defined for z-axis slabs")
        if repo_path:
            repo_path = os.path.expandvars(os.path.expanduser(repo_path))
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
        try:
            from pure_python import torch_pb as tp
            from pure_python.pb import derived_params
            from pure_python.dipole_correction import (
                EwaldDipoleMixer,
                cdipol_indmin_from_center,
                cdipol_potential_1d,
                valence_ion_dipole_cart,
            )
        except ImportError as exc:
            raise ImportError(
                "solvent_pb_backend='torch' requires cep-dip-python-pb (with "
                "torch_pb); pass --solvent_pb_repo or add it to PYTHONPATH"
            ) from exc
        self._tp = tp
        self._EwaldDipoleMixer = EwaldDipoleMixer
        self._cdipol_indmin_from_center = cdipol_indmin_from_center
        self._cdipol_potential_1d = cdipol_potential_1d
        self._valence_ion_dipole_cart = valence_ion_dipole_cart

        config_path = os.path.expandvars(os.path.expanduser(config_path))
        with open(config_path) as f:
            cfg = json.load(f)
        self.params = derived_params(cfg.get("solvation", cfg))
        self.grid_spacing = float(grid_spacing)
        self.fixsol_steps = int(fixsol_steps)
        self.tol = float(tol)
        self.max_outer = int(max_outer)
        self.cg_max_iter = int(cg_max_iter)
        self.coarse_init = bool(coarse_init)
        self.nuclear_sigma = float(nuclear_sigma)
        self.axis = int(axis)
        self.warm_start = bool(warm_start)
        self.warm_fixsol_steps = int(warm_fixsol_steps)
        self._grids: Dict = {}
        self._phi_cache: Dict = {}
        # Optional rank-independent warm-start cache: phi persisted per
        # sample_id to a shared directory, so warm start survives a data
        # shuffle that moves a structure across ranks (the in-memory dict is
        # per-process). When set, disk is authoritative (always the latest
        # solve); the in-memory dict is a same-rank fast path only.
        self._phi_cache_dir = (
            os.path.expandvars(os.path.expanduser(phi_cache_dir))
            if phi_cache_dir else None
        )
        if self._phi_cache_dir:
            os.makedirs(self._phi_cache_dir, exist_ok=True)
        self.last_diagnostics: Dict[str, float] = {}
        # Optional per-sample DFT baseline cache (neutral / dencor / phi_base
        # fields on the PB grid): replaces the Gaussian-baseline and
        # Gaussian-nuclei surrogates with the exact data-derived fields.
        # Memory-mapped so distributed ranks share physical pages.
        self._bl_arr = None
        self._bl_index: Dict[int, int] = {}
        self._bl_shape = None
        if baseline_cache:
            bl = os.path.expandvars(os.path.expanduser(baseline_cache))
            with open(os.path.join(bl, "baseline_index.json")) as f:
                self._bl_index = {int(k): int(v) for k, v in json.load(f).items()}
            with open(os.path.join(bl, "baseline_meta.json")) as f:
                meta = json.load(f)
            self._bl_shape = tuple(meta["pb_shape"])
            self._bl_arr = np.load(
                os.path.join(bl, "baseline_cache.npy"), mmap_mode="r"
            )

    def __getstate__(self) -> Dict:
        return dict(self._init_kwargs)

    def __setstate__(self, state: Dict) -> None:
        self.__init__(**state)

    def _grid_shape(self, cell: np.ndarray) -> tuple:
        lengths = np.linalg.norm(cell, axis=1)
        return tuple(
            _fft_friendly_even(int(math.ceil(l / self.grid_spacing)))
            for l in lengths
        )

    def _grid_for(self, cell_np: np.ndarray, shape, device):
        key = (shape, cell_np.tobytes(), str(device))
        g = self._grids.get(key)
        if g is None:
            g = self._tp.TorchGrid(
                cell_np, shape, device=device, dtype=torch.float64, rspec=True
            )
            # per-shape frequency vectors for structure factors
            nx, ny, nz = shape
            g._hx = torch.fft.fftfreq(nx, device=g.device, dtype=g.dtype) * nx
            g._hy = torch.fft.fftfreq(ny, device=g.device, dtype=g.dtype) * ny
            g._hz = torch.arange(nz // 2 + 1, device=g.device, dtype=g.dtype)
            self._grids[key] = g
        return g

    def _gto_spec_basis(self, grid, sigmas) -> torch.Tensor:
        """Per-channel reciprocal form factors of the normalized-multipole
        GTO basis, cached on the grid: [C, *spec_shape] complex with
        C = len(sigmas) * 9 (sigma-major, then lm with l = 0, 1, 2).

        FT[phi_lm^sigma](G) = (-i)^l A_l |G|^l Y_lm(G_hat) exp(-sigma^2 G^2/2),
        A_l = (2 pi)^{3/2} / (sqrt(4 pi/(2l+1)) 2^{(2l+1)/2} Gamma(l+3/2));
        the 'multipoles' normalization cancels sigma^{2l+3} exactly and gives
        A_0 = 1, so a monopole coefficient c contributes amplitude c at G -> 0
        — the same convention as the neutral/nuclear structure factors.
        Y_lm is evaluated with the same e3nn call (and axis permutation) as
        the real-space evaluator, so conventions match by construction.
        """
        key = tuple(float(s) for s in sigmas)
        cache = getattr(grid, "_gto_spec_basis", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        from e3nn.o3 import spherical_harmonics
        from graph_longrange.utils import permute_to_e3nn_convention
        from scipy.special import gamma as _gamma

        gvec = torch.stack([grid.gx, grid.gy, grid.gz], dim=-1) * (2.0 * math.pi)
        g2 = (gvec * gvec).sum(-1)
        gabs = torch.sqrt(g2)
        nz = g2 > 1.0e-24
        safe = torch.where(
            nz.unsqueeze(-1), gvec,
            torch.tensor([1.0, 0.0, 0.0], device=gvec.device, dtype=gvec.dtype),
        )
        Y = []
        for l in range(3):
            y = spherical_harmonics(
                l, permute_to_e3nn_convention(safe.reshape(-1, 3)), normalize=True
            )
            Y.append(y.reshape(*grid.spec_shape, 2 * l + 1))
        chans = []
        for s in key:
            damp = torch.exp(-0.5 * g2 * s * s)
            for l in range(3):
                A = (2.0 * math.pi) ** 1.5 / (
                    math.sqrt(4.0 * math.pi / (2 * l + 1))
                    * 2.0 ** ((2 * l + 1) / 2.0)
                    * float(_gamma((2 * l + 3) / 2.0))
                )
                radial = damp * gabs ** l * A
                if l > 0:
                    radial = radial * nz  # kills G=0 (and the padded dirs)
                phase = (-1j) ** l
                for m in range(2 * l + 1):
                    chans.append((phase * (radial * Y[l][..., m])).to(grid.cdtype))
        B = torch.stack(chans)
        grid._gto_spec_basis = (key, B)
        return B

    def _gto_net_density_g(
        self, grid, pos_frac: torch.Tensor, coeffs: torch.Tensor, sigmas
    ) -> torch.Tensor:
        """Values-amplitude spectrum of the model net density from its GTO
        multipole coefficients [n_atoms, n_sigmas, 9] — replaces the O(grid x
        atoms) pointwise evaluation with structure factors x form factors."""
        B = self._gto_spec_basis(grid, sigmas)
        tpi = 2.0 * math.pi
        ex = torch.exp(-1j * tpi * pos_frac[:, 0, None] * grid._hx[None, :])
        ey = torch.exp(-1j * tpi * pos_frac[:, 1, None] * grid._hy[None, :])
        ez = torch.exp(-1j * tpi * pos_frac[:, 2, None] * grid._hz[None, :])
        W = coeffs.reshape(coeffs.shape[0], -1).to(B.dtype)  # [a, C]
        S = torch.einsum("ac,ah,ak,al->chkl", W, ex, ey, ez)
        return (B * S).sum(dim=0)

    def _structure_factor(self, grid, pos_frac: torch.Tensor) -> torch.Tensor:
        tpi = 2.0 * math.pi
        ex = torch.exp(-1j * tpi * pos_frac[:, 0, None] * grid._hx[None, :])
        ey = torch.exp(-1j * tpi * pos_frac[:, 1, None] * grid._hy[None, :])
        ez = torch.exp(-1j * tpi * pos_frac[:, 2, None] * grid._hz[None, :])
        return torch.einsum("ah,ak,al->hkl", ex, ey, ez)

    @staticmethod
    def _restrict_half_t(a: torch.Tensor) -> torch.Tensor:
        return a[::2, ::2, ::2].contiguous()

    def _prolong_double_t(self, phi_c: torch.Tensor, grid_c, grid_f) -> torch.Tensor:
        spec_c = grid_c.fft(phi_c)
        spec_f = torch.zeros(grid_f.spec_shape, dtype=grid_f.cdtype, device=grid_f.device)
        ncx, ncy, ncz = grid_c.shape
        hx, hy, hz = ncx // 2, ncy // 2, ncz // 2
        spec_f[:hx, :hy, :hz] = spec_c[:hx, :hy, :hz]
        spec_f[-(hx - 1):, :hy, :hz] = spec_c[-(hx - 1):, :hy, :hz]
        spec_f[:hx, -(hy - 1):, :hz] = spec_c[:hx, -(hy - 1):, :hz]
        spec_f[-(hx - 1):, -(hy - 1):, :hz] = spec_c[-(hx - 1):, -(hy - 1):, :hz]
        return grid_f.ifft_real(spec_f)

    def solve_rho_ion_z(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        z_valence: torch.Tensor,
        total_charge: float,
        neutral_sigma: float,
        eval_net_density=None,
        sample_id: Optional[int] = None,
        radial_coeffs: Optional[torch.Tensor] = None,
        sigmas=None,
    ) -> Dict[str, torch.Tensor]:
        """Device-resident analogue of PBPlanarSolvent.solve_rho_ion_z.

        positions/cell/z_valence are torch tensors on the model device.
        The net density enters either spectrally via ``radial_coeffs``
        [n_atoms, n_sigmas, 9] + ``sigmas`` (fast path, exact reciprocal
        assembly) or through ``eval_net_density`` mapping cartesian points
        [m, 3] (device tensor) to net charge density [m] (device tensor,
        physics sign, e/A^3). Returns z / rho_ion_z / rho_bound_z as device
        tensors plus float moments.
        """
        t0 = perf_counter()
        device = positions.device
        cell_np = cell.detach().cpu().numpy().astype(float).reshape(3, 3)
        shape = self._grid_shape(cell_np)
        grid = self._grid_for(cell_np, shape, device)
        nx, ny, nz = shape
        volume = grid.volume
        dt = torch.float64

        pos64 = positions.detach().to(dt)
        cell64 = torch.as_tensor(cell_np, device=device, dtype=dt)
        pos_frac = torch.remainder(
            pos64 @ torch.linalg.inv(cell64), 1.0
        )

        # net density: spectral assembly from coefficients (fast path) or
        # pointwise evaluation on cached grid points (fallback)
        if radial_coeffs is not None:
            net_g = self._gto_net_density_g(
                grid, pos_frac, radial_coeffs.detach().to(dt), sigmas
            )
        else:
            if not hasattr(grid, "_cart_pts"):
                fx = torch.arange(nx, device=device, dtype=dt) / nx
                fy = torch.arange(ny, device=device, dtype=dt) / ny
                fz = torch.arange(nz, device=device, dtype=dt) / nz
                frac = torch.stack(
                    torch.meshgrid(fx, fy, fz, indexing="ij"), dim=-1
                ).reshape(-1, 3)
                grid._cart_pts = frac @ cell64
            net = eval_net_density(grid._cart_pts).to(dt).reshape(shape)
            net_g = grid.fft(net * volume)
        t_net = perf_counter()

        zv = z_valence.detach().to(dt)
        # baseline: exact per-sample DFT fields from the cache when
        # available, else Gaussian surrogates via structure factors.
        bl_row = (
            self._bl_index.get(sample_id)
            if (self._bl_arr is not None and sample_id is not None
                and self._bl_shape == shape)
            else None
        )
        if bl_row is not None:
            fields = torch.as_tensor(
                np.ascontiguousarray(self._bl_arr[bl_row]), device=device
            ).to(dt)
            neutral_v, phi_base = fields[0], fields[2]
            # NOTE: fields[1] (dencor, POTCAR partial core) is kept in the
            # cache format but NOT used — user testing showed it does not
            # affect the results (consistent with its ~4.6 e being localized
            # deep inside the cavity where density >> the NC_K isovalue).
            net_values = grid.ifft_real(net_g)
            n_e_values = neutral_v - net_values  # valence only
            n_e_density = torch.clamp(n_e_values / volume, min=0.0)
            # phi_base already contains Hartree(neutral) + exact local PSP
            cvhar = phi_base - grid.ifft_real(grid.l0_inv_op(net_g))
        else:
            # Gaussian surrogates (fallback / no cache): neutral baseline +
            # nuclei via structure factors grouped by zval.
            g_abs2 = grid.gsq * (2.0 * math.pi) ** 2
            damp_neutral = torch.exp(-0.5 * g_abs2 * float(neutral_sigma) ** 2)
            damp_nuclear = torch.exp(-0.5 * g_abs2 * float(self.nuclear_sigma) ** 2)
            neutral_g = torch.zeros(grid.spec_shape, dtype=grid.cdtype, device=device)
            nuclei_g = torch.zeros(grid.spec_shape, dtype=grid.cdtype, device=device)
            for v in torch.unique(zv):
                sf = self._structure_factor(grid, pos_frac[zv == v])
                neutral_g = neutral_g + float(v) * damp_neutral * sf
                nuclei_g = nuclei_g + (-float(v)) * damp_nuclear * sf
            n_e_g = neutral_g - net_g
            n_e_values = grid.ifft_real(n_e_g)
            # NOTE: positivity-clipping n_e for the solute potential was tried
            # and REVERTED — the clip changes the electron count while q_sol
            # stays nominal, breaking charge consistency and diverging the
            # Newton solve (residual 10..178, |mu_bound| ~ 1e15 at init). The
            # cavity input alone is clipped (log() needs positivity); the
            # Hartree charge keeps the model density as-is.
            n_e_density = torch.clamp(n_e_values / volume, min=0.0)
            cvhar = grid.ifft_real(grid.l0_inv_op(n_e_g + nuclei_g))
        t_dens = perf_counter()

        s_ion, s_diel, _ = self._tp.create_cavity_torch(n_e_density, grid, self.params)
        t_cav = perf_counter()

        q_sol = float(-total_charge)

        # valence+ion dipole via the reference routine on the plane profile
        # (identical by construction: it only uses the plane average).
        prof_ne = n_e_values.mean(dim=(0, 1)).detach().cpu().numpy()
        val_ion_dipole = self._valence_ion_dipole_cart(
            prof_ne.reshape(1, 1, nz),
            pos_frac.detach().cpu().numpy(),
            [float(x) for x in zv.detach().cpu().numpy()],
            [1] * int(pos64.shape[0]),
            cell_np,
        )
        val_ion_dipole[0:2] = 0.0

        # warm start: phi only — the dipole fix-point loop always runs the
        # cold protocol from a zero state so results are protocol-identical.
        cached = None
        if self.warm_start and sample_id is not None:
            if self._phi_cache_dir is not None:
                # disk is authoritative (rank-independent, always latest)
                p = os.path.join(self._phi_cache_dir, f"phi_{sample_id}.npy")
                try:
                    arr = np.load(p)
                    if tuple(arr.shape) == tuple(shape):
                        cached = {"phi": torch.from_numpy(arr), "shape": shape}
                except (OSError, ValueError):
                    cached = None
            else:
                cached = self._phi_cache.get(sample_id)
        warm = cached is not None and cached["shape"] == shape
        if warm:
            phi_total = cached["phi"].to(device=device, dtype=dt)
            n_steps = (
                self.warm_fixsol_steps
                if self.warm_fixsol_steps > 0
                else self.fixsol_steps
            )
        else:
            phi_total = torch.zeros(shape, dtype=dt, device=device)
            n_steps = self.fixsol_steps
        qsol_cache = 0.0
        dsol_z = 0.0

        mixer = self._EwaldDipoleMixer.fresh()
        indmin_z = self._cdipol_indmin_from_center(nz, 0.5)
        length_z = float(np.linalg.norm(cell_np[2]))
        center_abs_z = 0.5 * (cell_np[0, 2] + cell_np[1, 2] + cell_np[2, 2])
        z_mesh = grid.z_mesh  # [nx,ny,nz] cartesian z of grid points
        n_b = n_ion = None
        history = []
        # per-step timers: wall time + Newton outer-iteration count of each
        # fixsol step, and the accumulated dipole-correction (Ewald/cdipol)
        # share — shows where extra fixsol steps spend their time.
        t_steps: List[float] = []
        newton_outer: List[int] = []
        t_dip_acc = 0.0
        for step in range(n_steps):
            ts0 = perf_counter()
            dip = val_ion_dipole.copy()
            dip[2] += dsol_z - qsol_cache * center_abs_z
            _, ef_direct = mixer.ewald_dipol(dip, cell_np, 3)
            cvdip_z = torch.as_tensor(
                self._cdipol_potential_1d(nz, length_z, ef_direct[2], indmin_z),
                device=device, dtype=dt,
            )
            t_dip_acc += perf_counter() - ts0
            phi_sol = cvhar + cvdip_z[None, None, :]
            if step == 0 and not warm and self.coarse_init and all(
                n % 2 == 0 for n in shape
            ):
                grid_c = self._grid_for(
                    cell_np, tuple(n // 2 for n in shape), device
                )
                phi_c, _, _, _, _ = self._tp.solve_nlpb_for_phi_sol_torch(
                    torch.zeros(grid_c.shape, dtype=dt, device=device),
                    self._restrict_half_t(phi_sol),
                    self._restrict_half_t(s_ion),
                    self._restrict_half_t(s_diel),
                    grid_c, self.params, q_sol,
                    self.tol, self.max_outer, self.cg_max_iter,
                )
                phi_total = self._prolong_double_t(phi_c, grid_c, grid)
            phi_total, n_b, n_ion, _, history = self._tp.solve_nlpb_for_phi_sol_torch(
                phi_total, phi_sol, s_ion, s_diel, grid, self.params,
                q_sol, self.tol, self.max_outer, self.cg_max_iter,
            )
            solv = n_b + n_ion
            qsol_cache = float(solv.mean())  # forces device sync
            dsol_z = float((solv * z_mesh).mean())
            t_steps.append(perf_counter() - ts0)
            newton_outer.append(len(history))
        t_solve = perf_counter()

        rms_last = float(history[-1][1]) if history else float("nan")
        if (
            self.warm_start and sample_id is not None
            and rms_last == rms_last and rms_last < 10.0 * self.tol
        ):
            # never seed future solves from a diverged state
            phi_cpu = phi_total.detach().to(torch.float32).cpu()
            if self._phi_cache_dir is not None:
                # atomic write (temp + replace) so a concurrent reader on
                # another rank never sees a partial file
                p = os.path.join(self._phi_cache_dir, f"phi_{sample_id}.npy")
                tmp = f"{p}.tmp.{os.getpid()}.npy"  # .npy so np.save won't append
                np.save(tmp, phi_cpu.numpy())
                os.replace(tmp, p)
            else:
                self._phi_cache[sample_id] = {"phi": phi_cpu, "shape": shape}

        # Differentiable recompute (opt-in via requires_grad + baseline cache):
        # the forward loop above ran detached, so phi*/n_b/n_ion are unchanged
        # (forward byte-identical). Here we rebuild cvhar+cavity from the
        # grad-carrying net density, wrap the converged solve in the IFT
        # Function, and re-derive n_ion/n_b by autograd -> same VALUES, now with
        # gradient into radial_coeffs. The dipole self-consistency cvdip is held
        # at its converged value (detached; a small secondary term also detached
        # in the non-diff path).
        if (radial_coeffs is not None and radial_coeffs.requires_grad
                and bl_row is not None and history):
            net_g_d = self._gto_net_density_g(
                grid, pos_frac, radial_coeffs.to(dt), sigmas
            )
            net_values_d = grid.ifft_real(net_g_d)
            n_e_density_d = torch.clamp((neutral_v - net_values_d) / volume, min=0.0)
            cvhar_d = phi_base - grid.ifft_real(grid.l0_inv_op(net_g_d))
            s_ion_d, s_diel_d, _ = self._tp.create_cavity_torch(
                n_e_density_d, grid, self.params
            )
            r_b = (float(self.params["R_B"]) if float(self.params["R_B"]) > 0.0
                   else float(self.params["A_K"]))
            w_b_d = self._tp._normalized_gaussian_kernel_g(grid, r_b)
            phi_sol_d = cvhar_d + cvdip_z[None, None, :].detach()
            phi_solv_g_star = grid.fft(phi_total.detach() - phi_sol_d.detach())
            meta = {
                "tp": self._tp, "grid": grid, "params": self.params, "w_b": w_b_d,
                "q_sol": q_sol, "tol": self.tol, "cg_max": self.cg_max_iter,
            }
            phi_solv_g_d = _PBSolveFn.apply(
                phi_sol_d, s_ion_d, s_diel_d, phi_solv_g_star, meta
            )
            phi_total_d = grid.ifft_real(phi_solv_g_d) + phi_sol_d
            f_d = self._tp._field_quantities(
                phi_total_d, s_ion_d, s_diel_d, grid, self.params, w_b_d
            )
            n_ion, n_b = f_d["n_ion"], f_d["n_b"]

        rho_ion_z = -(n_ion / volume).mean(dim=(0, 1))
        rho_bound_z = -(n_b / volume).mean(dim=(0, 1))
        dz = length_z / nz
        area = volume / length_z
        z = torch.arange(nz, device=device, dtype=dt) * dz
        q_ion = float((rho_ion_z.sum() * dz * area).item())
        q_bound = float((rho_bound_z.sum() * dz * area).item())
        denom = q_ion if abs(q_ion) > 1.0e-12 else 1.0e-12
        layer_mean = float(((rho_ion_z * z).sum() * dz * area).item() / denom)
        mu_bound = float(((rho_bound_z * z).sum() * dz * area).item())

        self.last_diagnostics = {
            "t_net_density": t_net - t0,
            "t_baseline": t_dens - t_net,
            "t_cavity": t_cav - t_dens,
            "t_solve": t_solve - t_cav,
            "t_total": t_solve - t0,
            "t_dipole": round(t_dip_acc, 4),
            "t_steps": [round(t, 3) for t in t_steps],
            "newton_outer": newton_outer,
            "warm": bool(warm),
            "rms_last": rms_last,
            "q_ion": q_ion,
            "q_bound": q_bound,
            "layer_mean": layer_mean,
            "mu_bound": mu_bound,
        }
        return {
            "z": z,
            "rho_ion_z": rho_ion_z,
            "rho_bound_z": rho_bound_z,
            "height": length_z,
            "q_ion": q_ion,
            "q_bound": q_bound,
            "layer_mean": layer_mean,
            "mu_bound": mu_bound,
            "rms_last": rms_last,
        }


def resample_profile_periodic_torch(
    profile: torch.Tensor,
    height: float,
    num_grid: int,
    offset_half: bool,
) -> torch.Tensor:
    """Periodic linear resampling of a uniform-grid profile (torch, device)."""
    nz = profile.shape[0]
    j = torch.arange(num_grid, device=profile.device, dtype=profile.dtype)
    u = (j + (0.5 if offset_half else 0.0)) * (nz / float(num_grid))
    i0 = torch.floor(u).to(torch.long) % nz
    i1 = (i0 + 1) % nz
    w = (u - torch.floor(u)).to(profile.dtype)
    return profile[i0] * (1.0 - w) + profile[i1] * w


def resample_profile_periodic(
    z_src: np.ndarray,
    rho_src: np.ndarray,
    z_dst: np.ndarray,
    period: float,
) -> np.ndarray:
    """Periodic linear interpolation of a 1D profile onto new z points."""
    zs = np.concatenate([z_src, [z_src[0] + period]])
    rs = np.concatenate([rho_src, [rho_src[0]]])
    return np.interp(np.mod(z_dst, period), zs, rs)


def profiles_to_tensors(
    profiles: List[Optional[Dict[str, np.ndarray]]],
    heights: List[float],
    num_grid: int,
    offset_half: bool,
    device: torch.device,
    dtype: torch.dtype,
    key: str = "rho_ion_z",
) -> torch.Tensor:
    """Stack a per-graph z-profile onto a common [n_graphs, num_grid] tensor.

    Graphs without a profile (non-slab or failed solve) get zeros. The
    destination grid matches the conventions of the layer machinery:
    z_j = (j + 0.5) * H / num_grid when offset_half else j * H / num_grid.
    """
    out = np.zeros((len(profiles), num_grid), dtype=np.float64)
    for g, prof in enumerate(profiles):
        if prof is None:
            continue
        height = float(heights[g])
        j = np.arange(num_grid) + (0.5 if offset_half else 0.0)
        z_dst = j * height / num_grid
        out[g] = resample_profile_periodic(
            prof["z"], prof[key], z_dst, height
        )
    return torch.as_tensor(out, device=device, dtype=dtype)
