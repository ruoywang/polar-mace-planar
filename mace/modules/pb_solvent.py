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
