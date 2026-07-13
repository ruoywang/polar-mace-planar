"""Per-graph engine for the pb1d solvent route.

For one structure and one forward pass this assembles the live 3-D fields
(model net density via spectral GTO assembly + the DFT baseline cache),
derives the 1-D closure quantities (pb1d_closure), applies the residual
head, and runs the differentiable 1-D solve (pb1d_solver). Output format
mirrors the retired 3-D backend's solve_rho_ion_z so the model wiring can
consume it through the same hooks.

The GTO spectral assembly and baseline-cache access are lifted from the
retired pb_solvent.PBTorchBackend (validated in the 3-D era).
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch

from .pb1d_closure import closure_from_fields, solute_dipole_z
from .pb1d_solver import Solver1D

TPI = 2.0 * math.pi


def _fft_friendly_even(n: int) -> int:
    def ok(m: int) -> bool:
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        return m == 1

    m = n + (n % 2)
    while not ok(m):
        m += 2
    return m


def fourier_upsample(profile: torch.Tensor, factor: int) -> torch.Tensor:
    """Band-limited periodic upsampling of a 1-D profile (differentiable)."""
    nz = profile.shape[0]
    spec = torch.fft.rfft(profile)
    out = torch.zeros(nz * factor // 2 + 1, dtype=spec.dtype, device=spec.device)
    out[: spec.shape[0]] = spec
    return torch.fft.irfft(out * factor, n=nz * factor)


class PB1DBackend:
    def __init__(
        self,
        config_path: str,
        repo_path: Optional[str] = None,
        grid_spacing: float = 0.15,
        baseline_cache: Optional[str] = None,
        solve_upsample: int = 2,
        fixsol_steps: int = 5,
        tol: float = 1.0e-3,
        max_outer: int = 12,
        axis: int = 2,
    ) -> None:
        self._init_kwargs = {
            "config_path": config_path,
            "repo_path": repo_path,
            "grid_spacing": grid_spacing,
            "baseline_cache": baseline_cache,
            "solve_upsample": solve_upsample,
            "fixsol_steps": fixsol_steps,
            "tol": tol,
            "max_outer": max_outer,
            "axis": axis,
        }
        if int(axis) != 2:
            raise NotImplementedError("pb1d is defined for z-axis slabs")
        if repo_path:
            repo_path = os.path.expandvars(os.path.expanduser(repo_path))
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
        try:
            from pure_python import torch_pb as tp
            from pure_python.pb import derived_params
            from pure_python.dipole_correction import EwaldDipoleMixer
        except ImportError as exc:
            raise ImportError(
                "solvent_model='pb1d' requires cep-dip-python-pb (torch_pb); "
                "pass --solvent_pb_repo or add it to PYTHONPATH"
            ) from exc
        self._tp = tp
        self._EwaldDipoleMixer = EwaldDipoleMixer
        config_path = os.path.expandvars(os.path.expanduser(config_path))
        with open(config_path) as f:
            cfg = json.load(f)
        self.params = derived_params(cfg.get("solvation", cfg))
        self.grid_spacing = float(grid_spacing)
        self.solve_upsample = int(solve_upsample)
        self.fixsol_steps = int(fixsol_steps)
        self.tol = float(tol)
        self.max_outer = int(max_outer)
        self._grids: Dict = {}
        self._solvers: Dict = {}
        self._c_units: Dict = {}
        self.last_diagnostics: Dict[str, float] = {}
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

    # -- lifted grid / spectral-assembly machinery --------------------------
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
            nx, ny, nz = shape
            g._hx = torch.fft.fftfreq(nx, device=g.device, dtype=g.dtype) * nx
            g._hy = torch.fft.fftfreq(ny, device=g.device, dtype=g.dtype) * ny
            g._hz = torch.arange(nz // 2 + 1, device=g.device, dtype=g.dtype)
            self._grids[key] = g
        return g

    def _gto_spec_basis(self, grid, sigmas) -> torch.Tensor:
        key = tuple(float(s) for s in sigmas)
        cache = getattr(grid, "_gto_spec_basis", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        from e3nn.o3 import spherical_harmonics
        from graph_longrange.utils import permute_to_e3nn_convention
        from scipy.special import gamma as _gamma

        gvec = torch.stack([grid.gx, grid.gy, grid.gz], dim=-1) * TPI
        g2 = (gvec * gvec).sum(-1)
        gabs = torch.sqrt(g2)
        nzm = g2 > 1.0e-24
        safe = torch.where(
            nzm.unsqueeze(-1), gvec,
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
                    radial = radial * nzm
                phase = (-1j) ** l
                for m in range(2 * l + 1):
                    chans.append((phase * (radial * Y[l][..., m])).to(grid.cdtype))
        B = torch.stack(chans)
        grid._gto_spec_basis = (key, B)
        return B

    def _gto_net_density_g(self, grid, pos_frac, coeffs, sigmas) -> torch.Tensor:
        B = self._gto_spec_basis(grid, sigmas)
        ex = torch.exp(-1j * TPI * pos_frac[:, 0, None] * grid._hx[None, :])
        ey = torch.exp(-1j * TPI * pos_frac[:, 1, None] * grid._hy[None, :])
        ez = torch.exp(-1j * TPI * pos_frac[:, 2, None] * grid._hz[None, :])
        W = coeffs.reshape(coeffs.shape[0], -1).to(B.dtype)
        S = torch.einsum("ac,ah,ak,al->chkl", W, ex, ey, ez)
        return (B * S).sum(dim=0)

    def _c_unit(self, cell_np: np.ndarray) -> float:
        key = cell_np.tobytes()
        c = self._c_units.get(key)
        if c is None:
            m = self._EwaldDipoleMixer.fresh()
            m.dipolc_tmp[2] = 1.0
            _, ef = m.ewald_dipol(np.array([0.0, 0.0, 1.0]), cell_np, 3)
            c = float(ef[2])
            self._c_units[key] = c
        return c

    def _solver_for(self, cell_np: np.ndarray, nz_solve: int, device) -> Solver1D:
        key = (cell_np.tobytes(), nz_solve, str(device))
        s = self._solvers.get(key)
        if s is None:
            s = Solver1D(
                torch.tensor(cell_np, dtype=torch.float64, device=device),
                nz_solve, self.params, device=device,
            )
            self._solvers[key] = s
        return s

    # -- the per-graph forward ----------------------------------------------
    def solve_graph(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        z_valence: torch.Tensor,
        total_charge: float,
        sample_id: Optional[int],
        radial_coeffs: torch.Tensor,
        sigmas,
        node_feats: Optional[torch.Tensor] = None,
        head=None,
        q_tot: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        from torch.utils.checkpoint import checkpoint as _ckpt

        device = positions.device
        dt = torch.float64
        cell_np = cell.detach().cpu().numpy().astype(float).reshape(3, 3)
        shape = self._grid_shape(cell_np)
        grid = self._grid_for(cell_np, shape, device)
        nx, ny, nz = shape
        volume = grid.volume
        length_z = float(np.linalg.norm(cell_np[2]))

        bl_row = (
            self._bl_index.get(sample_id)
            if (self._bl_arr is not None and sample_id is not None
                and self._bl_shape == shape)
            else None
        )
        if bl_row is None:
            raise RuntimeError(
                f"pb1d requires the baseline cache for sample_id={sample_id} "
                f"(shape {shape}); no Gaussian-surrogate fallback in v1"
            )

        cell64 = torch.as_tensor(cell_np, device=device, dtype=dt)
        pos64 = positions.to(dt)
        pos_frac = torch.remainder(pos64 @ torch.linalg.inv(cell64), 1.0)
        want_grad = bool(radial_coeffs.requires_grad or (
            node_feats is not None and node_feats.requires_grad))

        fields = torch.as_tensor(
            np.ascontiguousarray(self._bl_arr[bl_row]), device=device
        ).to(dt)
        neutral_v, phi_base = fields[0], fields[2]

        def _assemble(coeffs):
            return self._gto_net_density_g(grid, pos_frac.detach(), coeffs.to(dt), sigmas)

        net_g = (
            _ckpt(_assemble, radial_coeffs, use_reentrant=False)
            if want_grad else _assemble(radial_coeffs)
        )
        net_values = grid.ifft_real(net_g)
        n_e_values = neutral_v - net_values
        n_e_density = torch.clamp(n_e_values / volume, min=0.0)
        cvhar3 = phi_base - grid.ifft_real(grid.l0_inv_op(net_g))

        def _closure(ne, cv):
            return closure_from_fields(ne, cv, grid, self.params, self._tp)

        if want_grad:
            keys = ("A_scr", "S_ion_z", "prior", "w_env", "u")

            def _closure_tuple(ne, cv):
                out = _closure(ne, cv)
                return tuple(out[k] for k in keys)

            vals = _ckpt(_closure_tuple, n_e_density, cvhar3, use_reentrant=False)
            clo = dict(zip(keys, vals))
        else:
            clo = _closure(n_e_density, cvhar3)

        cvhar_z = cvhar3.mean(dim=(0, 1))
        prof_ne_z = n_e_values.mean(dim=(0, 1))

        # upsample the solver inputs to dz <= 0.1 A (measured requirement)
        f = self.solve_upsample
        nz_s = nz * f
        cvhar_s = fourier_upsample(cvhar_z, f)
        s_ion_s = torch.clamp(fourier_upsample(clo["S_ion_z"], f), min=0.0)
        a_s = torch.clamp(fourier_upsample(clo["A_scr"], f), min=0.0)
        prior_s = fourier_upsample(clo["prior"], f)
        w_s = torch.clamp(fourier_upsample(clo["w_env"], f), min=0.0)
        u_s = torch.cumsum(w_s, dim=0)
        u_s = u_s / torch.clamp(u_s[-1], min=1.0e-30)

        # residual head (v1): coefficients from zone-pooled features
        delta_stats = {"c_absmax": 0.0, "dp_rms": 0.0}
        if head is not None and node_feats is not None:
            lz_t = float(cell_np[2, 2])
            z_atoms = pos_frac[:, 2] * lz_t
            z_grid_c = torch.arange(nz, device=device, dtype=dt) * (length_z / nz)
            coeff = head.coefficients(
                node_feats, z_atoms, clo["w_env"], clo["u"], clo["prior"],
                z_grid_c, length_z,
                q_tot if q_tot is not None else torch.tensor(total_charge, dtype=dt, device=device),
            )
            delta_p = head.delta_p(coeff, w_s, u_s, length_z)
            delta_stats = {
                "c_absmax": float(coeff.detach().abs().max()),
                "dp_rms": float(delta_p.detach().pow(2).mean().sqrt()),
            }
        else:
            delta_p = torch.zeros_like(prior_s)
        p_off = prior_s + delta_p

        val_dip_z = solute_dipole_z(prof_ne_z, pos_frac, z_valence.to(dt), cell64)
        q_sol = float(-total_charge)
        solver = self._solver_for(cell_np, nz_s, device)
        center_z = 0.5 * (cell_np[0, 2] + cell_np[1, 2] + cell_np[2, 2])
        nouth = nz_s // 2
        indmin = int((nouth + int(0.5 * nz_s) + 10 * nz_s) % nz_s + 1)
        out = solver.solve(
            cvhar_z=cvhar_s, s_ion=s_ion_s, a1=a_s, p_off=p_off, q_sol=q_sol,
            val_ion_dipole_z=val_dip_z, c_unit=self._c_unit(cell_np),
            center_z=center_z, indmin=indmin,
            fixsol_steps=self.fixsol_steps, tol=self.tol, max_outer=self.max_outer,
        )

        rho_ion_z = -(out["n_ion"] / volume)
        rho_bound_z = -(out["n_b"] / volume)
        dz = length_z / nz_s
        area = volume / length_z
        z = solver.z
        q_ion_t = rho_ion_z.sum() * dz * area
        denom = torch.where(torch.abs(q_ion_t) > 1.0e-12, q_ion_t,
                            torch.full_like(q_ion_t, 1.0e-12))
        layer_mean_t = (rho_ion_z * z).sum() * dz * area / denom
        mu_bound_t = (rho_bound_z * z).sum() * dz * area

        self.last_diagnostics = {
            "rms_last": float(out["rms_last"]),
            "n_outer": int(out["n_outer"]),
            "q_ion": float(q_ion_t.detach()),
            "layer_mean": float(layer_mean_t.detach()),
            "mu_bound": float(mu_bound_t.detach()),
            **delta_stats,
        }
        return {
            "z": z,
            "phi_z": out["phi"],
            "rho_ion_z": rho_ion_z,
            "rho_bound_z": rho_bound_z,
            "rho_layer_z": rho_ion_z + rho_bound_z,
            "height": length_z,
            "q_ion": float(q_ion_t.detach()),
            "layer_mean": float(layer_mean_t.detach()),
            "mu_bound": float(mu_bound_t.detach()),
            "q_ion_t": q_ion_t,
            "layer_mean_t": layer_mean_t,
            "mu_bound_t": mu_bound_t,
            "rms_last": float(out["rms_last"]),
            "prior_solve": prior_s,
            "delta_p": delta_p,
        }
