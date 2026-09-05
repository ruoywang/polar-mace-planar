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
        self._timing_on = bool(os.environ.get("MACE_PB1D_TIMING"))
        self.timings: Dict[str, float] = {}
        self.timing_calls = 0
        self._bl_arr = None
        self._bl_index: Dict[int, int] = {}
        self._bl_shape = None
        # lazy RAM cache of the two used fields (neutral, phi_base) as f32;
        # kills the ~80 ms/graph disk re-read measured in real training
        # (3 ranks hammering one 14 GB file). ~24 MB/sample.
        self._bl_ram: Dict[int, torch.Tensor] = {}
        self._rt_tables_path = None
        self._rt_tables = None
        self._bl_ram_max = int(os.environ.get("MACE_PB1D_PRELOAD_MAX", "512"))
        if os.environ.get("MACE_PB1D_NO_PRELOAD"):
            self._bl_ram_max = 0
        if baseline_cache:
            bl = os.path.expandvars(os.path.expanduser(baseline_cache))
            with open(os.path.join(bl, "baseline_index.json")) as f:
                self._bl_index = {int(k): int(v) for k, v in json.load(f).items()}
            with open(os.path.join(bl, "baseline_meta.json")) as f:
                meta = json.load(f)
            self._bl_shape = tuple(meta["pb_shape"])
            fields = list(meta.get("fields", ["neutral_values", "dencor_values", "phi_base_eV"]))
            self._bl_nfields = len(fields)
            self._bl_take = [fields.index("neutral_values"), fields.index("phi_base_eV")]
            self._bl_arr = np.load(
                os.path.join(bl, "baseline_cache.npy"), mmap_mode="r"
            )
            rt_path = os.path.join(bl, "runtime_baseline_tables.npz")
            self._rt_tables_path = rt_path if os.path.exists(rt_path) else None

    def _get_runtime_baseline(self):
        if self._rt_tables is None and self._rt_tables_path is not None:
            from .pb1d_runtime_baseline import RuntimeBaselineTables

            self._rt_tables = RuntimeBaselineTables(self._rt_tables_path)
        return self._rt_tables

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

    class _Phase:
        """Synced wall-clock timer accumulating into backend.timings.
        Gated by MACE_PB1D_TIMING; a no-op otherwise (zero overhead)."""
        def __init__(self, backend, name, device):
            self.backend, self.name, self.device = backend, name, device

        def __enter__(self):
            if self.backend._timing_on:
                import time
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                self.t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            if self.backend._timing_on:
                import time
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - self.t0
                self.backend.timings[self.name] = (
                    self.backend.timings.get(self.name, 0.0) + dt)
            return False

    def timing_report(self, reset=True):
        """Return {phase: total_ms} averaged per call; optionally reset."""
        n = max(1, self.timing_calls)
        rep = {k: 1e3 * v / n for k, v in sorted(self.timings.items())}
        rep["_calls"] = self.timing_calls
        if reset:
            self.timings = {}
            self.timing_calls = 0
        return rep

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
        node_z: Optional[torch.Tensor] = None,
        node_feats: Optional[torch.Tensor] = None,
        head=None,
        q_tot: Optional[torch.Tensor] = None,
        ckpt_closure: bool = True,
        probe_points: Optional[torch.Tensor] = None,
        s3d_coeffs: Optional[torch.Tensor] = None,
        s3d_sigmas=None,
        s3d_energy: bool = False,
        cav_energy: bool = False,
    ) -> Dict[str, torch.Tensor]:

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
        use_runtime_baseline = False
        if bl_row is None:
            rt = self._get_runtime_baseline()
            if rt is None or node_z is None:
                raise RuntimeError(
                    f"pb1d baseline unavailable for sample_id={sample_id} "
                    f"(shape {shape}): no cache row, and runtime tables "
                    f"{'missing' if rt is None else 'need node_z'}"
                )
            if not rt.matches(cell_np, shape):
                raise RuntimeError(
                    "runtime baseline tables were built for a different "
                    f"cell/grid (need shape {shape})"
                )
            use_runtime_baseline = True

        self.timing_calls += 1
        cell64 = torch.as_tensor(cell_np, device=device, dtype=dt)
        pos64 = positions.to(dt)
        pos_frac = torch.remainder(pos64 @ torch.linalg.inv(cell64), 1.0)

        if use_runtime_baseline:
            with self._Phase(self, "1_baseline", device):
                neutral_v, phi_base = self._rt_tables.fields(
                    pos_frac.detach(), node_z, device
                )
        else:
          with self._Phase(self, "1_baseline", device):
            ram = self._bl_ram.get(sample_id)
            if ram is None:
                ram = torch.from_numpy(
                    np.ascontiguousarray(self._bl_arr[bl_row][self._bl_take])
                )
                if len(self._bl_ram) < self._bl_ram_max:
                    if device.type == "cuda":
                        ram = ram.pin_memory()
                    self._bl_ram[sample_id] = ram
            fields = ram.to(device, non_blocking=True).to(dt)
            neutral_v, phi_base = fields[0], fields[1]

        # checkpointed when grad is needed: keeping the full-grid graph
        # resident OOMs a 40 GB A100 (measured); the ~85 ms recompute is cheap
        want_grad = bool(radial_coeffs.requires_grad or (
            node_feats is not None and node_feats.requires_grad))

        with self._Phase(self, "2_assembly", device):
            def _assemble(coeffs):
                return self._gto_net_density_g(grid, pos_frac.detach(), coeffs.to(dt), sigmas)

            if want_grad:
                from torch.utils.checkpoint import checkpoint as _ckpt
                net_g = _ckpt(_assemble, radial_coeffs, use_reentrant=False)
            else:
                net_g = _assemble(radial_coeffs)
        with self._Phase(self, "3_poisson", device):
            net_values = grid.ifft_real(net_g)
            n_e_values = neutral_v - net_values
            n_e_density = torch.clamp(n_e_values / volume, min=0.0)
            cvhar3 = phi_base - grid.ifft_real(grid.l0_inv_op(net_g))

        with self._Phase(self, "4_closure", device):
            # ckpt_closure=False (training steps): keep the closure graph
            # resident (~0.7 GB, freed at backward) — saves the ~100 ms
            # recompute. ckpt_closure=True (eval / default): checkpoint it —
            # the eval loop retains per-structure graphs long enough that
            # uncheckpointed closures OOM a 40 GB card (measured, prod400).
            if want_grad and ckpt_closure:
                from torch.utils.checkpoint import checkpoint as _ckpt
                keys = ("A_scr", "S_ion_z", "prior", "w_env", "u")

                def _closure_tuple(ne, cv):
                    out = closure_from_fields(ne, cv, grid, self.params, self._tp)
                    return tuple(out[k] for k in keys)

                vals = _ckpt(_closure_tuple, n_e_density, cvhar3, use_reentrant=False)
                clo = dict(zip(keys, vals))
            else:
                clo = closure_from_fields(n_e_density, cvhar3, grid, self.params, self._tp)

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
        phi_init = None
        if os.environ.get("MACE_PB1D_PHI_WARM") and sample_id is not None:
            stash = getattr(self, "_phi_warm_stash", None)
            if stash is not None and stash[0] == (sample_id, nz_s):
                phi_init = stash[1]
        with self._Phase(self, "5_solve1d", device):
            out = solver.solve(
                cvhar_z=cvhar_s, s_ion=s_ion_s, a1=a_s, p_off=p_off, q_sol=q_sol,
                val_ion_dipole_z=val_dip_z, c_unit=self._c_unit(cell_np),
                center_z=center_z, indmin=indmin,
                fixsol_steps=self.fixsol_steps, tol=self.tol, max_outer=self.max_outer,
                grad_passes=int(os.environ.get("MACE_PB1D_GRAD_PASSES", "1")),
                phi_init=phi_init,
            )
        if os.environ.get("MACE_PB1D_PHI_WARM") and sample_id is not None:
            self._phi_warm_stash = ((sample_id, nz_s), out["phi"].detach().clone())

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
        if self._timing_on:
            every = int(os.environ.get("MACE_PB1D_TIMING_EVERY", "100"))
            if self.timing_calls % every == 0:
                snap = {k: 1e3 * v / self.timing_calls
                        for k, v in sorted(self.timings.items())}
                total = sum(snap.values())
                msg = " ".join(f"{k}={v:.1f}" for k, v in snap.items())
                print(f"PB1DTIMING calls={self.timing_calls} "
                      f"fwd_total={total:.1f}ms/graph {msg}", flush=True)
        # ---- stage-2 energy terms -------------------------------------
        # Lagged-SCF convention (same as the 1-D compensation energy and the
        # dipole delta): the solvent state enters DETACHED; the only live
        # factor is cvhar3 in the solute-cross term (gradient reaches the
        # density coefficients, exactly like e1d's cross term). Units are
        # physical throughout: densities e/A^3, potentials eV via the
        # explicit spectral Poisson (validated against the DFT-side sizing
        # script). Grid GTO assembly = point evaluator * volume (measured
        # 2026-09-05, ratio/V = 1.0003; the 3e-4 is periodic-image tails).
        e_cav_t: Optional[torch.Tensor] = None
        e_s3d_t: Optional[torch.Tensor] = None
        if cav_energy or (s3d_energy and s3d_coeffs is not None):
            cav = getattr(grid, "_solv3d_cavity", None)
            if cav is not None:
                from .solvent3d import (
                    _grad_mag_periodic,
                    poisson_phi_periodic,
                    project_zero_total,
                )
                s_ion3_d, s_diel3_d = cav
                gsd3 = _grad_mag_periodic(s_diel3_d, cell64)
                dV = volume / float(gsd3.numel())
                if cav_energy:
                    # VASPsol++ CREATECAVITY (solvation.F): the area integrand
                    # is |grad S| of the UN-convolved shape function of the
                    # bare electron density (s_vdw, not s_diel), with the
                    # LVAC solvent-box mask applied AFTER the gradient:
                    # Acav = TAU * sum M*|grad s_vdw| * dV
                    p = self.params
                    ne_d = n_e_density.detach()
                    x_vdw = torch.log(torch.clamp(
                        ne_d / float(p["NC_K"]), min=float(p["N_MIN"])))
                    s_vdw = self._tp._shape_func(x_vdw, float(p["SIGMA_K"]))
                    gsv = _grad_mag_periodic(s_vdw, cell64)
                    if bool(p["LVAC"]) and float(p["SOL_Z1"]) > float(p["SOL_Z0"]):
                        m_sol = self._tp._smooth_box(
                            grid, float(p["SOL_Z0"]), float(p["SOL_Z1"]),
                            float(p["SOL_SIGMA"]),
                        ).clamp(0.0, 1.0)
                        gsv = gsv * m_sol
                    e_cav_t = float(p["TAU"]) * gsv.sum() * dV
                if s3d_energy and s3d_coeffs is not None:
                    with torch.no_grad():
                        env_b3 = gsd3 / torch.clamp(gsd3.max(), min=1.0e-30)
                        env_i3 = torch.clamp(s_ion3_d, 0.0, 1.0)
                        mb3 = grid.ifft_real(self._gto_net_density_g(
                            grid, pos_frac.detach(),
                            s3d_coeffs[:, 0].detach().to(dt), s3d_sigmas,
                        )) / volume
                        mi3 = grid.ifft_real(self._gto_net_density_g(
                            grid, pos_frac.detach(),
                            s3d_coeffs[:, 1].detach().to(dt), s3d_sigmas,
                        )) / volume
                        delta_b3 = project_zero_total(env_b3 * mb3, env_b3)
                        delta_i3 = project_zero_total(env_i3 * mi3, env_i3)
                        delta3 = delta_b3 + delta_i3
                        # self energy of the residual
                        phi_d3 = poisson_phi_periodic(delta3, cell64)
                        e_self = 0.5 * (delta3 * phi_d3).sum() * dV
                        # cross with the 1-D solvent profile (plane content)
                        delta_pl = delta3.mean(dim=(0, 1))
                        rho_l1 = (rho_ion_z + rho_bound_z).detach()
                        phi_l1 = poisson_phi_periodic(
                            rho_l1.view(1, 1, -1), cell64).view(-1)
                        f_up = rho_l1.shape[0] // delta_pl.shape[0]
                        e_x1d = (delta_pl * phi_l1[::f_up]).sum() * (
                            volume / float(delta_pl.shape[0]))
                    # cross with the solute potential: cvhar3 stays LIVE
                    # (VASP potential-energy-of-electron convention -> the
                    # electrostatic potential of the solute is -cvhar3;
                    # sign validated by the label-charge unit test)
                    e_xsol = -(delta3 * cvhar3).sum() * dV
                    e_s3d_t = e_xsol + e_x1d + e_self

        # solvent3d probe: detached envelopes + 1-D baselines at the sampled
        # label points (supervision only; no gradient path through the solve)
        solv3d = None
        if probe_points is not None:
            cav = getattr(grid, "_solv3d_cavity", None)
            if cav is not None:
                from .solvent3d import solvent3d_probe_fields
                with torch.no_grad():
                    solv3d = solvent3d_probe_fields(
                        s_ion3=cav[0], s_diel3=cav[1], cell=cell64,
                        points=probe_points.to(dt),
                        rho_bound_z=rho_bound_z.detach(),
                        rho_ion_z=rho_ion_z.detach(),
                    )
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
            "solv3d": solv3d,
            "e_cav": e_cav_t,
            "e_s3d": e_s3d_t,
        }
