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
        bl_energy: bool = False,
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
        # MACE_PB1D_DFORCE: energy-derivative-force mode (user experiment
        # 2026-09-07) — forces become the derivative of the energy through
        # the 1-D solvent response. The frozen per-sid baseline cannot carry
        # position gradients (measured artifact forces), so this mode uses
        # the runtime tables even when a cache row exists, and keeps
        # positions LIVE into the solve inputs below. Remaining truncations:
        # SCF features (scheme C cache) and layer_mean (float).
        live_resp = bool(os.environ.get("MACE_PB1D_DFORCE"))
        if live_resp and bl_row is not None:
            rt = self._get_runtime_baseline()
            if rt is not None and node_z is not None and rt.matches(cell_np, shape):
                bl_row = None
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
        # positions into the SOLVE inputs: detached by default (lagged-force
        # convention); LIVE in energy-derivative-force mode
        pf_in = pos_frac if live_resp else pos_frac.detach()

        if use_runtime_baseline:
            with self._Phase(self, "1_baseline", device):
                neutral_v, phi_base = self._rt_tables.fields(
                    pf_in, node_z, device
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
                return self._gto_net_density_g(grid, pf_in, coeffs.to(dt), sigmas)

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
            z_atoms = pf_in[:, 2] * lz_t
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

        val_dip_z = solute_dipole_z(
            prof_ne_z, pf_in, z_valence.to(dt), cell64)
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
        ion_dipole_t = (rho_ion_z * z).sum() * dz * area
        # layer_mean = ionic-layer center. The charge-weighted mean
        # (dipole/charge) is undefined at q_ion ~ 0 (neutral solvated
        # frames: the +/- ion layers cancel), so switch to the
        # |rho|-weighted center there; it always lies inside [0, H].
        # mu must use ion_dipole_t directly, never q_ion*layer_mean.
        # (ported from pmp-trainall; the old 1e-12-clamp version made the
        # health gate reject EVERY neutral solvated frame -> silent
        # planar fallback throughout the s3d gate and 500-ep production)
        q_abs_t = torch.abs(rho_ion_z).sum() * dz * area
        if float(torch.abs(q_ion_t)) > 1.0e-3:
            layer_mean_t = ion_dipole_t / q_ion_t
        elif float(q_abs_t) > 1.0e-9:
            layer_mean_t = (torch.abs(rho_ion_z) * z).sum() * dz * area / q_abs_t
        else:
            layer_mean_t = q_ion_t.detach() * 0.0 + 0.5 * length_z
        mu_bound_t = (rho_bound_z * z).sum() * dz * area

        # ---- baseline-coupling energy (audit 2026-09-08): the compensation
        # term couples only the NET model charge to the solvent profile; the
        # BASELINE solute charge x solvent coupling was missing (measured
        # -0.41 eV on the neutral cal vs its -0.62 eV 1-D gap). Definition:
        # E_bl = sum rho_solv(z) * (-<phi_base>(z)) * A dz with a ZERO-MEAN
        # potential — the G0 piece on charged frames (q_ion x reference) is
        # deliberately excluded until the potential-reference / dipole
        # bookkeeping audit settles its owner. Disjoint from the compensation
        # term (net-only) by construction. Lagged by default; LIVE under
        # MACE_PB1D_DFORCE like every other solve consumer.
        e_bl_t: Optional[torch.Tensor] = None
        if bl_energy:
            pbz_s = fourier_upsample(phi_base.mean(dim=(0, 1)), f)
            pbz_s = pbz_s - pbz_s.mean()
            e_bl_raw = ((rho_ion_z + rho_bound_z) * (-pbz_s)).sum() * dz * area
            e_bl_t = e_bl_raw if live_resp else e_bl_raw.detach()

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
        # ---- stage-2 energy terms (live, single checkpointed rebuild) ----
        # One rebuild carries every per-step-differentiable dependency:
        # density coefficients AND explicit atom positions (run_graphs passes
        # LIVE positions; each pre-existing solve consumer detaches itself)
        # -> net density -> cavity (E_cav area + envelopes) -> delta ->
        # couplings. Gradients reach the trunk through the coefficients
        # (network path) and through the GTO anchors (explicit path).
        # DECLARED frozen inputs (not recomputed per step, architecture-wide,
        # NOT covered by the returned forces): the per-sid neutral baseline
        # fields (neutral_v, phi_base) and the 1-D solve outputs inside the
        # 1-D cross term (lagged-SCF, same treatment as e1d).
        # Units physical throughout (validated: grid = point evaluator * V
        # to 3e-4; Poisson helper = DFT-side sizing script to 0.1%; E_cav
        # formula bitwise vs the VASPsol++ print, 433.82 = 433.82).
        e_cav_t: Optional[torch.Tensor] = None
        e_s3d_t: Optional[torch.Tensor] = None
        s3d_obs: Optional[Dict[str, torch.Tensor]] = None
        do_s3d = bool(s3d_energy and s3d_coeffs is not None)
        if cav_energy or do_s3d:
            from torch.utils.checkpoint import checkpoint as _ckpt2
            from .solvent3d import (
                normalized_gradient_envelope, poisson_phi_periodic)
            dV = volume / float(grid.ngrid)
            p = self.params
            m_ion3 = None
            if bool(p["LVAC"]) and float(p["SOL_Z1"]) > float(p["SOL_Z0"]):
                m_ion3 = self._tp._smooth_box(
                    grid,
                    float(p["SOL_Z0"]) + float(p["D_STERN"]),
                    float(p["SOL_Z1"]) - float(p["D_STERN"]),
                    float(p["SOL_SIGMA"]),
                )
            cb = s3d_coeffs[:, 0].to(dt) if do_s3d else pos_frac.new_zeros(0)
            ci = s3d_coeffs[:, 1].to(dt) if do_s3d else pos_frac.new_zeros(0)
            nz_pl = int(grid.shape[2])
            z_pl = torch.arange(nz_pl, device=pos_frac.device, dtype=dt) * (
                length_z / float(nz_pl))
            # frozen-envelope copies (the loss's convention): used for the
            # SUPERVISION-side projection ratio, so the loss's backward is
            # the exact derivative of its own frozen-envelope forward while
            # the energy keeps its fully-live ratio (values identical)
            cavd = getattr(grid, "_solv3d_cavity", None)
            env_bf = None
            env_if = None
            if do_s3d and cavd is not None:
                env_bf = normalized_gradient_envelope(cavd[1], cell64)
                env_if = torch.clamp(cavd[0], 0.0, 1.0)

            def _stage2_energy(cfd, cbt, cit, pf):
                # runtime mode: the baseline tables are pure-torch structure
                # factors -> regenerate the baseline LIVE in positions; the
                # per-sid cached baseline (training) stays the declared
                # frozen input
                if use_runtime_baseline:
                    neutral_e, phi_base_e = self._rt_tables.fields(
                        pf, node_z, pf.device)
                else:
                    neutral_e, phi_base_e = neutral_v, phi_base
                net_g2 = self._gto_net_density_g(grid, pf, cfd.to(dt), sigmas)
                ne2 = torch.clamp(
                    (neutral_e - grid.ifft_real(net_g2)) / volume, min=0.0)
                # cavity: LIVE only in runtime mode (Vcav-analog feedback and
                # cavity force with the differentiable baseline, tier-ii
                # validated). In training the cavity enters as values —
                # exactly the passing-gate (3417982/3419003-training-side)
                # semantics; the live-in-training variants degraded the
                # trunk (gates 3419752/3419977 F 62-81 vs 35 meV/A).
                ne_cav = ne2 if use_runtime_baseline else ne2.detach()
                s_ion3e, s_diel3e, s_cav3e = self._tp.create_cavity_torch(
                    ne_cav, grid, p)
                if m_ion3 is not None:
                    s_cav3e = s_cav3e * m_ion3
                # VASPsol++ solvation_nlpcm CREATECAVITY (solvation.F 1984):
                # A_cav = TAU * int |grad S_cav| dV, Stern mask doubled,
                # spectral gradient. |grad| recomposed with an eps floor:
                # sqrt(0) on the saturated plateaus has an infinite backward
                gx, gy, gz, _ = grid.grad_from_recip(grid.fft(s_cav3e))
                area = torch.sqrt(gx * gx + gy * gy + gz * gz + 1.0e-30).sum()
                if not do_s3d:
                    zero = area.new_zeros(())
                    zv = area.new_zeros(nz_pl)
                    zg = area.new_zeros((1, 1, 1))
                    return (area, zero, zero, zv, zv, zv, zv, zg, zg,
                            zg, zg, zg, zg)
                # LIVE envelopes (value-identical to the loss's detached
                # stash; here the density response is in the graph)
                env_b = normalized_gradient_envelope(s_diel3e, cell64)
                env_i = torch.clamp(s_ion3e, 0.0, 1.0)
                mb = grid.ifft_real(self._gto_net_density_g(
                    grid, pf, cbt, s3d_sigmas)) / volume
                mi = grid.ifft_real(self._gto_net_density_g(
                    grid, pf, cit, s3d_sigmas)) / volume
                raw_b = env_b * mb
                raw_i = env_i * mi
                # PER-PLANE projection (user design 2026-09-07): every z
                # plane of the residual sums to zero, so the 1-D pipeline
                # owns ALL plane-averaged content BY CONSTRUCTION and the
                # residual is purely lateral. Kills the loss-blind plane-
                # dipole junk at the source (measured -6..-14 e*A, 97%
                # spurious per the dipole-consistency check 3420722);
                # subsumes the scalar charge projection; the full 3-D
                # charge's dipole now equals the solver dipole exactly.
                # Env-weighted, so the correction lives inside the envelope
                # (no charge deposited on vacuum planes).
                r_b = raw_b.mean(dim=(0, 1)) / torch.clamp(
                    env_b.mean(dim=(0, 1)), min=1.0e-12)
                r_i = raw_i.mean(dim=(0, 1)) / torch.clamp(
                    env_i.mean(dim=(0, 1)), min=1.0e-12)
                # supervision-side PROJECTED GRID FIELDS (frozen envelopes,
                # live m): the loss interpolates THESE at its sampled points,
                # so loss and energy score literally the same discrete
                # projected field. Subtracting an interpolated r(z) from a
                # directly-evaluated GTO density is NOT equivalent between
                # planes (a pure plane mode is annihilated on the grid but
                # not at off-plane points — user-reproduced 2026-09-07).
                if env_bf is not None:
                    r_sup_b = (env_bf * mb).mean(dim=(0, 1)) / torch.clamp(
                        env_bf.mean(dim=(0, 1)), min=1.0e-12)
                    r_sup_i = (env_if * mi).mean(dim=(0, 1)) / torch.clamp(
                        env_if.mean(dim=(0, 1)), min=1.0e-12)
                    d_sup_b = env_bf * mb - r_sup_b[None, None, :] * env_bf
                    d_sup_i = env_if * mi - r_sup_i[None, None, :] * env_if
                else:
                    r_sup_b, r_sup_i = r_b, r_i
                    d_sup_b = raw_b - r_b[None, None, :] * env_b
                    d_sup_i = raw_i - r_i[None, None, :] * env_i
                # LAGGED delta in the energy integrals (same convention as
                # the 1-D solvent state; evidence matrix: five live-delta
                # gates unstable, value-coupled gates PASS; forensics
                # 3419335/3419347/3419395). Values exact and per-step
                # tracking; gradients flow through cvhar_e. The omitted
                # delta-anchor force channel is measured (12-36 meV/A total
                # MD-path gap, job 3420723).
                delta_b = raw_b - r_b[None, None, :] * env_b
                delta_i = raw_i - r_i[None, None, :] * env_i
                if not os.environ.get("MACE_S3D_LIVE_DELTA"):
                    # MACE_S3D_LIVE_DELTA is a MEASUREMENT-ONLY toggle for
                    # cost probes (wall time / peak memory of the live-delta
                    # response channel); production always trains lagged
                    # (five live-delta gates unstable, forensics 3419335+)
                    delta_b = delta_b.detach()
                    delta_i = delta_i.detach()
                delta = delta_b + delta_i
                e_self_raw = 0.5 * (delta * poisson_phi_periodic(delta, cell64)).sum()
                # solute cross: cvhar rebuilt from the same live assembly
                # (VASP electron-PE convention -> potential = -cvhar; sign
                # validated by the label-charge unit test). The 1-D cross
                # term vanishes identically under the per-plane projection.
                cvhar_e = phi_base_e - grid.ifft_real(grid.l0_inv_op(net_g2))
                e_xsol_raw = -(delta * cvhar_e).sum()
                # MACE_S3D_EXPORT_DELTA: diagnostics-only export of the exact
                # energy-side residual on the ENERGY grid. d_sup_* live on the
                # upsampled supervision grid, so they are the wrong field to
                # score energy-side statistics against. Off in production
                # (returns a 1-element dummy, so the tuple shape is fixed).
                # bound and ionic residuals are exported SEPARATELY: they
                # belong to different physics (bound charge is -div P, the
                # ionic charge is the diffuse screening layer) and a
                # diagnostic that merges them cannot be compared against
                # RHOB and RHOION as separate references.
                if os.environ.get("MACE_S3D_EXPORT_DELTA"):
                    d_out_b, d_out_i = delta_b.detach(), delta_i.detach()
                    # mb and mi are the coefficient-built GTO densities BEFORE
                    # any envelope multiplies them, so they are
                    # envelope-independent. Exporting them is what makes
                    # "same coefficients, different envelope" exact rather
                    # than reconstructed.
                    m_out_b, m_out_i = mb.detach(), mi.detach()
                else:
                    z1 = delta.new_zeros((1, 1, 1))
                    d_out_b, d_out_i = z1, z1
                    m_out_b, m_out_i = z1, z1
                return (area, e_xsol_raw, e_self_raw,
                        delta_b.mean(dim=(0, 1)), delta_i.mean(dim=(0, 1)),
                        r_sup_b, r_sup_i, d_sup_b, d_sup_i, d_out_b, d_out_i,
                        m_out_b, m_out_i)

            # positions in the energy assemblies: LIVE only in runtime mode,
            # where the baseline itself is differentiable and the position
            # derivatives are physical (tier-ii FD: 0.2-1.4%). In cached-
            # baseline (training) mode the frozen per-sid baseline makes the
            # grid position derivatives artifactual (tier-i FD measured up
            # to ~1 eV/A of frozen-baseline response), and gate 3419752
            # showed the force loss degrading (F 62-81 vs 35 meV/A) as the
            # growing delta rode that channel -> detach positions there,
            # exactly the passing-gate training physics.
            pf_e = pos_frac if use_runtime_baseline else pos_frac.detach()
            live = bool(want_grad or pf_e.requires_grad)
            if live:
                outs = _ckpt2(_stage2_energy, radial_coeffs, cb, ci,
                              pf_e, use_reentrant=False)
            else:
                outs = _stage2_energy(radial_coeffs, cb, ci, pf_e)
            (area, e_xsol_raw, e_self_raw, d_b_pl, d_i_pl,
             r_sup_b, r_sup_i, d_sup_b, d_sup_i,
             d_grid_b, d_grid_i, m_grid_b, m_grid_i) = outs
            if cav_energy:
                e_cav_t = float(p["TAU"]) * area * dV
            if do_s3d:
                # per-plane projection makes the residual's plane content —
                # and hence the 1-D cross term and its dipole — identically
                # zero; d_*_pl and mu_delta stay exported as diagnostics
                # (should read ~1e-15)
                delta_pl = d_b_pl + d_i_pl
                e_s3d_t = (e_xsol_raw + e_self_raw) * dV
                s3d_obs = {
                    "delta_b_pl": d_b_pl,
                    "delta_i_pl": d_i_pl,
                    "mu_delta": (delta_pl * z_pl).sum() * (volume / float(nz_pl)),
                    "r_sup_b": r_sup_b,
                    "r_sup_i": r_sup_i,
                    # loss-facing projected grid fields (LIVE in m)
                    "d_sup_b": d_sup_b,
                    "d_sup_i": d_sup_i,
                    # diagnostics only (detached floats): energy split
                    "e_xsol": float(e_xsol_raw.detach()) * dV,
                    "e_self": float(e_self_raw.detach()) * dV,
                }
                if os.environ.get("MACE_S3D_EXPORT_DELTA"):
                    s3d_obs["delta_b_grid"] = d_grid_b
                    s3d_obs["delta_i_grid"] = d_grid_i
                    s3d_obs["delta_grid"] = d_grid_b + d_grid_i
                    s3d_obs["m_b_grid"] = m_grid_b
                    s3d_obs["m_i_grid"] = m_grid_i

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
            "ion_dipole_t": ion_dipole_t,
            "layer_mean_t": layer_mean_t,
            "mu_bound_t": mu_bound_t,
            "rms_last": float(out["rms_last"]),
            # convergence PROVENANCE, not just the residual: iteration counts,
            # the caps they were tested against, and each loop's exit reason.
            # A small residual is consistent with converging on the criterion
            # AND with hitting the cap, so the residual alone is not evidence.
            "n_outer": int(out["n_outer"]) if out.get("n_outer") is not None
            else None,
            "solver_exit": out.get("solver_exit"),
            "prior_solve": prior_s,
            "delta_p": delta_p,
            "solv3d": solv3d,
            "e_cav": e_cav_t,
            "e_bl": e_bl_t,
            "e_s3d": e_s3d_t,
            "s3d_obs": s3d_obs,
        }
