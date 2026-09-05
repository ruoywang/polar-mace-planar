"""Residual-3D solvent charge: head, labels, loss (supervision-only stage).

Model: rho_ch(r) = rho_ch_1d(z) [detached broadcast of the final PB solve]
               + env_ch(r) [detached: bound |grad s_diel3|, ion s_ion3]
               * sum_atoms basis_(sigma,lm)(r - R_a) . c[a, ch, sigma, lm]
Channels: 0 = bound, 1 = ionic (output gated by total charge -> exactly zero
on neutral frames). Coefficients come from an equivariant linear readout of
the mixed node features; the ONLY gradient path of this loss into the model
is coefficients -> node features -> trunk. Baselines and envelopes are
detached in the backend probe (pb1d_backend.solve_graph, probe_points).

Labels: per-sid full-grid npy pair in the PHYSICS convention (e/A^3,
rho = -RHOB_or_RHOION/V_cell, C-order (nz, ny, nx)), manifest JSON with
per-channel signal mean-squares for normalization (weight 1.0 = parity,
same scheme as the other unit-normalized loss terms).
"""
from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

try:
    from graph_longrange.realspace_grid_integrals import (
        tensor_realspace_GTO_evaluation,
    )
except ImportError:  # pragma: no cover - required by PolarMACE
    tensor_realspace_GTO_evaluation = None


def _get(ref, key: str):
    if isinstance(ref, dict):
        return ref.get(key)
    return getattr(ref, key, None)


# ---------------------------------------------------------------------------
# probe fields: envelopes + 1-D baselines at sampled points (called by the
# backend under no_grad; everything here is supervision-side, detached)
# ---------------------------------------------------------------------------

def _interp3_periodic(field: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
    """Trilinear periodic interpolation; field [nx,ny,nz], frac [P,3]."""
    n = torch.tensor(field.shape, device=frac.device, dtype=frac.dtype)
    g = frac * n
    i0 = torch.floor(g).long()
    w = g - i0.to(frac.dtype)
    out = torch.zeros(frac.shape[0], device=field.device, dtype=field.dtype)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                idx = (i0 + torch.tensor([dx, dy, dz], device=i0.device)) % n.long()
                ww = ((w[:, 0] if dx else 1 - w[:, 0])
                      * (w[:, 1] if dy else 1 - w[:, 1])
                      * (w[:, 2] if dz else 1 - w[:, 2]))
                out += ww.to(field.dtype) * field[idx[:, 0], idx[:, 1], idx[:, 2]]
    return out


def _interp1_periodic(prof: torch.Tensor, zfrac: torch.Tensor) -> torch.Tensor:
    n = prof.shape[0]
    g = zfrac * n
    i0 = torch.floor(g).long() % n
    w = (g - torch.floor(g)).to(prof.dtype)
    return (1 - w) * prof[i0] + w * prof[(i0 + 1) % n]


def _grad_mag_periodic(field: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Central-difference |grad| on a periodic grid, general cell
    (cart = frac @ cell); field [nx,ny,nz]."""
    inv = torch.linalg.inv(cell)
    gf = [
        (torch.roll(field, -1, dims=ax) - torch.roll(field, 1, dims=ax))
        * (field.shape[ax] / 2.0)
        for ax in range(3)
    ]
    out = torch.zeros_like(field)
    for j in range(3):
        gj = gf[0] * inv[j, 0] + gf[1] * inv[j, 1] + gf[2] * inv[j, 2]
        out += gj * gj
    return torch.sqrt(out)


def solvent3d_probe_fields(
    s_ion3: torch.Tensor,
    s_diel3: torch.Tensor,
    cell: torch.Tensor,
    points: torch.Tensor,
    rho_bound_z: torch.Tensor,
    rho_ion_z: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    frac = torch.remainder(torch.linalg.solve(cell.T, points.T).T, 1.0)
    gsd3 = _grad_mag_periodic(s_diel3, cell)
    env_b = _interp3_periodic(gsd3, frac)
    env_b = env_b / torch.clamp(gsd3.max(), min=1e-30)
    env_i = _interp3_periodic(s_ion3, frac).clamp(0.0, 1.0)
    base_b = _interp1_periodic(rho_bound_z, frac[:, 2])
    base_i = _interp1_periodic(rho_ion_z, frac[:, 2])
    return {"env_b": env_b, "env_i": env_i, "base_b": base_b, "base_i": base_i}


# ---------------------------------------------------------------------------
# energy-path field helpers (stage 2). Explicit spectral electrostatics with
# its own constants — deliberately independent of the VASP-unit grid class,
# validated against the DFT-side sizing script (exp_neutralsolv_prep).
# ---------------------------------------------------------------------------

FIELD_CONSTANT_EVA = 14.39964546866782  # e^2/(4 pi eps0), eV*A


def poisson_phi_periodic(rho: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Periodic Poisson potential (eV) of a charge density rho (e/A^3) on a
    general-cell grid [nx,ny,nz]; the G=0 mode is dropped (zero-mean phi).
    Convention: positive rho -> positive phi (energy of a +1e test charge is
    rho-weighted phi; the caller owns sign bookkeeping)."""
    import math as _math
    nx, ny, nz = rho.shape
    rg = torch.fft.rfftn(rho)
    B = 2.0 * _math.pi * torch.linalg.inv(cell).T.to(rho.dtype)
    fx = torch.fft.fftfreq(nx, device=rho.device, dtype=rho.dtype) * nx
    fy = torch.fft.fftfreq(ny, device=rho.device, dtype=rho.dtype) * ny
    fz = torch.arange(nz // 2 + 1, device=rho.device, dtype=rho.dtype)
    G = (fx[:, None, None, None] * B[0]
         + fy[None, :, None, None] * B[1]
         + fz[None, None, :, None] * B[2])
    G2 = (G * G).sum(-1)
    G2[0, 0, 0] = 1.0
    pg = 4.0 * _math.pi * FIELD_CONSTANT_EVA * rg / G2
    pg[0, 0, 0] = 0.0
    return torch.fft.irfftn(pg, s=rho.shape)


def project_zero_total(delta: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
    """Charge-conservation projection: remove the net charge of delta by
    subtracting (sum delta / sum env) * env — the envelope is the
    redistribution shape, so the correction lives where charge is allowed.
    Grid-uniform dV cancels from the ratio."""
    denom = torch.clamp(env.sum(), min=1.0e-30)
    return delta - (delta.sum() / denom) * env


# ---------------------------------------------------------------------------
# head
# ---------------------------------------------------------------------------

class Solvent3DChargeHead(torch.nn.Module):
    """Equivariant linear readout: mixed node feats -> GTO coefficients
    [n_atoms, 2 channels, n_sigma, (l_max+1)^2], e3nn m-order (the layout the
    existing density evaluators consume). Zero-initialized; ion channel gated
    by the frame's total charge."""

    OUT_SCALE = 100.0

    def __init__(self, node_feats_irreps, sigmas, cueq_config=None):
        super().__init__()
        from e3nn import o3
        from .wrapper_ops import Linear

        self.sigmas = [float(s) for s in sigmas]
        n_out = 2 * len(self.sigmas)
        self.n_out = n_out
        irreps_out = o3.Irreps(f"{n_out}x0e + {n_out}x1o + {n_out}x2e")
        self.linear = Linear(
            o3.Irreps(node_feats_irreps), irreps_out, cueq_config=cueq_config
        )
        for p in self.linear.parameters():
            torch.nn.init.zeros_(p)

    def forward(self, node_feats: torch.Tensor, q_gate: torch.Tensor) -> torch.Tensor:
        flat = self.linear(node_feats)
        n = flat.shape[0]
        blocks = flat.new_zeros(n, self.n_out, 9)
        off = 0
        for ell in range(3):
            w = 2 * ell + 1
            blocks[:, :, ell * ell:(ell + 1) * (ell + 1)] = \
                flat[:, off:off + self.n_out * w].view(n, self.n_out, w)
            off += self.n_out * w
        c = blocks.view(n, 2, len(self.sigmas), 9) * self.OUT_SCALE
        return torch.stack([c[:, 0], c[:, 1] * q_gate.view(-1, 1, 1)], dim=1)


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

class Solvent3DGridTargets:
    """Full-grid label pack: manifest JSON with per-sid npy pairs (physics
    convention e/A^3, (nz,ny,nx)) + lattice meta + per-channel signal_ms."""

    def __init__(self, manifest_path: Union[str, Path], max_cache_size: int = 32):
        self.manifest_path = Path(manifest_path)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != "solvent3d_grid_npy_v1":
            raise ValueError(f"Unsupported solvent3d manifest {self.manifest_path}")
        self.signal_ms = {
            "b": float(payload["signal_ms"]["b"]),
            "i": float(payload["signal_ms"]["i"]),
        }
        self.entries = {int(k): v for k, v in payload["entries"].items()}
        self.max_cache_size = int(max_cache_size)
        self._cache: OrderedDict = OrderedDict()

    def __contains__(self, sample_id: int) -> bool:
        return int(sample_id) in self.entries

    def _load(self, sample_id: int):
        sample_id = int(sample_id)
        if sample_id in self._cache:
            self._cache.move_to_end(sample_id)
            return self._cache[sample_id]
        entry = self.entries[sample_id]

        def _p(key):
            p = Path(entry[key])
            return p if p.is_absolute() else self.manifest_path.parent / p

        # mmap + sequential materialization — measured on the training nodes
        # (2026-08-31, $SCRATCH is BeeGFS):
        #   * random mmap faults (1k sampled points): 3-5 s/graph cold
        #   * plain read() (np.load without mmap): bypasses the client-side
        #     cache entirely -> re-fetches ~57 GB/epoch forever (14.5 min/ep)
        #   * mmap sequential materialize: 0.0-0.1 s/file, repeats 1-2 ms
        rho_b = np.asarray(np.load(_p("path_b"), mmap_mode="r"))
        rho_i = np.asarray(np.load(_p("path_i"), mmap_mode="r"))
        with np.load(_p("meta_path")) as meta:
            lattice = np.asarray(meta["lattice"], dtype=np.float64)
        item = (rho_b, rho_i, lattice)
        self._cache[sample_id] = item
        self._cache.move_to_end(sample_id)
        while len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)
        return item

    def sample_points(self, sample_id, n_points, rng, dtype, device):
        rho_b, rho_i, lattice = self._load(sample_id)
        nz, ny, nx = rho_b.shape
        rand = rng if rng is not None else random
        linear = np.asarray(rand.sample(range(nz * ny * nx), int(n_points)),
                            dtype=np.int64)
        iz = linear // (ny * nx)
        rem = linear - iz * ny * nx
        iy = rem // nx
        ix = rem - iy * nx
        frac = np.column_stack([ix / float(nx), iy / float(ny), iz / float(nz)])
        points = frac @ lattice
        return (
            torch.as_tensor(points, dtype=dtype, device=device),
            torch.as_tensor(rho_b[iz, iy, ix], dtype=dtype, device=device),
            torch.as_tensor(rho_i[iz, iy, ix], dtype=dtype, device=device),
        )


class Solvent3DPointsTargets:
    """Presampled point pack (solvent3d_points_npy_v1): per sid one npy of
    rows (x, y, z, ref_b, ref_i) f32, drawn uniformly at build time. Each
    access reads ONE contiguous window — ~20 KB, cold-cache cost ~ms — which
    is the structural fix for BeeGFS, where nothing retains the 65 GB
    full-grid pack between epochs (measured 10-30 min/epoch of pure IO in
    both random-fault and sequential-read modes, 2026-08-31)."""

    def __init__(self, manifest_path: Union[str, Path], max_open: int = 64):
        self.manifest_path = Path(manifest_path)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != "solvent3d_points_npy_v1":
            raise ValueError(f"Unsupported solvent3d points manifest {self.manifest_path}")
        self.signal_ms = {
            "b": float(payload["signal_ms"]["b"]),
            "i": float(payload["signal_ms"]["i"]),
        }
        self.n_points = int(payload["n_points"])
        self.entries = {int(k): v for k, v in payload["entries"].items()}
        self.max_open = int(max_open)
        self._open: OrderedDict = OrderedDict()

    def __contains__(self, sample_id: int) -> bool:
        return int(sample_id) in self.entries

    def _mmap(self, sample_id: int):
        sample_id = int(sample_id)
        if sample_id in self._open:
            self._open.move_to_end(sample_id)
            return self._open[sample_id]
        p = Path(self.entries[sample_id]["path"])
        p = p if p.is_absolute() else self.manifest_path.parent / p
        arr = np.load(p, mmap_mode="r")
        self._open[sample_id] = arr
        while len(self._open) > self.max_open:
            self._open.popitem(last=False)
        return arr

    def sample_points(self, sample_id, n_points, rng, dtype, device):
        arr = self._mmap(sample_id)
        n = arr.shape[0]
        rand = rng if rng is not None else random
        off = rand.randrange(0, max(n - int(n_points), 1))
        rows = np.asarray(arr[off:off + int(n_points)])
        t = torch.as_tensor(rows, dtype=dtype, device=device)
        return t[:, 0:3].contiguous(), t[:, 3].contiguous(), t[:, 4].contiguous()


def load_solvent3d_targets(path: Union[str, Path]):
    """Manifest-format dispatch: points pack (preferred) or full-grid pack."""
    fmt = json.loads(Path(path).read_text(encoding="utf-8")).get("format")
    if fmt == "solvent3d_points_npy_v1":
        return Solvent3DPointsTargets(path)
    return Solvent3DGridTargets(path)


def attach_solvent3d_samples_to_batch(batch, loss_fn) -> None:
    targets = getattr(loss_fn, "solvent3d_targets", None)
    samples_per_graph = int(getattr(loss_fn, "solvent3d_samples", 0))
    if not targets or samples_per_graph <= 0:
        return
    sample_ids = _get(batch, "sample_id")
    if sample_ids is None:
        return
    rng = getattr(loss_fn, "solvent3d_rng", None)
    positions = batch["positions"] if isinstance(batch, dict) else batch.positions
    pts, rb, ri, gi = [], [], [], []
    for graph_idx, sid_value in enumerate(sample_ids.view(-1)):
        sid = int(sid_value.detach().cpu().item())
        if sid not in targets:
            continue  # frames without solvent labels (e.g. vacuum neutrals)
        points, ref_b, ref_i = targets.sample_points(
            sid, samples_per_graph, rng, positions.dtype, positions.device
        )
        pts.append(points)
        rb.append(ref_b)
        ri.append(ref_i)
        gi.append(torch.full((points.shape[0],), graph_idx,
                             device=positions.device, dtype=torch.long))
    if not pts:
        return
    batch.solv3d_points = torch.cat(pts, dim=0)
    batch.solv3d_ref_b = torch.cat(rb, dim=0)
    batch.solv3d_ref_i = torch.cat(ri, dim=0)
    batch.solv3d_graph_index = torch.cat(gi, dim=0)


# ---------------------------------------------------------------------------
# residuals / loss
# ---------------------------------------------------------------------------

def _gto_channels_at_points(points, positions, cell, coeffs, sigmas):
    """Sum of atom-centered GTO contributions per channel at the points.
    coeffs [nat, 2, n_sig, 9] (e3nn m-order); returns [2, P]. Same evaluator
    and xy-min-image convention as the density_3d loss."""
    if tensor_realspace_GTO_evaluation is None:
        raise ImportError("graph_longrange is required for the solvent3d loss")
    point_frac = torch.linalg.solve(cell.T.to(points.dtype), points.T).T
    atom_frac = torch.linalg.solve(cell.T.to(positions.dtype), positions.T).T.to(points.dtype)
    disp_frac = point_frac[:, None, :] - atom_frac[None, :, :]
    disp_frac[..., 0:2] = disp_frac[..., 0:2] - torch.round(disp_frac[..., 0:2])
    disp_cart = torch.einsum("...i,ij->...j", disp_frac, cell.to(points.dtype))
    out = points.new_zeros(2, points.shape[0])
    zero = points.new_zeros(3)
    for si, sg in enumerate(sigmas):
        for ell in range(3):
            basis = tensor_realspace_GTO_evaluation(
                disp_cart, ell, float(sg), zero, normalize="multipoles")
            out = out + torch.einsum(
                "pnm,ncm->cp", basis,
                coeffs[:, :, si, ell * ell:(ell + 1) * (ell + 1)].to(points.dtype),
            )
    return out


def solvent3d_residuals(ref, pred, sigmas):
    """Per-channel residuals over all valid sampled points, or None."""
    coeffs = pred.get("solvent3d_coeffs")
    points = _get(ref, "solv3d_points")
    if coeffs is None or points is None:
        return None
    env_b, env_i = pred.get("solv3d_env_b"), pred.get("solv3d_env_i")
    base_b, base_i = pred.get("solv3d_base_b"), pred.get("solv3d_base_i")
    valid = pred.get("solv3d_valid")
    if env_b is None or valid is None:
        return None
    gidx = _get(ref, "solv3d_graph_index")
    ref_b, ref_i = _get(ref, "solv3d_ref_b"), _get(ref, "solv3d_ref_i")
    ptr = ref["ptr"] if isinstance(ref, dict) else ref.ptr
    cells = ref["cell"] if isinstance(ref, dict) else ref.cell
    num_graphs = int(ptr.numel() - 1)
    cells = cells.view(num_graphs, 3, 3) if cells.dim() != 3 else cells
    positions = ref["positions"] if isinstance(ref, dict) else ref.positions
    res_b, res_i = [], []
    for g in range(num_graphs):
        m = (gidx == g) & valid
        if not bool(m.any()):
            continue
        a0, a1 = int(ptr[g].item()), int(ptr[g + 1].item())
        pr = _gto_channels_at_points(
            points[m], positions[a0:a1], cells[g], coeffs[a0:a1], sigmas)
        res_b.append(base_b[m] + env_b[m] * pr[0] - ref_b[m])
        res_i.append(base_i[m] + env_i[m] * pr[1] - ref_i[m])
    if not res_b:
        return None
    return torch.cat(res_b), torch.cat(res_i)


def mean_squared_error_solvent3d(ref, pred, signal_ms, sigmas,
                                 ddp: Optional[bool] = None):
    """Per-channel signal-ms-normalized MSE (weight 1.0 = parity per channel).

    DDP: the scoreable point count is DATA-dependent per rank (frames without
    labels, fallback solves), so every rank joins the same collective and all
    return None together only when the global count is zero — a rank-local
    early return here deadlocks DDP (de84363 lesson). The zero-count branch
    keeps a graph connection through the coefficients."""
    import torch.distributed as dist

    coeffs = pred.get("solvent3d_coeffs")
    if coeffs is None:
        return None  # config-level absence: identical on every rank
    res = solvent3d_residuals(ref, pred, sigmas)
    if res is None:
        sq_sum = coeffs.sum() * 0.0
        n_local = 0
    else:
        sq = (torch.square(res[0]) / signal_ms["b"]
              + torch.square(res[1]) / signal_ms["i"])
        sq_sum = sq.sum()
        n_local = int(sq.numel())
    ddp_flag = (dist.is_available() and dist.is_initialized()) if ddp is None else ddp
    if ddp_flag and dist.is_initialized():
        total = torch.tensor(float(n_local), device=coeffs.device, dtype=coeffs.dtype)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        if float(total.item()) < 0.5:
            return None  # global count is zero on every rank consistently
        return sq_sum * dist.get_world_size() / total
    if n_local == 0:
        return None
    return sq_sum / float(n_local)
