"""Rigid-shift test of the ML 3-D solvent charge against DFT on the FULL DFT grid (reviewer suggestion 2026-10-01).
For twin pairs (charged k, neutral k + 600): ML fields rebuilt on the 500 x 168 x 168 grid exactly as the loss does
(1-D profile of the final solve broadcast + per-plane-projected lateral residual, trilinear), DFT labels from
data/solvent3d_grid. The ML field is shifted rigidly by a Cartesian vector d (Fourier phase shift on the periodic
grid = trigonometric interpolation, no smoothing); the squared error against DFT is minimised over (a) dz only and
(b) (dx, dy, dz); the shift is also fitted for the lateral part alone (each field minus its own plane means) and for
the charged - neutral response. Reported before / after: L1 and L2 ratios, correlation, the norm ratio
||ML|| / ||DFT|| (shift-invariant), and the L1 in the DFT-positive / DFT-negative regions.
Usage: python shift_scan_3d.py <out.json>   (cwd with ./data, ./cal1_train.json);  env KIT_PAIRS "28 60 94 128 177 122"
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")

import numpy as np
import torch
from ase.io import read
from scipy.optimize import minimize

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils
from mace.modules.solvent3d import _interp1_periodic, _interp3_periodic

OUT = sys.argv[1] if len(sys.argv) > 1 else "shift_scan_3d.json"
MODEL = os.environ.get("KIT_MODEL", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_vsolv_fix/models/prod500_vsolv_fix.model")
PAIRS = [int(s) for s in os.environ.get("KIT_PAIRS", "28 60 94 128 177 122").replace("+", " ").split()]
ZMAX, XYMAX = 3.0, 2.0
torch.set_default_dtype(torch.float64)
dev = torch.device("cuda:0")
model = torch.load(MODEL, map_location=dev).to(dev); model.eval()
for p in model.parameters():
    p.requires_grad_(False)
backend = model._get_pb1d_backend()
man = json.load(open("data/solvent3d_grid_manifest.json")); ENT = {int(k): v for k, v in man["entries"].items()}
ATOMS = {}
for sp in ("train", "val", "test"):
    for a in read(f"data/{sp}.xyz", ":"):
        a.info["_split"] = sp; ATOMS[int(a.info["sample_id"])] = a
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin", "sample_id": "sample_id",
                                    "solvated": "solvated", "fermi_level": "Fermi", "potential": "potential_diff"}, arrays_keys={"forces": "forces"})
T0 = time.time()
print(f"model {os.path.basename(MODEL)}; pairs {PAIRS}; window |dz| <= {ZMAX} A, |dx|,|dy| <= {XYMAX} A", flush=True)


def label(sid):
    e = ENT[sid]
    rb = torch.as_tensor(np.asarray(np.load("data/" + e["path_b"], mmap_mode="r"), dtype=np.float64), device=dev)
    ri = torch.as_tensor(np.asarray(np.load("data/" + e["path_i"], mmap_mode="r"), dtype=np.float64), device=dev)
    with np.load("data/" + e["meta_path"]) as m:
        lat = np.asarray(m["lattice"], dtype=np.float64)
    return rb, ri, lat


def ml_fields(sid, shape):
    atoms = ATOMS[sid]
    cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(dev)
    hold = {}; orig = backend.solve_graph

    def spy(*a, **kw):
        res = orig(*a, **kw); hold["res"] = res; return res
    backend.solve_graph = spy
    try:
        with torch.no_grad():
            model(batch.to_dict(), training=False, compute_force=False)
    finally:
        backend.solve_graph = orig
    res = hold["res"]; obs = res["s3d_obs"]
    nz, ny, nx = shape
    out = {}
    iy, ix = torch.meshgrid(torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing="ij")
    fxy = torch.stack([ix.reshape(-1).double() / nx, iy.reshape(-1).double() / ny], dim=1)
    for ch, Bz, dg in (("b", res["rho_bound_z"], obs["d_sup_b"]), ("i", res["rho_ion_z"], obs["d_sup_i"])):
        M = torch.empty(shape, device=dev)
        for z0 in range(0, nz, 25):
            z1 = min(z0 + 25, nz)
            fz = torch.arange(z0, z1, device=dev, dtype=torch.float64) / nz
            frac = torch.cat([fxy.repeat(z1 - z0, 1), fz.repeat_interleave(ny * nx)[:, None]], dim=1)
            M[z0:z1] = (_interp1_periodic(Bz.double(), frac[:, 2]) + _interp3_periodic(dg.double(), frac)).reshape(z1 - z0, ny, nx)
        out[ch] = M
    return out


class Shifter:
    """Fourier phase shift of a (nz, ny, nx) periodic field by a Cartesian vector, general cell."""

    def __init__(self, shape, lat):
        nz, ny, nx = shape
        self.inv = torch.as_tensor(np.linalg.inv(lat), device=dev)          # frac = cart @ inv
        hz = torch.fft.fftfreq(nz, device=dev, dtype=torch.float64) * nz
        hy = torch.fft.fftfreq(ny, device=dev, dtype=torch.float64) * ny
        hx = torch.arange(nx // 2 + 1, device=dev, dtype=torch.float64)
        self.hz, self.hy, self.hx = hz[:, None, None], hy[None, :, None], hx[None, None, :]
        self.shape = shape

    def __call__(self, Fk, d):
        f = torch.as_tensor(np.asarray(d, dtype=np.float64), device=dev) @ self.inv     # fractional (fa, fb, fc)
        ph = torch.exp(-2j * np.pi * (self.hx * f[0] + self.hy * f[1] + self.hz * f[2]))
        return torch.fft.irfftn(Fk * ph, s=self.shape)


def metrics(M, D):
    d = M - D; pos, neg = D > 0, D < 0
    return dict(L1=float(d.abs().sum() / D.abs().sum()), L2=float((d * d).sum() / (D * D).sum()),
                corr=float((M * D).sum() / torch.sqrt((M * M).sum() * (D * D).sum())),
                L1pos=float(d[pos].abs().sum() / D[pos].abs().sum()) if bool(pos.any()) else None,
                L1neg=float(d[neg].abs().sum() / D[neg].abs().sum()) if bool(neg.any()) else None)


def scan(M, D, lat, label_):
    sh = Shifter(D.shape, lat); Mk = torch.fft.rfftn(M); Dk = torch.fft.rfftn(D)
    # integer-shift cross-correlation, windowed in Cartesian displacement
    cc = torch.fft.irfftn(Dk * torch.conj(Mk), s=D.shape).cpu().numpy()        # cc[s] = sum_r M(r) D(r + s)
    nz, ny, nx = D.shape
    iz = np.arange(nz); iy = np.arange(ny); ix = np.arange(nx)
    fz = np.where(iz <= nz // 2, iz, iz - nz) / nz; fy = np.where(iy <= ny // 2, iy, iy - ny) / ny; fx = np.where(ix <= nx // 2, ix, ix - nx) / nx
    FZ, FY, FX = np.meshgrid(fz, fy, fx, indexing="ij")
    cart = np.stack([FX, FY, FZ], axis=-1) @ lat                                  # (nz, ny, nx, 3) Cartesian displacement
    win = (np.abs(cart[..., 2]) <= ZMAX) & (np.abs(cart[..., 0]) <= XYMAX) & (np.abs(cart[..., 1]) <= XYMAX)
    ccw = np.where(win, cc, -np.inf); k = np.unravel_index(np.argmax(ccw), cc.shape); d0 = cart[k]
    winz = win & (FX == 0) & (FY == 0); ccz = np.where(winz, cc, -np.inf); kz = np.unravel_index(np.argmax(ccz), cc.shape); dz0 = cart[kz][2]
    obj3 = lambda d: float(((sh(Mk, d) - D) ** 2).sum())
    objz = lambda d: float(((sh(Mk, [0.0, 0.0, d[0]]) - D) ** 2).sum())
    rz = minimize(objz, [dz0], method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-14, maxiter=200))
    r3 = minimize(obj3, d0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-14, maxiter=600))
    m0 = metrics(M, D); mz = metrics(sh(Mk, [0.0, 0.0, float(rz.x[0])]), D); Ms = sh(Mk, r3.x); m3 = metrics(Ms, D)
    a = float((Ms * D).sum() / (Ms * Ms).sum()); ms = metrics(a * Ms, D)
    rec = dict(label=label_, norm_ratio=float(torch.sqrt((M * M).sum() / (D * D).sum())), zero=m0, dz=float(rz.x[0]), z_shift=mz,
               d=[float(v) for v in r3.x], xyz_shift=m3, scale=a, xyz_shift_scale=ms, n_eval=int(rz.nfev + r3.nfev))
    print(f"[{time.time() - T0:6.0f}s] {label_:28s} |ML|/|DFT| {rec['norm_ratio']:.3f} corr {m0['corr']:.3f} L1 {m0['L1']:.3f} L2 {m0['L2']:.3f}"
          f" | dz {rec['dz']:+.3f} -> L1 {mz['L1']:.3f} L2 {mz['L2']:.3f} | d ({rec['d'][0]:+.3f},{rec['d'][1]:+.3f},{rec['d'][2]:+.3f}) -> L1 {m3['L1']:.3f}"
          f" L2 {m3['L2']:.3f} corr {m3['corr']:.3f} (DFT>0 {m3['L1pos']}, <0 {m3['L1neg']}) | +scale {a:.3f} -> L1 {ms['L1']:.3f} L2 {ms['L2']:.3f}", flush=True)
    return rec


lateral = lambda F: F - F.mean(dim=(1, 2), keepdim=True)
RES = []
for k in PAIRS:
    recs = []
    Db, Di, lat = label(k); Dbn, Din, _ = label(k + 600)
    Mc = ml_fields(k, tuple(Db.shape)); Mn = ml_fields(k + 600, tuple(Db.shape))
    print(f"[{time.time() - T0:6.0f}s] pair {k}/{k + 600} ({ATOMS[k].info['_split']}, q {float(ATOMS[k].info['total_charge']):+.3f}) fields ready", flush=True)
    for lab, M, D in (("charged bound", Mc["b"], Db), ("charged ion", Mc["i"], Di), ("neutral bound", Mn["b"], Dbn),
                      ("response bound", Mc["b"] - Mn["b"], Db - Dbn), ("response ion", Mc["i"] - Mn["i"], Di - Din)):
        recs.append(scan(M, D, lat, lab))
        recs.append(scan(lateral(M), lateral(D), lat, lab + " LATERAL"))
    RES.append(dict(pair=k, split=ATOMS[k].info["_split"], q=float(ATOMS[k].info["total_charge"]), scans=recs))
    json.dump(RES, open(OUT, "w"), indent=1)
    del Mc, Mn, Db, Di, Dbn, Din; torch.cuda.empty_cache()
print(f"[{time.time() - T0:6.0f}s] done; GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
