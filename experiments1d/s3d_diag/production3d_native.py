"""Production full 3-D solvent charge on the DFT native grid, for the same-frame comparison table (reviewer 2026-10-02).
rho_b^prod(r) = B(z) + d_b(r): the final 1-D solve profile (rho_bound_z, 600 planes, _interp1_periodic in z) plus the per-plane
zero-mean lateral residual d_sup_b (model grid, _interp3_periodic) -- the assembly of solvent3d_full_eval.py -- evaluated at
the native grid points; same for the ion channel. Metrics vs native RHOB / RHOION (L1, lateral, plane average, norm,
bound-potential rms), the plane-average-only comparison, and the charged-minus-neutral response per pair.
Usage: python production3d_native.py <out.json>; env KIT_PAIRS, KIT_DEVICE, KIT_DFT, KIT_MODEL
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

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils
from mace.modules.solvent3d import _interp1_periodic, _interp3_periodic

OUT = sys.argv[1] if len(sys.argv) > 1 else "production3d_native.json"
MODEL = os.environ.get("KIT_MODEL", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_vsolv_fix/models/prod500_vsolv_fix.model")
PAIRS = [int(s) for s in os.environ.get("KIT_PAIRS", "1 52 152").replace("+", " ").split()]
DFT = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single")
devname = os.environ.get("KIT_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
dev = torch.device(devname)
if dev.type == "cpu":
    _jl = torch.jit.load
    torch.jit.load = lambda f, map_location=None, **kw: _jl(f, map_location="cpu", **kw)
model = torch.load(MODEL, map_location=dev).to(dev); model.eval()
for p in model.parameters():
    p.requires_grad_(False)
backend = model._get_pb1d_backend()
K_EVA = 14.39964546866782
ATOMS = {}
for sp in ("train", "val", "test"):
    for a in read(f"data/{sp}.xyz", ":"):
        a.info["_split"] = sp; ATOMS[int(a.info["sample_id"])] = a
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin", "sample_id": "sample_id",
                                    "solvated": "solvated", "fermi_level": "Fermi", "potential": "potential_diff"}, arrays_keys={"forces": "forces"})
T0 = time.time()


def forward(sid):
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
    res = hold["res"]; obs = res.get("s3d_obs")
    assert obs is not None and "d_sup_b" in obs, "forward produced no 3-D residual (s3d_obs)"
    return res["rho_bound_z"].detach().double(), res["rho_ion_z"].detach().double(), obs["d_sup_b"].detach().double(), obs["d_sup_i"].detach().double()


def dft_dir(sid):
    return f"{DFT}/1-44_GCE/cal_{sid}" if sid <= 200 else f"{DFT}/5-44_neutral_withsolv/cal_{sid - 600}"


def read_grid_fast(path):
    with open(path) as f:
        f.readline(); s = float(f.readline())
        lat = np.array([[float(x) for x in f.readline().split()] for _ in range(3)]) * s
        line = f.readline().split()
        try:
            counts = [int(x) for x in line]
        except ValueError:
            counts = [int(x) for x in f.readline().split()]
        nat = sum(counts); f.readline()
        for _ in range(nat):
            f.readline()
        f.readline()
        nx, ny, nz = [int(x) for x in f.readline().split()]
        need = nx * ny * nz
        rest = f.read()
    arr = np.fromstring(rest, sep=" ", dtype=np.float64)[:need]; del rest
    assert arr.size == need, (path, arr.size, need)
    return lat, np.ascontiguousarray(arr.reshape(nz, ny, nx))          # (nz, ny, nx)


def on_native(rb_z, ri_z, db, di, shape):
    """production fields at the native grid points -> numpy (nz, ny, nx)."""
    nz, ny, nx = shape
    iy, ix = torch.meshgrid(torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing="ij")
    fxy = torch.stack([ix.reshape(-1).double() / nx, iy.reshape(-1).double() / ny], dim=1)
    Mb = np.empty(shape); Mi = np.empty(shape)
    for z0 in range(0, nz, 25):
        z1 = min(z0 + 25, nz)
        fz = torch.arange(z0, z1, device=dev, dtype=torch.float64) / nz
        frac = torch.cat([fxy.repeat(z1 - z0, 1), fz.repeat_interleave(ny * nx)[:, None]], dim=1)
        Mb[z0:z1] = (_interp1_periodic(rb_z, frac[:, 2]) + _interp3_periodic(db, frac)).reshape(z1 - z0, ny, nx).cpu().numpy()
        Mi[z0:z1] = (_interp1_periodic(ri_z, frac[:, 2]) + _interp3_periodic(di, frac)).reshape(z1 - z0, ny, nx).cpu().numpy()
    return Mb, Mi


def poisson_np(rho_zyx, lat):
    nz, ny, nx = rho_zyx.shape
    rg = np.fft.rfftn(rho_zyx); Bm = 2.0 * np.pi * np.linalg.inv(lat).T
    hx = np.fft.rfftfreq(nx) * nx; hy = np.fft.fftfreq(ny) * ny; hz = np.fft.fftfreq(nz) * nz
    G = (hz[:, None, None, None] * Bm[2] + hy[None, :, None, None] * Bm[1] + hx[None, None, :, None] * Bm[0])
    G2 = (G * G).sum(-1); G2[0, 0, 0] = 1.0
    pg = 4.0 * np.pi * K_EVA * rg / G2; pg[0, 0, 0] = 0.0
    return np.fft.irfftn(pg, s=rho_zyx.shape)


def metrics(M, D, lat, dV, pD=None):
    d = M - D; pm, pd = M.mean(axis=(1, 2)), D.mean(axis=(1, 2)); Ml, Dl = M - pm[:, None, None], D - pd[:, None, None]
    pM = poisson_np(M, lat); pD = poisson_np(D, lat) if pD is None else pD
    return dict(L1=float(np.abs(d).sum() / np.abs(D).sum()), L2=float((d ** 2).sum() / (D ** 2).sum()),
                eps_lat=float(np.abs(Ml - Dl).sum() / np.abs(Dl).sum()), lat_sq=float(((Ml - Dl) ** 2).sum() / (Dl ** 2).sum()),
                lat_corr=float((Ml * Dl).sum() / np.sqrt((Ml * Ml).sum() * (Dl * Dl).sum())),
                eps_pa=float(np.abs(pm - pd).sum() / np.abs(pd).sum()), pa_corr=float(((pm - pm.mean()) * (pd - pd.mean())).sum() / np.sqrt(((pm - pm.mean()) ** 2).sum() * ((pd - pd.mean()) ** 2).sum())),
                net_err_e=float(d.sum()) * dV, net_ml_e=float(M.sum()) * dV, net_dft_e=float(D.sum()) * dV,
                phi_rms_err=float(np.sqrt(((pM - pD) ** 2).mean())), phi_rms_dft=float(np.sqrt((pD ** 2).mean())),
                norm_ratio=float(np.sqrt((M * M).sum() / (D * D).sum())), lat_norm_ratio=float(np.sqrt((Ml * Ml).sum() / (Dl * Dl).sum()))), pD


RES = []
for kpair in PAIRS:
    rec = dict(pair=kpair, split=ATOMS[kpair].info["_split"], frames={}, response={})
    keep = {}
    for sid in (kpair, kpair + 600):
        d = dft_dir(sid); t1 = time.time()
        lat, rb_raw = read_grid_fast(f"{d}/RHOB"); _, ri_raw = read_grid_fast(f"{d}/RHOION")
        shape = rb_raw.shape; V = float(abs(np.linalg.det(lat))); dV = V / rb_raw.size
        Db = -(rb_raw / V); Di = -(ri_raw / V); del rb_raw, ri_raw
        rb_z, ri_z, db, di = forward(sid)
        with torch.no_grad():
            Mb, Mi = on_native(rb_z, ri_z, db, di, shape)
            B_only = np.broadcast_to(_interp1_periodic(rb_z, torch.arange(shape[0], device=dev, dtype=torch.float64) / shape[0]).cpu().numpy()[:, None, None], shape).copy()
        pDb = poisson_np(Db, lat); pDi = poisson_np(Di, lat)
        fb, _ = metrics(Mb, Db, lat, dV, pDb); fi, _ = metrics(Mi, Di, lat, dV, pDi); ft, _ = metrics(Mb + Mi, Db + Di, lat, dV)
        fB, _ = metrics(B_only, Db, lat, dV, pDb)
        pa_only = dict(bound_pa_L1=float(np.abs(Mb.mean(axis=(1, 2)) - Db.mean(axis=(1, 2))).sum() / np.abs(Db.mean(axis=(1, 2))).sum()),
                       ion_pa_L1=float(np.abs(Mi.mean(axis=(1, 2)) - Di.mean(axis=(1, 2))).sum() / np.abs(Di.mean(axis=(1, 2))).sum()))
        rec["frames"][str(sid)] = dict(read_s=time.time() - t1, full3d_bound=fb, full3d_ion=fi, full3d_total=ft, plane_only_bound=fB, pa_only=pa_only,
                                       lateral_residual_norm_e2=float((Mb - B_only).__pow__(2).sum() * dV))
        keep[sid] = dict(b=Mb, i=Mi, Db=Db, Di=Di)
        print(f"[{time.time() - T0:5.0f}s] sid {sid} (q {float(ATOMS[sid].info['total_charge']):+.3f}) production full 3-D vs native: bound L1 {fb['L1']:.3f} lat {fb['eps_lat']:.3f} ({fb['lat_corr']:+.3f}) pa {fb['eps_pa']:.3f} norm {fb['norm_ratio']:.3f} phi {fb['phi_rms_err']:.3f} | "
              f"ion L1 {fi['L1']:.3f} pa {fi['eps_pa']:.3f} norm {fi['norm_ratio']:.3f} phi {fi['phi_rms_err']:.3f} | total L1 {ft['L1']:.3f} phi {ft['phi_rms_err']:.3f} | plane-only bound L1 {fB['L1']:.3f} (lat part of RHOB missing by construction) pa {fB['eps_pa']:.3f}", flush=True)
    dV = abs(np.linalg.det(lat)) / keep[kpair]["Db"].size
    rb, _ = metrics(keep[kpair]["b"] - keep[kpair + 600]["b"], keep[kpair]["Db"] - keep[kpair + 600]["Db"], lat, dV)
    ri, _ = metrics(keep[kpair]["i"] - keep[kpair + 600]["i"], keep[kpair]["Di"] - keep[kpair + 600]["Di"], lat, dV)
    rec["response"] = dict(bound=rb, ion=ri)
    print(f"[{time.time() - T0:5.0f}s] pair {kpair} production response vs DFT: bound L1 {rb['L1']:.3f} lat {rb['eps_lat']:.3f} ({rb['lat_corr']:+.2f}) pa {rb['eps_pa']:.3f} phi {rb['phi_rms_err']:.3f} | ion L1 {ri['L1']:.3f} pa {ri['eps_pa']:.3f} phi {ri['phi_rms_err']:.3f}", flush=True)
    RES.append(rec); json.dump(RES, open(OUT, "w"), indent=1); del keep
print(f"[{time.time() - T0:5.0f}s] done", flush=True)
