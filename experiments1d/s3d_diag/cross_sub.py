"""Cross-substitution of the plane-averaged (1-D) and lateral parts of the bound charge (reviewer 2026-10-02; no training,
no PB re-solve): on the DFT grid, with pa(F) = <F>_xy and lat(F) = F - pa(F),
    V0  = ML                              current model
    A   = pa(DFT) + lat(ML)               1-D fixed, lateral as predicted: what remains once the 1-D part is right
    Bf  = pa(ML)  + lat(DFT)              lateral perfect, 1-D as predicted: what the 1-D error alone costs
for charged / neutral frames and the charged - neutral response. Also: (1) negative-correction coverage of the residual:
at interface points where DFT has (almost) no bound charge but the broadcast 1-D part B(z) is large, the residual must
supply -B(z); report how much of that cancellation it supplies and how much of it falls where the envelope is ~0;
(2) half-charge thickness of DFT, residual and ML on ONE common column set. CPU-capable (KIT_DEVICE=cpu).
Usage: python cross_sub.py <out.json>   (cwd with ./data, ./cal1_train.json); env KIT_PAIRS (charged sids, '+' or space separated)
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
from mace.modules.solvent3d import _interp1_periodic, _interp3_periodic, normalized_gradient_envelope

OUT = sys.argv[1] if len(sys.argv) > 1 else "cross_sub.json"
MODEL = os.environ.get("KIT_MODEL", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_vsolv_fix/models/prod500_vsolv_fix.model")
PAIRS = [int(s) for s in os.environ.get("KIT_PAIRS", "28 30 43 60 61 62 69 79 83 94 128 134 148 153 159 177 180 185 186 189 122").replace("+", " ").split()]
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
man = json.load(open("data/solvent3d_grid_manifest.json")); ENT = {int(k): v for k, v in man["entries"].items()}
ATOMS = {}
for sp in ("train", "val", "test"):
    for a in read(f"data/{sp}.xyz", ":"):
        a.info["_split"] = sp; ATOMS[int(a.info["sample_id"])] = a
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin", "sample_id": "sample_id",
                                    "solvated": "solvated", "fermi_level": "Fermi", "potential": "potential_diff"}, arrays_keys={"forces": "forces"})
T0 = time.time(); K_EVA = 14.39964546866782
print(f"model {os.path.basename(MODEL)}; device {dev}; pairs {PAIRS}", flush=True)


def label(sid):
    e = ENT[sid]
    rb = np.asarray(np.load("data/" + e["path_b"], mmap_mode="r"), dtype=np.float64)
    with np.load("data/" + e["meta_path"]) as m:
        lat = np.asarray(m["lattice"], dtype=np.float64)
    return rb, lat


def capture(sid, shape):
    atoms = ATOMS[sid]
    cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(dev)
    hold = {}; orig = backend.solve_graph

    def spy(*a, **kw):
        res = orig(*a, **kw)
        cell_np = kw["cell"].detach().cpu().numpy().astype(float).reshape(3, 3)
        grid = backend._grid_for(cell_np, backend._grid_shape(cell_np), kw["positions"].device)
        hold.update(res=res, cell=cell_np, cav=grid._solv3d_cavity[1].clone())
        return res
    backend.solve_graph = spy
    try:
        with torch.no_grad():
            model(batch.to_dict(), training=False, compute_force=False)
    finally:
        backend.solve_graph = orig
    res = hold["res"]; cell64 = torch.as_tensor(hold["cell"], device=dev)
    env3 = normalized_gradient_envelope(hold["cav"], cell64)
    Bz = res["rho_bound_z"].detach().double(); d3 = res["s3d_obs"]["d_sup_b"].detach().double()
    nz, ny, nx = shape
    iy, ix = torch.meshgrid(torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing="ij")
    fxy = torch.stack([ix.reshape(-1).double() / nx, iy.reshape(-1).double() / ny], dim=1)
    D = np.empty(shape); E = np.empty(shape)
    with torch.no_grad():
        for z0 in range(0, nz, 25):
            z1 = min(z0 + 25, nz)
            fz = torch.arange(z0, z1, device=dev, dtype=torch.float64) / nz
            frac = torch.cat([fxy.repeat(z1 - z0, 1), fz.repeat_interleave(ny * nx)[:, None]], dim=1)
            D[z0:z1] = _interp3_periodic(d3, frac).reshape(z1 - z0, ny, nx).cpu().numpy()
            E[z0:z1] = _interp3_periodic(env3, frac).reshape(z1 - z0, ny, nx).cpu().numpy()
        Bd = _interp1_periodic(Bz, torch.arange(nz, device=dev, dtype=torch.float64) / nz).cpu().numpy()
    return dict(q=float(atoms.info["total_charge"]), Bz=Bd, d=D, env=E)


def poisson(rho, lat):
    nz, ny, nx = rho.shape
    rg = np.fft.rfftn(rho)                                           # axes (z, y, x); last axis x halved
    Bm = 2.0 * np.pi * np.linalg.inv(lat).T                          # rows = reciprocal vectors b_a, b_b, b_c
    hx = np.fft.rfftfreq(nx) * nx; hy = np.fft.fftfreq(ny) * ny; hz = np.fft.fftfreq(nz) * nz
    G = (hz[:, None, None, None] * Bm[2] + hy[None, :, None, None] * Bm[1] + hx[None, None, :, None] * Bm[0])
    G2 = (G * G).sum(-1); G2[0, 0, 0] = 1.0
    pg = 4.0 * np.pi * K_EVA * rg / G2; pg[0, 0, 0] = 0.0
    return np.fft.irfftn(pg, s=rho.shape)


def metrics(M, D, lat, dV):
    d = M - D; pm, pd = M.mean(axis=(1, 2)), D.mean(axis=(1, 2)); Ml, Dl = M - pm[:, None, None], D - pd[:, None, None]
    pM, pD = poisson(M, lat), poisson(D, lat)
    return dict(L1=float(np.abs(d).sum() / np.abs(D).sum()), L2=float((d ** 2).sum() / (D ** 2).sum()),
                corr=float((M * D).sum() / np.sqrt((M * M).sum() * (D * D).sum())),
                eps_lat=float(np.abs(Ml - Dl).sum() / np.abs(Dl).sum()), lat_sq=float(((Ml - Dl) ** 2).sum() / (Dl ** 2).sum()),
                eps_pa=float(np.abs(pm - pd).sum() / np.abs(pd).sum()), net_err_e=float(d.sum()) * dV,
                phi_rms_err=float(np.sqrt(((pM - pD) ** 2).mean())), phi_rms_dft=float(np.sqrt((pD ** 2).mean())),
                eself_ml=0.5 * float((M * pM).sum()) * dV, eself_dft=0.5 * float((D * pD).sum()) * dV)


def half_thick(F, cols, dz):
    pos = np.clip(F, 0, None).reshape(F.shape[0], -1)[:, cols]
    srt = -np.sort(-pos, axis=0); cum = np.cumsum(srt, axis=0); tot = cum[-1]
    n = (cum < 0.5 * tot[None, :]).sum(axis=0) + 1
    ok = tot > 0
    return float(np.median(n[ok]) * dz), float(ok.mean())


RES = []
for k in PAIRS:
    Dc, lat = label(k); Dn, _ = label(k + 600); shape = Dc.shape; nz = shape[0]; dz = lat[2, 2] / nz
    dV = abs(np.linalg.det(lat)) / Dc.size
    Cc = capture(k, shape); Cn = capture(k + 600, shape)
    rec = dict(pair=k, split=ATOMS[k].info["_split"], q=Cc["q"], frames={})
    MLc = Cc["Bz"][:, None, None] + Cc["d"]; MLn = Cn["Bz"][:, None, None] + Cn["d"]
    for sid, D, ML, C in ((k, Dc, MLc, Cc), (k + 600, Dn, MLn, Cn)):
        pa = lambda F: F.mean(axis=(1, 2))[:, None, None]
        A = pa(D) + (ML - pa(ML)); Bf = pa(ML) + (D - pa(D))
        fr = {"V0": metrics(ML, D, lat, dV), "A_dft1d_mllat": metrics(A, D, lat, dV), "B_ml1d_dftlat": metrics(Bf, D, lat, dV)}
        # negative-correction coverage: interface window from the plane-averaged |DFT|
        pz = np.abs(D).mean(axis=(1, 2)); win = pz > 0.01 * pz.max()
        Bz = C["Bz"]; nolayer = (np.abs(D) < 0.01 * np.abs(D).max()) & (Bz[:, None, None] > 0.1 * Bz.max()) & win[:, None, None]
        need = -Bz[:, None, None] * np.ones_like(D); have = C["d"]
        e = C["env"]
        cov = dict(n_points=int(nolayer.sum()), need_e=float(need[nolayer].sum()) * dV, have_e=float(have[nolayer].sum()) * dV,
                   ratio=float(have[nolayer].sum() / need[nolayer].sum()) if nolayer.any() else None,
                   share_env_lt_1e3=float((e[nolayer] < 1e-3).mean()) if nolayer.any() else None,
                   share_env_lt_1e2=float((e[nolayer] < 1e-2).mean()) if nolayer.any() else None,
                   ratio_where_env_ge_1e2=float(have[nolayer & (e >= 1e-2)].sum() / need[nolayer & (e >= 1e-2)].sum()) if (nolayer & (e >= 1e-2)).any() else None,
                   ratio_where_env_lt_1e3=float(have[nolayer & (e < 1e-3)].sum() / need[nolayer & (e < 1e-3)].sum()) if (nolayer & (e < 1e-3)).any() else None)
        # where DFT HAS the layer (positive), how much of the needed addition (DFT - B) does d supply?
        layer = (D > 0.1 * D.max()) & win[:, None, None]
        needp = (D - Bz[:, None, None])[layer]; havep = C["d"][layer]
        cov["layer_points"] = int(layer.sum()); cov["layer_ratio"] = float(havep.sum() / needp.sum()) if layer.any() else None
        cov["layer_corr"] = float(np.corrcoef(needp, havep)[0, 1]) if layer.sum() > 2 else None
        fr["coverage"] = cov
        # thickness on a common column set: columns with positive DFT charge in the window > 1 % of the strongest column
        q = np.clip(D[win], 0, None).sum(axis=0).reshape(-1); cols = np.where(q > 0.01 * q.max())[0]
        fr["thickness_common_cols"] = dict(n_cols=int(cols.size), DFT=half_thick(D[win], cols, dz), residual=half_thick(C["d"][win], cols, dz),
                                           ML=half_thick(ML[win], cols, dz), B1d=half_thick(np.broadcast_to(Bz[win][:, None, None], D[win].shape), cols, dz))
        rec["frames"][str(sid)] = fr
        print(f"[{time.time() - T0:5.0f}s] sid {sid} (q {C['q']:+.3f}): L1 V0 {fr['V0']['L1']:.3f} | DFT-1D+ML-lat {fr['A_dft1d_mllat']['L1']:.3f} | ML-1D+DFT-lat {fr['B_ml1d_dftlat']['L1']:.3f} ;"
              f" L2 {fr['V0']['L2']:.3f} / {fr['A_dft1d_mllat']['L2']:.3f} / {fr['B_ml1d_dftlat']['L2']:.3f} ; phi rms {fr['V0']['phi_rms_err']:.4f} / {fr['A_dft1d_mllat']['phi_rms_err']:.4f} / {fr['B_ml1d_dftlat']['phi_rms_err']:.4f} ;"
              f" cancel ratio {cov['ratio']:.3f} (env<1e-3 share {cov['share_env_lt_1e3']:.2f}, ratio there {cov['ratio_where_env_lt_1e3']}, env>=1e-2 ratio {cov['ratio_where_env_ge_1e2']:.3f}) ; layer add ratio {cov['layer_ratio']:.3f} corr {cov['layer_corr']:.3f} ;"
              f" thickness common {fr['thickness_common_cols']['n_cols']} cols: DFT {fr['thickness_common_cols']['DFT'][0]:.2f} d {fr['thickness_common_cols']['residual'][0]:.2f} (has + in {fr['thickness_common_cols']['residual'][1]:.2f}) ML {fr['thickness_common_cols']['ML'][0]:.2f}", flush=True)
    pa = lambda F: F.mean(axis=(1, 2))[:, None, None]
    Dr, Mr = Dc - Dn, MLc - MLn
    rec["response"] = {"V0": metrics(Mr, Dr, lat, dV), "A_dft1d_mllat": metrics(pa(Dr) + (Mr - pa(Mr)), Dr, lat, dV),
                       "B_ml1d_dftlat": metrics(pa(Mr) + (Dr - pa(Dr)), Dr, lat, dV)}
    r = rec["response"]
    print(f"[{time.time() - T0:5.0f}s] pair {k} response: L1 {r['V0']['L1']:.3f} / {r['A_dft1d_mllat']['L1']:.3f} / {r['B_ml1d_dftlat']['L1']:.3f} ; L2 {r['V0']['L2']:.3f} / {r['A_dft1d_mllat']['L2']:.3f} / {r['B_ml1d_dftlat']['L2']:.3f} ;"
          f" phi rms {r['V0']['phi_rms_err']:.4f} / {r['A_dft1d_mllat']['phi_rms_err']:.4f} / {r['B_ml1d_dftlat']['phi_rms_err']:.4f} (DFT {r['V0']['phi_rms_dft']:.3f})", flush=True)
    RES.append(rec); json.dump(RES, open(OUT, "w"), indent=1)
print(f"[{time.time() - T0:5.0f}s] done", flush=True)
