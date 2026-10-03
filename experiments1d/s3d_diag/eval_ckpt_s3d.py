"""Checkpoint evaluation of the 3-D solvent charge for the prod500_chargeinput_repair run (reviewer plan 2026-10-03, item 6), on
the fixed validation pairs, DFT native grid, for any .model file (baseline or new):
  bound / ion separately: eps_3d, lateral L1 + corr, plane-average L1, norm ratio, bound/ion/total potential rms error
  charged-minus-neutral response per pair: same metrics
  forbidden-zone charge: |rho| mass (e) and its ratio to int |rho_DFT| dV, with the MODEL cavity mask (the repair's own weight,
  M < 1e-3) and with the DFT-density cavity mask (same construction on the package-route DFT density)
  excess predicted charge where DFT is near zero: int_{|rho_DFT| < 1 % of max} |rho_model| (e) and the ratio to int |rho_DFT|
  representative xz slices (bound, charged frame of KIT_SLICE_PAIRS) saved as npz for a page with one common colour scale
Fields: the production assembly B(z) (final 1-D solve) + d_sup (the repaired lateral residual when the model has the repair),
evaluated at the native grid points (solvent3d_full_eval.py assembly).
Usage: python eval_ckpt_s3d.py <model.model> <out.json>; env KIT_PAIRS (default: the 20 validation pairs), KIT_DEVICE, KIT_DFT, KIT_SLICE_PAIRS
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
import mace.modules.pb1d_backend as PB
from mace.modules.solvent3d import _interp1_periodic, _interp3_periodic

MODEL = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "eval_ckpt_s3d.json"
PAIRS = [int(s) for s in os.environ.get("KIT_PAIRS", "28 30 43 60 61 62 69 79 83 94 128 134 148 153 159 177 180 185 186 189").replace("+", " ").split()]
SLICE_PAIRS = [int(s) for s in os.environ.get("KIT_SLICE_PAIRS", "28 128").replace("+", " ").split()]
DFT = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single")
MASK = dict(sigma=0.25, t0=1.0e-4, t1=1.0e-2)
torch.set_default_dtype(torch.float64)
dev = torch.device(os.environ.get("KIT_DEVICE", "cuda:0"))
if dev.type == "cpu":
    _jl = torch.jit.load
    torch.jit.load = lambda f, map_location=None, **kw: _jl(f, map_location="cpu", **kw)
model = torch.load(MODEL, map_location=dev).to(dev); model.eval()
for p in model.parameters():
    p.requires_grad_(False)
backend = model._get_pb1d_backend(); tp = backend._tp; params = backend.params
if hasattr(model, "solvent3d_mask") and isinstance(model.solvent3d_mask, dict):
    MASK.update({k: float(model.solvent3d_mask[k]) for k in ("sigma", "t0", "t1") if k in model.solvent3d_mask})
K_EVA = 14.39964546866782
man_n = json.load(open("data/density3d_net_grid_manifest_npy.json")); ENT_N = {int(k): v for k, v in man_n["entries"].items()}
ATOMS = {}
for sp in ("train", "val", "test"):
    for a in read(f"data/{sp}.xyz", ":"):
        a.info["_split"] = sp; ATOMS[int(a.info["sample_id"])] = a
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin", "sample_id": "sample_id",
                                    "solvated": "solvated", "fermi_level": "Fermi", "potential": "potential_diff"}, arrays_keys={"forces": "forces"})
T0 = time.time()
SL = os.environ.get("KIT_SLICES", "eval_slices"); os.makedirs(SL, exist_ok=True)
print(f"model {MODEL}; repair {getattr(model, 'solvent3d_repair', None)}; head charge_state_input {getattr(model.solvent3d_head, 'charge_state_input', None)} ion_gate {getattr(model.solvent3d_head, 'ion_gate', None)}; mask {MASK}; pairs {PAIRS}", flush=True)
CAP = {}
_orig_clo = PB.closure_from_fields


def clo_cap(n_e, cv, grid, params_, tp_):
    CAP["ne"] = n_e.detach().clone(); CAP["grid"] = grid
    return _orig_clo(n_e, cv, grid, params_, tp_)


PB.closure_from_fields = clo_cap


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
    cav = getattr(CAP["grid"], "_solv3d_cavity", None)
    return (res["rho_bound_z"].detach().double(), res["rho_ion_z"].detach().double(), obs["d_sup_b"].detach().double(), obs["d_sup_i"].detach().double(),
            cav, CAP["ne"], CAP["grid"], obs.get("repair"))


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
    return lat, np.ascontiguousarray(arr.reshape(nz, ny, nx))


def to_native(F3, shape):
    """model-grid field [nx,ny,nz] -> numpy (nz,ny,nx) at the native points (trilinear periodic)."""
    nz, ny, nx = shape
    iy, ix = torch.meshgrid(torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing="ij")
    fxy = torch.stack([ix.reshape(-1).double() / nx, iy.reshape(-1).double() / ny], dim=1)
    M = np.empty(shape)
    for z0 in range(0, nz, 25):
        z1 = min(z0 + 25, nz); fz = torch.arange(z0, z1, device=dev, dtype=torch.float64) / nz
        frac = torch.cat([fxy.repeat(z1 - z0, 1), fz.repeat_interleave(ny * nx)[:, None]], dim=1)
        M[z0:z1] = _interp3_periodic(F3, frac).reshape(z1 - z0, ny, nx).cpu().numpy()
    return M


def on_native(rb_z, ri_z, db, di, shape):
    nz, ny, nx = shape
    iy, ix = torch.meshgrid(torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing="ij")
    fxy = torch.stack([ix.reshape(-1).double() / nx, iy.reshape(-1).double() / ny], dim=1)
    Mb = np.empty(shape); Mi = np.empty(shape)
    for z0 in range(0, nz, 25):
        z1 = min(z0 + 25, nz); fz = torch.arange(z0, z1, device=dev, dtype=torch.float64) / nz
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
    return dict(eps_3d=float(np.abs(d).sum() / np.abs(D).sum()), eps_lat=float(np.abs(Ml - Dl).sum() / np.abs(Dl).sum()),
                lat_corr=float((Ml * Dl).sum() / np.sqrt((Ml * Ml).sum() * (Dl * Dl).sum())), eps_pa=float(np.abs(pm - pd).sum() / np.abs(pd).sum()),
                norm_ratio=float(np.sqrt((M * M).sum() / (D * D).sum())), net_ml_e=float(M.sum()) * dV, net_dft_e=float(D.sum()) * dV,
                phi_rms_err=float(np.sqrt(((pM - pD) ** 2).mean())), phi_rms_dft=float(np.sqrt((pD ** 2).mean()))), pD


RES = dict(meta=dict(model=MODEL, pairs=PAIRS, mask=MASK, repair=bool(getattr(model, "solvent3d_repair", False)),
                     charge_state_input=bool(getattr(model.solvent3d_head, "charge_state_input", False)), ion_gate=bool(getattr(model.solvent3d_head, "ion_gate", True))), frames={}, pairs={})
for kpair in PAIRS:
    keep = {}
    for sid in (kpair, kpair + 600):
        t1 = time.time(); d = dft_dir(sid)
        lat, rb_raw = read_grid_fast(f"{d}/RHOB"); _, ri_raw = read_grid_fast(f"{d}/RHOION")
        shape = rb_raw.shape; V = float(abs(np.linalg.det(lat))); dV = V / rb_raw.size
        Db = -(rb_raw / V); Di = -(ri_raw / V); del rb_raw, ri_raw
        rb_z, ri_z, db, di, cav, ne_m, grid, rep = forward(sid)
        with torch.no_grad():
            Mb, Mi = on_native(rb_z, ri_z, db, di, shape)
            # masks: model cavity (the repair's weight) and DFT-density cavity (package route), both -> native grid
            Mw_b = PB._s3d_allowed_weight(cav[1], grid, MASK); Mw_i = PB._s3d_allowed_weight(torch.clamp(cav[0], 0, 1), grid, MASK)
            bl_row = backend._bl_index.get(sid); neutral_v = torch.from_numpy(np.ascontiguousarray(backend._bl_arr[bl_row][backend._bl_take][0])).to(dev).double()
            mshape = tuple(ne_m.shape)
            net_dft = torch.as_tensor(np.asarray(np.load(ENT_N[sid]["path"], mmap_mode="r"), dtype=np.float64), device=dev).permute(2, 1, 0).contiguous()
            nx_, ny_, nz_ = mshape
            ii, jj, kk = torch.meshgrid(torch.arange(nx_, device=dev), torch.arange(ny_, device=dev), torch.arange(nz_, device=dev), indexing="ij")
            fm = torch.stack([ii.reshape(-1) / nx_, jj.reshape(-1) / ny_, kk.reshape(-1) / nz_], dim=1).double()
            net_m = _interp3_periodic(net_dft, fm).reshape(mshape) * float(grid.volume); del net_dft
            ne_dft = torch.clamp((neutral_v - net_m) / float(grid.volume), min=0.0); del net_m, neutral_v
            si_d, sd_d, _ = tp.create_cavity_torch(ne_dft, grid, params); del ne_dft
            Md_b = PB._s3d_allowed_weight(sd_d, grid, MASK); Md_i = PB._s3d_allowed_weight(torch.clamp(si_d, 0, 1), grid, MASK)
            zone_mb = to_native(Mw_b, shape) < 1e-3; zone_mi = to_native(Mw_i, shape) < 1e-3; zone_db = to_native(Md_b, shape) < 1e-3; zone_di = to_native(Md_i, shape) < 1e-3
        pDb = poisson_np(Db, lat); pDi = poisson_np(Di, lat)
        fb, _ = metrics(Mb, Db, lat, dV, pDb); fi, _ = metrics(Mi, Di, lat, dV, pDi); ft, _ = metrics(Mb + Mi, Db + Di, lat, dV)
        absDb = float(np.abs(Db).sum() * dV); absDi = float(np.abs(Di).sum() * dV)
        near0_b = np.abs(Db) < 0.01 * np.abs(Db).max(); near0_i = np.abs(Di) < 0.01 * np.abs(Di).max()
        zones = dict(bound_zone_model_e=float(np.abs(Mb[zone_mb]).sum() * dV), bound_zone_dftcav_e=float(np.abs(Mb[zone_db]).sum() * dV), bound_dft_in_dftcav_zone_e=float(np.abs(Db[zone_db]).sum() * dV),
                     ion_zone_model_e=float(np.abs(Mi[zone_mi]).sum() * dV), ion_zone_dftcav_e=float(np.abs(Mi[zone_di]).sum() * dV), ion_dft_in_dftcav_zone_e=float(np.abs(Di[zone_di]).sum() * dV),
                     bound_abs_dft_e=absDb, ion_abs_dft_e=absDi, bound_near0_excess_e=float(np.abs(Mb[near0_b]).sum() * dV), ion_near0_excess_e=float(np.abs(Mi[near0_i]).sum() * dV),
                     bound_near0_dft_e=float(np.abs(Db[near0_b]).sum() * dV), near0_frac_b=float(near0_b.mean()), zone_frac_model_b=float(zone_mb.mean()), zone_frac_dftcav_b=float(zone_db.mean()))
        zones.update(bound_zone_model_ratio=zones["bound_zone_model_e"] / absDb, bound_zone_dftcav_ratio=zones["bound_zone_dftcav_e"] / absDb, bound_near0_excess_ratio=zones["bound_near0_excess_e"] / absDb,
                     ion_zone_model_ratio=zones["ion_zone_model_e"] / max(absDi, 1e-30), ion_near0_excess_ratio=zones["ion_near0_excess_e"] / max(absDi, 1e-30))
        RES["frames"][str(sid)] = dict(n_atoms=len(ATOMS[sid]), q=float(ATOMS[sid].info["total_charge"]), bound=fb, ion=fi, total=ft, zones=zones, repair=rep, seconds=time.time() - t1)
        keep[sid] = dict(b=Mb, i=Mi, Db=Db, Di=Di)
        if kpair in SLICE_PAIRS:
            iy = shape[1] // 2
            np.savez_compressed(f"{SL}/sid{sid}_xz.npz", b_model=Mb[:, iy, :], b_dft=Db[:, iy, :], i_model=Mi[:, iy, :], i_dft=Di[:, iy, :], zone_model=zone_mb[:, iy, :], zone_dftcav=zone_db[:, iy, :], lat=lat)
        print(f"[{time.time() - T0:5.0f}s] sid {sid} (q {RES['frames'][str(sid)]['q']:+.2f}): bound eps3d {fb['eps_3d']:.3f} lat {fb['eps_lat']:.3f} ({fb['lat_corr']:+.2f}) pa {fb['eps_pa']:.3f} phi {fb['phi_rms_err']:.3f} | ion eps3d {fi['eps_3d']:.3f} pa {fi['eps_pa']:.3f} | total phi {ft['phi_rms_err']:.3f} | "
              f"zone(model) b {zones['bound_zone_model_e']:.4f} e ({zones['bound_zone_model_ratio']:.4f} of {absDb:.3f}) zone(DFT cav) b {zones['bound_zone_dftcav_e']:.4f} ({zones['bound_zone_dftcav_ratio']:.4f}; DFT itself {zones['bound_dft_in_dftcav_zone_e']:.4f}) | near-zero excess b {zones['bound_near0_excess_e']:.4f} ({zones['bound_near0_excess_ratio']:.4f}; DFT {zones['bound_near0_dft_e']:.4f}) i {zones['ion_near0_excess_e']:.4f} | repair {rep}", flush=True)
    dV = abs(np.linalg.det(lat)) / keep[kpair]["Db"].size
    rb, _ = metrics(keep[kpair]["b"] - keep[kpair + 600]["b"], keep[kpair]["Db"] - keep[kpair + 600]["Db"], lat, dV)
    ri, _ = metrics(keep[kpair]["i"] - keep[kpair + 600]["i"], keep[kpair]["Di"] - keep[kpair + 600]["Di"], lat, dV)
    rt, _ = metrics((keep[kpair]["b"] + keep[kpair]["i"]) - (keep[kpair + 600]["b"] + keep[kpair + 600]["i"]), (keep[kpair]["Db"] + keep[kpair]["Di"]) - (keep[kpair + 600]["Db"] + keep[kpair + 600]["Di"]), lat, dV)
    RES["pairs"][str(kpair)] = dict(bound=rb, ion=ri, total=rt)
    print(f"[{time.time() - T0:5.0f}s] pair {kpair} response: bound eps3d {rb['eps_3d']:.3f} lat {rb['eps_lat']:.3f} ({rb['lat_corr']:+.2f}) pa {rb['eps_pa']:.3f} phi {rb['phi_rms_err']:.3f} | ion eps3d {ri['eps_3d']:.3f} pa {ri['eps_pa']:.3f} | total phi {rt['phi_rms_err']:.3f}", flush=True)
    json.dump(RES, open(OUT, "w"), indent=1, default=float); del keep
# summary
def med(path, rows):
    vals = []
    for r in rows:
        v = r
        for p in path: v = v[p]
        vals.append(v)
    return float(np.median(vals))
fc = [RES["frames"][str(k)] for k in PAIRS]; fn = [RES["frames"][str(k + 600)] for k in PAIRS]; pr = [RES["pairs"][str(k)] for k in PAIRS]
RES["summary"] = dict(
    charged_bound_eps3d=med(["bound", "eps_3d"], fc), charged_bound_lat=med(["bound", "eps_lat"], fc), charged_bound_pa=med(["bound", "eps_pa"], fc), charged_ion_eps3d=med(["ion", "eps_3d"], fc),
    neutral_bound_eps3d=med(["bound", "eps_3d"], fn), neutral_bound_lat=med(["bound", "eps_lat"], fn),
    resp_bound_eps3d=med(["bound", "eps_3d"], pr), resp_bound_lat=med(["bound", "eps_lat"], pr), resp_bound_corr=med(["bound", "lat_corr"], pr), resp_ion_eps3d=med(["ion", "eps_3d"], pr),
    phi_total_charged=med(["total", "phi_rms_err"], fc), phi_total_neutral=med(["total", "phi_rms_err"], fn), phi_total_resp=med(["total", "phi_rms_err"], pr), phi_bound_charged=med(["bound", "phi_rms_err"], fc),
    zone_model_b_e=med(["zones", "bound_zone_model_e"], fc), zone_model_b_ratio=med(["zones", "bound_zone_model_ratio"], fc), zone_dftcav_b_e=med(["zones", "bound_zone_dftcav_e"], fc), zone_dftcav_b_ratio=med(["zones", "bound_zone_dftcav_ratio"], fc),
    near0_excess_b_e=med(["zones", "bound_near0_excess_e"], fc), near0_excess_b_ratio=med(["zones", "bound_near0_excess_ratio"], fc), near0_excess_i_e=med(["zones", "ion_near0_excess_e"], fc))
json.dump(RES, open(OUT, "w"), indent=1, default=float)
print(f"[{time.time() - T0:5.0f}s] SUMMARY (medians, {len(PAIRS)} pairs): " + ", ".join(f"{k} {v:.4f}" for k, v in RES["summary"].items()), flush=True)
print("done", flush=True)
