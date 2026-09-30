"""Full-grid evaluation of the model's 3-D solvent charge against the DFT label grids (user request 2026-09-30).

The model's 3-D solvent charge per channel (b = bound, i = ionic) is the field the solvent3d loss scores:
    rho_ch(r) = rho_ch_1d(z)          (1-D PB profile of the final solve, periodic linear interpolation in z)
              + d_sup_ch(r)           (per-plane-projected lateral residual on the model grid, trilinear)
exactly as mace.modules.solvent3d.solvent3d_residuals builds its prediction at the sampled points; here the
points are EVERY point of the DFT label grid (500 x 168 x 168). The solve is captured by wrapping
backend.solve_graph (returned dict of the LAST call of a forward = the stage-2 solve); nothing is re-solved.
Labels: data/solvent3d_grid (physics convention, e/A^3, rho = -RHOX/V, (nz, ny, nx), f32).

Per frame and channel (b, i, t = b + i), with d = ML - DFT and dV the grid volume element:
  eps_3d  = sum|d| / sum|DFT|                    (+ the numerator abs_err_e = sum|d| dV in e)
  eps_lat = sum|d_perp| / sum|DFT_perp|          (x_perp = x - <x>_xy of that plane, taken separately for ML and DFT)
  eps_pa  = sum|<d>_xy| / sum|<DFT>_xy|          (the plane-averaged part)
  net_err = sum d dV                             (e)
  rmse    = sqrt(mean d^2)                       (e/A^3, same quantity as the training metric)
Per z plane: err_z = sum_xy |d| dA, ref_z = sum_xy |DFT| dA, lat_err_z = sum_xy |d_perp| dA (e/A), plane means.
Twin pairs (charged k, solvated neutral k + 600, same geometry): the same metrics on
  Delta = rho(charged) - rho(neutral) for ML and DFT.
Saved: metrics.json, perz.npz, slices.npz (xz plane through the Ni atom, xy planes at the DFT bound / ionic
peak planes, for every frame and pair), volumes.npz (stride-2 3-D fields of the representative frames).

Usage: python solvent3d_full_eval.py <out_dir>        (run from a dir with ./data -> the training data dir)
Env: KIT_MODEL (model file), KIT_SPLIT (default val), KIT_EXTRA_PAIRS (default "122" = sid 122 / 722 from test),
     KIT_DEVICE (cuda / cpu), KIT_MAX_FRAMES (smoke test: stop after this many labelled frames).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")

import numpy as np
import torch
from ase.io import read

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils
from mace.modules.solvent3d import _interp1_periodic, _interp3_periodic

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "s3d_eval")
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ.get("KIT_MODEL", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_vsolv_fix/models/prod500_vsolv_fix.model")
SPLIT = os.environ.get("KIT_SPLIT", "val")
EXTRA = [int(s) for s in os.environ.get("KIT_EXTRA_PAIRS", "122").split()] if os.environ.get("KIT_EXTRA_PAIRS", "122").strip() else []
MAXF = int(os.environ.get("KIT_MAX_FRAMES", "0"))
dev = os.environ.get("KIT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
device = torch.device(dev)
if device.type == "cpu":
    # e3nn codegen submodules are TorchScript buffers saved on CUDA; their __setstate__ calls
    # torch.jit.load(buffer) without map_location, which fails on a node without a GPU driver
    _jit_load = torch.jit.load
    torch.jit.load = lambda f, map_location=None, **kw: _jit_load(f, map_location="cpu", **kw)
model = torch.load(MODEL, map_location=device).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
backend = model._get_pb1d_backend()
print(f"model {os.path.basename(MODEL)}; device {device}; vsolv_input={getattr(model, 'solvent_pb1d_vsolv_input', False)}; "
      f"solvent3d_energy={getattr(model, 'solvent3d_energy', None)}; split {SPLIT}; extra pairs {EXTRA}", flush=True)

man = json.load(open("data/solvent3d_grid_manifest.json"))
assert man["format"] == "solvent3d_grid_npy_v1", man["format"]
ENT = {int(k): v for k, v in man["entries"].items()}
LBL = Path("data").resolve()

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin",
               "sample_id": "sample_id", "solvated": "solvated", "fermi_level": "Fermi", "potential": "potential_diff"},
    arrays_keys={"forces": "forces"})


def group_of(sid):
    return "NiN44 charged" if sid <= 200 else "NiN88 charged" if sid <= 400 else "NiN44 vacuum neutral" if sid <= 600 else "NiN44 solvated neutral"


def load_label(sid):
    e = ENT[sid]
    rb = np.asarray(np.load(LBL / e["path_b"], mmap_mode="r"), dtype=np.float64)
    ri = np.asarray(np.load(LBL / e["path_i"], mmap_mode="r"), dtype=np.float64)
    with np.load(LBL / e["meta_path"]) as m:
        lat = np.asarray(m["lattice"], dtype=np.float64)
    return rb, ri, lat


def model_fields(atoms):
    """Forward one frame; return the ML bound / ionic 3-D charge on the DFT label grid (nz, ny, nx)."""
    cfgm = mace_data.config_from_atoms(atoms, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfgm, z_table=z_table, cutoff=float(model.r_max))]
    batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    holder = {}
    orig = backend.solve_graph

    def spy(*a, **kw):
        res = orig(*a, **kw)
        holder["res"] = res; holder["n"] = holder.get("n", 0) + 1
        return res
    backend.solve_graph = spy
    try:
        with torch.no_grad():
            out = model(batch.to_dict(), training=False, compute_force=False)
    finally:
        backend.solve_graph = orig
    res = holder["res"]
    obs = res.get("s3d_obs")
    if obs is None or "d_sup_b" not in obs:
        raise RuntimeError("stage-2 solve returned no s3d_obs / d_sup fields")
    return (res["rho_bound_z"].detach(), res["rho_ion_z"].detach(), obs["d_sup_b"].detach(), obs["d_sup_i"].detach(),
            holder["n"], out)


def on_label_grid(rb_z, ri_z, db, di, shape):
    nz, ny, nx = shape
    mb = np.empty(shape, dtype=np.float64); mi = np.empty(shape, dtype=np.float64)
    iy, ix = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij")
    fxy = torch.stack([ix.reshape(-1).to(torch.float64) / nx, iy.reshape(-1).to(torch.float64) / ny], dim=1)
    CH = 25
    for z0 in range(0, nz, CH):
        z1 = min(z0 + CH, nz)
        fz = (torch.arange(z0, z1, device=device, dtype=torch.float64) / nz)
        frac = torch.cat([fxy.repeat(z1 - z0, 1), fz.repeat_interleave(ny * nx)[:, None]], dim=1)
        vb = _interp1_periodic(rb_z.to(torch.float64), frac[:, 2]) + _interp3_periodic(db.to(torch.float64), frac)
        vi = _interp1_periodic(ri_z.to(torch.float64), frac[:, 2]) + _interp3_periodic(di.to(torch.float64), frac)
        mb[z0:z1] = vb.reshape(z1 - z0, ny, nx).cpu().numpy(); mi[z0:z1] = vi.reshape(z1 - z0, ny, nx).cpu().numpy()
    return mb, mi


def metrics(M, D, dV, dA):
    d = M - D
    pm_m, pm_d = M.mean(axis=(1, 2)), D.mean(axis=(1, 2))
    dp = d - (pm_m - pm_d)[:, None, None]
    Dp = D - pm_d[:, None, None]
    nxy = M.shape[1] * M.shape[2]
    r = dict(abs_err_e=float(np.abs(d).sum() * dV), ref_abs_e=float(np.abs(D).sum() * dV),
             lat_abs_err_e=float(np.abs(dp).sum() * dV), lat_ref_abs_e=float(np.abs(Dp).sum() * dV),
             pa_abs_err_e=float(np.abs(pm_m - pm_d).sum() * nxy * dV), pa_ref_abs_e=float(np.abs(pm_d).sum() * nxy * dV),
             net_ml_e=float(M.sum() * dV), net_dft_e=float(D.sum() * dV), rmse=float(np.sqrt((d * d).mean())),
             max_abs_dft=float(np.abs(D).max()), max_abs_err=float(np.abs(d).max()))
    r["net_err_e"] = r["net_ml_e"] - r["net_dft_e"]
    for k, a, b in (("eps_3d", "abs_err_e", "ref_abs_e"), ("eps_lat", "lat_abs_err_e", "lat_ref_abs_e"), ("eps_pa", "pa_abs_err_e", "pa_ref_abs_e")):
        r[k] = r[a] / r[b] if r[b] > 0 else float("nan")
    perz = dict(err_z=np.abs(d).sum(axis=(1, 2)) * dA, ref_z=np.abs(D).sum(axis=(1, 2)) * dA,
                lat_err_z=np.abs(dp).sum(axis=(1, 2)) * dA, pm_ml=pm_m, pm_dft=pm_d)
    return r, perz


def geometry(atoms, lat, shape):
    nz, ny, nx = shape
    frac = np.remainder(atoms.get_scaled_positions(wrap=False), 1.0)
    sym = np.array(atoms.get_chemical_symbols())
    ni = np.where(sym == "Ni")[0]
    iy_ni = int(np.round(frac[ni[0], 1] * ny)) % ny if ni.size else ny // 2
    return frac, sym, iy_ni


def slice_pack(tag, fields, atoms, lat, shape, zb, zi, iy_ni):
    """xz plane through the Ni atom (fixed iy) and xy planes at the bound / ionic peak planes."""
    pk = {}
    for name, F in fields.items():
        pk[f"{tag}|{name}|xz"] = F[:, iy_ni, :].astype(np.float32)
        pk[f"{tag}|{name}|xyb"] = F[zb].astype(np.float32)
        pk[f"{tag}|{name}|xyi"] = F[zi].astype(np.float32)
    pk[f"{tag}|meta"] = np.array([iy_ni, zb, zi, shape[0], shape[1], shape[2]], dtype=np.int64)
    pk[f"{tag}|lat"] = lat.astype(np.float64)
    pk[f"{tag}|frac"] = np.remainder(atoms.get_scaled_positions(wrap=False), 1.0)
    pk[f"{tag}|sym"] = np.array(atoms.get_chemical_symbols())
    return pk


frames = read(f"data/{SPLIT}.xyz", ":")
by_sid = {int(a.info["sample_id"]): a for a in frames}
test_by_sid = {int(a.info["sample_id"]): a for a in read("data/test.xyz", ":")} if EXTRA else {}
jobs = []   # (kind, sid_c, sid_n) ; single frames have sid_n None
for s in sorted(by_sid):
    if s <= 200 and (s + 600) in by_sid and s in ENT and (s + 600) in ENT:
        jobs.append(("pair", s, s + 600))
for s in sorted(by_sid):
    if 200 < s <= 400 and s in ENT:
        jobs.append(("single", s, None))
for s in EXTRA:
    if s in test_by_sid and (s + 600) in test_by_sid:
        jobs.append(("pair_extra", s, s + 600))
print(f"{len(jobs)} jobs: {sum(j[0] == 'pair' for j in jobs)} {SPLIT} twin pairs, {sum(j[0] == 'single' for j in jobs)} NiN88 frames, "
      f"{sum(j[0] == 'pair_extra' for j in jobs)} extra pairs", flush=True)

rows, pair_rows, perz, slices = [], [], {}, {}
done = 0; t0 = time.time()


def eval_frame(a, sid, split_tag):
    rb, ri, lat = load_label(sid)
    assert np.allclose(lat, a.cell[:], atol=1e-4), f"sid {sid}: label lattice != frame cell"
    shape = rb.shape
    tf0 = time.time()
    rb_z, ri_z, db, di, ncall, out = model_fields(a)
    tf1 = time.time()
    mb, mi = on_label_grid(rb_z, ri_z, db, di, shape)
    tf2 = time.time()
    vol = abs(float(np.linalg.det(lat))); dV = vol / float(np.prod(shape)); dA = vol / lat[2, 2] / (shape[1] * shape[2])
    rec = dict(sid=sid, group=group_of(sid), split=split_tag, n_solve_calls=int(ncall), t_forward_s=round(tf1 - tf0, 2), t_grid_s=round(tf2 - tf1, 2),
               q=float(a.info.get("total_charge", 0.0)), grid=list(shape), model_grid=list(db.shape), model_nz_1d=int(rb_z.shape[0]))
    pz = {}
    for ch, M, D in (("b", mb, rb), ("i", mi, ri), ("t", mb + mi, rb + ri)):
        r, p = metrics(M, D, dV, dA)
        rec[ch] = r
        for k, v in p.items():
            pz[f"{sid}|{ch}|{k}"] = v.astype(np.float32)
    ref_zb = np.abs(rb).sum(axis=(1, 2)); ref_zi = np.abs(ri).sum(axis=(1, 2))
    zb, zi = int(np.argmax(ref_zb)), int(np.argmax(ref_zi))
    frac, sym, iy_ni = geometry(a, lat, shape)
    rec.update(zb_plane=zb, zi_plane=zi, zb=float(zb * lat[2, 2] / shape[0]), zi=float(zi * lat[2, 2] / shape[0]), iy_ni=iy_ni)
    pk = slice_pack(str(sid), {"dft_b": rb, "ml_b": mb, "dft_i": ri, "ml_i": mi}, a, lat, shape, zb, zi, iy_ni)
    return rec, pz, pk, (mb, mi, rb, ri, lat, dV, dA, zb, zi, iy_ni)


for kind, sc, sn in jobs:
    try:
        src = test_by_sid if kind == "pair_extra" else by_sid
        tag = "test" if kind == "pair_extra" else SPLIT
        recc, pzc, pkc, gc = eval_frame(src[sc], sc, tag)
        rows.append(recc); perz.update(pzc); slices.update(pkc); done += 1
        print(f"[{time.time() - t0:6.0f} s] sid {sc:3d} {recc['group']:24s} eps3d b {recc['b']['eps_3d']:.3f} i {recc['i']['eps_3d']:.3f} "
              f"t {recc['t']['eps_3d']:.3f} | lat b {recc['b']['eps_lat']:.3f} | net err b {recc['b']['net_err_e']:+.4f} i {recc['i']['net_err_e']:+.4f} e "
              f"| rmse b {recc['b']['rmse']:.3e} i {recc['i']['rmse']:.3e} | model grid {recc['model_grid']} calls {recc['n_solve_calls']} | forward {recc['t_forward_s']} s grid {recc['t_grid_s']} s", flush=True)
        if sn is not None:
            recn, pzn, pkn, gn = eval_frame(src[sn], sn, tag)
            rows.append(recn); perz.update(pzn); slices.update(pkn); done += 1
            print(f"[{time.time() - t0:6.0f} s] sid {sn:3d} {recn['group']:24s} eps3d b {recn['b']['eps_3d']:.3f} i {recn['i']['eps_3d']:.3f} "
                  f"t {recn['t']['eps_3d']:.3f} | lat b {recn['b']['eps_lat']:.3f} | net err b {recn['b']['net_err_e']:+.4f} i {recn['i']['net_err_e']:+.4f} e "
                  f"| abs err i {recn['i']['abs_err_e']:.4f} e (ref {recn['i']['ref_abs_e']:.4f})", flush=True)
            mbc, mic, rbc, ric, lat, dV, dA, zb, zi, iy_ni = gc
            mbn, min_, rbn, rin = gn[0], gn[1], gn[2], gn[3]
            prec = dict(sid_c=sc, sid_n=sn, split=tag)
            for ch, M, D in (("b", mbc - mbn, rbc - rbn), ("i", mic - min_, ric - rin), ("t", (mbc + mic) - (mbn + min_), (rbc + ric) - (rbn + rin))):
                r, p = metrics(M, D, dV, dA)
                prec[ch] = r
                for k, v in p.items():
                    perz[f"d{sc}|{ch}|{k}"] = v.astype(np.float32)
            pair_rows.append(prec)
            if kind == "pair_extra" or sc == jobs[0][1]:
                slices.update(slice_pack(f"d{sc}", {"dft_b": rbc - rbn, "ml_b": mbc - mbn, "dft_i": ric - rin, "ml_i": mic - min_},
                                         src[sc], lat, rbc.shape, zb, zi, iy_ni))
            print(f"          pair {sc}/{sn}: Delta eps3d b {prec['b']['eps_3d']:.3f} i {prec['i']['eps_3d']:.3f} t {prec['t']['eps_3d']:.3f} | "
                  f"Delta net err t {prec['t']['net_err_e']:+.4f} e (DFT {prec['t']['net_dft_e']:+.4f})", flush=True)
            del gn
        del gc
    except Exception:
        print(f"FAILED job {kind} {sc}/{sn}:\n{traceback.format_exc()}", flush=True)
    json.dump(dict(model=os.path.basename(MODEL), split=SPLIT, rows=rows, pairs=pair_rows, partial=True), open(OUT / "metrics_partial.json", "w"))
    if MAXF and done >= MAXF:
        print(f"KIT_MAX_FRAMES={MAXF} reached, stopping the frame loop", flush=True)
        break

# representative frames: charged frames of the split, eps_3d of the total charge closest to the median, and the maximum
ch_rows = [r for r in rows if r["split"] == SPLIT and r["group"] in ("NiN44 charged", "NiN88 charged")]
reps = {}
if ch_rows:
    e = np.array([r["t"]["eps_3d"] for r in ch_rows])
    med = float(np.median(e))
    reps["median"] = int(ch_rows[int(np.argmin(np.abs(e - med)))]["sid"])
    reps["max"] = int(ch_rows[int(np.argmax(e))]["sid"])
    print(f"representative frames: median eps3d(t) {med:.3f} -> sid {reps['median']}; max {e.max():.3f} -> sid {reps['max']}", flush=True)
if 122 in [r["sid"] for r in rows]:
    reps["pair_page"] = 122
vols = {}
for role, sid in reps.items():
    try:
        a = by_sid.get(sid) or test_by_sid.get(sid)
        rb, ri, lat = load_label(sid)
        rb_z, ri_z, db, di, _, _ = model_fields(a)
        mb, mi = on_label_grid(rb_z, ri_z, db, di, rb.shape)
        for name, F in (("dft_b", rb), ("ml_b", mb), ("dft_i", ri), ("ml_i", mi)):
            vols[f"{sid}|{name}"] = F[::2, ::2, ::2].astype(np.float32)
        vols[f"{sid}|lat"] = lat; vols[f"{sid}|frac"] = np.remainder(a.get_scaled_positions(wrap=False), 1.0)
        vols[f"{sid}|sym"] = np.array(a.get_chemical_symbols())
        print(f"volume saved for {role} sid {sid}", flush=True)
    except Exception:
        print(f"FAILED volume {role} {sid}:\n{traceback.format_exc()}", flush=True)

json.dump(dict(model=os.path.basename(MODEL), split=SPLIT, rows=rows, pairs=pair_rows, representative=reps,
               definitions=__doc__), open(OUT / "metrics.json", "w"), indent=1)
np.savez_compressed(OUT / "perz.npz", **perz)
np.savez_compressed(OUT / "slices.npz", **slices)
np.savez_compressed(OUT / "volumes.npz", **vols)


def summary(rs, label):
    for ch in ("b", "i", "t"):
        e3 = np.array([r[ch]["eps_3d"] for r in rs]); el = np.array([r[ch]["eps_lat"] for r in rs])
        ne = np.array([r[ch]["net_err_e"] for r in rs]); ae = np.array([r[ch]["abs_err_e"] for r in rs])
        print(f"  {label:26s} {ch}: n {len(rs):2d}  eps3d median {np.median(e3):.3f} [{e3.min():.3f}, {e3.max():.3f}]  eps_lat median {np.median(el):.3f}  "
              f"abs err median {np.median(ae):.4f} e  net err mean {ne.mean():+.4f} rms {np.sqrt((ne ** 2).mean()):.4f} e", flush=True)


print("===== summary =====")
for g in ("NiN44 charged", "NiN88 charged", "NiN44 solvated neutral"):
    rs = [r for r in rows if r["group"] == g and r["split"] == SPLIT]
    if rs:
        summary(rs, f"{SPLIT} {g}")
prs = [p for p in pair_rows if p["split"] == SPLIT]
if prs:
    summary(prs, f"{SPLIT} pairs (charged-neutral)")
print(f"wrote {OUT}/metrics.json perz.npz slices.npz volumes.npz; {done} frames in {time.time() - t0:.0f} s", flush=True)
print("all_done", flush=True)
