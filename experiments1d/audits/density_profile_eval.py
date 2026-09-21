"""Plane-averaged net charge density of a checkpoint against the DFT density grid (the
density_3d training target), for the same pairs ebl_audit.py uses (20 val pairs + stride-6
train pairs). What the v_new gate is judged on besides E/F (user 2026-09-21): density tail,
charge response, solvent profile (the last one comes from ebl_audit.py).

Per frame the model's GTO density (charge_density_radial_coefficients, the training sigmas)
is evaluated on every grid point of the valid planes of the DFT grid (177-178 planes of
168x168, z window about 3.6-19.6 A) and plane-averaged: n(z) in e/A^3, "net" = electrons
(negative) + smeared ion charge (positive), the target's own convention. Reported:
  * whole window: L1(model-DFT) in electrons, 3-D rmse over the valid region (same quantity
    as the validation density_3d rmse, but on all points instead of 1024 samples);
  * tails: below the lowest atom - 1 A and above the highest atom + 1 A, the integral of
    model-DFT (electrons; the ion part is the same convention on both sides, so this is the
    electron-tail error) and the L1 there;
  * charge response of each pair (k charged, k+600 neutral, identical geometry): the pair
    difference dn(z) is pure electron density; L1(model-DFT), its integral, its centroid,
    and the share of the extra charge sitting in the metal region (z <= top solute atom + 1 A).
Env: KIT_RUN, KIT_MODEL_OBJ, KIT_CKPT, KIT_TAG, KIT_OUT, KIT_STRIDE (6), KIT_PAIRS,
KIT_VSOLV=1 for a checkpoint trained with the stage-1 effective-potential input.
Writes <OUT>/density_profile_<TAG>.npz with the per-frame profiles.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.environ.get(
    "KIT_MACE_REPO",
    "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.pb1d_backend as PB
from mace.modules.loss import _gto_density_at_points_axis2_pbc, _parse_density_sigmas

RUN = os.environ["KIT_RUN"]
TAG = os.environ.get("KIT_TAG", "density_profile")
OUT = os.path.abspath(os.environ.get("KIT_OUT", os.path.join(RUN, "logs")))
STRIDE = int(os.environ.get("KIT_STRIDE", "6"))
PLANES_PER_CHUNK = int(os.environ.get("KIT_PLANES_PER_CHUNK", "2"))
TAIL_OFFSET = 1.0   # A beyond the outermost atom
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")
os.chdir(RUN)

PAIRS_VAL = [28, 30, 43, 60, 61, 62, 69, 79, 83, 94,
             128, 134, 148, 153, 159, 177, 180, 185, 186, 189]

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
MO, CK = os.environ["KIT_MODEL_OBJ"], os.environ["KIT_CKPT"]
model = torch.load(f=MO, map_location=device).to(device)
model.load_state_dict(torch.load(CK, map_location=device)["model"], strict=True)
EPOCH = int(CK.rsplit("epoch-", 1)[1].split(".")[0])
model.eval(); model._pb1d_epoch = EPOCH
if os.environ.get("KIT_VSOLV", "") not in ("", "0"):
    model.solvent_pb1d_vsolv_input = True
for p in model.parameters():
    p.requires_grad_(False)
cfg_yaml = yaml.safe_load(open("config_pb1d.yaml"))
SIGMAS = _parse_density_sigmas(cfg_yaml["density_3d_sigma"])
manifest = json.load(open(cfg_yaml["density_3d_file"]))["entries"]
print(f"[{TAG}] {os.path.basename(MO)} @ epoch {EPOCH}; solvent_pb1d_vsolv_input={getattr(model, 'solvent_pb1d_vsolv_input', False)}; "
      f"density sigmas {SIGMAS}; LIVE_POS={os.environ.get('MACE_PB1D_LIVE_POS','unset')} GRAD_PASSES={os.environ.get('MACE_PB1D_GRAD_PASSES','unset')}")

_cap = {}
_bk = PB.PB1DBackend.solve_graph
def _wrap_bk(self, *a, **k):
    _cap["b"] = self
    return _bk(self, *a, **k)
PB.PB1DBackend.solve_graph = _wrap_bk


def _evict():
    b = _cap.get("b")
    if b is not None:
        for nm in ("_bl_ram", "_grids", "_solvers", "_c_units"):
            d = getattr(b, nm, None)
            if isinstance(d, dict):
                d.clear()
    torch.cuda.empty_cache()


z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin",
               "sample_id": "sample_id", "fermi_level": "Fermi", "potential": "potential_diff",
               "solvated": "solvated"},
    arrays_keys={"forces": "forces"})
atoms_by_sid, train_order = {}, []
for split in ("train", "val"):
    p = os.path.join(RUN, "data", f"{split}.xyz")
    if os.path.exists(p):
        for a in read(p, ":"):
            s = int(a.info["sample_id"]); atoms_by_sid[s] = a
            if split == "train":
                train_order.append(s)
if os.environ.get("KIT_PAIRS"):
    pairs = [int(x) for x in os.environ["KIT_PAIRS"].split(",") if x.strip()]
else:
    tp = sorted({(s if s <= 200 else s - 600) for s in train_order[::STRIDE]})
    pairs = list(PAIRS_VAL) + [k for k in tp if k not in PAIRS_VAL]
pairs = [k for k in pairs if k in atoms_by_sid and (k + 600) in atoms_by_sid and str(k) in manifest and str(k + 600) in manifest]


def dft_planes(sid):
    e = manifest[str(sid)]
    meta = np.load(e["meta_path"])
    lattice = np.asarray(meta["lattice"], float); valid_iz = np.asarray(meta["valid_iz"], np.int64)
    mm = np.load(e["path"], mmap_mode="r")
    nz, ny, nx = mm.shape
    lo, hi = int(valid_iz[0]), int(valid_iz[-1])
    assert valid_iz.size == hi - lo + 1, "non-contiguous valid_iz"
    sub = np.array(mm[lo:hi + 1], dtype=np.float32, copy=True)
    del mm
    return sub, lattice, valid_iz, nz, ny, nx


def run(sid):
    a = atoms_by_sid[sid]
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
    coeffs = pred["charge_density_radial_coefficients"].detach()
    sub, lattice, valid_iz, nz, ny, nx = dft_planes(sid)
    cell = torch.as_tensor(lattice, dtype=torch.float64, device=device)
    pos = b["positions"].detach(); zs = b["atomic_numbers"].to(torch.long)
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")   # iy rows, ix cols -> matches rho[iz, iy, ix]
    fx = (ix.ravel() / nx).astype(float); fy = (iy.ravel() / ny).astype(float)
    n_pred = np.zeros(valid_iz.size); n_dft = sub.reshape(valid_iz.size, -1).mean(axis=1).astype(float)
    sq = 0.0; ab = 0.0
    with torch.no_grad():
        for c0 in range(0, valid_iz.size, PLANES_PER_CHUNK):
            c1 = min(c0 + PLANES_PER_CHUNK, valid_iz.size)
            frac = np.concatenate([np.column_stack([fx, fy, np.full(fx.size, valid_iz[j] / nz)]) for j in range(c0, c1)])
            pts = torch.as_tensor(frac @ lattice, dtype=torch.float64, device=device)
            rho = _gto_density_at_points_axis2_pbc(points=pts, density_coefficients=coeffs, positions=pos,
                                                   atomic_numbers=zs, cell=cell, sigma=SIGMAS)
            rho = rho.view(c1 - c0, -1)
            ref = torch.as_tensor(sub[c0:c1].reshape(c1 - c0, -1), dtype=torch.float64, device=device)
            n_pred[c0:c1] = rho.mean(dim=1).cpu().numpy()
            d = rho - ref
            sq += float((d * d).sum()); ab += float(d.abs().sum())
    npts = valid_iz.size * ny * nx
    area = float(abs(np.cross(lattice[0], lattice[1])[2])); dz = float(lattice[2, 2]) / nz
    z = valid_iz / nz * float(lattice[2, 2])
    zat = np.asarray(a.positions[:, 2], float)
    sym = np.array(a.get_chemical_symbols()); solute = ~np.isin(sym, ["O", "H"])
    rec = {"sid": sid, "nat": len(a), "q": float(a.info.get("total_charge", 0.0)),
           "state": ("charged" if abs(float(a.info.get("total_charge", 0.0))) > 1e-6 else "neutral") + (" NiN44" if len(a) == 207 else " NiN88"),
           "z": z, "n_pred": n_pred, "n_dft": n_dft, "area": area, "dz": dz,
           "z_atom_min": float(zat.min()), "z_atom_max": float(zat.max()), "z_solute_max": float(zat[solute].max()),
           "rmse3d": float(np.sqrt(sq / npts)), "mae3d": ab / npts}
    w = area * dz
    d = n_pred - n_dft
    rec["L1_window_e"] = float(np.abs(d).sum() * w)
    rec["N_window_model"] = float(n_pred.sum() * w); rec["N_window_dft"] = float(n_dft.sum() * w)
    lo = z < rec["z_atom_min"] - TAIL_OFFSET; hi = z > rec["z_atom_max"] + TAIL_OFFSET
    for nm, m in (("bottom", lo), ("top", hi)):
        rec[f"tail_{nm}_planes"] = int(m.sum())
        rec[f"tail_{nm}_dN_e"] = float(d[m].sum() * w)          # model - DFT electrons in the tail
        rec[f"tail_{nm}_L1_e"] = float(np.abs(d[m]).sum() * w)
        rec[f"tail_{nm}_N_dft_e"] = float(n_dft[m].sum() * w)
        rec[f"tail_{nm}_N_model_e"] = float(n_pred[m].sum() * w)
    del pred, b
    _evict()
    return rec


t0 = time.time(); F = {}
for i, k in enumerate(pairs):
    for sid in (k, k + 600):
        F[sid] = run(sid)
    if (i + 1) % 10 == 0:
        print(f"   {i+1}/{len(pairs)} pairs, {time.time()-t0:.0f} s", flush=True)
print(f"   all {len(pairs)} pairs ({2*len(pairs)} frames) in {time.time()-t0:.0f} s")

# pair charge response
P = {}
for k in pairs:
    c, n = F[k], F[k + 600]
    assert np.allclose(c["z"], n["z"]) and abs(c["area"] - n["area"]) < 1e-9
    w = c["area"] * c["dz"]; z = c["z"]
    dm = c["n_pred"] - n["n_pred"]; dd = c["n_dft"] - n["n_dft"]
    metal = z <= c["z_solute_max"] + TAIL_OFFSET
    rec = {"k": k, "state": c["state"].replace("charged ", ""), "q": c["q"],
           "dN_model": float(dm.sum() * w), "dN_dft": float(dd.sum() * w),
           "L1_e": float(np.abs(dm - dd).sum() * w), "L1_dft_e": float(np.abs(dd).sum() * w),
           "zc_model": float((dm * z).sum() / dm.sum()) if abs(dm.sum()) > 1e-9 else float("nan"),
           "zc_dft": float((dd * z).sum() / dd.sum()) if abs(dd.sum()) > 1e-9 else float("nan"),
           "metal_share_model": float(dm[metal].sum() / dm.sum()) if abs(dm.sum()) > 1e-9 else float("nan"),
           "metal_share_dft": float(dd[metal].sum() / dd.sum()) if abs(dd.sum()) > 1e-9 else float("nan"),
           "dn_model": dm, "dn_dft": dd, "z": z}
    P[k] = rec

np.savez(os.path.join(OUT, f"density_profile_{TAG}.npz"),
         frames=np.array(list(F.values()), dtype=object), pairs=np.array(list(P.values()), dtype=object), epoch=EPOCH,
         sigmas=np.array(SIGMAS), tail_offset=TAIL_OFFSET)


def mean(x): x = np.asarray(x, float); x = x[np.isfinite(x)]; return float(x.mean()) if x.size else float("nan")
def rms(x): x = np.asarray(x, float); x = x[np.isfinite(x)]; return float(np.sqrt((x ** 2).mean())) if x.size else float("nan")


allf = list(F.values())
print(f"\n=== per state, plane-averaged net density vs DFT grid (epoch {EPOCH}, {len(allf)} frames) ===")
print(f"   {'state':14s} {'n':>3} {'rmse3d':>9} {'L1 win':>8} {'N win m/dft':>17} {'tail_bot dN':>11} {'L1':>7} {'N_dft':>8} {'tail_top dN':>11} {'L1':>7} {'N_dft':>8}   (e/A^3; electrons)")
for st in sorted({r["state"] for r in allf}):
    R = [r for r in allf if r["state"] == st]
    print(f"   {st:14s} {len(R):>3} {mean([r['rmse3d'] for r in R]):9.5f} {mean([r['L1_window_e'] for r in R]):8.4f} "
          f"{mean([r['N_window_model'] for r in R]):+8.4f}/{mean([r['N_window_dft'] for r in R]):+8.4f} "
          f"{mean([r['tail_bottom_dN_e'] for r in R]):+11.5f} {mean([r['tail_bottom_L1_e'] for r in R]):7.5f} {mean([r['tail_bottom_N_dft_e'] for r in R]):+8.5f} "
          f"{mean([r['tail_top_dN_e'] for r in R]):+11.5f} {mean([r['tail_top_L1_e'] for r in R]):7.5f} {mean([r['tail_top_N_dft_e'] for r in R]):+8.5f}")
print(f"   tails: below lowest atom - {TAIL_OFFSET} A / above highest atom + {TAIL_OFFSET} A, within the grid window "
      f"({mean([r['tail_bottom_planes'] for r in allf]):.0f} / {mean([r['tail_top_planes'] for r in allf]):.0f} planes of {mean([r['dz'] for r in allf]):.3f} A)")

allp = list(P.values())
print(f"\n=== charge response of the pairs: dn(z) = n(charged) - n(neutral), model vs DFT ({len(allp)} pairs) ===")
print(f"   {'system':6s} {'n':>3} {'L1(m-d)':>8} {'L1 dft':>7} {'dN model':>9} {'dN dft':>8} {'zc model':>9} {'zc dft':>7} {'metal share m/dft':>18}   (electrons; A)")
for st in sorted({r["state"] for r in allp}):
    R = [r for r in allp if r["state"] == st]
    print(f"   {st:6s} {len(R):>3} {mean([r['L1_e'] for r in R]):8.4f} {mean([r['L1_dft_e'] for r in R]):7.4f} "
          f"{mean([r['dN_model'] for r in R]):+9.4f} {mean([r['dN_dft'] for r in R]):+8.4f} "
          f"{mean([r['zc_model'] for r in R]):9.3f} {mean([r['zc_dft'] for r in R]):7.3f} "
          f"{mean([r['metal_share_model'] for r in R]):8.3f}/{mean([r['metal_share_dft'] for r in R]):<8.3f}")
print(f"   {'k':>4} {'sys':>5} {'q':>7} {'L1(m-d)':>8} {'dN m':>7} {'dN d':>7} {'zc m':>6} {'zc d':>6} {'metal m':>7} {'metal d':>7}")
for r in allp:
    print(f"   {r['k']:>4} {r['state']:>5} {r['q']:+7.3f} {r['L1_e']:8.4f} {r['dN_model']:+7.3f} {r['dN_dft']:+7.3f} {r['zc_model']:6.2f} {r['zc_dft']:6.2f} {r['metal_share_model']:7.3f} {r['metal_share_dft']:7.3f}")
print("\nDONE")
