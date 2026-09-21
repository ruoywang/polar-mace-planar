"""Paired charging energy of a trained model, term by term, against the DFT
side's own printed solvation pieces -- no training, no flag forcing.

User request 2026-09-21 (after the cavity-derivative discussion):
  1. confirm from the LOADED model object whether the cavity / residual-3D /
     baseline-coupling energies are on, and print one frame's actual values;
  2. on the real charging pairs, take the charged-minus-neutral difference of
     every additive model term and of the DFT's printed A_cav / A_corr, and
     relate them to the paired residual (sign, size, correlation);
  3. (inventory of explicit vs learned terms is written in the ledger, not here.)

Model terms are the ten of charging_reconcile.py (closure-checked against the
model's own total). DFT side: A_cav, A_corr, Ecorr, Ecorr_band from log.out,
energy(sigma->0) / E-fermi / NELECT from OUTCAR, read as printed and NOT mapped
onto model terms except for the one candidate correspondence A_cav <-> cavity
energy, which is reported as a candidate.

  KIT_RUN        run dir (data/{train,val}.xyz)
  KIT_MODEL_OBJ  model object (.model) of an identical-architecture run
  KIT_CKPT       epoch checkpoint (EMA state dict, loaded strict=True)
  KIT_TAG        label for the printout / npz name
  KIT_DFT        DFT payload root (default /scratch/08384/tg876840/tmp/2-NiN_single)
  KIT_PAIRS      comma list of pair ids k (charged k, neutral k+600); default = the
                 20 val pairs + every KIT_STRIDE-th train frame's pair (default 6)
  KIT_OUT        where the npz goes
"""
from __future__ import annotations

import glob
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ.get(
    "KIT_MACE_REPO",
    "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.pb1d_backend as PB

RUN = os.environ["KIT_RUN"]
TAG = os.environ.get("KIT_TAG", "pair_terms")
DFT = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single")
OUT = os.environ.get("KIT_OUT", os.path.join(RUN, "logs"))
STRIDE = int(os.environ.get("KIT_STRIDE", "6"))
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")   # per-frame eviction below

PAIRS_VAL = [28, 30, 43, 60, 61, 62, 69, 79, 83, 94,
             128, 134, 148, 153, 159, 177, 180, 185, 186, 189]


def _rss_gb():
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS"):
                    return float(ln.split()[1]) / 1048576.0
    except OSError:
        pass
    return float("nan")


# ---------------------------------------------------------------- DFT side
def dftdir(sid):
    if 1 <= sid <= 200:
        return f"{DFT}/1-44_GCE/cal_{sid}"
    if 601 <= sid <= 800:
        return f"{DFT}/5-44_neutral_withsolv/cal_{sid - 600}"
    return None


def dft_terms(sid):
    d = dftdir(sid)
    out = {k: float("nan") for k in ("sigma0", "efermi", "nelect", "a_cav",
                                     "a_corr", "ecorr", "ecorr_band")}
    if d is None:
        return out
    try:
        oc = open(f"{d}/OUTCAR", errors="ignore").read()
        for key, pat in (("sigma0", r"energy\(sigma->0\)\s*=\s*([-\d.]+)"),
                         ("nelect", r"NELECT\s*=\s*([\d.]+)"),
                         ("efermi", r"E-fermi\s*:\s*([-\d.]+)")):
            m = re.findall(pat, oc)
            out[key] = float(m[-1]) if m else float("nan")
        lg = open(f"{d}/log.out", errors="ignore").read()
        for key, pat in (("a_cav", r"A_cav:\s*([-\d.Ee+]+)"),
                         ("a_corr", r"A_corr=\s*([-\d.Ee+]+)"),
                         ("ecorr", r"Ecorr=\s*([-\d.Ee+]+)"),
                         ("ecorr_band", r"Ecorr_band=\s*([-\d.Ee+]+)")):
            m = re.findall(pat, lg)
            out[key] = float(m[-1]) if m else float("nan")
    except OSError:
        pass
    return out


# ---------------------------------------------------------------- model
device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
MO = os.environ["KIT_MODEL_OBJ"]
CK = os.environ["KIT_CKPT"]
model = torch.load(f=MO, map_location=device).to(device)
model.load_state_dict(torch.load(CK, map_location=device)["model"], strict=True)
EPOCH = int(CK.rsplit("epoch-", 1)[1].split(".")[0])
model.eval()
model._pb1d_epoch = EPOCH
for p in model.parameters():
    p.requires_grad_(False)

print(f"[{TAG}] model object {os.path.basename(MO)}, weights {os.path.basename(CK)} "
      f"(epoch {EPOCH}); env LIVE_POS={os.environ.get('MACE_PB1D_LIVE_POS','unset')} "
      f"GRAD_PASSES={os.environ.get('MACE_PB1D_GRAD_PASSES','unset')}")
print(f"\n=== STEP 1a: switches AS LOADED from the model object (nothing forced) ===")
for k in ("solvent_model", "solvent_cavity_energy", "solvent3d_energy",
          "solvent_baseline_coupling", "solvent_pb1d_fresh_stage1",
          "solvent_pb1d_warmup_encounters", "num_recursion_steps",
          "solvent_center_mean_shift"):
    print(f"   {k:32s} = {getattr(model, k, '<absent>')}")
print(f"   solvent3d_head present         = {getattr(model, 'solvent3d_head', None) is not None}")
print(f"   pb1d_head present              = {getattr(model, 'pb1d_head', None) is not None}")
print(f"   local_electron_energy present  = {getattr(model, 'local_electron_energy', None) is not None}")

_backend = {}
_bk = PB.PB1DBackend.solve_graph
def _wrap_bk(self, *a, **k):
    _backend["b"] = self
    return _bk(self, *a, **k)
PB.PB1DBackend.solve_graph = _wrap_bk


def _evict():
    b = _backend.get("b")
    if b is not None:
        for nm in ("_bl_ram", "_grids", "_solvers", "_c_units"):
            d = getattr(b, nm, None)
            if isinstance(d, dict):
                d.clear()
    torch.cuda.empty_cache()


z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff",
               "solvated": "solvated"},
    arrays_keys={"forces": "forces"},
)
atoms_by_sid, split_by_sid, train_order = {}, {}, []
for split in ("train", "val"):
    p = os.path.join(RUN, "data", f"{split}.xyz")
    if not os.path.exists(p):
        continue
    for a in read(p, ":"):
        s = int(a.info["sample_id"])
        atoms_by_sid[s] = a
        split_by_sid[s] = split
        if split == "train":
            train_order.append(s)

if os.environ.get("KIT_PAIRS"):
    pairs = [int(x) for x in os.environ["KIT_PAIRS"].split(",") if x.strip()]
else:
    train_pairs = sorted({(s if s <= 200 else s - 600) for s in train_order[::STRIDE]})
    pairs = list(PAIRS_VAL) + [k for k in train_pairs if k not in PAIRS_VAL]
pairs = [k for k in pairs if k in atoms_by_sid and (k + 600) in atoms_by_sid]
print(f"\n   pairs: {len(pairs)} with both frames in train/val "
      f"({sum(1 for k in pairs if split_by_sid[k]=='val' and split_by_sid[k+600]=='val')} wholly in val)")

TERMS = [
    ("e0 (atomic, per-species)", "e0"),
    ("energy head (inter_e)", "interaction_energy"),
    ("local electron energy", "electron_energy"),
    ("charge branch (absent)", "charge_branch_energy"),
    ("solute electrostatic", "electrostatic_energy"),
    ("1D solvent compensation", "compensation_periodic_1d_energy"),
    ("slab dipole correction", "compensation_slab_correction_energy"),
    ("external field . dipole", "extfield_dipole"),
    ("cavity energy tau*A", "cavity_energy_g"),
    ("3D solvent energy", "solvent3d_energy_g"),
    ("baseline coupling E_bl", "baseline_coupling_energy_g"),
]


def run(sid):
    a = atoms_by_sid[sid]
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
    nat = int(b["ptr"][1] - b["ptr"][0])
    heads = (b["head"][b["batch"]] if "head" in b.to_dict() else torch.zeros_like(b["batch"]))
    ar = torch.arange(nat, device=device)
    e0v = model.atomic_energies_fn(b["node_attrs"])
    t = {"e0": float((e0v[ar, heads] if e0v.dim() > 1 else e0v).sum())}
    for _, k in TERMS:
        if k == "e0":
            continue
        if k == "extfield_dipole":
            ef, ed = pred.get("external_field"), pred.get("explicit_dipole")
            t[k] = float((ef * ed).sum()) if ef is not None and ed is not None else 0.0
        else:
            v = pred.get(k)
            t[k] = float(v.sum()) if v is not None else 0.0
    t["_total"] = float(pred["energy"].sum())
    t["_label"] = float(a.info["energy"])
    t["_nat"] = nat
    t["_q"] = float(a.info.get("total_charge", 0.0))
    del pred, b
    _evict()
    return t


# ---------------------------------------------------------------- run
t0 = time.time()
frames = {}
worst_closure = 0.0
for i, k in enumerate(pairs):
    for sid in (k, k + 600):
        frames[sid] = run(sid)
        s = sum(frames[sid][kk] for _, kk in TERMS)
        worst_closure = max(worst_closure, abs(s - frames[sid]["_total"]))
    if i == 0:
        c, n = frames[k], frames[k + 600]
        print(f"\n=== STEP 1b: one pair's actual term values (sid {k} charged / sid {k+600} neutral), eV ===")
        print(f"   {'term':>30} {'neutral':>14} {'charged':>14} {'difference':>14}")
        for lbl, kk in TERMS:
            z = "   (exact zero)" if c[kk] == 0.0 and n[kk] == 0.0 else ""
            print(f"   {lbl:>30} {n[kk]:+14.6f} {c[kk]:+14.6f} {c[kk]-n[kk]:+14.6f}{z}")
        print(f"   {'model total':>30} {n['_total']:+14.6f} {c['_total']:+14.6f} {c['_total']-n['_total']:+14.6f}")
        print(f"   {'DFT label':>30} {n['_label']:+14.6f} {c['_label']:+14.6f} {c['_label']-n['_label']:+14.6f}")
    if (i + 1) % 10 == 0:
        print(f"   {i+1}/{len(pairs)} pairs, {time.time()-t0:.0f} s, RSS {_rss_gb():.2f} GB", flush=True)
print(f"   all {len(pairs)} pairs in {time.time()-t0:.0f} s; closure of the eleven terms vs the "
      f"model total: worst |diff| {worst_closure:.3e} eV [{'PASS' if worst_closure < 1e-6 else 'FAIL'}]")

# ---------------------------------------------------------------- tables
rows = []
for k in pairs:
    c, n, dc, dn = frames[k], frames[k + 600], dft_terms(k), dft_terms(k + 600)
    r = {"k": k, "split": split_by_sid[k] if split_by_sid[k] == split_by_sid[k + 600] else "mixed",
         "dE_model": c["_total"] - n["_total"], "dE_dft": c["_label"] - n["_label"]}
    r["resid"] = r["dE_model"] - r["dE_dft"]
    for _, kk in TERMS:
        r["d_" + kk] = c[kk] - n[kk]
    for kk in ("a_cav", "a_corr", "ecorr", "ecorr_band", "efermi", "sigma0"):
        r["dft_d_" + kk] = dc[kk] - dn[kk]
    r["dft_a_cav_c"], r["dft_a_cav_n"] = dc["a_cav"], dn["a_cav"]
    rows.append(r)

np.savez(os.path.join(OUT, f"pair_terms_{TAG}.npz"),
         rows=np.array(rows, dtype=object), epoch=EPOCH,
         frames=np.array([{**{kk: v for kk, v in f.items()}, "sid": s} for s, f in frames.items()], dtype=object))


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or x[m].std() == 0 or y[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def rms(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.sqrt((x ** 2).mean())) if x.size else float("nan")


for split in ("val", "train"):
    R = [r for r in rows if r["split"] == split]
    if not R:
        continue
    res = np.array([r["resid"] for r in R])
    print(f"\n=== STEP 2 ({split}, {len(R)} pairs): charged-minus-neutral differences, eV ===")
    print(f"   paired residual dE_model - dE_dft: rmse {rms(res):.4f}  bias {res.mean():+.4f}  "
          f"std {res.std():.4f}  ({(res>0).sum()} positive, {(res<0).sum()} negative)")
    print(f"\n   per pair (model term differences; DFT dA_cav is the DFT's own printed cavity term):")
    hdr = (f"   {'k':>4} {'resid':>8} | {'d cav':>8} {'DFT dAcav':>9} | {'d E_bl':>8} {'d 3D':>8} "
           f"{'d 1Dcomp':>8} {'d slab':>8} {'d solES':>8} {'d inter':>8} {'d locEl':>8} | {'DFT dAcorr':>10}")
    print(hdr)
    for r in R:
        print(f"   {r['k']:>4} {r['resid']:+8.3f} | {r['d_cavity_energy_g']:+8.4f} {r['dft_d_a_cav']:+9.4f} | "
              f"{r['d_baseline_coupling_energy_g']:+8.3f} {r['d_solvent3d_energy_g']:+8.3f} "
              f"{r['d_compensation_periodic_1d_energy']:+8.3f} {r['d_compensation_slab_correction_energy']:+8.3f} "
              f"{r['d_electrostatic_energy']:+8.3f} {r['d_interaction_energy']:+8.3f} {r['d_electron_energy']:+8.3f} | "
              f"{r['dft_d_a_corr']:+10.4f}")
    print(f"\n   summary over the {len(R)} pairs: mean / rms of each difference, and its correlation with the residual")
    print(f"   {'term':>30} {'mean':>10} {'rms':>10} {'corr(resid)':>12}")
    for lbl, kk in TERMS:
        d = np.array([r["d_" + kk] for r in R])
        if np.all(d == 0.0):
            print(f"   {lbl:>30} {'0':>10} {'0':>10} {'--':>12}   (exact zero on every pair)")
        else:
            print(f"   {lbl:>30} {d.mean():+10.4f} {rms(d):10.4f} {corr(d, res):+12.3f}")
    dcm = np.array([r["d_cavity_energy_g"] for r in R])
    dcd = np.array([r["dft_d_a_cav"] for r in R])
    m = np.isfinite(dcd)
    agree = int(np.sum(np.sign(dcm[m]) == np.sign(dcd[m])))
    print(f"\n   CANDIDATE correspondence cavity energy <-> DFT A_cav ({m.sum()} pairs with log.out):")
    print(f"     model d(cavity): mean {dcm[m].mean():+.4f}  rms {rms(dcm[m]):.4f} eV")
    print(f"     DFT   dA_cav   : mean {dcd[m].mean():+.4f}  rms {rms(dcd[m]):.4f} eV")
    print(f"     same sign on {agree}/{m.sum()} pairs; corr(model d cav, DFT dA_cav) = {corr(dcm[m], dcd[m]):+.3f}")
    gap = dcm[m] - dcd[m]
    print(f"     gap d(cavity) - dA_cav: mean {gap.mean():+.4f}  rms {rms(gap):.4f} eV; "
          f"corr(gap, resid) = {corr(gap, res[m]):+.3f}")
    res2 = res[m] - gap
    print(f"     IF the model's cavity difference were replaced by the DFT's (candidate only): residual rmse "
          f"{rms(res[m]):.4f} -> {rms(res2):.4f}, bias {res[m].mean():+.4f} -> {res2.mean():+.4f}")
    # absolute per-frame cavity energies against DFT A_cav
    ac_m = np.array([[frames[r['k']]['cavity_energy_g'], frames[r['k']+600]['cavity_energy_g']] for r in R]).ravel()
    ac_d = np.array([[r['dft_a_cav_c'], r['dft_a_cav_n']] for r in R]).ravel()
    mm = np.isfinite(ac_d)
    print(f"     per-frame absolute: model tau*A mean {ac_m[mm].mean():.4f}, DFT A_cav mean {ac_d[mm].mean():.4f} eV; "
          f"model - DFT mean {(ac_m[mm]-ac_d[mm]).mean():+.4f}, rms {rms(ac_m[mm]-ac_d[mm]):.4f} eV, "
          f"corr {corr(ac_m[mm], ac_d[mm]):+.3f}")
    dac = np.array([r["dft_d_a_corr"] for r in R]); dsl = np.array([r["d_compensation_slab_correction_energy"] for r in R])
    print(f"   DFT dA_corr vs model d(slab dipole correction) (the 09-10 candidate): mean {dac[m].mean():+.4f} vs {dsl[m].mean():+.4f}, "
          f"corr {corr(dac[m], dsl[m]):+.3f}, rms diff {rms(dac[m]-dsl[m]):.4f} eV")
print("\nDONE")
