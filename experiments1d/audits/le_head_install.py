"""Install the head that was fitted on the paired residual, then check what the
paired objective could not see: each state's ABSOLUTE energy, and the forces.
(user-specified follow-on, 2026-09-11; runs only because the full-set paired
error dropped from 4.9295 to 0.1359 eV on val.)

THE BLIND SPOT BEING TESTED. The fit minimised (dE_LE - target)^2 over pairs.
Any term equal in the two states -- a structure function C(R) of the geometry
alone -- cancels exactly in dE and is therefore completely unconstrained by
that objective, while it still shifts both absolute energies and, through
dC/dR, the forces. So a 36x drop in the paired error licenses nothing about the
model until this is measured. Concretely: the head's output was about -0.78 eV
per frame before the fit and the fitted head must now produce a -4.9 eV
DIFFERENCE, but nothing in the objective decided where the common level sits.

WHAT IS REPORTED, both arms on the same frames:
  arm A = the model as trained (original head weights)
  arm B = the same model with the paired-fitted head installed
  per frame: E - label in eV and meV/atom, force rmse against the DFT forces,
             and the head's own contribution le
  aggregated CHARGED and NEUTRAL separately, because the earlier probe showed
             the two states carry errors of opposite sign (+17.67 against
             -5.87 meV/atom), so a single average would hide the effect
  the common-mode shift (le_new - le_old) averaged over each pair, which is
             exactly the component the paired fit left free

GATE: the paired dE of the installed model must reproduce the cached fit's
prediction. If it does not, the cached fit does not transfer to the live model
and nothing else here is interpretable.
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get(
    "KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get(
    "KIT_MACE_REPO",
    "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.pb1d_backend as PB

RUN = os.environ.get("KIT_RUN",
                     "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/gate_le")
FITTED = os.environ.get(
    "KIT_FITTED",
    "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/exp_le_headfit/"
    "head_fitted.pt")
WHICH = os.environ.get("KIT_WHICH", "final")       # final | valbest
NTRAIN = int(os.environ.get("KIT_NTRAIN", "10"))   # train pairs to also check
NVAL = int(os.environ.get("KIT_NVAL", "0"))        # 0 = all val pairs

os.chdir(RUN)
device = torch_tools.init_device(os.environ.get("KIT_DEVICE", "cuda"))
torch_tools.set_default_dtype("float64")

mp = [p for p in sorted(glob.glob(os.path.join(RUN, "models", "*.model")))
      if "compiled" not in os.path.basename(p)][-1]
cp = sorted(glob.glob(os.path.join(RUN, "checkpoints", "*_epoch-*.pt")),
            key=lambda p: int(p.rsplit("epoch-", 1)[1].split(".")[0]))[-1]
model = torch.load(f=mp, map_location=device).to(device)
ck = torch.load(cp, map_location=device)
model.load_state_dict(ck["model"] if "model" in ck else ck.model, strict=False)
model.eval()
model.solvent3d_energy = True
model.solvent_cavity_energy = True
model.solvent_baseline_coupling = True
head = model.local_electron_energy

F = torch.load(FITTED, map_location=device)
w_orig = {k: v.detach().clone() for k, v in head.state_dict().items()}
w_fit = F[WHICH]
VA, TR = F["pairs_val"], F["pairs_train"][:NTRAIN]
if NVAL:
    VA = VA[:NVAL]
print(f"model   {os.path.basename(mp)} @ {os.path.basename(cp)}")
print(f"fitted  {FITTED} [{WHICH}], step {F['step_final']} / valbest step "
      f"{F['step_valbest']}, a_fit {F['a_fit']:+.6f} eV/electron")
print(f"frames  {len(VA)} val pairs + {len(TR)} train pairs")
dw = max(float((w_fit[k] - w_orig[k]).abs().max()) for k in w_orig)
print(f"install max |w_fit - w_orig| = {dw:.4e}  (0 would mean nothing to test)")

kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "sample_id": "sample_id", "Fermi": "Fermi",
               "fermi_level": "Fermi", "solvated": "solvated",
               "potential": "potential_diff", "head": "head"},
    arrays_keys={"forces": "forces"})
atoms, split_of = {}, {}
for sp in ("train", "val", "test"):
    f = os.path.join(RUN, "data", f"{sp}.xyz")
    if not os.path.exists(f):
        continue
    for a in read(f, ":"):
        s = a.info.get("sample_id")
        if s is not None:
            atoms[int(s)] = a
            split_of[int(s)] = sp
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

cap = {}
head.register_forward_hook(
    lambda m, a, k, o: cap.__setitem__("le", o.detach().clone()),
    with_kwargs=True)
_bd = {}
_bk = PB.PB1DBackend.solve_graph


def _wrap(self, *a, **k):
    _bd["b"] = self
    return _bk(self, *a, **k)


PB.PB1DBackend.solve_graph = _wrap


def _evict():
    b = _bd.get("b")
    if b is not None:
        for at in ("_bl_ram", "_grids", "_solvers", "_c_units"):
            d = getattr(b, at, None)
            if isinstance(d, dict):
                d.clear()


def one(sid, want_force=True):
    a = atoms[sid]
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(
        ds, batch_size=1))).to(device)
    cap.clear()
    pred = model(b.to_dict(), compute_force=want_force, training=False)
    e = float(pred["energy"].sum())
    fr = None
    if want_force and pred.get("forces") is not None:
        fm = pred["forces"].detach().cpu().numpy()
        fd = np.asarray(a.get_array("forces"))
        fr = float(np.sqrt(((fm - fd) ** 2).mean()))
        fmax = float(np.abs(fm - fd).max())
    else:
        fmax = float("nan")
    out = dict(e=e, label=float(a.info["energy"]), nat=len(a),
               q=float(a.info.get("total_charge", 0.0)),
               le=float(cap["le"].sum()), frmse=fr, fmax=fmax)
    _evict()
    return out


def sweep(pairs, tag):
    rows = {}
    for arm, w in (("A_orig", w_orig), ("B_fitted", w_fit)):
        head.load_state_dict(w)
        rr = {}
        for sc, sn in pairs:
            rr[sc] = one(sc)
            rr[sn] = one(sn)
        rows[arm] = rr
        print(f"   {tag}: arm {arm} done ({2 * len(pairs)} frames)", flush=True)
    return rows


print("\n" + "=" * 78)
print("running both arms with forces")
R = {}
R["val"] = sweep(VA, "val")
if TR:
    R["train"] = sweep(TR, "train")
head.load_state_dict(w_orig)

for split, pairs in (("val", VA), ("train", TR)):
    if split not in R:
        continue
    A, B = R[split]["A_orig"], R[split]["B_fitted"]
    print("\n" + "=" * 78)
    print(f"{split.upper()}  ({len(pairs)} pairs)")

    print("\n   GATE: does the installed head reproduce the cached fit's "
          f"paired prediction?")
    dles = []
    for sc, sn in pairs:
        dles.append((B[sc]["le"] - B[sn]["le"]))
    dles = np.array(dles)
    tgt = np.array([(B[sc]["label"] - B[sn]["label"])
                    - ((A[sc]["e"] - A[sc]["le"]) - (A[sn]["e"] - A[sn]["le"]))
                    for sc, sn in pairs])
    err = dles - tgt
    print(f"     installed d(le) mean {dles.mean():+.4f} eV, target mean "
          f"{tgt.mean():+.4f} eV, rmse of the difference "
          f"{float((err ** 2).mean() ** 0.5):.4f} eV")
    print(f"     (the head-only fit reported val rmse 0.1359 eV; a large "
          f"disagreement here would mean the cached fit does not transfer)")

    print(f"\n   ABSOLUTE energy error per frame, by state")
    print(f"   {'state':>8} {'arm':>10} {'mean meV/at':>12} "
          f"{'rms meV/at':>11} {'mean eV':>10} {'worst eV':>10}")
    for state, pick in (("charged", 0), ("neutral", 1)):
        for arm, D in (("A_orig", A), ("B_fitted", B)):
            v = [D[p[pick]] for p in pairs]
            pa = np.array([1000.0 * (x["e"] - x["label"]) / x["nat"] for x in v])
            ev = np.array([x["e"] - x["label"] for x in v])
            print(f"   {state:>8} {arm:>10} {pa.mean():+12.2f} "
                  f"{float((pa ** 2).mean() ** 0.5):11.2f} {ev.mean():+10.4f} "
                  f"{ev[np.abs(ev).argmax()]:+10.4f}")

    print(f"\n   FORCES rmse against DFT (eV/A)")
    print(f"   {'state':>8} {'arm':>10} {'mean rmse':>10} {'worst rmse':>11} "
          f"{'worst |dF|':>11}")
    for state, pick in (("charged", 0), ("neutral", 1)):
        for arm, D in (("A_orig", A), ("B_fitted", B)):
            v = [D[p[pick]] for p in pairs]
            fr = np.array([x["frmse"] for x in v])
            fx = np.array([x["fmax"] for x in v])
            print(f"   {state:>8} {arm:>10} {fr.mean():10.4f} {fr.max():11.4f} "
                  f"{fx.max():11.4f}")

    print(f"\n   THE COMPONENT THE PAIRED FIT LEFT FREE: common-mode le shift")
    cm_o = np.array([0.5 * (A[sc]["le"] + A[sn]["le"]) for sc, sn in pairs])
    cm_n = np.array([0.5 * (B[sc]["le"] + B[sn]["le"]) for sc, sn in pairs])
    nat0 = A[pairs[0][0]]["nat"]
    shift = float(cm_n.mean() - cm_o.mean())
    print(f"     mean of (le_charged + le_neutral)/2: "
          f"{cm_o.mean():+.4f} eV before, {cm_n.mean():+.4f} eV after, "
          f"shift {shift:+.4f} eV = {1000.0 * shift / nat0:+.2f} meV/atom")
    print(f"     spread of that shift across pairs: rms "
          f"{float(((cm_n - cm_o - (cm_n - cm_o).mean()) ** 2).mean() ** 0.5):.4f} eV "
          f"-- a pure constant would show zero spread, and a geometry-dependent "
          f"C(R) shows up here")

    print(f"\n   per-frame table")
    print(f"   {'sid':>5} {'q':>7} {'E-lab A':>9} {'E-lab B':>9} "
          f"{'F rmse A':>9} {'F rmse B':>9} {'le A':>9} {'le B':>9}")
    for sc, sn in pairs:
        for s in (sc, sn):
            print(f"   {s:>5} {A[s]['q']:+7.3f} {A[s]['e'] - A[s]['label']:+9.4f} "
                  f"{B[s]['e'] - B[s]['label']:+9.4f} {A[s]['frmse']:9.4f} "
                  f"{B[s]['frmse']:9.4f} {A[s]['le']:+9.4f} {B[s]['le']:+9.4f}")
