"""Can a charge-aware branch learn what the existing model is still missing?
(user-specified action 2, 2026-09-10. Target is the RESIDUAL, not the full integral.)

WHY THE TARGET IS NOT int mu dN. The user's correction, and it is decisive:
mu = dE/dN and Delta E = int mu dN, so the integral IS the whole charging
energy and it already CONTAINS whatever the existing electrostatic and solvent
terms contribute. Adding the full integral on top of the model total would
double-count them. Worse, the "relation" that looked so strong is close to a
tautology: eps_Delta = Delta E_model - int mu dN, and Delta E_model is only
7.6% of the truth, so eps_Delta is approximately -int mu dN whatever the model
does. The trapezoid beating the endpoints is a fact about numerical
integration, not about the model. So the endpoint average remains valuable
EVIDENCE that the missing quantity tracks a chemical-potential integral, and
is not proof of an independent missing term.

The paired target is therefore the RESIDUAL the existing model leaves:

    dE_correction = dE_DFT - dE_existing

CONVENTIONS, settled from source and settings rather than from field names,
which is where the previous round overreached:
  no constant-potential mode anywhere -- 0 of 800 INCARs set EFERMI_ref or a
  const-pot tag, its default in solvation.F is 0, L_const_pot is
  (EFERMI_ref < 0) hence FALSE, and the OUTCAR prints EFERMI_ref = 0.000000;
  Ecorr_sol = A_corr - Ecorr_band - q_sol*EFERMI_ref IS added into TOTEN
  (electron_all.F:456, electron.F:590), so the solvation free energy is inside
  the label while the reservoir term, present in the formula, is identically
  zero for these runs;
  the label is energy(sigma->0), exactly VASP's ISMEAR=1 extrapolation of the
  printed TOTEN (ratio 1/3 to 3e-9), so nothing was added on top of it;
  the stored Fermi is the raw OUTCAR E-fermi, and the reference zero is the
  RIGHT vacuum, pinned at +2.3e-4 and +2.0e-5 eV in the two states with a
  spread of 3.6e-6, so mu is on a unified scale across a pair to 2.2e-4 eV.
  The LEFT vacuum moves 3.46 eV between the states, which is the slab's
  response to the added charge and not a reference problem.

WHAT THE FIT MAY AND MAY NOT USE. The readout features are IDENTICAL between
the two members of a pair (1.9e-16 relative), so any function of features
alone contributes exactly zero to a paired difference. A correction can only
act through the electron count. The ladder, all linear in their parameters and
all fitted by least squares on the TRAIN pairs and scored on VAL:

  1  a*dN                     one scalar: a rigid per-electron energy
  2  a*dN + b*dN^2            the capacitor form; note the trapezoid mu_bar*dN
                              IS this form, since mu(N) linear in N gives
                              exactly a quadratic, so this is the honest
                              parametric version of that observation
  3  (a + w.h)*dN             feature-modulated, h the pooled readout features:
                              a per-structure per-electron energy, which is
                              what a charge-aware branch would actually be
  4  mu_bar*dN                A CONTROL that uses extra reference information
                              (the DFT Fermi levels of both states). It is NOT
                              an oracle and NOT an upper bound: the trapezoid
                              is only an approximation to the integral, and
                              nothing guarantees it beats a fitted residual
                              model. Reported to see whether the extra
                              information helps at all -- and in the first
                              pass it LOST, which is itself the proof that
                              calling it a bound was wrong.

Reported on val: residual bias AND residual spread separately, because the
last round showed those two answer different questions and only reporting the
RMS ratio hides it -- eps_Delta was nearly a pure offset, so removing 97.3% of
the offset and 55.4% of the spread both round to "94% explained".
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get(
    "KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get(
    "KIT_MACE_REPO",
    "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.pb1d_backend as PB

CKDIR = os.environ.get("KIT_CKPT_DIR", "checkpoints")
DATADIR = os.environ.get("KIT_DATA_DIR", "data")
OUT = os.environ.get("KIT_OUT", ".")
CACHE = os.path.join(OUT, "charge_branch_cache.npz")
# the per-sample caches are sized for training, where the same samples recur
# every epoch; a one-pass sweep over 400 distinct samples is the case they
# were never meant for and they leak monotonically there
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")


def _rss_gb():
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS"):
                    return float(ln.split()[1]) / 1048576.0
    except OSError:
        pass
    return float("nan")


if os.path.exists(CACHE):
    z = np.load(CACHE)
    print(f"  reusing {CACHE}: {z['dN'].shape[0]} pairs")
else:
    model = torch.load(f=os.path.join(CKDIR, "s3d_gate_bl2_run-123.model"),
                       map_location=device).to(device)
    ck = torch.load(os.path.join(CKDIR, "s3d_gate_bl2_run-123_epoch-33.pt"),
                    map_location=device)
    model.load_state_dict(ck["model"] if "model" in ck else ck.model)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.solvent3d_energy = True
    model.solvent_cavity_energy = True
    model.solvent_baseline_coupling = True

    _backend = {}
    _bk = PB.PB1DBackend.solve_graph
    def _wrap(self, *a, **k):
        _backend["b"] = self
        return _bk(self, *a, **k)
    PB.PB1DBackend.solve_graph = _wrap

    grab = []
    hooks = [r.register_forward_pre_hook(
        lambda m, args: grab.append(args[0].detach().to(torch.float64)))
        for r in model.readouts]

    z_table = utils.AtomicNumberTable([int(x) for x in model.atomic_numbers])
    kspec = KeySpecification(
        info_keys={"energy": "energy", "total_charge": "total_charge",
                   "total_spin": "total_spin", "sample_id": "sample_id",
                   "fermi_level": "Fermi", "potential": "potential_diff",
                   "solvated": "solvated"},
        arrays_keys={"forces": "forces"},
    )
    atoms_by_sid, split_by_sid = {}, {}
    for sp in ("train", "val", "test"):
        for a in read(os.path.join(DATADIR, f"{sp}.xyz"), ":"):
            s = int(a.info["sample_id"])
            atoms_by_sid[s] = a
            split_by_sid[s] = sp

    def run(sid):
        grab.clear()
        a = atoms_by_sid[sid]
        cfg = mace_data.config_from_atoms(a, key_specification=kspec)
        ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                               cutoff=float(model.r_max))]
        b = next(iter(torch_geometric.dataloader.DataLoader(
            ds, batch_size=1))).to(device)
        with torch.no_grad():
            pred = model(b.to_dict(), compute_force=False, training=False)
        # POOLED features: sum over atoms. A paired difference of any function
        # of features alone is exactly zero, so pooling loses nothing here and
        # keeps the cache small.
        h = torch.cat([g.sum(dim=0) for g in grab]).cpu().numpy()
        bk = _backend.get("b")
        if bk is not None:
            for nm in ("_bl_ram", "_grids", "_solvers", "_c_units"):
                d = getattr(bk, nm, None)
                if isinstance(d, dict):
                    d.clear()
        torch.cuda.empty_cache()
        return (float(pred["energy"].sum()), float(a.info["energy"]),
                float(a.info.get("total_charge", 0.0)),
                float(a.info.get("Fermi", float("nan"))), h)

    pairs = [k for k in range(1, 201)
             if k in atoms_by_sid and (600 + k) in atoms_by_sid]
    print(f"  {len(pairs)} pairs; caching pooled features and energies "
          f"(start RSS {_rss_gb():.2f} GB)", flush=True)
    rec = {k: [] for k in ("k", "dN", "de_dft", "de_mod", "mu_bar", "h")}
    spl = []
    for i, k in enumerate(pairs):
        ec, dc, qc, fc, hc = run(k)
        en, dn_, qn, fn, hn = run(600 + k)
        rec["k"].append(k)
        rec["dN"].append(-qc)
        rec["de_dft"].append(dc - dn_)
        rec["de_mod"].append(ec - en)
        rec["mu_bar"].append(0.5 * (fc + fn))
        rec["h"].append(hc)
        spl.append(split_by_sid[k])
        if (i + 1) % 25 == 0:
            print(f"   [{i+1}/{len(pairs)}] RSS {_rss_gb():.2f} GB", flush=True)
        # the paired features must agree; if they do not, a feature-only term
        # could act on a difference and the ladder's premise changes
        if i == 0:
            d0 = float(np.abs(hc - hn).max() / max(np.abs(hn).max(), 1e-30))
            print(f"   [gate] pooled features agree across the pair: relative "
                  f"{d0:.3e} -- a feature-only term contributes exactly zero "
                  f"to a paired difference", flush=True)
    np.savez(CACHE, k=np.array(rec["k"]), dN=np.array(rec["dN"]),
             de_dft=np.array(rec["de_dft"]), de_mod=np.array(rec["de_mod"]),
             mu_bar=np.array(rec["mu_bar"]), h=np.array(rec["h"]),
             split=np.array(spl))
    for hh in hooks:
        hh.remove()
    z = np.load(CACHE)
    print(f"  cache written: {CACHE}")

dN, de_dft, de_mod = z["dN"], z["de_dft"], z["de_mod"]
mu_bar, h, spl = z["mu_bar"], z["h"], z["split"]
y = de_dft - de_mod                      # THE TARGET: what is still missing
tr, va = spl == "train", spl == "val"
print(f"\n  target dE_correction = dE_DFT - dE_existing over {len(y)} pairs: "
      f"mean {y.mean():+.4f}, rmse {np.sqrt((y*y).mean()):.4f}, spread "
      f"{y.std():.4f} eV; dN from {dN.min():.3f} to {dN.max():.3f}")
print(f"  splits: train {int(tr.sum())}, val {int(va.sum())}, "
      f"test {int((spl=='test').sum())}")

hs = (h - h[tr].mean(0)) / (h[tr].std(0) + 1e-12)
LAD = [("1  a*dN", np.stack([dN], 1)),
       ("2  a*dN + b*dN^2", np.stack([dN, dN * dN], 1)),
       ("3  (a + w.h)*dN", np.concatenate([dN[:, None],
                                           hs * dN[:, None]], 1))]
print(f"\n[RESULT] fitted on {int(tr.sum())} TRAIN pairs, scored on "
      f"{int(va.sum())} VAL pairs. Bias and spread are reported separately "
      f"because they answer different questions.")
print(f"  {'model':>22} {'par':>4} {'val bias':>10} {'val rmse':>10} "
      f"{'val spread':>11} {'bias removed':>13} {'spread removed':>15}")
b0, s0 = y[va].mean(), y[va].std()
print(f"  {'(nothing)':>22} {0:4d} {b0:+10.4f} "
      f"{np.sqrt((y[va]**2).mean()):10.4f} {s0:11.4f} {'-':>13} {'-':>15}")
for nm, X in LAD:
    w, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
    r = y - X @ w
    print(f"  {nm:>22} {X.shape[1]:4d} {r[va].mean():+10.4f} "
          f"{np.sqrt((r[va]**2).mean()):10.4f} {r[va].std():11.4f} "
          f"{100*(1-abs(r[va].mean())/max(abs(b0),1e-30)):12.1f}% "
          f"{100*(1-r[va].std()/max(s0,1e-30)):14.1f}%")
r_or = y - mu_bar * dN
print(f"  {'4  mu_bar*dN (control)':>22} {'-':>4} {r_or[va].mean():+10.4f} "
      f"{np.sqrt((r_or[va]**2).mean()):10.4f} {r_or[va].std():11.4f} "
      f"{100*(1-abs(r_or[va].mean())/max(abs(b0),1e-30)):12.1f}% "
      f"{100*(1-r_or[va].std()/max(s0,1e-30)):14.1f}%")
print(f"\n  Row 4 is a CONTROL using extra reference information, the DFT "
      f"Fermi levels of both states. It is not an oracle and not an upper "
      f"bound -- the trapezoid only approximates the integral and nothing\n  "
      f"guarantees it beats a fitted model; it is here to show whether that "
      f"extra information helps at all. Row 2 is the honest parametric form "
      f"of the same idea,\n"
      f"  since mu linear in N makes the trapezoid exactly a quadratic. Row 3 "
      f"asks whether the correction needs to know the structure or only the "
      f"electron count.")
# ======================================================================
# MODEL SELECTION -- row 3 as printed above is not a model. 1153 parameters
# against 160 training pairs is under-determined, so lstsq returns whatever
# minimum-norm vector it likes, and the score sits on 20 val points where the
# relative standard error of an rmse is 1/sqrt(2n) = 15.8%. A 31% gain is
# about two standard errors. Three things settle it, all from the cache:
#   ridge with the penalty chosen by K-fold CV inside TRAIN, so row 3 becomes
#   a real model rather than a min-norm artefact;
#   a bootstrap CI on the val scores, so the margin is read against its own
#   resolution instead of at face value;
#   and a SHUFFLE control -- the same fit with the features permuted across
#   pairs. If shuffled features buy a similar gain, the gain is fitting noise
#   and nothing about structure.
# The Fermi CONTROL also gets its best linear rescaling MEASURED here rather than
# estimated from the 200-pair correlation.
# ======================================================================
rng = np.random.default_rng(0)
itr = np.where(tr)[0]
iva = np.where(va)[0]


def ridge_fit(X, yy, idx, alpha):
    A = X[idx]
    G = A.T @ A + alpha * np.eye(A.shape[1])
    return np.linalg.solve(G, A.T @ yy[idx])


def kfold_alpha(X, yy, idx, alphas, k=5):   # retained, unused: the
    """penalty is now chosen on VAL per the user's protocol, not by CV inside
    TRAIN. Kept so the alternative protocol is one call away if wanted."""
    perm = rng.permutation(len(idx))
    folds = np.array_split(perm, k)
    best, ba = None, None
    for al in alphas:
        err = []
        for f in range(k):
            te = idx[folds[f]]
            trn = idx[np.concatenate([folds[j] for j in range(k) if j != f])]
            w = ridge_fit(X, yy, trn, al)
            err.append(((X[te] @ w - yy[te]) ** 2).mean())
        m = float(np.mean(err))
        if best is None or m < best:
            best, ba = m, al
    return ba


# THE USER'S PROTOCOL: fit on TRAIN, choose the ridge penalty on VAL, and
# hold TEST back until a form is chosen. Selecting on val means the val score
# is no longer an unbiased estimate of that form -- which is exactly why test
# is sealed -- and the script says so rather than quietly reporting val as if
# it were clean. KIT_FINAL_FORM breaks the seal, once, for a named form.
ALPHAS = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
X3 = LAD[2][1]
_sel = []
for al in ALPHAS:
    w = ridge_fit(X3, y, itr, al)
    r = y - X3 @ w
    _sel.append((float(np.sqrt((r[iva] ** 2).mean())), al))
_sel.sort()
a3 = _sel[0][1]
print(f"\n  ridge penalty selected ON VAL for the feature form: alpha "
      f"{a3:g}; val rmse across the grid " +
      ", ".join(f"{al:g}:{v:.4f}" for v, al in sorted(_sel, key=lambda t: t[1])))
print(f"  feature dimension {X3.shape[1]} against {int(tr.sum())} training "
      f"pairs, so regularisation is required rather than optional.")
w3 = ridge_fit(X3, y, itr, a3)
r3 = y - X3 @ w3
X1 = LAD[0][1]
w1 = ridge_fit(X1, y, itr, 1e-8)
r1 = y - X1 @ w1
print(f"\n[MODEL SELECTION] row 3 with ridge, penalty chosen ON VAL "
      f"(alpha {a3:g}), so its val score is SELECTION-CONTAMINATED and is a "
      f"model-choice number, not a clean estimate. TEST stays sealed.")
print(f"   {'variant':>34} {'val rmse':>10} {'val spread':>11}")
print(f"   {'1  a*dN (reference)':>34} {np.sqrt((r1[iva]**2).mean()):10.4f} "
      f"{r1[iva].std():11.4f}")
print(f"   {'3  (a + w.h)*dN, ridge':>34} {np.sqrt((r3[iva]**2).mean()):10.4f} "
      f"{r3[iva].std():11.4f}")

# shuffle control: permute the feature block across pairs, keep dN intact
sh = []
for _ in range(20):
    q = rng.permutation(len(y))
    Xs = np.concatenate([dN[:, None], hs[q] * dN[:, None]], 1)
    ws = ridge_fit(Xs, y, itr, a3)
    rs = y - Xs @ ws
    sh.append(np.sqrt((rs[iva] ** 2).mean()))
sh = np.array(sh)
print(f"   {'3  with features SHUFFLED x20':>34} {sh.mean():10.4f} "
      f"{'':>11}  (min {sh.min():.4f}, max {sh.max():.4f})")
print(f"   -> real features beat shuffled by "
      f"{100*(1-np.sqrt((r3[iva]**2).mean())/sh.mean()):.1f}%; if that is near "
      f"zero the structure dependence is fitting noise.")

# bootstrap the val score so the margin is read against its own resolution
def boot(rr, n=2000):
    v = rr[iva]
    out = [np.sqrt((v[rng.integers(0, len(v), len(v))] ** 2).mean())
           for _ in range(n)]
    return np.percentile(out, [2.5, 50, 97.5])


for nm, rr in (("1  a*dN", r1), ("3  ridge", r3)):
    lo, md, hi = boot(rr)
    print(f"   bootstrap val rmse {nm:>12}: median {md:.4f}, 95% CI "
          f"[{lo:.4f}, {hi:.4f}]")
# PAIRED bootstrap: both models must be scored on the SAME resample, or the
# difference picks up two independent sampling noises instead of one
v1, v3 = r1[iva], r3[iva]
diffs = []
for _ in range(2000):
    i = rng.integers(0, len(iva), len(iva))
    diffs.append(np.sqrt((v1[i] ** 2).mean()) - np.sqrt((v3[i] ** 2).mean()))
d_lo, d_md, d_hi = np.percentile(diffs, [2.5, 50, 97.5])
print(f"   bootstrap of the DIFFERENCE (row1 - row3): median {d_md:+.4f}, "
      f"95% CI [{d_lo:+.4f}, {d_hi:+.4f}] -- "
      f"{'excludes zero, so the gain is real' if d_lo > 0 else 'INCLUDES ZERO, so the gain is not established'}")

# the Fermi control's best linear rescaling, MEASURED rather than estimated
Xo = (mu_bar * dN)[:, None]
wo = ridge_fit(Xo, y, itr, 1e-8)
ro = y - Xo @ wo
print(f"\n   Fermi control rescaled, scale {float(wo[0]):.4f} fitted on "
      f"train: val "
      f"rmse {np.sqrt((ro[iva]**2).mean()):.4f}, spread {ro[iva].std():.4f} "
      f"-- against the unscaled control's {np.sqrt((r_or[iva]**2).mean()):.4f} "
      f"and {r_or[iva].std():.4f}")
print(f"   so rescaling the Fermi-based control "
      f"{'does not rescue it' if ro[iva].std() > r1[iva].std() else 'rescues it'}: "
      f"one fitted constant per electron is still "
      f"{ro[iva].std()/max(r1[iva].std(),1e-30):.2f}x better in spread.")

print(f"\n[THE THREE METRICS THE USER ASKED FOR, val pairs, eV]")
print(f"  {'form':>26} {'rmse':>9} {'bias':>9} {'mean-removed':>13} "
      f"{'all three better?':>18}")
_base = (float(np.sqrt((y[iva] ** 2).mean())), float(y[iva].mean()),
         float(y[iva].std()))
print(f"  {'(no correction)':>26} {_base[0]:9.4f} {_base[1]:+9.4f} "
      f"{_base[2]:13.4f} {'-':>18}")
_prev = None
for nm, rr in (("a*dN", r1), ("a*dN + b*dN^2", y - LAD[1][1] @ np.linalg.lstsq(
        LAD[1][1][tr], y[tr], rcond=None)[0]), ("(a + w.h)*dN ridge", r3)):
    m = (float(np.sqrt((rr[iva] ** 2).mean())), float(rr[iva].mean()),
         float(rr[iva].std()))
    ok = (m[0] < _base[0] and abs(m[1]) < abs(_base[1]) and m[2] < _base[2])
    step = ("" if _prev is None else
            ("  (better than the previous row on all three)"
             if (m[0] < _prev[0] and abs(m[1]) < abs(_prev[1])
                 and m[2] < _prev[2]) else
             "  (NOT better than the previous row on all three)"))
    print(f"  {nm:>26} {m[0]:9.4f} {m[1]:+9.4f} {m[2]:13.4f} "
          f"{('yes' if ok else 'no'):>18}{step}")
    _prev = m
print(f"  The ladder is read stepwise: a*dN improving means a per-electron "
      f"energy baseline is the first thing missing; b*dN^2 improving further "
      f"means charging curvature is also needed; h improving further means\n"
      f"  the correction depends on the configuration. What this says is which "
      f"EXPRESSION is effective enough -- the fitted coefficients are not "
      f"thereby identified as independent physical terms.")

_final = os.environ.get("KIT_FINAL_FORM")
if _final:
    ite = np.where(spl == "test")[0]
    pick = {"1": (LAD[0][1], 1e-8), "2": (LAD[1][1], 1e-8), "3": (X3, a3)}[_final]
    w = ridge_fit(pick[0], y, itr, pick[1])
    rf = y - pick[0] @ w
    print(f"\n[SEALED TEST OPENED for form {_final}] {len(ite)} test pairs: "
          f"rmse {np.sqrt((rf[ite]**2).mean()):.4f}, bias {rf[ite].mean():+.4f}"
          f", mean-removed {rf[ite].std():.4f} eV. This is the clean estimate; "
          f"it is valid once and only for this form.")
else:
    print(f"\n  TEST IS SEALED: {int((spl=='test').sum())} pairs held back and "
          f"not scored. Set KIT_FINAL_FORM=1|2|3 to open it, once, after the "
          f"form is chosen.")

print(f"\n[READING] a correction that removes the BIAS but not the SPREAD is a "
      f"constant per electron and does not make the energy easier to learn "
      f"frame by frame. One that removes both is what the user asked for.\n"
      f"  And if row 3 beats row 2 materially, the missing quantity depends on "
      f"the structure and not only on how many electrons were added.")
print("DONE")
