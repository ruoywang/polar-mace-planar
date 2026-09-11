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
  4  mu_bar*dN                NOT A MODEL -- it uses the DFT Fermi levels of
                              both states, so it is an ORACLE and bounds what
                              a perfect charge-aware branch could reach

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
print(f"  {'4  mu_bar*dN (ORACLE)':>22} {'-':>4} {r_or[va].mean():+10.4f} "
      f"{np.sqrt((r_or[va]**2).mean()):10.4f} {r_or[va].std():11.4f} "
      f"{100*(1-abs(r_or[va].mean())/max(abs(b0),1e-30)):12.1f}% "
      f"{100*(1-r_or[va].std()/max(s0,1e-30)):14.1f}%")
print(f"\n  Row 4 is NOT a model: it uses the DFT Fermi levels of BOTH states, "
      f"so it bounds what a perfect charge-aware branch could reach and cannot "
      f"be deployed. Row 2 is the honest parametric form of the same idea,\n"
      f"  since mu linear in N makes the trapezoid exactly a quadratic. Row 3 "
      f"asks whether the correction needs to know the structure or only the "
      f"electron count.")
print(f"\n[READING] a correction that removes the BIAS but not the SPREAD is a "
      f"constant per electron and does not make the energy easier to learn "
      f"frame by frame. One that removes both is what the user asked for.\n"
      f"  And if row 3 beats row 2 materially, the missing quantity depends on "
      f"the structure and not only on how many electrons were added.")
print("DONE")
