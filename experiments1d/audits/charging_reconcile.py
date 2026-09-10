"""Charging-energy reconciliation: where does the 5.635 eV actually sit?
(user-specified design, 2026-09-10. sid1/sid601 first, then the 20 val pairs.)

WHAT THIS CORRECTS. My A/B cache called E_model - node_energy the "solvent
remainder". It is not: reading the assembly in extensions.py, that difference
also contains the SOLUTE electrostatic energy, the optional electron energy
and the boundary/dipole corrections, so it cannot be charged to the solvent as
a block. And the +5.635 eV it was going to be compared against is not a
solvent number either -- it is the error in the whole system's charged-minus-
neutral energy difference, and the explicit part (catalyst plus explicit
water) changes its electron distribution when charge is added, which moves
solute electrostatics and other electronic energies too.

  Delta E   = E_charged - E_neutral        (same geometry, same cell)
  eps_Delta = Delta E_model - Delta E_DFT

THE MODEL SIDE HAS TEN ADDITIVE TERMS, taken from the assembly rather than
guessed (extensions.py: total_energy = e0 + inter_e, then + electron_energy,
then + electrostatic_energy + compensation_periodic_1d_energy +
compensation_slab_correction_energy + external_field . explicit_dipole, then
+ cavity_energy_g + solvent3d_energy_g + baseline_coupling_energy_g). Each is
reported for neutral, charged and their difference, their sum is checked
against the model's own total, and anything not enabled is printed as an
explicit zero rather than omitted.

THE DFT SIDE IS DECOMPOSED BY ITS OWN FORMULA AND NOT MAPPED ONTO THE MODEL'S.
The label is identified against the OUTCAR fields rather than assumed, and the
printed solvation pieces (A_cav, A_corr, Ecorr, Ecorr_band) are reported as
printed. Two things are deliberately NOT done, per the user: the DFT solvent
free energy is not equated with the model's cross+self, and the network energy
head is not equated with any DFT electronic energy term. Whatever cannot be
matched with confidence stays in a remainder with a name.

THE STRUCTURAL CHECK THAT COULD SETTLE THE A/B PREMISE. e0 is a per-species
constant and the two frames have identical species, so e0 cancels in Delta E
exactly. If the readout INPUT FEATURES are also identical between the two
states -- which they are unless the charge or potential reaches them -- then
inter_e cancels too, and retraining the energy head cannot move the paired
charging energy at all. That is decided here by comparing the captured
features directly, not by reading the architecture.

One measured fact that frames the whole thing: NELECT is 661 for the charged
frame and 660 for the neutral one, so this is an electron-ADDITION energy, and
the label carries no reservoir term, so it is canonical rather than grand
canonical.
"""
import math
import os
import re
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

DFT = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single")
CKDIR = os.environ.get("KIT_CKPT_DIR", "checkpoints")
DATADIR = os.environ.get("KIT_DATA_DIR", "data")
# The user's main table is the 20 pairs that sit wholly inside val. KIT_PAIRS
# overrides the list so a machine holding a different slice of the DFT payload
# can still test the electron-energy hypothesis on the pairs it has -- but the
# output states which SPLIT each pair belongs to, because a table built on
# train pairs is evidence about a structural absence and is NOT the user's
# validation table, and the two must not be confused in a report.
PAIRS_VAL = [28, 30, 43, 60, 61, 62, 69, 79, 83, 94,
             128, 134, 148, 153, 159, 177, 180, 185, 186, 189]
if os.environ.get("KIT_PAIRS"):
    PAIRS_VAL = [int(x) for x in os.environ["KIT_PAIRS"].split(",") if x.strip()]


def dftdir(sid):
    if 1 <= sid <= 200:
        return f"{DFT}/1-44_GCE/cal_{sid}"
    if 601 <= sid <= 800:
        return f"{DFT}/5-44_neutral_withsolv/cal_{sid - 600}"
    return None


def dft_terms(sid):
    """Everything the DFT run PRINTED, with no mapping onto model terms."""
    d = dftdir(sid)
    out = {}
    oc = open(f"{d}/OUTCAR", errors="ignore").read()
    for key, pat in (("toten", r"free  energy   TOTEN\s*=\s*([-\d.]+)"),
                     ("noentropy", r"energy  without entropy=\s*([-\d.]+)"),
                     ("sigma0", r"energy\(sigma->0\)\s*=\s*([-\d.]+)"),
                     ("nelect", r"NELECT\s*=\s*([\d.]+)"),
                     ("efermi", r"E-fermi\s*:\s*([-\d.]+)")):
        m = re.findall(pat, oc)
        out[key] = float(m[-1]) if m else float("nan")
    lg = open(f"{d}/log.out", errors="ignore").read()
    for key, pat in (("a_cav", r"A_cav:\s*([-\d.Ee+]+)"),
                     ("a_corr", r"A_corr=\s*([-\d.Ee+]+)"),
                     ("ecorr", r"Ecorr=\s*([-\d.Ee+]+)"),
                     ("ecorr_band", r"Ecorr_band=\s*([-\d.Ee+]+)"),
                     ("dele_z", r"Dele\s*=\s*\S+\s+\S+\s+([-\d.Ee+]+)")):
        m = re.findall(pat, lg)
        out[key] = float(m[-1]) if m else float("nan")
    return out


# ---- availability census FIRST: refuse to start rather than crash halfway.
# The DFT payload is not the same on every machine that can run this, and a
# script that prints four steps and then throws FileNotFoundError inside the
# fifth is worse than one that says up front what it cannot do. This is the
# guard lateral_bound_swap.py grew two rounds ago and which did not carry
# over -- the workstation caught it before running.
def _have(sid):
    d = dftdir(sid)
    return (d is not None and os.path.exists(f"{d}/OUTCAR")
            and os.path.exists(f"{d}/log.out"))


_pairs_all = list(PAIRS_VAL)
PAIRS_VAL = [k for k in _pairs_all if _have(k) and _have(600 + k)]
_missing = [(k, "charged" if not _have(k) else "neutral")
            for k in _pairs_all if k not in PAIRS_VAL]
print(f"  DFT availability census: step 1 pair (1/601) "
      f"{'present' if _have(1) and _have(601) else 'MISSING'}; "
      f"complete val pairs {len(PAIRS_VAL)} of {len(_pairs_all)}")
if _missing:
    mc = [k for k, w in _missing if w == "charged"]
    mn = [k for k, w in _missing if w == "neutral"]
    print(f"    incomplete pairs: {len(_missing)} -- missing charged side for "
          f"{mc if mc else 'none'}, missing neutral side for "
          f"{mn if mn else 'none'}")
if not (_have(1) and _have(601)):
    raise SystemExit("  STOP: steps 1-4 need the sid1/sid601 pair and this "
                     "machine does not hold it. Run where the payload is "
                     "rather than substituting another pair.")
if not PAIRS_VAL:
    print(f"    -> STEP 5 WILL BE SKIPPED on this machine, and skipped is not "
          f"'done with fewer pairs': 0 complete pairs means the table cannot "
          f"be formed at all. Steps 1-4 still run and are self-contained.",
          flush=True)

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
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

grab = {}
def mk_hook(i):
    def _h(mod, args):
        grab.setdefault(i, []).append(args[0].detach().clone())
    return _h
hooks = [r.register_forward_pre_hook(mk_hook(i))
         for i, r in enumerate(model.readouts)]
emb = getattr(model, "embedding_readout", None)
if emb is not None:
    hooks.append(emb.register_forward_pre_hook(mk_hook(-1)))

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff",
               "solvated": "solvated"},
    arrays_keys={"forces": "forces"},
)
atoms_by_sid, split_by_sid = {}, {}
for split in ("train", "val", "test"):
    for a in read(os.path.join(DATADIR, f"{split}.xyz"), ":"):
        s = int(a.info["sample_id"])
        atoms_by_sid[s] = a
        split_by_sid[s] = split

# the ten additive model terms, in assembly order
TERMS = [
    ("e0 (atomic, per-species)", "e0"),
    ("energy head (inter_e)", "interaction_energy"),
    ("electron energy (optional)", "electron_energy"),
    ("solute electrostatic", "electrostatic_energy"),
    ("1D solvent compensation", "compensation_periodic_1d_energy"),
    ("slab dipole correction", "compensation_slab_correction_energy"),
    ("external field . dipole", "extfield_dipole"),
    ("cavity energy", "cavity_energy_g"),
    ("3D solvent energy", "solvent3d_energy_g"),
    ("baseline coupling E_bl", "baseline_coupling_energy_g"),
]


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
    nat = int(b["ptr"][1] - b["ptr"][0])
    heads = (b["head"][b["batch"]] if "head" in b.to_dict()
             else torch.zeros_like(b["batch"]))
    ar = torch.arange(nat, device=device)
    e0v = model.atomic_energies_fn(b["node_attrs"])
    e0 = float((e0v[ar, heads] if e0v.dim() > 1 else e0v).sum())
    t = {"e0": e0}
    for _, k in TERMS:
        if k == "e0":
            continue
        if k == "extfield_dipole":
            ef = pred.get("external_field")
            ed = pred.get("explicit_dipole")
            t[k] = (float((ef * ed).sum()) if ef is not None and ed is not None
                    else 0.0)
        else:
            v = pred.get(k)
            t[k] = float(v.sum()) if v is not None else 0.0
    t["_total"] = float(pred["energy"].sum())
    t["_label"] = float(a.info["energy"])
    t["_nat"] = nat
    t["_q"] = float(a.info.get("total_charge", 0.0))
    t["_fermi"] = float(a.info.get("Fermi", float("nan")))
    t["_feats"] = {i: torch.cat(v, 0).clone() for i, v in grab.items()}
    return t


print(f"=================== STEP 1: the sid1/sid601 pair "
      f"===================")
a1, a6 = atoms_by_sid[1], atoms_by_sid[601]
dpos = float(np.abs(a1.get_positions() - a6.get_positions()).max())
dcell = float(np.abs(np.array(a1.get_cell()) - np.array(a6.get_cell())).max())
same_sp = list(a1.get_chemical_symbols()) == list(a6.get_chemical_symbols())
print(f"   geometry max |dpos| {dpos:.3e} A, cell max |dcell| {dcell:.3e} A, "
      f"species identical {same_sp}, splits {split_by_sid[1]}/"
      f"{split_by_sid[601]}")
d1, d6 = dft_terms(1), dft_terms(601)
print(f"\n   LABEL DEFINITION, identified against the OUTCAR rather than "
      f"assumed:")
for sid, dd, a in ((1, d1, a1), (601, d6, a6)):
    lbl = float(a.info["energy"])
    which = [k for k in ("toten", "noentropy", "sigma0")
             if abs(dd[k] - lbl) < 1e-6]
    print(f"     sid {sid:>3}: label {lbl:+.8f} == OUTCAR "
          f"{which if which else 'NONE OF toten/noentropy/sigma0'}; "
          f"TOTEN {dd['toten']:+.8f}, no-entropy {dd['noentropy']:+.8f}, "
          f"sigma->0 {dd['sigma0']:+.8f}")
    print(f"              NELECT {dd['nelect']:.4f}, E-fermi "
          f"{dd['efermi']:+.4f} eV, total_charge "
          f"{float(a.info.get('total_charge',0.0)):+.2f}")
print(f"     -> NELECT differs by {d1['nelect']-d6['nelect']:+.4f}, so this is "
       f"an electron-ADDITION energy, and the label is a plain VASP energy "
       f"with NO reservoir term: canonical, not grand canonical.")

m1, m6 = run(1), run(601)
dE_m = m1["_total"] - m6["_total"]
dE_d = m1["_label"] - m6["_label"]
print(f"\n   RAW TOTALS AND THE DIFFERENCE")
print(f"     {'':>10} {'charged (sid 1)':>18} {'neutral (sid 601)':>18} "
      f"{'difference':>14}")
print(f"     {'model':>10} {m1['_total']:+18.6f} {m6['_total']:+18.6f} "
      f"{dE_m:+14.6f}")
print(f"     {'DFT':>10} {m1['_label']:+18.6f} {m6['_label']:+18.6f} "
      f"{dE_d:+14.6f}")
print(f"     eps per frame: charged {m1['_total']-m1['_label']:+.6f}, neutral "
      f"{m6['_total']-m6['_label']:+.6f} eV")
print(f"     eps_Delta = {dE_m - dE_d:+.6f} eV   "
      f"(the earlier table implied +5.6350 from 20.843 and -6.379 meV/atom "
      f"x 207; agreement {abs((dE_m-dE_d)-5.6350):.4f} eV)")

print(f"\n=================== STEP 2: every model term "
      f"===================")
print(f"   {'term':>30} {'neutral':>14} {'charged':>14} {'difference':>14}")
tot_n = tot_c = 0.0
for lbl, k in TERMS:
    vn, vc = m6[k], m1[k]
    tot_n += vn; tot_c += vc
    z = "   (not enabled: exact zero)" if vn == 0.0 and vc == 0.0 else ""
    print(f"   {lbl:>30} {vn:+14.6f} {vc:+14.6f} {vc-vn:+14.6f}{z}")
print(f"   {'SUM of the ten':>30} {tot_n:+14.6f} {tot_c:+14.6f} "
      f"{tot_c-tot_n:+14.6f}")
print(f"   {'model total':>30} {m6['_total']:+14.6f} {m1['_total']:+14.6f} "
      f"{dE_m:+14.6f}")
cl = max(abs(tot_n - m6["_total"]), abs(tot_c - m1["_total"]))
print(f"   [{'PASS' if cl < 1e-6 else 'FAIL'}] closure: the ten terms sum to "
      f"the model's own total, worst |diff| {cl:.3e} eV -- so the "
      f"decomposition is complete and nothing is hidden in a remainder")

print(f"\n=================== STEP 3: the DFT side, by its own formula "
      f"===================")
print(f"   {'printed quantity':>22} {'neutral':>14} {'charged':>14} "
      f"{'difference':>14}")
for k, nm in (("sigma0", "energy(sigma->0)"), ("toten", "free energy TOTEN"),
              ("noentropy", "energy w/o entropy"), ("a_cav", "A_cav"),
              ("a_corr", "A_corr"), ("ecorr", "Ecorr"),
              ("ecorr_band", "Ecorr_band"), ("dele_z", "Dele_z (dipole)"),
              ("efermi", "E-fermi"), ("nelect", "NELECT")):
    print(f"   {nm:>22} {d6[k]:+14.6f} {d1[k]:+14.6f} {d1[k]-d6[k]:+14.6f}")
print(f"\n   NOT DONE HERE, deliberately: the DFT solvent free energy is not "
      f"equated with the model's cross+self, and the energy head is not "
      f"equated with any DFT electronic term. The one correspondence that is\n"
      f"   defensible on its face is A_cav against the model's cavity energy, "
      f"reported as a CANDIDATE and not folded into any total:")
print(f"     dA_cav {d1['a_cav']-d6['a_cav']:+.6f} eV against d(cavity "
      f"energy) {m1['cavity_energy_g']-m6['cavity_energy_g']:+.6f} eV, "
      f"difference "
      f"{(m1['cavity_energy_g']-m6['cavity_energy_g'])-(d1['a_cav']-d6['a_cav']):+.6f}")
print(f"   Everything else on the DFT side stays in ONE named remainder: "
      f"'DFT terms whose additive role and model counterpart are both "
      f"unverified'. Its size is not quotable as an error.")

print(f"\n=================== STEP 4: can the energy head move Delta E? "
      f"===================")
ks = sorted(set(m1["_feats"]) | set(m6["_feats"]))
worst = 0.0
for k in ks:
    f1, f6 = m1["_feats"].get(k), m6["_feats"].get(k)
    if f1 is None or f6 is None or f1.shape != f6.shape:
        print(f"   block {k}: shapes {None if f1 is None else tuple(f1.shape)}"
              f" vs {None if f6 is None else tuple(f6.shape)} -- DIFFER")
        worst = float("inf"); continue
    d = float((f1 - f6).abs().max())
    rel = d / max(float(f6.abs().max()), 1e-30)
    worst = max(worst, rel)
    print(f"   readout block {k:>3}: features {tuple(f1.shape)}, max |diff| "
          f"{d:.3e}, relative {rel:.3e}")
print(f"   d(inter_e) across the pair: "
      f"{m1['interaction_energy']-m6['interaction_energy']:+.6f} eV")
if worst < 1e-12:
    print(f"   VERDICT: the readout inputs are IDENTICAL between the two "
          f"states, so the energy head contributes the same amount to both and "
          f"CANCELS EXACTLY in Delta E. Retraining that head alone therefore\n"
          f"   cannot change the paired charging energy at all, whatever the "
          f"target -- which invalidates the A/B refit as a route to this error "
          f"and is why this check came first.")
else:
    print(f"   VERDICT: the readout inputs DIFFER between the two states "
          f"(worst relative {worst:.3e}), so the head can move Delta E and "
          f"retraining it is not excluded. Its actual leverage is "
          f"d(inter_e) above.")

print(f"\n=================== STEP 5: the validation pairs "
      f"===================")
if not PAIRS_VAL:
    print(f"   SKIPPED: 0 of {len(_pairs_all)} val pairs are complete on this "
          f"machine. Not a reduced table -- no table. The step-1 pair is the "
          f"only complete one here and it is already reported above.")
    for h in hooks:
        h.remove()
    print("\nDONE (steps 1-4 only)")
    raise SystemExit(0)
if len(PAIRS_VAL) < len(_pairs_all):
    print(f"   SUBSET: {len(PAIRS_VAL)} of {len(_pairs_all)} pairs complete "
          f"here; the bias and RMSE below cover only those and must be "
          f"reported as a subset.")
# A HYPOTHESIS TO TEST HERE, not a conclusion, and it came out of the one
# pair the workstation ran. The four terms that cannot contribute to Delta E
# are e0 (per-species, identical species) and inter_e (identical features, so
# the descriptor is charge-blind), plus two disabled terms. So the model has NO
# channel for the energy of the ADDED ELECTRON ITSELF -- no integral mu dN --
# and on that pair the missing amount matched the mean Fermi level to 0.20%:
#   eps_Delta = +5.635095, so the model is short by -5.635095 eV
#   mean Fermi x dN = (-3.8745 + -7.3736)/2 x 1 = -5.6240 eV, residual -0.0110
# The endpoints do NOT match (-1.76 and +1.74), so it is specifically the
# TRAPEZOID. One pair cannot distinguish that from coincidence; twenty pairs
# with dN spread over 0.8 to 1.3 can, because a varying dN makes the relation
# a one-parameter prediction rather than a single number. The columns below
# test it and the residual RMSE says how much it explains.
_sp = {}
for k in PAIRS_VAL:
    _sp[k] = (split_by_sid.get(k, "?"), split_by_sid.get(600 + k, "?"))
_allval = all(v == ("val", "val") for v in _sp.values())
print(f"   pair set: {PAIRS_VAL}")
print(f"   splits:   " + ", ".join(f"{k}:{a}/{b}" for k, (a, b) in _sp.items()))
if _allval:
    print(f"   -> these are the user's 20 validation pairs.")
else:
    print(f"   -> NOT the user's validation table: some or all of these pairs "
          f"are TRAINING pairs. Evidence about a structural absence still "
          f"counts here, but this table must not be reported as the "
          f"validation result.")
print(f"   {'pair':>6} {'q':>7} {'dN':>6} {'DFT dE':>11} {'model dE':>11} "
      f"{'eps_D':>10} {'mu_bar.dN':>11} {'residual':>10}")
rows = []
for k in PAIRS_VAL:
    mc, mn = run(k), run(600 + k)
    de_m = mc["_total"] - mn["_total"]
    de_d = mc["_label"] - mn["_label"]
    dN = -mc["_q"]
    mu = 0.5 * (mc["_fermi"] + mn["_fermi"])
    res = (de_m - de_d) + mu * dN
    rows.append((k, mc["_q"], dN, de_d, de_m, de_m - de_d, mu * dN, res, mu,
                 mc["_fermi"], mn["_fermi"]))
    print(f"   {k:>6} {mc['_q']:+7.2f} {dN:+6.2f} {de_d:+11.6f} "
          f"{de_m:+11.6f} {de_m-de_d:+10.6f} {mu*dN:+11.6f} {res:+10.6f}")
e = np.array([r[5] for r in rows])
mu_dn = np.array([r[6] for r in rows])
res = np.array([r[7] for r in rows])
print(f"   {'bias':>6} {'':>7} {'':>6} {'':>11} {'':>11} {e.mean():+10.6f} "
      f"{mu_dn.mean():+11.6f} {res.mean():+10.6f}")
print(f"   {'RMSE':>6} {'':>7} {'':>6} {'':>11} {'':>11} "
      f"{np.sqrt((e*e).mean()):10.6f} {'':>11} {np.sqrt((res*res).mean()):10.6f}")
print(f"   {'min/max':>6} {'':>7} {'':>6} {'':>11} {'':>11} "
      f"{e.min():+.3f}/{e.max():+.3f} {'':>11} {res.min():+.3f}/{res.max():+.3f}")
print(f"\n   THE ELECTRON-ENERGY HYPOTHESIS, tested rather than asserted. "
      f"eps_Delta RMSE {np.sqrt((e*e).mean()):.4f} eV; after crediting "
      f"mu_bar x dN the residual RMSE is {np.sqrt((res*res).mean()):.4f} eV, "
      f"so the trapezoid integral of the\n   chemical potential accounts for "
      f"{100*(1-np.sqrt((res*res).mean())/max(np.sqrt((e*e).mean()),1e-30)):.1f}% "
      f"of the charging-energy error in RMS. Correlation between eps_Delta and "
      f"-mu_bar x dN: {np.corrcoef(e, -mu_dn)[0,1]:+.4f}.")
_sl = float(np.polyfit(-mu_dn, e, 1)[0]) if len(rows) > 2 else float('nan')
print(f"   Regressing eps_Delta on -mu_bar x dN gives slope {_sl:+.4f} "
      f"(1.0000 would mean the missing energy IS the electron's own energy at "
      f"the average chemical potential).")
# endpoint controls: trapezoid against either endpoint is the whole question,
# and on the single pair the endpoints missed by -1.76 and +1.74 eV while the
# mean missed by -0.011. Computed from the stored Fermi values, no extra runs.
dn = np.array([r[2] for r in rows])
fc = np.array([r[9] for r in rows])
fn = np.array([r[10] for r in rows])
for nm, mu_alt in (("charged-endpoint Fermi", fc),
                   ("neutral-endpoint Fermi", fn),
                   ("trapezoid mean", 0.5 * (fc + fn))):
    r_alt = e + mu_alt * dn
    print(f"   crediting the {nm:>22}: residual RMSE "
          f"{np.sqrt((r_alt*r_alt).mean()):8.4f} eV, bias "
          f"{r_alt.mean():+8.4f}")
print(f"   Only the trapezoid is expected to work if the missing quantity is "
      f"an integral of the chemical potential over the electron count; an "
      f"endpoint working equally well would mean it is something else.",
      flush=True)
print(f"\n   eV per pair. The sid1/sid601 value above ({dE_m-dE_d:+.4f}) sits "
      f"at percentile "
      f"{100.0*float((e < (dE_m-dE_d)).mean()):.0f} of these 20, so the first "
      f"pair is {'representative' if abs((dE_m-dE_d)-e.mean()) < e.std() else 'NOT representative'}"
      f" of the group.")
np.savez(os.path.join(os.environ.get("KIT_OUT", "."),
                      "charging_reconcile.npz"),
         pair=np.array([r[0] for r in rows]), q=np.array([r[1] for r in rows]),
         dN=dn, dft=np.array([r[3] for r in rows]),
         model=np.array([r[4] for r in rows]), err=e, mu_dn=mu_dn,
         residual=res, fermi_charged=fc, fermi_neutral=fn)
for h in hooks:
    h.remove()
print("\nDONE")
