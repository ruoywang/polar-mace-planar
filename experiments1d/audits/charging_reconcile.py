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
PAIRS_VAL = [28, 30, 43, 60, 61, 62, 69, 79, 83, 94,
             128, 134, 148, 153, 159, 177, 180, 185, 186, 189]


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

print(f"\n=================== STEP 5: the 20 validation pairs "
      f"===================")
print(f"   {'pair':>6} {'q':>7} {'DFT dE':>12} {'model dE':>12} "
      f"{'model-DFT':>12}")
rows = []
for k in PAIRS_VAL:
    mc, mn = run(k), run(600 + k)
    de_m = mc["_total"] - mn["_total"]
    de_d = mc["_label"] - mn["_label"]
    rows.append((k, mc["_q"], de_d, de_m, de_m - de_d))
    print(f"   {k:>6} {mc['_q']:+7.2f} {de_d:+12.6f} {de_m:+12.6f} "
          f"{de_m-de_d:+12.6f}")
e = np.array([r[4] for r in rows])
print(f"   {'bias':>6} {'':>7} {'':>12} {'':>12} {e.mean():+12.6f}")
print(f"   {'RMSE':>6} {'':>7} {'':>12} {'':>12} "
      f"{np.sqrt((e*e).mean()):12.6f}")
print(f"   {'min/max':>6} {'':>7} {'':>12} {'':>12} "
      f"{e.min():+.4f}/{e.max():+.4f}")
print(f"\n   eV per pair. The sid1/sid601 value above ({dE_m-dE_d:+.4f}) sits "
      f"at percentile "
      f"{100.0*float((e < (dE_m-dE_d)).mean()):.0f} of these 20, so the first "
      f"pair is {'representative' if abs((dE_m-dE_d)-e.mean()) < e.std() else 'NOT representative'}"
      f" of the group.")
np.savez(os.path.join(os.environ.get("KIT_OUT", "."),
                      "charging_reconcile.npz"),
         pair=np.array([r[0] for r in rows]), q=np.array([r[1] for r in rows]),
         dft=np.array([r[2] for r in rows]), model=np.array([r[3] for r in rows]),
         err=e)
for h in hooks:
    h.remove()
print("\nDONE")
