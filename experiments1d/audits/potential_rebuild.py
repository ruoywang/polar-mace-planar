"""Can the potential be rebuilt at all -- and does it need the model's solute input?
(user-specified control, 2026-09-10. Existing fields only; no training, no fitting.)

WHAT THIS REPLACES. The previous round concluded "both inputs are exonerated,
so the potential construction must be wrong". The user rejected that and three
corrections stand behind this script:

  1. The full self-energy IMPROVED, 0.0520 -> 0.0368 eV; what worsened is the
     mutual term inside it. The cross energy improved too, 0.3953 -> 0.0223.
     So the gain cannot be written off as error cancellation. The sufficient
     reason to pause was the potential alone, and only that.
  2. The inputs are NOT excluded. A better charge L1 and a better dipole do
     not imply that every spatial component which determines the potential is
     better, and a band rms does not establish it either. Worse, the dipole I
     reported was the first moment of the SOLVENT charge, while the feedback
     is driven by dip_z = val_ion_dipole_z + dsol_z - q*center_z -- it
     contains the MODEL's SOLUTE dipole, which was never checked. And holding
     cvhar_z fixed across the two cases says nothing about whether cvhar_z is
     right.
  3. The 24.9% and 37.6% are shares of the CHANGE the modification caused,
     not shares of the DFT error. And the solver fixes the potential's
     constant by the ionic electroneutrality condition -- the G=0 row of the
     residual is mean(n_b + n_ion) + q_sol = 0, and n_ion depends on phi's
     mean through n_work -- so when the shape moves the constant must move
     with it. A G=0 change therefore does NOT imply a second bookkeeping
     error. That "two items" conclusion is withdrawn.

SCOPE, also from the user: the `total` in the previous table is the 1-D
solvent electrostatic energy at a FIXED bare-solute potential. It contains
neither the dipole correction nor the G=0 term under discussion, so "96%
closed" must not be read as the full DFT energy being repaired.

THE CONTROL. Hold ONE solvent charge -- the DFT bound plus ionic -- and rebuild
the total potential twice, changing only the solute input:

  solute input                          question
  DFT solute potential and dipole       can all-reference inputs rebuild the
                                        DFT total potential at all?
  model solute potential and dipole     how much error does swapping in the
                                        model's inputs add?

Both rows compute the dipole correction from their OWN complete inputs, since
that is what the solver does. Mean-removed potentials are compared first; the
constant is checked separately against the ionic electroneutrality condition
rather than folded into the shape comparison.

  all-DFT rebuilds and the model's inputs do not -> the solute input is the
      priority.
  all-DFT does not rebuild                       -> potential assembly,
      boundary and grid conventions.
  both rebuild                                   -> back to the residual
      charge error and the self-consistent response.

THE DFT SOLUTE INPUT IS CONSTRUCTED INDEPENDENTLY, never back-derived from the
total potential it is used to test. The backend builds

  cvhar3 = phi_base - l0_inv(net_g),   net = neutral_v - n_e

so cvhar is l0_inv(cores - electrons), the solute Hartree potential with the
model's electron density. Substituting the DFT density in the same assembly,
the reference fields phi_base and neutral_v CANCEL out of the difference:

  cvhar_DFT = cvhar_model + l0_inv(n_e_DFT - n_e_model)

which needs only the two electron densities. PHI never enters. The DFT solute
dipole comes from the same solute_dipole_z the model uses, called with the DFT
electron profile.
"""
import json
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

from pure_python.pb import derived_params

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.pb1d_backend as PB
import mace.modules.pb1d_closure as CL
from mace.modules.pb1d_solver import (Solver1D, cdipol_potential_1d,
                                      ion_density_values)

DFT = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single")
SID, DFTDIR, TAG = 1, f"{DFT}/1-44_GCE/cal_1", "NiN44 q=-1.00"
CFG = os.environ.get(
    "KIT_PB_CONFIG",
    "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/cal1_train.json")
CKDIR = os.environ.get("KIT_CKPT_DIR", "checkpoints")
DATADIR = os.environ.get("KIT_DATA_DIR", "data")

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
with open(CFG) as fh:
    params = derived_params(json.load(fh).get("solvation", {}))
print(f"=================== {TAG} ===================")
print(f"  one solvent charge (DFT bound + ionic) for both rows; only the "
      f"SOLUTE input changes. Existing fields only, no fitting.", flush=True)


def read_grid(path):
    f = open(path); f.readline(); s = float(f.readline())
    lat = np.array([[float(x) for x in f.readline().split()]
                    for _ in range(3)]) * s
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
    need = nx * ny * nz; vals = []
    while len(vals) < need:
        vals.extend(float(x) for x in f.readline().split())
    return lat, torch.tensor(np.ascontiguousarray(
        np.array(vals[:need]).reshape(nz, ny, nx).transpose(2, 1, 0)))


def fresample(p, n):
    m = p.shape[0]
    if m == n:
        return p
    sp = torch.fft.rfft(p)
    keep = min(sp.shape[0], n // 2 + 1)
    out = torch.zeros(n // 2 + 1, dtype=sp.dtype, device=sp.device)
    out[:keep] = sp[:keep]
    return torch.fft.irfft(out, n=n) * (n / m)


def hi_share(p, cut):
    sp = torch.fft.rfft(p)
    e = sp[1:].real ** 2 + sp[1:].imag ** 2
    return float(e[cut:].sum() / torch.clamp(e.sum(), min=1e-300))


# ---- DFT native fields (plane profiles only) ----
lat, chg = read_grid(f"{DFTDIR}/CHGCAR")
Vd = float(abs(np.linalg.det(lat)))
nzd = chg.shape[2]
ne_val_dft_n = chg.to(device).mean(dim=(0, 1)).clone()   # density * V, as
del chg                                                  # n_e_values is
_, phi_raw = read_grid(f"{DFTDIR}/PHI")
phi_dft_n = phi_raw.to(device).mean(dim=(0, 1)).clone(); del phi_raw
_, rb = read_grid(f"{DFTDIR}/RHOB")
rho_b_n = (-(rb.to(device)) / Vd).mean(dim=(0, 1)).clone(); del rb
_, ri = read_grid(f"{DFTDIR}/RHOION")
rho_i_n = (-(ri.to(device)) / Vd).mean(dim=(0, 1)).clone(); del ri
torch.cuda.empty_cache()

# ---- one model call ----
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

cap = {}
_sv = Solver1D.solve
def wrap_sv(self, **kw):
    out = _sv(self, **kw)
    cap["solver"] = self; cap["kw"] = dict(kw)
    cap["phi"] = out["phi"].detach().clone()
    cap["phi_sol"] = out["phi_sol"].detach().clone()
    cap["n_b"] = out["n_b"].detach().clone()
    cap["n_ion"] = out["n_ion"].detach().clone()
    cap["dsol_z"] = out.get("dsol_z")
    return out
Solver1D.solve = wrap_sv
_sd = CL.solute_dipole_z
def wrap_sd(n_e_values_z, pos_frac, z_valence, cell, width=4.0):
    cap["prof_ne_z"] = n_e_values_z.detach().clone()
    cap["pos_frac"] = pos_frac.detach().clone()
    cap["z_val"] = z_valence.detach().clone()
    cap["cell64"] = cell.detach().clone()
    return _sd(n_e_values_z, pos_frac, z_valence, cell, width)
CL.solute_dipole_z = wrap_sd
PB.solute_dipole_z = wrap_sd

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff",
               "solvated": "solvated"},
    arrays_keys={"forces": "forces"},
)
frame = None
for spl in ("train", "val"):
    for a in read(os.path.join(DATADIR, f"{spl}.xyz"), ":"):
        if int(a.info.get("sample_id", -1)) == SID:
            frame = a; break
    if frame is not None:
        break
cfg = mace_data.config_from_atoms(frame, key_specification=kspec)
ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                       cutoff=float(model.r_max))]
b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
with torch.no_grad():
    model(b.to_dict(), compute_force=False, training=False)

solver, kw = cap["solver"], cap["kw"]
nz_s = kw["cvhar_z"].shape[0]
lz, volume = solver.lz, solver.volume
dz, area = lz / nz_s, volume / lz
l0i, zs = solver.l0_inv, solver.z
c_unit, center_z = float(kw["c_unit"]), float(kw["center_z"])
indmin, q_sol = int(kw["indmin"]), float(kw["q_sol"])
nz_c = cap["prof_ne_z"].shape[0]
print(f"\n  solve grid {nz_s}, closure/density grid {nz_c}, DFT native {nzd}; "
      f"c_unit {c_unit:.6e}, center_z {center_z:.4f} A, indmin {indmin}, "
      f"q_sol {q_sol:+.6f}", flush=True)


def phi_of_values(x):
    """l0_inv on a 'values' profile (density * cell volume), matching
    grid.l0_inv_op reduced to the gx=gy=0 line."""
    return torch.fft.irfft(torch.fft.rfft(x) * l0i, n=nz_s)


def counts_of(rho):
    return -(rho * volume)          # rho_phys = -(counts/volume)


def rebuild(cvhar, dip_solute, chg_counts, lbl):
    """The solver's own assembly: phi = cvhar + cvdip(total dipole) +
    l0_inv(solvent charge + q_sol), in the ELECTRON convention."""
    dsol = float((chg_counts * zs).mean())
    qs = float(chg_counts.mean())
    dip = float(dip_solute) + dsol - qs * center_z
    dip = max(-20.0, min(20.0, dip))
    cvdip = cdipol_potential_1d(nz_s, lz, c_unit * dip, indmin, device)
    src = chg_counts + q_sol
    return dict(lbl=lbl, phi=cvhar + cvdip + phi_of_values(src),
                dip_tot=dip, dip_solute=float(dip_solute), dsol=dsol)


# ======================================================================
# GATE -- the assembly itself, on the model's OWN inputs and OWN charge
# ======================================================================
chg_model = cap["n_b"] + cap["n_ion"]
g0 = rebuild(kw["cvhar_z"], kw["val_ion_dipole_z"], chg_model, "model self")
d0 = g0["phi"] - cap["phi"]
gA = float((d0 - d0.mean()).abs().max())
print(f"\n[GATE] the assembly reproduces the model's own converged potential "
      f"from its own inputs and its own solvent charge")
print(f"   shape: max abs deviation after removing the constant "
      f"{gA:.3e} eV; the constant itself {float(d0.mean()):+.6f} eV")
print(f"   [{'PASS' if gA < 1e-9 else 'FAIL'}] -- this validates cvdip, "
      f"l0_inv, indmin, c_unit, center_z, the dipole mixing and every sign at "
      f"once. If it fails nothing below can be read.")
if gA >= 1e-9:
    print(f"   GATE FAILED -- stopping the reading here.", flush=True)

# ======================================================================
# the two solute inputs
# ======================================================================
ne_model = fresample(cap["prof_ne_z"], nz_s)
ne_dft = fresample(ne_val_dft_n, nz_s)
# cvhar_DFT = cvhar_model + l0_inv(n_e_DFT - n_e_model): phi_base and
# neutral_v cancel, so PHI never enters this construction
cvhar_dft = kw["cvhar_z"] + phi_of_values(ne_dft - ne_model)
# call the UNWRAPPED function so the captured model profile is not clobbered
dip_dft = _sd(fresample(ne_val_dft_n, nz_c), cap["pos_frac"],
              cap["z_val"], cap["cell64"])
dip_dft_native = _sd(ne_val_dft_n, cap["pos_frac"], cap["z_val"],
                     cap["cell64"])
print(f"\n[THE TWO SOLUTE INPUTS]")
# every difference is reported RELATIVE to the reference's own scale as well
# as absolutely: "the count matches to 1e-4 while the profile differs at
# 6.985" compares two quantities in different units with no scale, and gives a
# reader no way to judge which is small. The relative pair is the argument.
_nc_m, _nc_d = float(ne_model.mean()), float(ne_dft.mean())
_ne_rms = float(ne_dft.pow(2).mean().sqrt())
_ne_dif = float((ne_dft - ne_model).pow(2).mean().sqrt())
_rel_cnt = abs(_nc_m - _nc_d) / max(abs(_nc_d), 1e-30)
_rel_shp = _ne_dif / max(_ne_rms, 1e-30)
print(f"   electron COUNT: model {_nc_m:.5f}, DFT {_nc_d:.5f} (values = "
      f"density * V, so the profile mean is the electron count); difference "
      f"{abs(_nc_m-_nc_d):.2e}, RELATIVE {_rel_cnt:.2e}")
print(f"   electron SHAPE: profile rms {_ne_rms:.6e}, difference rms "
      f"{_ne_dif:.5f}, RELATIVE {100*_rel_shp:.3f}%")
print(f"     -> the shape error is {_rel_shp/max(_rel_cnt,1e-30):.1e} times "
      f"the count error in relative terms, which is what makes this a SHAPE "
      f"error in the predicted density and not a charge error")
_cv_d = float(cvhar_dft.pow(2).mean().sqrt())
_cv_dif = float((cvhar_dft - kw["cvhar_z"]).pow(2).mean().sqrt())
print(f"   solute POTENTIAL: cvhar_model rms "
      f"{float(kw['cvhar_z'].pow(2).mean().sqrt()):.5f}, cvhar_DFT {_cv_d:.5f}"
      f", difference rms {_cv_dif:.5f} eV, RELATIVE "
      f"{100*_cv_dif/max(_cv_d,1e-30):.3f}%")
# model MINUS reference, stated as such: the previous version computed
# reference minus model and labelled it "the model's error", so its sign and
# its label disagreed and that discrepancy was passed onward
_dip_err = float(kw["val_ion_dipole_z"]) - float(dip_dft)
print(f"   solute DIPOLE: model {float(kw['val_ion_dipole_z']):+.5f}, DFT "
      f"{float(dip_dft):+.5f} (on the {nz_c} grid) and "
      f"{float(dip_dft_native):+.5f} (native {nzd}) e A")
print(f"     -> MODEL MINUS REFERENCE = {_dip_err:+.5f} e A, i.e. the model's "
      f"dipole is {'more negative' if _dip_err < 0 else 'more positive'} by "
      f"{abs(_dip_err):.5f}")
print(f"   grid reach: the model's cvhar is upsampled from {nz_c} so it "
      f"carries no mode above {nz_c//2}; share of cvhar_DFT's spectral energy "
      f"above that cut {100*hi_share(cvhar_dft, nz_c//2):.2f}%", flush=True)

# ======================================================================
# the control: one solvent charge, two solute inputs
# ======================================================================
rho_b = fresample(rho_b_n, nz_s)
rho_i = fresample(rho_i_n, nz_s)
chg_dft = counts_of(rho_b + rho_i)
phi_ref = fresample(phi_dft_n, nz_s)          # electron convention, as stored
rows = [rebuild(cvhar_dft, dip_dft, chg_dft, "DFT solute potential + dipole"),
        rebuild(kw["cvhar_z"], kw["val_ion_dipole_z"], chg_dft,
                "model solute potential + dipole")]
print(f"\n[RESULT] one solvent charge (DFT bound + ionic, net "
      f"{float((rho_b+rho_i).sum()*dz*area):+.5f} e), two solute inputs. "
      f"Mean-removed, since the constant is checked separately below.")
print(f"  {'solute input':>34} {'L1':>9} {'max':>9} {'rms':>9} "
      f"{'45 A amp':>9} {'dipole':>9}")
for r in rows:
    d = (r["phi"] - phi_ref)
    d = d - d.mean()
    sp = torch.fft.rfft(d) / nz_s
    print(f"  {r['lbl']:>34} {float(d.abs().sum()*dz):9.4f} "
          f"{float(d.abs().max()):9.4f} {float(d.pow(2).mean().sqrt()):9.5f} "
          f"{float(2*torch.sqrt(sp[1].real**2+sp[1].imag**2)):9.5f} "
          f"{r['dip_tot']:+9.4f}")
rA, rB = rows[0], rows[1]
dA = rA["phi"] - phi_ref; dA = dA - dA.mean()
dB = rB["phi"] - phi_ref; dB = dB - dB.mean()
eA, eB = float(dA.pow(2).mean().sqrt()), float(dB.pow(2).mean().sqrt())
ref_rms = float((phi_ref - phi_ref.mean()).pow(2).mean().sqrt())
print(f"\n  all-DFT rebuild error {eA:.5f} eV rms, which is "
      f"{100*eA/max(ref_rms,1e-30):.2f}% of the potential's own rms "
      f"({ref_rms:.5f} eV)")
print(f"  swapping in the model's solute input takes it to {eB:.5f} eV, "
      f"a factor of {eB/max(eA,1e-30):.2f}")
print(f"\n  the four relative errors, each against its own reference scale, "
      f"in the order the assembly applies them:")
print(f"    electron count   {_rel_cnt:.2e}")
print(f"    density shape    {100*_rel_shp:.3f}%")
print(f"    cvhar            {100*_cv_dif/max(_cv_d,1e-30):.3f}%")
print(f"    rebuilt potential {100*eB/max(ref_rms,1e-30):.3f}%")
print(f"  Each is larger than the last, and l0_inv sits between the second "
      f"and third while the dipole correction sits between the third and "
      f"fourth. NO SHARES ARE ATTRIBUTED: the two contributions were never\n"
      f"  separated, and treating the printed parts of a total as a "
      f"decomposition is exactly what went wrong last round.")

# ---- the constant, by the ionic electroneutrality condition ----
print(f"\n[THE CONSTANT] checked against the condition the solver actually "
      f"uses, not folded into the shape. The residual's G=0 row is "
      f"mean(n_b + n_ion) + q_sol = 0 and n_ion depends on phi's mean through "
      f"n_work,\n  so the constant is set by ionic electroneutrality and MUST "
      f"move when the shape moves -- which is why a G=0 change does not imply "
      f"a separate bookkeeping error.")


def ion_total(phi, c):
    n = ion_density_values(phi + c, kw["s_ion"], solver.params, volume)
    return float(-(n / volume).sum() * dz * area)


tgt = float(rho_i.sum() * dz * area)
print(f"   target ionic total from DFT RHOION {tgt:+.6f} e")
for r in rows + [dict(lbl="DFT total potential (stored)", phi=phi_ref)]:
    lo, hi = -20.0, 20.0
    f_lo, f_hi = ion_total(r["phi"], lo) - tgt, ion_total(r["phi"], hi) - tgt
    if f_lo * f_hi > 0:
        print(f"   {r['lbl']:>34}: no neutralising offset in [-20, 20] eV "
              f"(f {f_lo:+.3e} .. {f_hi:+.3e})")
        continue
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (ion_total(r["phi"], mid) - tgt) * f_lo > 0:
            lo = mid
        else:
            hi = mid
    c = 0.5 * (lo + hi)
    print(f"   {r['lbl']:>34}: offset {c:+.6f} eV, its own mean "
          f"{float(r['phi'].mean()):+.5f}, mean after the offset "
          f"{float(r['phi'].mean())+c:+.5f} vs the DFT potential's "
          f"{float(phi_ref.mean()):+.5f}")
print(f"   (the stored DFT total potential's own offset is the calibration: "
      f"it should be ~0, and each rebuild's offset says how far its constant "
      f"sits from neutrality.)", flush=True)

print(f"\n[READING]")
print(f"  all-DFT rebuilds and the model's input does not -> the SOLUTE INPUT "
      f"is the priority.")
print(f"  all-DFT does not rebuild -> the potential ASSEMBLY, boundary and "
      f"grid conventions.")
print(f"  both rebuild -> back to the residual charge error and the "
      f"self-consistent response.")
print(f"  Nothing here is a share of the DFT error; these are rebuild errors "
      f"against the stored DFT total potential with one solvent charge held "
      f"fixed. Plane-averaged 1-D only.", flush=True)
np.savez(os.path.join(os.environ.get("KIT_OUT", "."),
                      "potential_rebuild_arrays.npz"),
         z=np.arange(nz_s) * dz, phi_ref=phi_ref.cpu().numpy(),
         phi_dft_solute=rA["phi"].cpu().numpy(),
         phi_model_solute=rB["phi"].cpu().numpy(),
         cvhar_model=kw["cvhar_z"].cpu().numpy(),
         cvhar_dft=cvhar_dft.cpu().numpy(),
         ne_model=ne_model.cpu().numpy(), ne_dft=ne_dft.cpu().numpy(),
         rho_b=rho_b.cpu().numpy(), rho_i=rho_i.cpu().numpy(),
         phi_model_own=cap["phi"].cpu().numpy())
print(f"\n  arrays saved to potential_rebuild_arrays.npz")
print("DONE")
