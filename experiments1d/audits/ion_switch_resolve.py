"""Does the ionic-switch improvement survive a self-consistent re-solve?
(user-specified single control, 2026-09-09, charged frame.)

WHAT THIS FOLLOWS FROM, and two corrections to my own reading of it.

The no-re-solve 2x2 (job 3427096) found that with the total potential held
fixed the ionic layer position follows s_ion: swapping the potential alone
left the displacement at +0.375 A, swapping the switch alone collapsed it to
+0.025 A.

CORRECTION 1, and it withdraws a claim of mine. I wrote that both single
swaps made the energy worse and called it an error cancellation. That mixed in
a change of total charge: the two mixed cells carry +1.1317 and +0.8863 e, not
the reference's ~1 e, because with the potential held and no re-solve nothing
enforces electroneutrality. Per unit charge the picture is different:
  model -2.0802 | potential only -2.0601 | switch only -1.9631
  both -1.9428  | DFT reference about -1.9385   (eV per e)
So swapping the switch takes the attraction per unit charge most of the way to
the reference, and the worse TOTAL energy is a charge-magnitude effect. The
cancellation claim does not survive. The division is diagnostic only and is
NOT a suggestion to rescale the charge.

CORRECTION 2, also withdrawn. I attributed the difference against job
3426875/3426892 to "the re-solve pushing the position back". That earlier run
substituted s_ion, a1 AND p_off, with the DFT density first interpolated
through the model grid -- three differences and a different construction, not
one. The comparison cannot carry that attribution.

SO THIS IS THE ONE CONTROL THAT DOES: take the DFT s_ion that worked in the
2x2, exactly as constructed there (native-grid cavity, plane-averaged, then
band-limited resampled to the solve grid), feed it into the solver unchanged,
hold every other input at the baseline, and RE-SOLVE self-consistently. Check
first that the total ionic charge returns to about 1 e, then whether the
position and the coupling improvements hold.

  improvement holds  -> investigate and fix along the ionic switch's
                        generation path;
  position drifts    -> only then is there direct evidence to look at the
                        self-consistent feedback and the boundary handling.

Stated plainly because it matters for what may be concluded: replacing only
s_ion while leaving a1 and p_off from the model's own cavity is a controlled
single-factor intervention, NOT a physically consistent model. It isolates the
switch's effect through the solve; it does not represent a better solvent
model.
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
from mace.modules.pb1d_solver import Solver1D, ion_density_values

DFT = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single")
SID, DFTDIR, TAG = 1, f"{DFT}/1-44_GCE/cal_1", "NiN44 q=-1.00"
CKDIR = os.environ.get("KIT_CKPT_DIR", "checkpoints")
DATADIR = os.environ.get("KIT_DATA_DIR", "data")
SHIFTS = np.arange(-2.0, 2.0001, 0.025)

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

cap = {}
_clo = PB.closure_from_fields
def wrap_clo(n_e, cv, grid, params, tp_):
    cap["grid"] = grid; cap["params"] = params
    return _clo(n_e, cv, grid, params, tp_)
PB.closure_from_fields = wrap_clo
_sv = Solver1D.solve
def wrap_sv(self, **kw):
    out = _sv(self, **kw)
    cap["solver"] = self; cap["kw"] = dict(kw)
    return out
Solver1D.solve = wrap_sv
_sg = PB.PB1DBackend.solve_graph
def wrap_sg(self, *a, **k):
    out = _sg(self, *a, **k)
    cap["rho_ion_z"] = out["rho_ion_z"].detach().clone()
    cap["rho_bound_z"] = out["rho_bound_z"].detach().clone()
    return out
PB.PB1DBackend.solve_graph = wrap_sg

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff",
               "solvated": "solvated"},
    arrays_keys={"forces": "forces"},
)
frame = None
for split in ("train", "val"):
    for a in read(os.path.join(DATADIR, f"{split}.xyz"), ":"):
        if int(a.info.get("sample_id", -1)) == SID:
            frame = a; break
    if frame is not None:
        break


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


def fshift(p, d, lz):
    n = p.shape[0]
    k = torch.fft.rfftfreq(n, d=lz / n, device=p.device) * (2 * math.pi)
    return torch.fft.irfft(torch.fft.rfft(p) * torch.exp(-1j * k * d), n=n)


cfg = mace_data.config_from_atoms(frame, key_specification=kspec)
ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                       cutoff=float(model.r_max))]
b = next(iter(torch_geometric.dataloader.DataLoader(
    ds, batch_size=1))).to(device)
with torch.no_grad():
    model(b.to_dict(), compute_force=False, training=False)      # ONCE

solver = cap["solver"]; kw = cap["kw"]; grid = cap["grid"]
params = cap["params"]
shm = tuple(grid.shape)
cellm = torch.as_tensor(grid.cell, dtype=torch.float64, device=device)
lz = float(torch.linalg.norm(cellm[2])); volume = grid.volume
nz_s = kw["cvhar_z"].shape[0]; dz = lz / nz_s; area = volume / lz
print(f"=================== {TAG} ===================")
print(f"  one full model call; solve grid nz_s {nz_s}", flush=True)

# ---- the DFT s_ion, built EXACTLY as in job 3427096 --------------------
lat, chg = read_grid(f"{DFTDIR}/CHGCAR")
_, ri = read_grid(f"{DFTDIR}/RHOION")
Vd = float(abs(np.linalg.det(lat)))
shd = tuple(chg.shape)
from pure_python import torch_pb as tp
gd = tp.TorchGrid(lat, shd, device=str(device), dtype=torch.float64,
                  rspec=True)
ne_d = torch.clamp(chg.to(device) / Vd, min=0.0)
si_d3, _sd, _ = tp.create_cavity_torch(ne_d, gd, params)
si_d3 = torch.clamp(si_d3, 0.0, 1.0)
sion_dft = torch.clamp(fresample(si_d3.mean(dim=(0, 1)), nz_s), 0.0, 1.0)
rho_ref = fresample((-(ri.to(device)) / Vd).mean(dim=(0, 1)), nz_s)
del chg, ri, ne_d
print(f"  DFT s_ion: native cavity, plane-averaged, band-limited "
      f"{shd[2]} -> {nz_s} -- the same construction as job 3427096, fed in "
      f"unchanged", flush=True)
print(f"  NOTE: only s_ion is replaced. a1 and p_off stay at the model's own "
      f"cavity values, so this is a controlled single-factor intervention and "
      f"NOT a physically consistent solvent model.", flush=True)

phi_score = -(kw["cvhar_z"] - kw["cvhar_z"].mean())


def metrics(p):
    e = float((p * phi_score).sum() * dz * area)
    q = float(p.sum() * dz * area)
    rows = []
    for d in SHIFTS:
        ps = fshift(p, float(d), lz)
        num = float((rho_ref * ps).sum()); den = float((ps * ps).sum())
        sc = num / max(den, 1e-300)
        rows.append((float(d), float((rho_ref - sc * ps).norm()
                                     / max(float(rho_ref.norm()), 1e-30))))
    bst = min(rows, key=lambda r: r[1])
    return dict(net=q, absq=float(p.abs().sum() * dz * area), cross=e,
                per_q=e / q if abs(q) > 1e-9 else float("nan"),
                l1=float((p - rho_ref).abs().sum() * dz * area),
                shift=bst[0], resid=bst[1],
                edge=abs(abs(bst[0]) - float(SHIFTS[-1])) < 1e-9)


# ---- baseline re-solve, then the s_ion-substituted re-solve -----------
with torch.no_grad():
    outA = _sv(solver, **kw)
rA = -(outA["n_ion"] / volume)
kwB = dict(kw); kwB["s_ion"] = sion_dft
with torch.no_grad():
    outB = _sv(solver, **kwB)
rB = -(outB["n_ion"] / volume)
exA = outA.get("solver_exit"); exB = outB.get("solver_exit")

# the no-re-solve switch-only cell, for the comparison this run is about
rNo = ion_density_values(-outA["phi"], sion_dft, params, volume) / volume

# ---- the three whole-system comparisons -------------------------------
_, rb_raw = read_grid(f"{DFTDIR}/RHOB")
rb_ref = fresample((-(rb_raw.to(device)) / Vd).mean(dim=(0, 1)), nz_s)
del rb_raw
_, phi_raw = read_grid(f"{DFTDIR}/PHI")
phi_dft = fresample((-(phi_raw.to(device))).mean(dim=(0, 1)), nz_s)
del phi_raw
l0i = solver.l0_inv


def phi_of(rho):
    """Potential of a 1-D charge profile through the solver's own l0_inv, so
    the self-energy of a SUM carries the mutual term."""
    return torch.fft.irfft(torch.fft.rfft(rho * volume) * l0i, n=nz_s)


def selfE(rho):
    return 0.5 * float((rho * phi_of(rho)).sum() * dz * area)


# convention check: l0_inv applied to the solver's own charges must give
# phi - phi_sol up to the dropped G=0 constant
_chk = phi_of((outA["n_b"] + outA["n_ion"]) / volume + kw["q_sol"] / volume)
_tgt = outA["phi"] - outA["phi_sol"]
_c = float((_tgt - _chk).mean())
print(f"\n  [convention] l0_inv reconstruction of phi - phi_sol: max residual "
      f"after removing the G=0 constant "
      f"{float((_tgt - _chk - _c).abs().max()):.3e} eV (constant {_c:+.4f}) "
      f"-- small means the charge-to-potential operator and its sign are as "
      f"used here", flush=True)


def whole(rb, ri, phi_tot, lbl):
    rt = rb + ri
    rt_ref = rb_ref + rho_ref
    cross = float((rt * phi_score).sum() * dz * area)
    se = selfE(rt)
    se_parts = selfE(rb) + selfE(ri)
    cr_ref = float((rt_ref * phi_score).sum() * dz * area)
    se_ref = selfE(rt_ref)
    dphi = phi_tot - phi_dft
    return dict(lbl=lbl,
                l1=float((rt - rt_ref).abs().sum() * dz * area),
                mx=float((rt - rt_ref).abs().max()),
                net=float(rt.sum() * dz * area),
                cross=cross, self=se, tot=cross + se,
                mutual=se - se_parts,
                d_tot=(cross + se) - (cr_ref + se_ref),
                pl1=float(dphi.abs().sum() * dz), pmx=float(dphi.abs().max()),
                prms=float(dphi.pow(2).mean().sqrt()))

print(f"\n  GATES")
g1 = float((rA - fresample(cap["rho_ion_z"], nz_s)).abs().max())
print(f"   [{'PASS' if g1 < 1e-12 else 'FAIL'}] baseline re-solve reproduces "
      f"the model's own rho_ion: max abs {g1:.3e} e/A^3")
same = (kw["cvhar_z"] is kwB["cvhar_z"]) and (kw["q_sol"] == kwB["q_sol"])
print(f"   [{'PASS' if same else 'FAIL'}] every other input identical "
      f"(same tensor objects, q_sol {kw['q_sol']:+.6f})")
def _ok(e):
    return ((e or {}).get("fix_exit") == "tol"
            and (e or {}).get("newton_exit") == "tol")
print(f"   [{'PASS' if _ok(exA) and _ok(exB) else 'FAIL'}] both solves exit "
      f"on their criteria: A {(exA or {}).get('fix_exit')}/"
      f"{(exA or {}).get('newton_exit')}, B {(exB or {}).get('fix_exit')}/"
      f"{(exB or {}).get('newton_exit')}", flush=True)
allok = g1 < 1e-12 and same and _ok(exA) and _ok(exB)
if not allok:
    print(f"   A GATE FAILED -- no reading offered.", flush=True)

mref = metrics(rho_ref)
print(f"\n[RESULT] ionic channel, one fixed scoring potential")
print(f"  {'case':>34} {'net (e)':>9} {'cross':>9} {'eV per e':>9} "
      f"{'shift':>8} {'resid':>7} {'L1':>9}")
for lbl, m in (("DFT reference", mref),
               ("baseline, model s_ion, re-solved", metrics(rA)),
               ("DFT s_ion, NO re-solve", metrics(rNo)),
               ("DFT s_ion, RE-SOLVED", metrics(rB))):
    f = "  <-- shift at scan edge" if m["edge"] else ""
    print(f"  {lbl:>34} {m['net']:+9.4f} {m['cross']:+9.4f} "
          f"{m['per_q']:+9.4f} {m['shift']:+8.3f} {100*m['resid']:6.1f}% "
          f"{m['l1']:9.5f}{f}", flush=True)
if allok:
    mA, mB, mN = metrics(rA), metrics(rB), metrics(rNo)
    print(f"\n  total charge restored by the re-solve: {mN['net']:+.4f} -> "
          f"{mB['net']:+.4f} e (reference {mref['net']:+.4f})")
    print(f"  does the improvement hold? displacement {mA['shift']:+.3f} -> "
          f"{mB['shift']:+.3f} A (no-re-solve gave {mN['shift']:+.3f}); "
          f"eV per e {mA['per_q']:+.4f} -> {mB['per_q']:+.4f} "
          f"(reference {mref['per_q']:+.4f}); profile error "
          f"{100*mA['resid']:.1f}% -> {100*mB['resid']:.1f}%")
    print(f"  (improvement holding points at the ionic switch's generation "
          f"path; the position drifting back is what would give direct\n"
          f"   evidence for the self-consistent feedback and the boundary "
          f"handling instead.)", flush=True)

    rt_ref = rb_ref + rho_ref
    wref = dict(lbl="DFT reference",
                l1=0.0, mx=0.0, net=float(rt_ref.sum() * dz * area),
                cross=float((rt_ref * phi_score).sum() * dz * area),
                self=selfE(rt_ref),
                tot=float((rt_ref * phi_score).sum() * dz * area) + selfE(rt_ref),
                mutual=selfE(rt_ref) - selfE(rb_ref) - selfE(rho_ref),
                d_tot=0.0, pl1=0.0, pmx=0.0, prms=0.0)
    wA = whole(fresample(cap["rho_bound_z"], nz_s), rA, -outA["phi"],
               "baseline, model s_ion")
    wB = whole(fresample(-(outB["n_b"] / volume), nz_s) if False
               else -(outB["n_b"] / volume), rB, -outB["phi"],
               "DFT s_ion, re-solved")
    print(f"\n[WHOLE SYSTEM] bound + ionic together, one grid, one reference")
    print(f"  {'case':>24} {'net (e)':>9} {'chg L1':>9} {'chg max':>9} "
          f"{'cross':>9} {'self':>8} {'total':>9} {'d total':>9}")
    for w in (wref, wA, wB):
        print(f"  {w['lbl']:>24} {w['net']:+9.4f} {w['l1']:9.5f} "
              f"{w['mx']:9.2e} {w['cross']:+9.4f} {w['self']:+8.4f} "
              f"{w['tot']:+9.4f} {w['d_tot']:+9.4f}", flush=True)
    print(f"  mutual bound-ion term in the self-energy (why the sum must be "
          f"used, not the two separate self-energies):")
    for w in (wref, wA, wB):
        print(f"    {w['lbl']:>24} {w['mutual']:+.4f} eV", flush=True)
    print(f"\n  {'case':>24} {'phi L1 (eV A)':>14} {'phi max (eV)':>13} "
          f"{'phi rms (eV)':>13}")
    for w in (wA, wB):
        print(f"  {w['lbl']:>24} {w['pl1']:14.4f} {w['pmx']:13.4f} "
              f"{w['prms']:13.5f}", flush=True)
    np.savez(os.path.join(os.environ.get("KIT_OUT", "."),
                          "ion_switch_resolve_arrays.npz"),
             z=np.arange(nz_s) * dz,
             rho_bound_ref=rb_ref.cpu().numpy(),
             rho_ion_ref=rho_ref.cpu().numpy(),
             phi_dft=phi_dft.cpu().numpy(),
             phi_score=phi_score.cpu().numpy(),
             rho_bound_A=fresample(cap["rho_bound_z"], nz_s).cpu().numpy(),
             rho_ion_A=rA.cpu().numpy(), phi_A=(-outA["phi"]).cpu().numpy(),
             rho_bound_B=(-(outB["n_b"] / volume)).cpu().numpy(),
             rho_ion_B=rB.cpu().numpy(), phi_B=(-outB["phi"]).cpu().numpy(),
             s_ion_model=kw["s_ion"].cpu().numpy(),
             s_ion_dft=sion_dft.cpu().numpy())
    print(f"\n  arrays saved to ion_switch_resolve_arrays.npz "
          f"(z, both charge channels and the total potential for all three "
          f"cases, plus both ionic switches)", flush=True)
print("DONE")
