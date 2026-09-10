"""Is the bound channel driven wrong, or does it respond wrong?
(user-specified single cheap test, 2026-09-10, charged frame, no re-solve.)

WHAT THIS FOLLOWS FROM. The whole-system check showed that fixing s_ion alone
buys nothing in aggregate. But the user then extracted a conclusion I had
missed: cross couplings ADD, so the bound channel's coupling error is
(total - ionic), and both channels in fact improved:

  coupling error, model - reference (eV)   baseline   after the s_ion swap
    bound                                   +0.5371          +0.4141
    ionic                                   -0.1417          -0.0170
    total                                   +0.3954          +0.3971

The two errors have OPPOSITE signs -- the bound channel under-attracts, the
ionic over-attracts -- so each getting closer to the reference reduced their
mutual cancellation and left the total slightly worse. My earlier reading,
"both cross and self got worse", was true of the totals and wrong at the
channel level, which is the level that carries the information. In particular
the result may NOT be read as "the bound charge was degraded and ate the ionic
gain". Whether the bound DISTRIBUTION improved still needs the arrays compared
directly, which is separate from the coupling.

Scope, stated because the previous run's table did not: that whole-system
block is the PLANE-AVERAGED 1-D electrostatic total. It does not contain the
full lateral 3-D error.

THE TEST. Hold the bound response exactly as the model has it -- the
coefficients a1 and the background p_off, both taken from the captured
baseline -- and change only the potential that DRIVES it, from the model's own
total potential to the DFT total potential. Compute the bound charge directly,
n_b = B @ phi + nb_off, and compare against the plane-averaged DFT RHOB. No
re-solve, so nothing else can move.

  the bound distribution visibly recovers -> the generation of the total
      potential is the priority;
  it stays clearly wrong                  -> the bound RESPONSE itself is,
      including the cavity, the 1-D closure and the learned correction.

That separates "the potential driving it is wrong" from "its response to the
potential is wrong", which is a sharper question than extending the basis or
retraining, and any later fix is still to be judged on the aggregate charge,
potential and energy.

Three things are measured rather than assumed.

The convention. The solver's phi is in the ELECTRON-ENERGY convention -- the
backend's rho = -(n_b/volume) and rho = -(n_ion/volume) exist to convert it,
and ion_potential_2x2 measured physical = -out["phi"] at 6.5e-19 against
2.5e-03 for the flip -- which is the SAME convention as the stored PHI files,
where physical = -PHI_raw was likewise measured. So the DFT potential enters
B @ phi as the plane-averaged PHI_raw with NO negation. That is not taken on
trust: the relative sign of the two profiles is measured here by correlating
them (mean removed, since a constant reference difference is not a sign), and
the bound charge is reported for BOTH signs so nothing hinges on the choice.

The reference zero. n_b = B @ phi + nb_off is linear in phi, so a constant
offset would matter unless B annihilates constants. B = V * WB @ D @ diag(a1)
@ D @ WB ends in a derivative acting on a smoothed constant, so it should
annihilate them exactly; |B @ 1| against |B @ phi| is printed to settle that
by measurement rather than by reading the formula.

Whether the test has any room to act. If the model's total potential were
already close to the DFT one, swapping it could not change the bound charge
much and the test would be near-vacuous. The difference between the two
driving potentials is therefore reported alongside the result.
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
from mace.modules.pb1d_solver import Solver1D

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
    cap["phi_tot"] = out["phi"].detach().clone()
    cap["n_b"] = out["n_b"].detach().clone()
    return out
Solver1D.solve = wrap_sv
_sg = PB.PB1DBackend.solve_graph
def wrap_sg(self, *a, **k):
    out = _sg(self, *a, **k)
    cap["rho_bound_z"] = out["rho_bound_z"].detach().clone()
    cap["rho_ion_z"] = out["rho_ion_z"].detach().clone()
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
lz = float(torch.linalg.norm(torch.as_tensor(
    grid.cell, dtype=torch.float64, device=device)[2]))
volume = grid.volume
nz_s = kw["cvhar_z"].shape[0]; dz = lz / nz_s; area = volume / lz
print(f"=================== {TAG} ===================")
print(f"  one full model call; solve grid nz_s {nz_s}. Bound response held "
      f"fixed: a1 and p_off from the captured baseline. Only the DRIVING "
      f"potential changes. No re-solve.", flush=True)

B = solver.bound_matrix(kw["a1"])
nb_off = solver.bound_offset(kw["p_off"])


def rho_b_of(phi):
    """The bound channel exactly as the solver forms it, converted to physical
    density the way the backend does: rho = -(n_b / volume)."""
    return -((B @ phi + nb_off) / volume)


# ---- DFT side ---------------------------------------------------------
lat, phi_raw = read_grid(f"{DFTDIR}/PHI")
_, rb_raw = read_grid(f"{DFTDIR}/RHOB")
Vd = float(abs(np.linalg.det(lat)))
# the solver's phi is ELECTRON-ENERGY convention, the same as the stored file,
# so PHI_raw goes in WITHOUT negation -- verified by the gate below
phi_dft = fresample(phi_raw.to(device).mean(dim=(0, 1)), nz_s)
rb_ref = fresample((-(rb_raw.to(device)) / Vd).mean(dim=(0, 1)), nz_s)
del phi_raw, rb_raw
phi_score = -(kw["cvhar_z"] - kw["cvhar_z"].mean())

# ---- gates and the three measurements --------------------------------
rb_model = fresample(cap["rho_bound_z"], nz_s)
g = float((rho_b_of(cap["phi_tot"]) - rb_model).abs().max())
print(f"\n  GATES AND MEASURED CONVENTIONS")
print(f"   [{'PASS' if g < 1e-12 else 'FAIL'}] the model's own potential "
      f"reproduces the model's own bound charge through B and nb_off: max abs "
      f"{g:.3e} e/A^3 -- the response really is n_b = B @ phi + nb_off")
ones = torch.ones_like(cap["phi_tot"])
bo = float((B @ ones).abs().max())
scale = float((B @ cap["phi_tot"]).abs().max())
print(f"   [reference zero] max |B @ 1| {bo:.3e} against max |B @ phi| "
      f"{scale:.3e}, ratio {bo/max(scale,1e-30):.2e} -- tiny means the bound "
      f"channel cannot see a constant offset in phi, so the electrolyte "
      f"reference zero cannot affect this comparison at all")


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / max(float(a.norm() * b.norm()), 1e-30))


cp, cm = corr(phi_dft, cap["phi_tot"]), corr(-phi_dft, cap["phi_tot"])
sgn_ok = cp > 0.5 and cm < -0.5
print(f"   [{'PASS' if sgn_ok else 'FAIL'}] relative sign measured by "
      f"correlation with the model's own total potential (mean removed): "
      f"+PHI_raw {cp:+.4f}, -PHI_raw {cm:+.4f} -- +PHI_raw is the solver's "
      f"convention, as the separation shows; both are reported below anyway")
d0 = cap["phi_tot"] - phi_dft
d0c = d0 - d0.mean()
print(f"   [room to act] the two driving potentials differ by: L1 "
      f"{float(d0.abs().sum() * dz):.4f}, max {float(d0.abs().max()):.4f}, "
      f"rms {float(d0.pow(2).mean().sqrt()):.5f} eV; with the constant "
      f"removed rms {float(d0c.pow(2).mean().sqrt()):.5f} eV (constant "
      f"{float(d0.mean()):+.4f}) -- a near-zero rms would make the swap "
      f"vacuous", flush=True)
if g >= 1e-12 or not sgn_ok:
    print(f"   A GATE FAILED -- no reading offered.", flush=True)


def metrics(p, lbl):
    cross = float((p * phi_score).sum() * dz * area)
    cross_ref = float((rb_ref * phi_score).sum() * dz * area)
    rows = []
    for d in SHIFTS:
        ps = fshift(p, float(d), lz)
        num = float((rb_ref * ps).sum()); den = float((ps * ps).sum())
        sc = num / max(den, 1e-300)
        rows.append((float(d), sc,
                     float((rb_ref - sc * ps).norm()
                           / max(float(rb_ref.norm()), 1e-30))))
    bst = min(rows, key=lambda r: r[2])
    return dict(lbl=lbl, net=float(p.sum() * dz * area),
                absq=float(p.abs().sum() * dz * area),
                cross=cross, gap=cross - cross_ref,
                l1=float((p - rb_ref).abs().sum() * dz * area),
                mx=float((p - rb_ref).abs().max()),
                shift=bst[0], scale=bst[1], resid=bst[2],
                edge=abs(abs(bst[0]) - float(SHIFTS[-1])) < 1e-9)


mref = metrics(rb_ref, "DFT reference RHOB")
mA = metrics(rho_b_of(cap["phi_tot"]), "model response, model phi")
mB = metrics(rho_b_of(phi_dft), "model response, DFT phi")
mF = metrics(rho_b_of(-phi_dft), "  (control: DFT phi, flipped)")
print(f"\n[RESULT] bound channel, response held fixed, driving potential "
      f"swapped, no re-solve")
print(f"  {'case':>28} {'net (e)':>9} {'int|.|':>8} {'cross':>9} "
      f"{'gap':>9} {'L1':>9} {'max dev':>9} {'shift':>8} {'resid':>7}")
for m in (mref, mA, mB, mF):
    f = "  <-- shift at scan edge" if m["edge"] else ""
    print(f"  {m['lbl']:>28} {m['net']:+9.4f} {m['absq']:8.4f} "
          f"{m['cross']:+9.4f} {m['gap']:+9.4f} {m['l1']:9.5f} "
          f"{m['mx']:9.2e} {m['shift']:+8.3f} {100*m['resid']:6.1f}%{f}",
          flush=True)
print(f"\n  CROSS-CHECK against the accepted whole-system run: the bound "
      f"channel's coupling gap there was +0.5371 eV for this same baseline "
      f"(derived as total minus ionic, since cross couplings add).\n  This "
      f"run's baseline gap is {mA['gap']:+.4f} eV, difference "
      f"{mA['gap'] - 0.5371:+.4f} eV -- the same quantity computed a second "
      f"way, so a mismatch would mean one of the two is wrong.")
print(f"\n  swapping the driving potential: coupling gap {mA['gap']:+.4f} -> "
      f"{mB['gap']:+.4f} eV, charge L1 {mA['l1']:.5f} -> {mB['l1']:.5f} e, "
      f"max deviation {mA['mx']:.3e} -> {mB['mx']:.3e}, displacement "
      f"{mA['shift']:+.3f} -> {mB['shift']:+.3f} A, profile error "
      f"{100*mA['resid']:.1f}% -> {100*mB['resid']:.1f}%")
print(f"  (a visible recovery makes the generation of the total potential the "
       f"priority; staying clearly wrong makes the bound RESPONSE itself the\n"
       f"   priority -- cavity, 1-D closure and the learned correction. This "
       f"is the plane-averaged 1-D channel only; the lateral 3-D error is not\n"
       f"   in any of these numbers.)", flush=True)
np.savez(os.path.join(os.environ.get("KIT_OUT", "."),
                      "bound_response_test_arrays.npz"),
         z=np.arange(nz_s) * dz, rho_b_ref=rb_ref.cpu().numpy(),
         rho_b_model_phi=rho_b_of(cap["phi_tot"]).cpu().numpy(),
         rho_b_dft_phi=rho_b_of(phi_dft).cpu().numpy(),
         rho_b_dft_phi_flipped=rho_b_of(-phi_dft).cpu().numpy(),
         phi_model=cap["phi_tot"].cpu().numpy(),
         phi_dft=phi_dft.cpu().numpy(), phi_score=phi_score.cpu().numpy())
print(f"\n  arrays saved to bound_response_test_arrays.npz", flush=True)
print("DONE")
