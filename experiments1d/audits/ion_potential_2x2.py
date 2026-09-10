"""Model against DFT TOTAL potential, crossed with the ionic cavity.
(user-specified next step 2026-09-09, charged frame, direct substitution only.)

THREE CORRECTIONS carried in from the review, because each one changed what
this script has to measure:

1. The previous sensitivity scan does NOT show the model's reference zero is
   wrong. It shifted the potential rigidly and let the total ionic charge run
   to about +-7 e, whereas the real PB solve carries an electroneutrality
   constraint -- different conditions. Under the constraint a potential error
   redistributes charge rather than multiplying the total. And the response
   contains both ion signs plus saturation, so "times 7" was never a valid
   summary. No sensitivity number is used as evidence here.
2. The 0.414 eV figure from the failed isolation may NOT be imported. That was
   the difference between two MODEL calls' solute potentials -- neither
   model-against-DFT, nor the final total potential the ions actually see.
   Three different quantities; I had conflated them.
3. The order-of-operations script computed only L1 and a max deviation, never
   the displacement or the coupling error that averaging introduces. So
   "averaging cannot explain 0.4 A and 7.4%" was not measured. It is measured
   here, on the existing profiles, with the shift-scan metric.

THE EXPERIMENT: a 2x2 of total potential source against ionic cavity source,
fed to the ionic response, on one grid, with one sign convention and one
electrolyte reference. The two diagonal cells are each independently
validated, which is what makes the off-diagonal cells informative:
  model phi + model s_ion  must reproduce the model's own rho_ion_z;
  DFT phi   + DFT s_ion    reproduces RHOION exactly (job 3426981, 0.00%,
                           required reference offset exactly zero).
Reported per cell: net charge, profile displacement by the shift-scan metric,
profile error, and coupling energy -- the coupling always against ONE common
solute reference potential, which is a scoring field and not part of the
intervention.

The reading, as set: only if swapping in the DFT total potential visibly
restores the ionic layer position is there direct evidence to concentrate on
how the model produces that potential, and only then does separating the
bound response from the boundary handling become the next question.
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
    cap["cv3"] = cv.detach().clone()
    return _clo(n_e, cv, grid, params, tp_)
PB.closure_from_fields = wrap_clo
_sv = Solver1D.solve
def wrap_sv(self, **kw):
    out = _sv(self, **kw)
    cap["kw"] = dict(kw)
    cap["phi_tot"] = out["phi"].detach().clone()      # the FINAL total potential
    cap["n_ion"] = out["n_ion"].detach().clone()
    return out
Solver1D.solve = wrap_sv
_sg = PB.PB1DBackend.solve_graph
def wrap_sg(self, *a, **k):
    out = _sg(self, *a, **k)
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
    """Band-limited periodic resample of a 1-D profile to length n."""
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
    model(b.to_dict(), compute_force=False, training=False)

grid = cap["grid"]; params = cap["params"]; kw = cap["kw"]
shm = tuple(grid.shape)
cellm = torch.as_tensor(grid.cell, dtype=torch.float64, device=device)
lz = float(torch.linalg.norm(cellm[2])); volume = grid.volume
nz_s = kw["cvhar_z"].shape[0]; dz = lz / nz_s; area = volume / lz
# SIGN CONVENTION, and the first run of this script got it wrong. The solver
# works in the ELECTRON-ENERGY convention (the same as the stored PHI files),
# which is why the backend writes rho_ion_z = -(n_ion/volume): n_work is odd,
# so that minus converts the convention. The DFT side here uses -PHI_raw,
# i.e. PHYSICAL. Feeding the solver's phi straight in crossed the two
# conventions and produced exactly the negated ionic charge -- caught by the
# gate, net -1.0000 e against the model's own +1.0000 e. One convention only:
# convert the model's total potential to physical by negating it.
phi_m = -cap["phi_tot"]                     # model total potential, PHYSICAL
sion_m = kw["s_ion"]                        # model ionic switch, nz_s
print(f"=================== {TAG} ===================")
print(f"  solve grid nz_s {nz_s}; params LNLION {params['LNLION']}, "
      f"theta_b {params['theta_b']:.6e}, ZBETA {params['ZBETA']:.6e}",
      flush=True)

# ---- DFT side, brought to the same grid, same sign, same reference --------
lat, phif = read_grid(f"{DFTDIR}/PHI")
_, ri = read_grid(f"{DFTDIR}/RHOION")
_, chg = read_grid(f"{DFTDIR}/CHGCAR")
Vd = float(abs(np.linalg.det(lat)))
shd = tuple(phif.shape)
from pure_python import torch_pb as tp
gd = tp.TorchGrid(lat, shd, device=str(device), dtype=torch.float64,
                  rspec=True)
ne_d = torch.clamp(chg.to(device) / Vd, min=0.0)
si_d3, _sd, _ = tp.create_cavity_torch(ne_d, gd, params)
si_d3 = torch.clamp(si_d3, 0.0, 1.0)
# sign established by measurement in job 3426981: phi = -PHI_raw, offset zero
phi_d3 = -(phif.to(device))
rho_ref3 = -(ri.to(device)) / Vd
del chg, phif, ri, ne_d
phi_d = fresample(phi_d3.mean(dim=(0, 1)), nz_s)
sion_d = torch.clamp(fresample(si_d3.mean(dim=(0, 1)), nz_s), 0.0, 1.0)
rho_ref = fresample(rho_ref3.mean(dim=(0, 1)), nz_s)
print(f"  DFT profiles band-limited resampled {shd[2]} -> {nz_s}; "
      f"reference net ionic charge on the solve grid "
      f"{float(rho_ref.sum()*dz*area):+.6f} e "
      f"(native {float(rho_ref3.sum()*Vd/float(np.prod(shd))):+.6f})",
      flush=True)

# one common solute reference potential for every coupling number below
phi_ref_sol = -(kw["cvhar_z"] - kw["cvhar_z"].mean())
print(f"  coupling scored against ONE common solute reference potential "
      f"(the model's cvhar_z, zero-mean); it is a scoring field, not part of "
      f"the intervention", flush=True)


def rho1(phi_z, s_z):
    return ion_density_values(phi_z, s_z, params, volume) / volume


def metrics(p, ref):
    e = float((p * phi_ref_sol).sum() * dz * area)
    e_ref = float((ref * phi_ref_sol).sum() * dz * area)
    rows = []
    for d in SHIFTS:
        ps = fshift(p, float(d), lz)
        num = float((ref * ps).sum()); den = float((ps * ps).sum())
        sc = num / max(den, 1e-300)
        rows.append((float(d), sc,
                     float((ref - sc * ps).norm() / max(float(ref.norm()),
                                                        1e-30))))
    b = min(rows, key=lambda r: r[2])
    edge = abs(abs(b[0]) - float(SHIFTS[-1])) < 1e-9
    return dict(net=float(p.sum() * dz * area),
                absq=float(p.abs().sum() * dz * area),
                cross=e, gap=e_ref - e,
                l1=float((p - ref).abs().sum() * dz * area),
                shift=b[0], scale=b[1], resid=b[2], edge=edge)


# ---- gate: the two diagonal cells must each be right ---------------------
rho_model = fresample(cap["rho_ion_z"], nz_s)
d_mm = rho1(phi_m, sion_m)
g1 = float((d_mm - rho_model).abs().max())
d_dd = rho1(phi_d, sion_d)
g2 = float((d_dd - rho_ref).abs().sum() * dz * area)
g2r = g2 / max(float(rho_ref.abs().sum() * dz * area), 1e-30)
# convention self-check, so the sign is established by measurement here too
_chk_phys = float((rho1(phi_m, sion_m) - rho_model).abs().max())
_chk_elec = float((rho1(-phi_m, sion_m) - rho_model).abs().max())
print(f"\n  [convention] |rho1(phi_physical) - model rho_ion| max "
      f"{_chk_phys:.3e}; with the sign flipped {_chk_elec:.3e} -- the smaller "
      f"one identifies the convention, and it must be the physical one",
      flush=True)
print(f"\n  GATES")
print(f"   [{'PASS' if g1 < 1e-12 else 'FAIL'}] model phi + model s_ion "
      f"reproduces the model's own rho_ion: max abs {g1:.3e} e/A^3")
print(f"   [{'PASS' if g2r < 0.05 else 'FAIL'}] DFT phi + DFT s_ion "
      f"reproduces RHOION on this grid: L1 {g2:.6f} e = {100*g2r:.2f}%",
      flush=True)
ok = g1 < 1e-12 and g2r < 0.05
if not ok:
    print(f"   A GATE FAILED -- no reading of the 2x2 is offered.", flush=True)

# ---- the 2x2 -------------------------------------------------------------
print(f"\n[2x2] ionic response: total potential source x ionic cavity source")
print(f"  {'cell':>26} {'net (e)':>9} {'int|.|':>8} {'cross':>9} "
      f"{'gap':>9} {'L1':>9} {'shift':>8} {'resid':>7}")
cells = (("model phi, model s_ion", phi_m, sion_m),
         ("DFT phi,   model s_ion", phi_d, sion_m),
         ("model phi, DFT s_ion  ", phi_m, sion_d),
         ("DFT phi,   DFT s_ion  ", phi_d, sion_d))
res = {}
for lbl, ph, si in cells:
    m = metrics(rho1(ph, si), rho_ref)
    res[lbl] = m
    flag = "  <-- shift at scan edge, censored" if m["edge"] else ""
    print(f"  {lbl:>26} {m['net']:+9.4f} {m['absq']:8.4f} {m['cross']:+9.4f} "
          f"{m['gap']:+9.4f} {m['l1']:9.5f} {m['shift']:+8.3f} "
          f"{100*m['resid']:6.1f}%{flag}", flush=True)
if ok:
    a = res["model phi, model s_ion"]; b_ = res["DFT phi,   model s_ion"]
    c = res["model phi, DFT s_ion  "]
    print(f"\n  swapping the TOTAL POTENTIAL only: shift {a['shift']:+.3f} -> "
          f"{b_['shift']:+.3f} A, profile error {100*a['resid']:.1f}% -> "
          f"{100*b_['resid']:.1f}%, gap {a['gap']:+.4f} -> {b_['gap']:+.4f} eV")
    print(f"  swapping the CAVITY only:          shift {a['shift']:+.3f} -> "
          f"{c['shift']:+.3f} A, profile error {100*a['resid']:.1f}% -> "
          f"{100*c['resid']:.1f}%, gap {a['gap']:+.4f} -> {c['gap']:+.4f} eV")
    print(f"  (the reading set for this: only a visible recovery of the layer "
          f"POSITION when the DFT total potential is swapped in is direct\n"
          f"   evidence to concentrate on how the model produces that "
          f"potential.)", flush=True)

# ---- the two metrics the order-of-operations run never computed ----------
print(f"\n[ORDER OF OPERATIONS] the displacement and coupling error that "
      f"averaging introduces, which the previous run did not measure")
r3_native = tp.ion_density_values_from_phi_torch(phi_d3, si_d3, gd,
                                                 params) / Vd
p_3d = fresample(r3_native.mean(dim=(0, 1)), nz_s)
p_1d = rho1(phi_d, sion_d)
m3 = metrics(p_3d, rho_ref); m1 = metrics(p_1d, rho_ref)
print(f"  {'case':>34} {'shift (A)':>10} {'resid':>8} {'cross':>9} "
      f"{'gap':>9} {'L1':>9}")
for lbl, m in (("A: 3-D pointwise, then averaged", m3),
               ("B: averaged first, then 1-D", m1)):
    print(f"  {lbl:>34} {m['shift']:+10.3f} {100*m['resid']:7.1f}% "
          f"{m['cross']:+9.4f} {m['gap']:+9.4f} {m['l1']:9.5f}", flush=True)
print(f"  averaging introduces: displacement {m1['shift']-m3['shift']:+.3f} A, "
      f"profile error {100*(m1['resid']-m3['resid']):+.1f} points, coupling "
      f"{m1['cross']-m3['cross']:+.4f} eV")
print(f"  (against the model's own ionic discrepancy, whose displacement and "
      f"coupling gap appear in the 2x2 above -- compare them there rather\n"
      f"   than against an L1, which is what the previous reading did wrongly.)",
      flush=True)
print("DONE")
