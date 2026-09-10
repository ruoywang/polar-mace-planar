"""Does averaging BEFORE computing the ionic charge lose information?
(user-specified next step 2026-09-09, charged frame only, no re-solve.)

The two implementations are the SAME formula:
  rho_ion = -n_max * invBETA * ZBETA * s_ion * n_work(ZBETA * phi)
with n_work = theta*sinh(x)/(1+theta*(cosh(x)-1)) under LNLION with saturation.
pb1d_solver.ion_density_values is documented as a port of
torch_pb.ion_density_values_from_phi_torch. So the ONLY difference between the
3-D and 1-D use is the order of operations: pointwise-then-average against
average-then-pointwise. Under a nonlinear response those are not equivalent,
and this measures the difference directly.

  A. 3-D: evaluate the full response pointwise on the DFT total potential and
     the DFT ionic cavity, then plane-average. Must reproduce RHOION -- that
     is the gate, and if it fails nothing else here may be interpreted.
  B. 1-D: plane-average the SAME fields first, then evaluate the same formula,
     and compare the resulting profile.

THE REFERENCE ZERO, which must not be handled casually. n_work vanishes at
phi = 0, so the electrolyte reference is where the ionic charge vanishes --
but on a CHARGED slab the bulk ionic charge is NOT zero (93.7% of |RHOION|
sits at s_ion > 0.9, carrying the +1.000 e that compensates the solute), so
"phi = 0 in the bulk" is false here. The solver fixes the reference by charge
neutrality instead, so that is the condition used: solve one scalar offset c
such that the reconstructed TOTAL ionic charge equals the reference total.
One integral is matched; the pointwise distribution is left free and is
therefore a genuine test. The mean is NEVER subtracted, and c is reported
along with a sensitivity scan so no conclusion rests on it silently.

THE SIGN CONVENTION is measured rather than assumed: both phi = -PHI_raw and
phi = +PHI_raw are tried, each with its own charge-neutrality offset, and the
one that reproduces RHOION is identified. The stored files are in the
electron-energy convention (= minus the physical potential), so -PHI_raw is
expected, but expected is not verified.

Needs no model and no re-solve: params come from the PB config directly.
Native DFT grid throughout, so no interpolation enters.
"""
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get(
    "KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))

from pure_python import torch_pb as tp
from pure_python.pb import derived_params

DFT = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single")
FRAME = (f"{DFT}/1-44_GCE/cal_1", "NiN44 q=-1.00")
CFG = os.environ.get("KIT_PB_CONFIG", "cal1_train.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


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


dftdir, tag = FRAME
with open(CFG) as fh:
    params = derived_params(json.load(fh).get("solvation", {}))
print(f"=================== {tag} ===================")
print(f"  params: LION {params['LION']}, LNLION {params['LNLION']}, "
      f"theta_b {params['theta_b']:.6e}, ZBETA {params['ZBETA']:.6e}, "
      f"n_max {params['n_max']:.6e}, invBETA {params['invBETA']:.6e}",
      flush=True)

lat, chg = read_grid(f"{dftdir}/CHGCAR")
_, phif = read_grid(f"{dftdir}/PHI")
_, ri = read_grid(f"{dftdir}/RHOION")
V = float(abs(np.linalg.det(lat)))
shape = tuple(chg.shape)
grid = tp.TorchGrid(lat, shape, device=str(device), dtype=torch.float64,
                    rspec=True)
ne = torch.clamp(chg.to(device) / V, min=0.0)
s_ion, s_diel, _ = tp.create_cavity_torch(ne, grid, params)
s_ion = torch.clamp(s_ion, 0.0, 1.0)
rho_ref = -(ri.to(device)) / V                 # physical density, e/A^3
del chg, ri
lz = float(np.linalg.norm(lat[2])); nz = shape[2]
dz = lz / nz; area = V / lz
Q_ref = float(rho_ref.sum() * V / float(np.prod(shape)))
print(f"  grid {shape}, V {V:.2f} A^3; reference total ionic charge "
      f"{Q_ref:+.6f} e; int|rho_ion| "
      f"{float(rho_ref.abs().sum() * V / float(np.prod(shape))):.6f} e",
      flush=True)


def rho3(phi):
    """The full 3-D response, pointwise, in physical density units."""
    return tp.ion_density_values_from_phi_torch(phi, s_ion, grid, params) / V


def rho1(phi_z, s_z):
    """The same formula on plane-averaged inputs (the 1-D path)."""
    x = float(params["ZBETA"]) * phi_z
    th = float(params["theta_b"])
    if bool(params["LNLION"]) and th > 0.0:
        xc = torch.clamp(x, -100.0, 100.0)
        den = 1.0 + th * (torch.cosh(xc) - 1.0)
        nw = torch.where(torch.abs(x) > 100.0, torch.sign(x),
                         torch.where(torch.abs(x) < math.sqrt(th) * 2.0e-4,
                                     th * x, th * torch.sinh(xc) / den))
    elif bool(params["LNLION"]):
        nw = torch.sinh(torch.clamp(x, -100.0, 100.0))
    else:
        nw = x
    return -(float(params["n_max"]) * float(params["invBETA"])
             * float(params["ZBETA"])) * s_z * nw


def total(rho):
    return float(rho.sum() * V / float(np.prod(shape)))


def fit_offset(phi0, lo=-20.0, hi=20.0):
    """One scalar c from charge neutrality: total(rho3(phi0 + c)) = Q_ref.
    The mean is never subtracted; this is the condition the solver uses."""
    f = lambda c: total(rho3(phi0 + c)) - Q_ref
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return None, flo, fhi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi), flo, fhi


print(f"\n[SIGN] both conventions tried, each with its own neutrality offset")
best = None
for sgn, name in ((-1.0, "phi = -PHI_raw (electron-energy convention)"),
                  (+1.0, "phi = +PHI_raw")):
    phi0 = sgn * phif.to(device)
    c, flo, fhi = fit_offset(phi0)
    if c is None:
        print(f"  {name}: no neutrality offset in [-20,20] eV "
              f"(residual {flo:+.3e} .. {fhi:+.3e} e) -- rejected", flush=True)
        continue
    r3 = rho3(phi0 + c)
    err = float((r3 - rho_ref).abs().sum() * V / float(np.prod(shape)))
    rel = err / max(float(rho_ref.abs().sum() * V / float(np.prod(shape))),
                    1e-30)
    print(f"  {name}: offset c {c:+.6f} eV, total {total(r3):+.6f} e "
          f"(target {Q_ref:+.6f}); pointwise int|d rho| {err:.6f} e = "
          f"{100*rel:.2f}% of int|rho_ref|", flush=True)
    if best is None or rel < best[0]:
        best = (rel, sgn, c, name)

if best is None:
    print("\n  NO sign convention admits a neutrality offset -- stopping.")
    raise SystemExit(0)
rel, sgn, c, name = best
phi = sgn * phif.to(device) + c
print(f"\n  chosen: {name}, c {c:+.6f} eV, reproduction error "
      f"{100*rel:.2f}% of int|rho_ref|")
GATE = rel < 0.05
print(f"  [GATE] 3-D reconstruction reproduces RHOION to better than 5%: "
      f"{'PASS' if GATE else 'FAIL'}", flush=True)

print(f"\n[SENSITIVITY] how much the reproduction depends on c")
for dc in (-0.2, -0.05, 0.0, 0.05, 0.2):
    r3 = rho3(phi + dc)
    e = float((r3 - rho_ref).abs().sum() * V / float(np.prod(shape)))
    print(f"  c {c+dc:+.4f} eV: total {total(r3):+.6f} e, int|d rho| "
          f"{e:.6f} e", flush=True)

# ---------------- A against B ----------------
r3 = rho3(phi)
p3 = r3.mean(dim=(0, 1))
pref = rho_ref.mean(dim=(0, 1))
phi_z = phi.mean(dim=(0, 1)); s_z = s_ion.mean(dim=(0, 1))
p1_same = rho1(phi_z, s_z)
# and with its own neutrality offset, so only the SHAPE differs
def tot1(p):
    return float(p.sum() * dz * area)
lo, hi = -20.0, 20.0
g = lambda cc: tot1(rho1(phi_z + cc, s_z)) - Q_ref
if g(lo) * g(hi) <= 0:
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if g(lo) * g(mid) <= 0:
            hi = mid
        else:
            lo = mid
    c1 = 0.5 * (lo + hi)
    p1_own = rho1(phi_z + c1, s_z)
else:
    c1 = float("nan"); p1_own = None

print(f"\n[A vs B] plane profiles, e/A^3, and integrals over the cell")
print(f"  {'case':>34} {'total (e)':>11} {'int|.| (e)':>11} "
       f"{'L1 vs ref':>11} {'max dev':>10}")
def row(lbl, p):
    l1 = float((p - pref).abs().sum() * dz * area)
    print(f"  {lbl:>34} {tot1(p):+11.6f} {float(p.abs().sum()*dz*area):11.6f} "
          f"{l1:11.6f} {float((p-pref).abs().max()):10.3e}", flush=True)
    return l1
row("DFT reference RHOION", pref)
l3 = row("A: 3-D pointwise, then averaged", p3)
l1s = row("B: averaged first, same offset", p1_same)
if p1_own is not None:
    l1o = row("B: averaged first, own offset", p1_own)
    print(f"  (the 1-D path needs its own offset c1 {c1:+.6f} eV against "
          f"{c:+.6f} to carry the same total charge; the difference is itself "
          f"a consequence of the order of operations)")
print(f"\n  order-of-operations penalty: L1 of the 1-D path is "
      f"{l1s/max(l3,1e-30):.2f}x the 3-D path's ({l1s:.6f} against "
      f"{l3:.6f} e)")
if p1_own is not None:
    print(f"  with its own offset: {l1o/max(l3,1e-30):.2f}x ({l1o:.6f} e)")
print(f"\n  READING, per the user's criteria: if A reproduces and B clearly "
      f"fails, the averaging approximation is the priority; if both\n"
      f"  reproduce, look next at the model's potential and its boundary "
      f"handling. Sign, units and the electrolyte reference zero are\n"
      f"  held by the neutrality condition above -- the mean was never "
      f"subtracted.", flush=True)
print("DONE")
