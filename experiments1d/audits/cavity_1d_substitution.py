"""1-D cavity substitution with a re-solve (user-fixed next step 2026-09-09).

Step 2's 3-D result, restated with the aggregate the user computed and which I
had failed to form: cross and self must be added, because e_s3d = e_xsol +
e_self is what enters the energy. Charged frame, original coefficients:

  reference                     -2.2417 + 0.6829 = -1.559 eV
  model cavity                  -0.9350 + 0.6118 = -0.323 eV
  cavity rebuilt from DFT n_e   -2.3437 + 1.8794 = -0.464 eV

So the deviation goes 1.236 -> 1.094 eV, an 11.5% improvement. The cross
attraction strengthens a lot and the extra self-energy cancels most of the
gain. Reporting cross and self apart, as I did, made a 11% effect read like a
fix.

THIS EXPERIMENT asks whether the cavity ALSO misplaces the background charge,
which the 3-D experiment could not see because it never re-solved. The
intervention: substitute the density the CAVITY is built from and re-run the
1-D solve, with model parameters and the explicit solute input held fixed.

That the substitution touches only the cavity is verified, not assumed:
pb1d_closure.closure_from_fields uses its n_e_density argument in exactly one
place, tp.create_cavity_torch, and everything downstream consumes s_ion3,
s_diel3 and phi_sol -- and phi_sol, the solute side, is untouched. The
potential used for every coupling number below is additionally taken from the
UNSUBSTITUTED run and reused, so it is fixed by construction rather than by
argument, and the two runs' solute potentials are compared as a check.

What it is meant to answer, from the user's list:
  - does the ionic layer's displacement, measured at about 0.42 A in the
    shift scan, shrink;
  - does the 1-D bound charge distribution improve, and does the 0.525 eV
    plane-average coupling gap improve WITH it.

Two solves per frame, so the solver exit reason and iteration counts are
printed for both: a warm start from the first solve's phi could in principle
bias the second, and both exiting on their criterion is what rules that out.

Paths come from env vars with the LS6 values as defaults.
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

DFT = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single")
FRAMES = [(1, f"{DFT}/1-44_GCE/cal_1", "NiN44 q=-1.00"),
          (601, f"{DFT}/5-44_neutral_withsolv/cal_1", "neutral")]
CKDIR = os.environ.get("KIT_CKPT_DIR", "checkpoints")
DATADIR = os.environ.get("KIT_DATA_DIR", "data")

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
SUB = {"ne": None}          # when set, replaces the density the cavity uses
_solve = PB.PB1DBackend.solve_graph
def wrap_solve(self, *a, **k):
    out = _solve(self, *a, **k)
    cap["rho_bound_z"] = out["rho_bound_z"].detach().clone()
    cap["rho_ion_z"] = out["rho_ion_z"].detach().clone()
    cap["solver_exit"] = out.get("solver_exit")
    return out
PB.PB1DBackend.solve_graph = wrap_solve
_clo = PB.closure_from_fields
def wrap_clo(n_e, cv, grid, params, tp_):
    cap["cvhar3"] = cv.detach().clone(); cap["grid"] = grid
    cap["params"] = params
    if SUB["ne"] is not None:
        n_e = SUB["ne"]      # ONLY the cavity input; phi_sol (cv) untouched
    else:
        cap["n_e"] = n_e.detach().clone()
    return _clo(n_e, cv, grid, params, tp_)
PB.closure_from_fields = wrap_clo

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff",
               "solvated": "solvated"},
    arrays_keys={"forces": "forces"},
)
want = {sid: None for sid, _, _ in FRAMES}
for split in ("train", "val"):
    for a in read(os.path.join(DATADIR, f"{split}.xyz"), ":"):
        sid = int(a.info.get("sample_id", -1))
        if sid in want and want[sid] is None:
            want[sid] = a
    if all(v is not None for v in want.values()):
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


def to_shape(t, shape):
    if tuple(t.shape) == tuple(shape):
        return t
    return torch.nn.functional.interpolate(
        t[None, None].to(torch.float64), size=tuple(shape), mode="trilinear",
        align_corners=False)[0, 0].contiguous()


def res1d(prof, n):
    if prof.shape[0] == n:
        return prof
    return torch.nn.functional.interpolate(
        prof[None, None].to(torch.float64), size=(n,), mode="linear",
        align_corners=False)[0, 0]


def run(a):
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(
        ds, batch_size=1))).to(device)
    with torch.no_grad():
        model(b.to_dict(), compute_force=False, training=False)


def exitline(ex, tag):
    if not ex:
        return f"  {tag}: solver exit not exported"
    w = ("" if ex.get("fix_exit") == "tol" and ex.get("newton_exit") == "tol"
         else "   <-- A LOOP EXITED ON ITS CAP")
    return (f"  {tag}: fixed-point {ex.get('fix_exit')} after "
            f"{ex.get('fix_steps')} steps (cap {ex.get('fix_steps_cap')}), "
            f"Newton {ex.get('newton_exit')} with {ex.get('newton_total')} "
            f"outer (cap {ex.get('newton_cap')}/call){w}")


def stats(dref, prof, phi_z, dz, area, nzarr):
    e = float((prof * phi_z).sum() * dz * area)
    q = float(prof.abs().sum() * dz * area)
    net = float(prof.sum() * dz * area)
    wm = float((prof.abs() * nzarr).sum()
               / torch.clamp(prof.abs().sum(), min=1e-30))
    num = float((dref * prof).sum()); den = float(dref.norm() * prof.norm())
    corr = num / max(den, 1e-30)
    sc = num / max(float((prof * prof).sum()), 1e-300)
    resid = float((dref - sc * prof).norm() / max(float(dref.norm()), 1e-30))
    return e, q, net, wm, corr, sc, resid


for sid, dftdir, tag in FRAMES:
    a = want[sid]
    print(f"\n=================== {tag} ===================", flush=True)

    # ---- solve 1: the model's own cavity -------------------------------
    SUB["ne"] = None
    cap.clear()
    run(a)
    grid = cap["grid"]; shm = tuple(grid.shape)
    cellm = torch.as_tensor(grid.cell, dtype=torch.float64, device=device)
    lz = float(torch.linalg.norm(cellm[2])); nz = shm[2]; dz = lz / nz
    area = grid.volume / lz
    # the potential every coupling below uses, taken ONCE from this run
    phi_ref = -(cap["cvhar3"] - cap["cvhar3"].mean())
    phi_z = phi_ref.mean(dim=(0, 1))
    ne_model = cap["n_e"].to(device)
    mb0 = res1d(cap["rho_bound_z"].to(device).to(torch.float64), nz)
    mi0 = res1d(cap["rho_ion_z"].to(device).to(torch.float64), nz)
    ex0 = cap.get("solver_exit")
    cv0 = cap["cvhar3"].clone()

    # ---- the DFT density on the model grid, cavity input only ----------
    latc, chg = read_grid(f"{dftdir}/CHGCAR")
    Vc = float(abs(np.linalg.det(latc)))
    ne_dft = to_shape(torch.clamp(chg.to(device) / Vc, min=0.0), shm).to(device)
    del chg

    # ---- solve 2: cavity built from the DFT density ---------------------
    SUB["ne"] = ne_dft
    cap.clear()
    run(a)
    mb1 = res1d(cap["rho_bound_z"].to(device).to(torch.float64), nz)
    mi1 = res1d(cap["rho_ion_z"].to(device).to(torch.float64), nz)
    ex1 = cap.get("solver_exit")
    dcv = float((cap["cvhar3"] - cv0).abs().max())
    SUB["ne"] = None

    print(exitline(ex0, "solve 1, model cavity"), flush=True)
    print(exitline(ex1, "solve 2, DFT-built cavity"), flush=True)
    print(f"  [check] solute potential identical across the two solves: max "
          f"|d cvhar3| {dcv:.3e} eV (the substitution must not touch it)",
          flush=True)
    print(f"  [check] cavity input differs as intended: max |ne_dft - "
          f"ne_model| {float((ne_dft - ne_model).abs().max()):.3e} e/A^3",
          flush=True)

    # ---- DFT reference profiles ----------------------------------------
    _, rb = read_grid(f"{dftdir}/RHOB")
    _, ri = read_grid(f"{dftdir}/RHOION")
    Vd = float(abs(np.linalg.det(latc)))
    rb_z = res1d((-(rb.to(device)) / Vd).mean(dim=(0, 1)), nz)
    ri_z = res1d((-(ri.to(device)) / Vd).mean(dim=(0, 1)), nz)
    del rb, ri
    zarr = torch.arange(nz, device=device, dtype=torch.float64) * dz

    for nm, dref, p0, p1 in (("BOUND", rb_z, mb0, mb1),
                             ("IONIC", ri_z, mi0, mi1)):
        ed, qd, nd, wd, _, _, _ = stats(dref, dref, phi_z, dz, area, zarr)
        e0, q0, n0, w0, c0, s0, r0 = stats(dref, p0, phi_z, dz, area, zarr)
        e1, q1, n1, w1, c1, s1, r1 = stats(dref, p1, phi_z, dz, area, zarr)
        print(f"\n[{nm}] plane-averaged channel, one fixed potential")
        print(f"  {'case':>18} {'int|.|':>9} {'net':>9} {'coupling':>10} "
              f"{'/DFT':>7} {'mean z':>8} {'shift':>8} {'corr':>8} "
              f"{'scale':>8} {'resid':>7}")
        print(f"  {'DFT reference':>18} {qd:9.4f} {nd:+9.4f} {ed:+10.4f} "
              f"{1.0:7.3f} {wd:8.3f} {'-':>8} {1.0:8.4f} {1.0:8.4f} {0.0:7.3f}")
        for lbl, e, q, n, w, c, s, r in (("model cavity", e0, q0, n0, w0, c0, s0, r0),
                                         ("DFT-built cavity", e1, q1, n1, w1, c1, s1, r1)):
            print(f"  {lbl:>18} {q:9.4f} {n:+9.4f} {e:+10.4f} "
                  f"{e/ed if abs(ed) > 1e-30 else float('nan'):7.3f} "
                  f"{w:8.3f} {w-wd:+8.3f} {c:8.4f} {s:8.4f} {r:7.3f}",
                  flush=True)
        gap0 = ed - e0; gap1 = ed - e1
        den = gap0
        frac = ((gap0 - gap1) / den * 100.0) if abs(den) > 1.0e-12 else float("nan")
        fs = f"{frac:.1f}%" if frac == frac else "n/a (gap below 1e-12)"
        print(f"  coupling gap: model cavity {gap0:+.4f} eV -> DFT-built "
              f"{gap1:+.4f} eV, {fs} of it closed", flush=True)
        print(f"  displacement: model cavity {w0-wd:+.3f} A -> DFT-built "
              f"{w1-wd:+.3f} A; residual after best scale {100*r0:.1f}% -> "
              f"{100*r1:.1f}% of |DFT|", flush=True)
    print(f"\n  (shift is the charge-weighted mean z minus the DFT one. The "
          f"shift-scan measured about 0.42 A of displacement on the ionic\n"
          f"   channel; whether it shrinks here is the question this run was "
          f"built for. Coupling and displacement are reported together so\n"
          f"   neither is read alone, and the residual says whether the shape "
          f"improved rather than just the position.)", flush=True)
    del phi_ref, phi_z, ne_model, ne_dft, mb0, mi0, mb1, mi1, rb_z, ri_z, cv0
    import gc; gc.collect(); torch.cuda.empty_cache()
print("DONE")
