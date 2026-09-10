"""1-D cavity comparison with the inputs actually held fixed (user-specified
minimal fix, 2026-09-09).

The previous attempt (job 3426811) is judged INCOMPLETE, not evidence. It
called the full model twice, and fixing model parameters and atomic positions
does not fix the model's PREDICTED charge and potential; the solute potential
differed by 4.142e-01 eV (charged) and 4.265e-01 (neutral) between the two
solves. The model also carries a cross-call solvent cache, so the difference
may include call-order effects and cannot even be called physical feedback
yet. Both solves converging says only that each solved acceptably; it does
not repair an uncontrolled input.

Also corrected: the previous script reported the CHARGE-CENTROID difference
and compared it against "about 0.42 A", which came from the rigid-shift scan.
Those are different metrics. Displacement here uses the shift-scan metric.

THE FIX, as specified. Run the full model ONCE. Capture the last PB call's
inputs. Then call the low-level solver DIRECTLY, twice, differing only in the
cavity-derived arguments, from the same initial state.

Why that is the whole difference and nothing leaks: solver.solve takes
cvhar_z, s_ion, a1, p_off, q_sol, val_ion_dipole_z, c_unit, center_z, indmin
and the iteration parameters. Of these only s_ion, a1 and p_off derive from
the cavity, via clo["S_ion_z"], clo["A_scr"] and clo["prior"] upsampled by
fourier_upsample and p_off = prior_s + delta_p. Everything else is passed
through as the SAME TENSOR OBJECT to both groups, so the solute potential,
the field derived from it, the total charge and the reference zero are
identical by construction rather than by argument.

One trap this avoids: a1 = plane_mean(a3_scr) and a3_scr depends on BOTH the
cavity AND phi_sol, so the substituted a1 may NOT be taken from a second full
model call -- it would carry that call's shifted potential. It is recomputed
here by calling closure_from_fields with the DFT density and RUN 1's phi_sol.

GATES, all of which must pass before any number below is interpreted:
  G1 the baseline group reproduces the model's own profiles;
  G2 cvhar_z, q_sol and val_ion_dipole_z are identical between groups;
  G3 the field derived from cvhar_z is identical, at the same reference zero;
  G4 both solves exit on their criteria.
If any gate fails the script says so and does not offer a reading.

Then only the three intended quantities: the ionic layer displacement by the
shift-scan metric, the bound charge profile error, and the ABSOLUTE
cross-energy gap in eV.
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
from mace.modules.pb1d_backend import fourier_upsample
from mace.modules.pb1d_solver import Solver1D

DFT = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single")
FRAMES = [(1, f"{DFT}/1-44_GCE/cal_1", "NiN44 q=-1.00"),
          (601, f"{DFT}/5-44_neutral_withsolv/cal_1", "neutral")]
CKDIR = os.environ.get("KIT_CKPT_DIR", "checkpoints")
DATADIR = os.environ.get("KIT_DATA_DIR", "data")
SHIFTS = np.arange(-1.5, 1.5001, 0.025)

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
    out = _clo(n_e, cv, grid, params, tp_)
    cap["n_e"] = n_e.detach().clone()
    cap["cv"] = cv.detach().clone()
    cap["grid"] = grid; cap["params"] = params; cap["tp"] = tp_
    cap["clo"] = {k: (v.detach().clone() if torch.is_tensor(v) else v)
                  for k, v in out.items()}
    return out
PB.closure_from_fields = wrap_clo

_sv = Solver1D.solve
def wrap_solve(self, **kw):
    cap["solver"] = self
    cap["kw"] = dict(kw)
    return _sv(self, **kw)
Solver1D.solve = wrap_solve

_sg = PB.PB1DBackend.solve_graph
def wrap_sg(self, *a, **k):
    out = _sg(self, *a, **k)
    cap["rho_bound_z"] = out["rho_bound_z"].detach().clone()
    cap["rho_ion_z"] = out["rho_ion_z"].detach().clone()
    cap["delta_p"] = out["delta_p"].detach().clone()
    cap["exit0"] = out.get("solver_exit")
    cap["backend"] = self
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


def res1d(p, n):
    if p.shape[0] == n:
        return p
    return torch.nn.functional.interpolate(
        p[None, None].to(torch.float64), size=(n,), mode="linear",
        align_corners=False)[0, 0]


def fshift(prof, d, lz):
    n = prof.shape[0]
    k = torch.fft.rfftfreq(n, d=lz / n, device=prof.device) * (2 * math.pi)
    return torch.fft.irfft(torch.fft.rfft(prof) * torch.exp(-1j * k * d), n=n)


def shift_scan(dref, prof, phi, dz, area, lz, label):
    """The shift-scan metric, the same one that produced the 0.42 A figure --
    NOT the charge-centroid difference the previous script reported."""
    e_ref = float((dref * phi).sum() * dz * area)
    e_0 = float((prof * phi).sum() * dz * area)
    q_ref = float(dref.abs().sum() * dz * area)
    if abs(e_ref) < 1.0e-2 or q_ref < 1.0e-2:
        print(f"    {label}: SKIPPED, reference coupling {e_ref:+.6f} eV on "
              f"{q_ref:.4f} e -- ratios and shifts on this are meaningless",
              flush=True)
        return None
    rows = []
    for d in SHIFTS:
        ms = fshift(prof, float(d), lz)
        e_s = float((ms * phi).sum() * dz * area)
        num = float((dref * ms).sum()); den = float((ms * ms).sum())
        sc = num / max(den, 1e-300)
        resid = float((dref - sc * ms).norm() / max(float(dref.norm()), 1e-30))
        rows.append((float(d), e_s, sc, resid))
    b_e = min(rows, key=lambda r: abs(r[1] / e_ref - 1.0))
    b_r = min(rows, key=lambda r: r[3])
    r_un = [r for r in rows if abs(r[0]) < 1e-9][0]
    print(f"    {label}: unshifted coupling {e_0:+.4f} ({e_0/e_ref:.3f} x "
          f"ref {e_ref:+.4f}), ABSOLUTE gap {e_ref - e_0:+.4f} eV")
    print(f"      best-by-coupling shift {b_e[0]:+.3f} A -> {b_e[1]:+.4f} "
          f"({b_e[1]/e_ref:.3f} x); best-by-residual shift {b_r[0]:+.3f} A, "
          f"scale {b_r[2]:.4f}, residual {100*b_r[3]:.1f}% "
          f"(unshifted {100*r_un[3]:.1f}%)", flush=True)
    return dict(e_ref=e_ref, e_0=e_0, gap=e_ref - e_0, shift_e=b_e[0],
                shift_r=b_r[0], resid=b_r[3], resid_un=r_un[3])


from pure_python import torch_pb as tp

for sid, dftdir, tag in FRAMES:
    a = want[sid]
    print(f"\n=================== {tag} ===================", flush=True)
    cap.clear()
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(
        ds, batch_size=1))).to(device)
    with torch.no_grad():
        model(b.to_dict(), compute_force=False, training=False)   # ONCE

    solver = cap["solver"]; kw = cap["kw"]; grid = cap["grid"]
    be = cap["backend"]; f = int(be.solve_upsample)
    shm = tuple(grid.shape)
    cellm = torch.as_tensor(grid.cell, dtype=torch.float64, device=device)
    lz = float(torch.linalg.norm(cellm[2]))
    volume = grid.volume
    nz_s = kw["cvhar_z"].shape[0]
    dz_s = lz / nz_s; area = volume / lz
    print(f"  one full model call; solve grid nz_s {nz_s} (upsample f={f}), "
          f"captured kwargs: {sorted(kw.keys())}", flush=True)

    # ---- group A: the captured inputs verbatim -------------------------
    with torch.no_grad():
        outA = _sv(solver, **kw)
    rbA = -(outA["n_b"] / volume); riA = -(outA["n_ion"] / volume)
    exA = outA.get("solver_exit")

    # ---- group B: only the cavity-derived arguments replaced -----------
    latc, chg = read_grid(f"{dftdir}/CHGCAR")
    Vc = float(abs(np.linalg.det(latc)))
    ne_dft = to_shape(torch.clamp(chg.to(device) / Vc, min=0.0), shm).to(device)
    del chg
    with torch.no_grad():
        # RUN 1's phi_sol, so the substituted a1 cannot carry a shifted
        # potential. This is the trap the previous attempt fell into.
        cloB = _clo(ne_dft, cap["cv"], grid, cap["params"], tp)
    kwB = dict(kw)
    kwB["s_ion"] = torch.clamp(fourier_upsample(cloB["S_ion_z"], f), min=0.0)
    kwB["a1"] = torch.clamp(fourier_upsample(cloB["A_scr"], f), min=0.0)
    kwB["p_off"] = fourier_upsample(cloB["prior"], f) + cap["delta_p"]
    with torch.no_grad():
        outB = _sv(solver, **kwB)
    rbB = -(outB["n_b"] / volume); riB = -(outB["n_ion"] / volume)
    exB = outB.get("solver_exit")

    # ---- GATES ---------------------------------------------------------
    gates = []
    d1b = float((res1d(rbA, cap["rho_bound_z"].shape[0])
                 - cap["rho_bound_z"]).abs().max())
    d1i = float((res1d(riA, cap["rho_ion_z"].shape[0])
                 - cap["rho_ion_z"]).abs().max())
    gates.append(("G1 baseline group reproduces the model's own profiles",
                  max(d1b, d1i) < 1.0e-12,
                  f"max abs bound {d1b:.3e}, ionic {d1i:.3e} e/A^3"))
    same = all(kw[k] is kwB[k] for k in ("cvhar_z", "q_sol",
                                         "val_ion_dipole_z")
               if not isinstance(kw[k], float))
    dcv = float((kw["cvhar_z"] - kwB["cvhar_z"]).abs().max())
    gates.append(("G2 solute potential, total charge and valence dipole "
                  "identical between groups",
                  dcv == 0.0 and kw["q_sol"] == kwB["q_sol"],
                  f"same tensor objects {same}; max |d cvhar_z| {dcv:.3e} eV; "
                  f"q_sol {kw['q_sol']:+.6f} vs {kwB['q_sol']:+.6f}"))
    ez_a = torch.diff(kw["cvhar_z"], append=kw["cvhar_z"][:1]) / dz_s
    ez_b = torch.diff(kwB["cvhar_z"], append=kwB["cvhar_z"][:1]) / dz_s
    dez = float((ez_a - ez_b).abs().max())
    gates.append(("G3 field from that potential identical, same reference "
                  "zero",
                  dez == 0.0,
                  f"max |d dphi/dz| {dez:.3e} eV/A; mean cvhar_z "
                  f"{float(kw['cvhar_z'].mean()):+.6f} both"))
    okA = (exA or {}).get("fix_exit") == "tol" and (exA or {}).get("newton_exit") == "tol"
    okB = (exB or {}).get("fix_exit") == "tol" and (exB or {}).get("newton_exit") == "tol"
    gates.append(("G4 both solves exit on their criteria", okA and okB,
                  f"A fixed-point {(exA or {}).get('fix_exit')} / Newton "
                  f"{(exA or {}).get('newton_exit')}; B "
                  f"{(exB or {}).get('fix_exit')} / "
                  f"{(exB or {}).get('newton_exit')}"))
    print(f"\n  GATES")
    allok = True
    for name, ok, detail in gates:
        allok = allok and bool(ok)
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}",
              flush=True)
    print(f"  cavity input differs as intended: max |ne_dft - ne_model| "
          f"{float((ne_dft - cap['n_e'].to(device)).abs().max()):.3e} e/A^3",
          flush=True)
    if not allok:
        print(f"\n  A GATE FAILED -- the control is not established and no "
              f"reading of the numbers below is offered. Fix the gate first.",
              flush=True)

    # ---- the three intended quantities ---------------------------------
    _, rb = read_grid(f"{dftdir}/RHOB")
    _, ri = read_grid(f"{dftdir}/RHOION")
    Vd = float(abs(np.linalg.det(latc)))
    rb_ref = res1d((-(rb.to(device)) / Vd).mean(dim=(0, 1)), nz_s)
    ri_ref = res1d((-(ri.to(device)) / Vd).mean(dim=(0, 1)), nz_s)
    del rb, ri
    phi_s = -(kw["cvhar_z"] - kw["cvhar_z"].mean())
    if allok:
        for nm, dref, pa, pb in (("BOUND", rb_ref, rbA, rbB),
                                 ("IONIC", ri_ref, riA, riB)):
            print(f"\n  [{nm}] shift-scan metric, one fixed potential")
            sa = shift_scan(dref, pa, phi_s, dz_s, area, lz, "model cavity")
            sb = shift_scan(dref, pb, phi_s, dz_s, area, lz, "DFT cavity  ")
            if sa and sb:
                print(f"    -> displacement {sa['shift_r']:+.3f} A -> "
                      f"{sb['shift_r']:+.3f} A (by residual), "
                      f"{sa['shift_e']:+.3f} -> {sb['shift_e']:+.3f} "
                      f"(by coupling)")
                print(f"    -> profile error {100*sa['resid']:.1f}% -> "
                      f"{100*sb['resid']:.1f}% of |DFT|")
                print(f"    -> ABSOLUTE coupling gap {sa['gap']:+.4f} eV -> "
                      f"{sb['gap']:+.4f} eV", flush=True)
    del ne_dft, rb_ref, ri_ref, phi_s, rbA, riA, rbB, riB
    import gc; gc.collect(); torch.cuda.empty_cache()
print("DONE")
