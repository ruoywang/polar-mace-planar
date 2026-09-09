"""Is the 1-D plane-average deficit a rigid displacement? (2026-09-09)

The workstation's z-resolved table localises the charged frame's 0.525 eV
plane-average deficit precisely, and the structure it shows is a DISPLACED
INTERFACE DIPOLE LAYER rather than a diffuse error:

  z (A)        <phi>   DFT rho    mdl rho   mdl/DFT   gap (eV)
  15.0-16.5   -1.430   1.17e-03   1.12e-03    0.96     +0.166
  16.5-18.0   -3.901   1.96e-03   1.76e-03    0.90     +0.225
  18.0-19.5   -3.540   1.60e-04   5.03e-05    0.31     +0.110
  19.5-21.0   -2.943  -6.54e-04  -9.29e-04    1.42     +0.234

Those four bins carry +0.734 eV, and everything from 21 A outward gives back
-0.141 eV, landing on +0.525. The positive peak is 10% low, the crossing bin
is 69% low and the negative lobe is 42% high -- i.e. the sign change of the
profile happens at too small a z. With the charge-weighted mean z shifted by
-0.423 A and the potential falling steeply across that band (-1.43 to -3.90
V), a rigidly displaced dipole layer would produce exactly this pattern.

So: scan a rigid z-shift of the model profile and see how much of the deficit
it removes. Shift is applied by Fourier phase, so it is exact and sub-grid.
For each shift the optimal single scale is available in closed form, so three
nested questions get answered at once:

  shift alone            -> is the deficit pure displacement?
  shift + optimal scale  -> displacement plus a uniform amplitude error?
  residual after both    -> what is left is genuine shape error.

If shift alone recovers most of the 0.525 eV then the defect is a 1-D
interface-position error, the cavity's switching position is the natural
suspect (its transition-shell gradient is already measured 24% weak), and no
3-D basis work addresses it. If it recovers little, the profile shape is
wrong and the position is a red herring. Either way this is a measurement,
not an attribution: nothing here shows what MOVED the layer.

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
_solve = PB.PB1DBackend.solve_graph
def wrap_solve(self, *a, **k):
    out = _solve(self, *a, **k)
    cap["rho_bound_z"] = out["rho_bound_z"].detach().clone()
    cap["rho_ion_z"] = out["rho_ion_z"].detach().clone()
    return out
PB.PB1DBackend.solve_graph = wrap_solve
_clo = PB.closure_from_fields
def wrap_clo(n_e, cv, grid, params, tp_):
    cap["cvhar3"] = cv.detach().clone(); cap["grid"] = grid
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


def res1d(prof, n):
    if prof.shape[0] == n:
        return prof
    return torch.nn.functional.interpolate(
        prof[None, None].to(torch.float64), size=(n,), mode="linear",
        align_corners=False)[0, 0]


def fshift(prof, d, lz):
    """Rigid periodic shift by d angstrom via Fourier phase (sub-grid exact)."""
    n = prof.shape[0]
    k = torch.fft.rfftfreq(n, d=lz / n, device=prof.device) * (2 * math.pi)
    return torch.fft.irfft(torch.fft.rfft(prof)
                           * torch.exp(-1j * k * d), n=n)


for sid, dftdir, tag in FRAMES:
    a = want[sid]
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(
        ds, batch_size=1))).to(device)
    cap.clear()
    with torch.no_grad():
        model(b.to_dict(), compute_force=False, training=False)
    grid = cap["grid"]
    shm = tuple(grid.shape)
    cellm = torch.as_tensor(grid.cell, dtype=torch.float64, device=device)
    lz = float(torch.linalg.norm(cellm[2])); nz = shm[2]; dz = lz / nz
    area = grid.volume / lz
    phi_z = (-(cap["cvhar3"] - cap["cvhar3"].mean())).mean(dim=(0, 1))

    latd, rb = read_grid(f"{dftdir}/RHOB")
    _, ri = read_grid(f"{dftdir}/RHOION")
    Vd = float(abs(np.linalg.det(latd)))
    rb_z = res1d((-(rb.to(device)) / Vd).mean(dim=(0, 1)), nz)
    ri_z = res1d((-(ri.to(device)) / Vd).mean(dim=(0, 1)), nz)
    mb_z = res1d(cap["rho_bound_z"].to(device).to(torch.float64), nz)
    mi_z = res1d(cap["rho_ion_z"].to(device).to(torch.float64), nz)
    del rb, ri
    print(f"\n=================== {tag} ===================", flush=True)

    for nm, dref, mprof in (("BOUND", rb_z, mb_z), ("IONIC", ri_z, mi_z)):
        e_ref = float((dref * phi_z).sum() * dz * area)
        e_0 = float((mprof * phi_z).sum() * dz * area)
        q_ref = float(dref.abs().sum() * dz * area)
        if abs(e_ref) < 1.0e-2 or q_ref < 1.0e-2:
            print(f"\n[{nm}] SKIPPED: reference coupling {e_ref:+.6f} eV on "
                  f"{q_ref:.4f} e of reference charge. Ratios on this are "
                  f"meaningless -- the neutral ionic channel printed a full "
                  f"scan and a 'ratio -0.155' against a +0.0006 eV reference "
                  f"before this guard was tightened (the old test was "
                  f"|coupling| < 1e-6, which such a channel passes).",
                  flush=True)
            continue
        print(f"\n[{nm}] reference coupling {e_ref:+.4f} eV, model as-is "
              f"{e_0:+.4f} eV (ratio {e_0/e_ref:.3f}), deficit "
              f"{e_ref-e_0:+.4f} eV")
        rows = []
        for d in SHIFTS:
            ms = fshift(mprof, float(d), lz)
            e_s = float((ms * phi_z).sum() * dz * area)
            num = float((dref * ms).sum()); den = float((ms * ms).sum())
            sc = num / max(den, 1e-300)
            e_ss = sc * e_s
            resid = float((dref - sc * ms).norm()
                          / max(float(dref.norm()), 1e-300))
            rows.append((float(d), e_s, e_s / e_ref, sc, e_ss / e_ref, resid))
        best_shift = max(rows, key=lambda r: -abs(r[2] - 1.0))
        best_resid = min(rows, key=lambda r: r[5])
        print(f"  {'shift (A)':>10} {'coupling':>10} {'/ref':>7} "
              f"{'opt scale':>10} {'scaled/ref':>11} {'resid':>7}")
        for r in rows[::6]:
            print(f"  {r[0]:10.3f} {r[1]:+10.4f} {r[2]:7.3f} {r[3]:10.4f} "
                  f"{r[4]:11.3f} {r[5]:7.3f}", flush=True)
        # the deficit is SIGNED: max(deficit, 1e-30) turns a negative one
        # into 1e-30 and the percentage overflows to ~1e31 (found on the
        # workstation 2026-09-09, printed -5.2e31% where the answer is 98.9%)
        _den = e_ref - e_0
        _rec = ((best_shift[1] - e_0) / _den * 100.0
                if abs(_den) > 1.0e-9 else float("nan"))
        _rec_s = (f"{_rec:.1f}%" if _rec == _rec
                  else "n/a (deficit below 1e-9 eV)")
        print(f"  BEST by coupling ratio: shift {best_shift[0]:+.3f} A gives "
              f"{best_shift[1]:+.4f} eV = {best_shift[2]:.3f} x reference "
              f"(from {e_0/e_ref:.3f} unshifted), recovering {_rec_s} of the "
              f"{_den:+.4f} eV deficit with shift ALONE", flush=True)
        print(f"  BEST by profile residual: shift {best_resid[0]:+.3f} A, "
              f"scale {best_resid[3]:.4f}, residual {100*best_resid[5]:.1f}% "
              f"of |DFT| (unshifted residual "
              f"{100*rows[len(rows)//2][5]:.1f}% at shift "
              f"{rows[len(rows)//2][0]:+.3f})", flush=True)
    del phi_z, rb_z, ri_z, mb_z, mi_z
    import gc; gc.collect(); torch.cuda.empty_cache()
print("DONE")
