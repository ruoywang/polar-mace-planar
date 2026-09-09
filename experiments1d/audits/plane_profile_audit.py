"""Where in z does the 1-D plane-averaged coupling lose its energy?
(follow-up to envelope_alignment, 2026-09-09)

Why this exists. The envelope_alignment run on the workstation established
something that changes the priority order, and it is easy to misread. Split
the bound charge into its xy-plane average and its lateral remainder:

  charged frame   plane average: magnitude ratio 1.004, ENERGY ratio 0.496
                  lateral      : magnitude ratio 0.629, ENERGY ratio 0.417
  neutral frame   plane average: magnitude 0.972, energy 1.020
                  lateral      : magnitude 0.770, energy 1.648

Read on magnitude alone the shortfall looks purely lateral. It is not. On the
charged frame the 1-D plane-averaged part carries the right amount of charge
to 0.4% and still delivers only 49.6% of its own attraction, so 0.525 eV of
the 1.832 eV missing bound attraction -- 28.7% -- sits in the 1-D channel,
not in the 3-D residual. That matters because of a standing constraint: the
3-D fit is answerable for the lateral shortfall ONLY and must never be
charged with the 1-D plane-average error. Fitting harder in 3-D cannot
recover that 0.525 eV, and a 3-D fit allowed to try would be rewarded for
compensating a 1-D defect.

The plane-average coupling is a purely 1-D integral,

  E_pm = A * sum_z <rho>_xy(z) * <phi>_xy(z) * dz

so if the magnitude is right and the energy is half, the z-PROFILE is
misplaced relative to the potential. This script prints the three profiles
and the per-z and cumulative contributions so the loss can be located in z
rather than inferred. It also does the ionic channel, where the charged frame
over-attracts by 7.5%.

Also measured, because the same asymmetry needs a mechanism: the model's
plane-average couples at half strength on the charged frame and exactly on
the neutral one, and the only structural difference between the two is that
the ion channel is multiplied by q_tot and so is identically zero when
neutral. The z-resolved comparison of the two frames is the cheapest way to
see whether the charged frame's bound profile is displaced or merely scaled.

Costs one forward pass per frame plus reading two DFT files. Paths come from
env vars with the LS6 values as defaults.
"""
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
NBIN = 30            # z bins for the printed table
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


def resample(prof, n):
    if prof.shape[0] == n:
        return prof
    return torch.nn.functional.interpolate(
        prof[None, None].to(torch.float64), size=(n,), mode="linear",
        align_corners=False)[0, 0]


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
    lz = float(torch.linalg.norm(cellm[2]))
    area = grid.volume / lz
    phi_m = -(cap["cvhar3"] - cap["cvhar3"].mean())
    phi_z = phi_m.mean(dim=(0, 1))              # <phi>_xy(z), model potential
    nz = shm[2]; dz = lz / nz

    latd, rb = read_grid(f"{dftdir}/RHOB")
    _, ri = read_grid(f"{dftdir}/RHOION")
    Vd = float(abs(np.linalg.det(latd)))
    rb_z = (-(rb.to(device)) / Vd).mean(dim=(0, 1))   # <RHOB>_xy on DFT nz
    ri_z = (-(ri.to(device)) / Vd).mean(dim=(0, 1))
    rb_z = resample(rb_z, nz); ri_z = resample(ri_z, nz)
    mb_z = resample(cap["rho_bound_z"].to(device).to(torch.float64), nz)
    mi_z = resample(cap["rho_ion_z"].to(device).to(torch.float64), nz)
    del rb, ri

    ex = cap.get("solver_exit") or {}
    print(f"\n=================== {tag} ===================", flush=True)
    if ex:
        print(f"  solver: fixed-point {ex.get('fix_exit')} after "
              f"{ex.get('fix_steps')} steps (min {ex.get('fix_steps_min')}, "
              f"cap {ex.get('fix_steps_cap')}), res "
              f"{ex.get('fix_res', float('nan')):.3e} vs "
              f"{ex.get('fix_tol', float('nan')):.3e} | Newton "
              f"{ex.get('newton_exit')}, {ex.get('newton_total')} outer "
              f"(cap {ex.get('newton_cap')}/call), rms "
              f"{ex.get('newton_rms_last', float('nan')):.3e} vs tol "
              f"{ex.get('newton_tol', float('nan')):.3e}", flush=True)
        if ex.get("fix_exit") != "tol" or ex.get("newton_exit") != "tol":
            print(f"  WARNING: a loop exited on its cap, not its criterion",
                  flush=True)
    else:
        print(f"  solver exit not exported -- code predates the provenance "
              f"commit; pull and rerun if this matters", flush=True)

    for nm, dz_prof, mz_prof in (("BOUND", rb_z, mb_z),
                                 ("IONIC", ri_z, mi_z)):
        ed = float((dz_prof * phi_z).sum() * dz * area)
        em = float((mz_prof * phi_z).sum() * dz * area)
        qd = float(dz_prof.abs().sum() * dz * area)
        qm = float(mz_prof.abs().sum() * dz * area)
        print(f"\n[{nm}] plane-averaged channel, model potential")
        print(f"  int|rho|  DFT {qd:.4f}  model {qm:.4f}  ratio "
              f"{qm/max(qd,1e-30):.3f}")
        print(f"  coupling  DFT {ed:+.4f}  model {em:+.4f}  ratio "
              f"{em/ed if abs(ed)>1e-30 else float('nan'):.3f} eV")
        print(f"  net       DFT {float(dz_prof.sum()*dz*area):+.4f}  model "
              f"{float(mz_prof.sum()*dz*area):+.4f} e")
        # per-bin and cumulative, so the loss is located rather than inferred
        edge = torch.linspace(0, nz, NBIN + 1).round().long()
        cum_d = cum_m = 0.0
        print(f"  {'z (A)':>13} {'<phi>':>8} {'DFT rho':>10} {'mdl rho':>10} "
              f"{'DFT dE':>9} {'mdl dE':>9} {'dE gap':>9} {'cum gap':>9}")
        for k in range(NBIN):
            i0, i1 = int(edge[k]), int(edge[k + 1])
            if i1 <= i0:
                continue
            sl = slice(i0, i1)
            e_d = float((dz_prof[sl] * phi_z[sl]).sum() * dz * area)
            e_m = float((mz_prof[sl] * phi_z[sl]).sum() * dz * area)
            cum_d += e_d; cum_m += e_m
            print(f"  {i0*dz:6.2f}-{i1*dz:<6.2f} {float(phi_z[sl].mean()):8.3f} "
                  f"{float(dz_prof[sl].mean()):10.2e} "
                  f"{float(mz_prof[sl].mean()):10.2e} "
                  f"{e_d:+9.4f} {e_m:+9.4f} {e_m-e_d:+9.4f} "
                  f"{cum_m-cum_d:+9.4f}", flush=True)
        # is the model profile displaced in z, or just scaled?
        wd = float((dz_prof.abs() * torch.arange(nz, device=device) * dz).sum()
                   / torch.clamp(dz_prof.abs().sum(), min=1e-30))
        wm = float((mz_prof.abs() * torch.arange(nz, device=device) * dz).sum()
                   / torch.clamp(mz_prof.abs().sum(), min=1e-30))
        num = float((dz_prof * mz_prof).sum())
        den = float(dz_prof.norm() * mz_prof.norm())
        print(f"  charge-weighted mean z: DFT {wd:.3f} A, model {wm:.3f} A, "
              f"shift {wm-wd:+.3f} A;  profile correlation "
              f"{num/max(den,1e-30):+.4f}", flush=True)
        # best single scale factor, to separate "scaled" from "misplaced"
        s_opt = num / max(float((mz_prof * mz_prof).sum()), 1e-30)
        resid = float((dz_prof - s_opt * mz_prof).norm() / max(
            float(dz_prof.norm()), 1e-30))
        print(f"  best single scale for the model profile {s_opt:+.4f}; "
              f"residual after scaling {100*resid:.1f}% of |DFT| "
              f"-- a small residual means merely scaled, a large one means "
              f"misplaced in z", flush=True)
    del phi_m, phi_z, rb_z, ri_z, mb_z, mi_z
    import gc; gc.collect(); torch.cuda.empty_cache()
print("DONE")
