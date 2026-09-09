"""Is the envelope where the charge is? (follow-up to the cavity comparison
3426068, 2026-09-09)

The cavity comparison settled two things and raised one mechanism:

  1. The cavity is essentially right: the two cavities disagree on 0.44%
     (charged) / 0.53% (neutral) of the cell.
  2. The true solvent charge is NOT where "the cavity is open". On the native
     grid, 83% of the raw solvent charge sits in the 5% of volume where
     0.01 < s_diel < 0.90 -- the dielectric transition shell -- and by
     distance, 74.6% of it sits 1.5-2.5 A from the nearest solute atom, where
     the mean s_diel is only 0.18. This also corrects the earlier
     model-grid claim "90% of the lateral charge within 1.5 A": on the native
     grid the 0-1.5 A region holds 18.5% of the raw charge and contributes
     cross energy of the OPPOSITE sign (+0.811 eV against a total of -5.96).
  3. The residual IS present in the closed region (|delta| 0.839 e where both
     cavities are closed), so it is not excluded there. What fails is the
     distribution: same total charge, 37% less attraction.

That points at a specific mechanism worth measuring rather than assuming. The
bound channel's envelope is env_b = |grad s_diel| / max, which by construction
peaks where the cavity function is steepest. If that peak does not coincide
with the 1.5-2.5 A shell that carries the charge, then reaching the reference
cross energy requires large coefficients in a region where the envelope is
small, and large coefficients cost self-energy -- which is exactly the
observed obstruction on the charged frames (minimum attainable self-energy
2.28-7.79 x the reference, while neutral reaches it comfortably).

So: bin the envelope and the charge over the SAME distance bins and the same
s_diel bins, and report the shares side by side plus their ratio. A charge
share far above the envelope share in the shell is the mechanism; comparable
shares refute it.

Grid note: model quantities (env_b, env^0.5, s_ion, delta) are binned on the
model grid (100 x 100 x 300); the DFT raw charge is binned on its native grid
(168 x 168 x 500). Shares are dimensionless, so they are comparable, but no
pointwise difference is taken between the two grids.

Frames: sid 1 charged, sid 601 neutral.  cwd = gate_bl.
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/work/08384/tg876840/ls6/repos/cep-dip-python-pb")
sys.path.insert(0, "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy")

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.pb1d_backend as PB
from mace.modules.solvent3d import (
    normalized_gradient_envelope, poisson_phi_periodic)

GCE44 = "/scratch/08384/tg876840/tmp/2-NiN_single/1-44_GCE"
NEU = "/scratch/08384/tg876840/tmp/2-NiN_single/5-44_neutral_withsolv"
FRAMES = [(1, f"{GCE44}/cal_1", "NiN44 q=-1.00"),
          (601, f"{NEU}/cal_1", "neutral")]
DB = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0, 1.0e9]
SB = [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
EPS_SQ = 1.0e-3

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
model = torch.load(f="checkpoints/s3d_gate_bl2_run-123.model",
                   map_location=device).to(device)
ck = torch.load("checkpoints/s3d_gate_bl2_run-123_epoch-33.pt",
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
    if out.get("s3d_obs") is not None and "delta_grid" in out["s3d_obs"]:
        cap["delta"] = out["s3d_obs"]["delta_grid"].detach().clone()
    return out
PB.PB1DBackend.solve_graph = wrap_solve
_clo = PB.closure_from_fields
def wrap_clo(n_e, cv, grid, params, tp_):
    cap["cvhar3"] = cv.detach().clone()
    cap["grid"] = grid
    cap["params"] = params
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
    for a in read(f"data/{split}.xyz", ":"):
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


def dist_to_atoms(shape, cell, fpos, budget=4.0e7):
    gi = torch.arange(shape[0], device=device, dtype=torch.float64) / shape[0]
    gj = torch.arange(shape[1], device=device, dtype=torch.float64) / shape[1]
    gk = torch.arange(shape[2], device=device, dtype=torch.float64) / shape[2]
    out = torch.empty(shape, dtype=torch.float64, device=device)
    step = max(1, int(budget / (shape[0] * shape[1] * max(len(fpos), 1))))
    for k0 in range(0, shape[2], step):
        fg = torch.stack(torch.meshgrid(gi, gj, gk[k0:k0 + step],
                                        indexing="ij"), dim=-1)
        df = fg[..., None, :] - fpos[None, None, None, :, :]
        df = df - torch.round(df)
        out[:, :, k0:k0 + step] = torch.linalg.norm(
            df @ cell, dim=-1).min(dim=-1).values
        del fg, df
    return out


def shares(field, mask_list, dV):
    tot = float(field.abs().sum() * dV)
    return [100.0 * float((field.abs() * m).sum() * dV) / max(tot, 1e-30)
            for m in mask_list], tot


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
    Vm = grid.volume; dVm = Vm / float(np.prod(shm))
    phi_m = -(cap["cvhar3"] - cap["cvhar3"].mean())
    cavd = grid._solv3d_cavity
    s_diel_m = torch.clamp(cavd[1], 0.0, 1.0)
    s_ion_m = torch.clamp(cavd[0], 0.0, 1.0)
    env_b = normalized_gradient_envelope(cavd[1], cellm)
    env_sq = env_b / torch.sqrt(env_b + EPS_SQ)
    env_sq = env_sq / torch.clamp(env_sq.max(), min=1e-30)
    delta = cap.get("delta")
    pos = torch.as_tensor(a.get_positions(), device=device, dtype=torch.float64)
    fpos = pos @ torch.linalg.inv(cellm)
    dm = dist_to_atoms(shm, cellm, fpos)
    print(f"\n=============== {tag} ===============", flush=True)

    # DFT raw charge on its own native grid, same distance bins
    latd, rb = read_grid(f"{dftdir}/RHOB")
    _, ri = read_grid(f"{dftdir}/RHOION")
    Vd = float(abs(np.linalg.det(latd)))
    shd = tuple(rb.shape)
    dVd = Vd / float(np.prod(shd))
    celld = torch.tensor(latd, device=device)
    n_solv_d = -((rb + ri).to(device)) / Vd
    dd = dist_to_atoms(shd, celld, pos @ torch.linalg.inv(celld))
    del rb, ri

    mm = [(dm >= DB[k]) & (dm < DB[k + 1]) for k in range(len(DB) - 1)]
    md = [(dd >= DB[k]) & (dd < DB[k + 1]) for k in range(len(DB) - 1)]
    q_sh, q_tot = shares(n_solv_d, md, dVd)
    eb_sh, _ = shares(env_b, mm, dVm)
    es_sh, _ = shares(env_sq, mm, dVm)
    si_sh, _ = shares(s_ion_m, mm, dVm)
    dl_sh = shares(delta, mm, dVm)[0] if delta is not None else [float("nan")] * len(mm)
    print(f"  raw solvent charge int|rho| {q_tot:.3f} e "
          f"(native grid {shd}); model grid {shm}")
    print(f"\n  {'dist (A)':>11} {'vol%(m)':>8} {'charge%':>8} {'env_b%':>8} "
          f"{'env^0.5%':>9} {'s_ion%':>8} {'delta%':>8} "
          f"{'chg/env_b':>10} {'chg/env^.5':>11}")
    for k in range(len(DB) - 1):
        hi = "inf" if DB[k + 1] > 1e8 else f"{DB[k+1]:.1f}"
        vol = 100.0 * float(mm[k].sum()) / float(np.prod(shm))
        r1 = q_sh[k] / max(eb_sh[k], 1e-12)
        r2 = q_sh[k] / max(es_sh[k], 1e-12)
        print(f"  {DB[k]:4.1f}-{hi:<6} {vol:8.2f} {q_sh[k]:8.2f} "
              f"{eb_sh[k]:8.2f} {es_sh[k]:9.2f} {si_sh[k]:8.2f} "
              f"{dl_sh[k]:8.2f} {r1:10.2f} {r2:11.2f}", flush=True)
    print(f"  (chg/env ratio >> 1 means the charge is where the envelope is "
          f"weak: reaching the reference cross energy there needs large\n"
          f"   coefficients, and large coefficients cost self-energy -- the "
          f"observed charged-frame obstruction. ~1 refutes the mechanism.)")

    mm2 = [(s_diel_m >= SB[k]) & (s_diel_m < SB[k + 1])
           for k in range(len(SB) - 1)]
    print(f"\n  {'s_diel bin':>12} {'vol%':>7} {'env_b%':>8} {'env^0.5%':>9} "
          f"{'delta%':>8} {'<|phi|>':>8}")
    for k in range(len(SB) - 1):
        if not bool(mm2[k].any()):
            continue
        eb = 100.0 * float((env_b * mm2[k]).sum() * dVm) / max(
            float(env_b.sum() * dVm), 1e-30)
        es = 100.0 * float((env_sq * mm2[k]).sum() * dVm) / max(
            float(env_sq.sum() * dVm), 1e-30)
        dl = (100.0 * float((delta.abs() * mm2[k]).sum() * dVm)
              / max(float(delta.abs().sum() * dVm), 1e-30)
              ) if delta is not None else float("nan")
        print(f"  {SB[k]:5.2f}-{SB[k+1]:<6.2f} "
              f"{100.0*float(mm2[k].sum())/float(np.prod(shm)):7.2f} "
              f"{eb:8.2f} {es:9.2f} {dl:8.2f} "
              f"{float(phi_m[mm2[k]].abs().mean()):8.3f}", flush=True)
    print(f"  (the cavity comparison put 83% of the raw charge in "
          f"0.01 < s_diel < 0.90; compare the env_b column against that)")

    # where is the envelope's own peak, in distance?
    pk = float(dm[env_b > 0.9 * env_b.max()].mean()) if bool(
        (env_b > 0.9 * env_b.max()).any()) else float("nan")
    wmean_b = float((env_b * dm).sum() / torch.clamp(env_b.sum(), min=1e-30))
    wmean_s = float((env_sq * dm).sum() / torch.clamp(env_sq.sum(), min=1e-30))
    wmean_q = float((n_solv_d.abs() * dd).sum()
                    / torch.clamp(n_solv_d.abs().sum(), min=1e-30))
    print(f"\n  envelope-weighted mean distance: env_b {wmean_b:.2f} A, "
          f"env^0.5 {wmean_s:.2f} A;  charge-weighted {wmean_q:.2f} A; "
          f"env_b peak at {pk:.2f} A")
    del dm, dd, n_solv_d, mm, md, mm2, env_b, env_sq, s_diel_m, s_ion_m, phi_m
    if delta is not None:
        del delta
    import gc; gc.collect(); torch.cuda.empty_cache()
print("DONE")
