"""Charge/potential mismatch, accounted correctly (user corrections
2026-09-09, replacing the withdrawn version of this script).

Four corrections are applied here, all of them mine to own:

1. WITHDRAWN: "the cavity is excluded". The previous classification called
   s_diel <= 0.5 "closed", which puts the charge-carrying transition region
   (mean s_diel ~ 0.18) inside "closed". Agreement of that binary label does
   NOT show that the continuous s_diel and, more to the point, its gradient
   agree -- and the bound charge is -div P, so it depends on the gradient.
   A 0.43%-of-volume mismatch that carries 27% of the cross-energy gap cannot
   be dismissed for being small in volume. Neither direction is established:
   the cavity is not exonerated and it is not convicted.

2. FIXED: an arithmetic error. int|bg + delta| is NOT int|bg| + int|delta| --
   background and residual can cancel, and the triangle inequality only gives
   an upper bound. The claim "the model already puts about the right amount of
   charge there" rested on 0.592 + 0.839 ~ 1.407 and is withdrawn. This script
   forms the SUM FIELD FIRST and then integrates its absolute value.

3. WITHDRAWN: the charge-share / envelope-share ratio as a pass-fail
   criterion. The envelope is not a prediction of the charge, so there is no
   reason for the two shares to agree; and large coefficients do not
   necessarily imply large self-energy, which is a functional of the final
   charge distribution. The previous version also compared RHOB+RHOION
   against the BOUND channel's envelope, remixing the two charge types it had
   just separated. The ratio is kept below as a POSITION diagnostic only, and
   each charge type is compared against its own switch.

4. WITHDRAWN: "the earlier 90% was an interpolation artifact". The earlier
   figure was 90% of the lateral charge within 2.5 A (17.8% + 10.9% of the
   cell volume), not within 1.5 A. The native grid gives 26.8% + 61.5% =
   88.2% within 2.5 A, which agrees. There is no interpolation artifact; the
   restatement was a misreading of my own table.

What survives from the cavity comparison, and is worth keeping: the bound
charge concentrates in the dielectric transition region (83% of int|RHOB| in
the 5% of volume with 0.01 < s_diel < 0.90) while the ionic charge sits in
the bulk (93.7% of int|RHOION| at s_ion > 0.90). The two channels live on
different length scales.

Measured here:

  T1  Per region, bound and ionic kept apart: net and absolute charge of the
      model's TOTAL field for that channel (1-D background + residual, summed
      before taking the absolute value) against the DFT reference, plus the
      components int|bg| and int|delta| listed separately and never added.
  T2  The bound part split into plane-average and lateral, on both sides.
      This separates the part of the gap that belongs to the 1-D pipeline
      (plane averages) from the part the 3-D residual is responsible for
      (lateral) -- the 3-D fit must never be charged with the 1-D error.
  T3  In the regions that actually carry the cross-energy gap, the CONTINUOUS
      cavity value and its gradient magnitude, model against DFT. This is the
      test that the binary open/closed classification failed to perform.
  T4  Position diagnostic only: where the bound charge sits versus where
      env_b has its weight, and where the ionic charge sits versus s_ion. No
      pass/fail attached.

Grid note: model fields live on 100 x 100 x 300, the DFT reference on
168 x 168 x 500. Charge and cavity comparisons are made on the model grid
(where the model's energy is actually evaluated) with the DFT fields
interpolated, which smooths sharp features; the DFT-side distributions are
ALSO reported on their native grid, uninterpolated, so the smoothing cannot
contaminate them.

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
    _grad_mag_periodic, normalized_gradient_envelope)

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
    cap["rho_bound_z"] = out["rho_bound_z"].detach().clone()
    cap["rho_ion_z"] = out["rho_ion_z"].detach().clone()
    o = out.get("s3d_obs")
    if o is not None and "delta_b_grid" in o:
        cap["delta_b"] = o["delta_b_grid"].detach().clone()
        cap["delta_i"] = o["delta_i_grid"].detach().clone()
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


def to_shape(t, shape):
    if tuple(t.shape) == tuple(shape):
        return t
    return torch.nn.functional.interpolate(
        t[None, None].to(torch.float64), size=tuple(shape), mode="trilinear",
        align_corners=False)[0, 0].contiguous()


def dist_to_atoms(shape, cell, fpos, budget=3.0e7):
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


def bins_of(field, edges):
    return [(field >= edges[k]) & (field < edges[k + 1])
            for k in range(len(edges) - 1)]


def lbl(edges, k):
    hi = "inf" if edges[k + 1] > 1e8 else f"{edges[k+1]:g}"
    return f"{edges[k]:g}-{hi}"


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
    gm_m = _grad_mag_periodic(cavd[1], cellm)
    env_b = normalized_gradient_envelope(cavd[1], cellm)
    env_sq = env_b / torch.sqrt(env_b + EPS_SQ)
    env_sq = env_sq / torch.clamp(env_sq.max(), min=1e-30)
    print(f"\n=================== {tag} ===================", flush=True)

    # model channel totals: 1-D background broadcast + 3-D residual
    def bcast(prof):
        p = prof.to(device).to(torch.float64)
        if p.shape[0] != shm[2]:
            p = torch.nn.functional.interpolate(
                p[None, None], size=(shm[2],), mode="linear",
                align_corners=False)[0, 0]
        return p[None, None, :].expand(shm).contiguous()

    bg_b = bcast(cap["rho_bound_z"])
    bg_i = bcast(cap["rho_ion_z"])
    db = cap["delta_b"].to(device)
    di = cap["delta_i"].to(device)
    assert tuple(db.shape) == shm and tuple(di.shape) == shm
    mb_tot = bg_b + db          # model bound total
    mi_tot = bg_i + di          # model ionic total
    print(f"  [check] residual plane means: bound "
          f"{float(db.mean(dim=(0,1)).abs().max()):.2e}, ionic "
          f"{float(di.mean(dim=(0,1)).abs().max()):.2e} e/A^3 "
          f"(per-plane projection -> both must be ~0)", flush=True)
    print(f"  [check] int|bg_b| {float(bg_b.abs().sum()*dVm):.4f}, "
          f"int|delta_b| {float(db.abs().sum()*dVm):.4f}, "
          f"int|bg_b+delta_b| {float(mb_tot.abs().sum()*dVm):.4f} e "
          f"-- the third is the only one that means anything; the first two "
          f"must never be added", flush=True)

    # DFT reference on its native grid
    latd, rb = read_grid(f"{dftdir}/RHOB")
    _, ri = read_grid(f"{dftdir}/RHOION")
    _, chg = read_grid(f"{dftdir}/CHGCAR")
    Vd = float(abs(np.linalg.det(latd)))
    shd = tuple(rb.shape)
    dVd = Vd / float(np.prod(shd))
    celld = torch.tensor(latd, device=device)
    db_d = -(rb.to(device)) / Vd            # DFT bound
    di_d = -(ri.to(device)) / Vd            # DFT ionic
    ne_d = torch.clamp(chg.to(device) / Vd, min=0.0)
    from pure_python import torch_pb as tp
    gd = tp.TorchGrid(latd, shd, device=str(device), dtype=torch.float64,
                      rspec=True)
    s_ion_d, s_diel_d, _ = tp.create_cavity_torch(ne_d, gd, cap["params"])
    s_diel_d = torch.clamp(s_diel_d, 0.0, 1.0)
    s_ion_d = torch.clamp(s_ion_d, 0.0, 1.0)
    gm_d = _grad_mag_periodic(s_diel_d, celld)
    del rb, ri, chg, ne_d

    pos = torch.as_tensor(a.get_positions(), device=device,
                          dtype=torch.float64)
    dm = dist_to_atoms(shm, cellm, pos @ torch.linalg.inv(cellm))
    dd = dist_to_atoms(shd, celld, pos @ torch.linalg.inv(celld))

    # DFT fields on the model grid for same-grid comparisons
    db_dm = to_shape(db_d, shm)
    di_dm = to_shape(di_d, shm)
    s_diel_dm = torch.clamp(to_shape(s_diel_d, shm), 0.0, 1.0)
    gm_dm = to_shape(gm_d, shm)

    mmd = bins_of(dm, DB)

    # ---------------- T1: bound and ionic, kept apart ----------------
    for nm, mtot, mbg, mdel, dref in (
            ("BOUND  (model bg_b+delta_b  vs  DFT RHOB)", mb_tot, bg_b, db, db_dm),
            ("IONIC  (model bg_i+delta_i  vs  DFT RHOION)", mi_tot, bg_i, di, di_dm)):
        print(f"\n[T1] {nm}", flush=True)
        print(f"  {'dist (A)':>10} {'vol%':>6} | {'DFT net':>8} {'DFT |q|':>8} "
              f"{'DFT cross':>10} | {'mdl net':>8} {'mdl |q|':>8} "
              f"{'mdl cross':>10} | {'int|bg|':>8} {'int|del|':>9}")
        for k, m in enumerate(mmd):
            if not bool(m.any()):
                continue
            print(f"  {lbl(DB,k):>10} "
                  f"{100.0*float(m.sum())/float(np.prod(shm)):6.2f} | "
                  f"{float((dref*m).sum()*dVm):+8.3f} "
                  f"{float((dref.abs()*m).sum()*dVm):8.3f} "
                  f"{float((dref*phi_m*m).sum()*dVm):+10.3f} | "
                  f"{float((mtot*m).sum()*dVm):+8.3f} "
                  f"{float((mtot.abs()*m).sum()*dVm):8.3f} "
                  f"{float((mtot*phi_m*m).sum()*dVm):+10.3f} | "
                  f"{float((mbg.abs()*m).sum()*dVm):8.3f} "
                  f"{float((mdel.abs()*m).sum()*dVm):9.3f}", flush=True)
        print(f"  {'TOTAL':>10} {100.0:6.2f} | "
              f"{float(dref.sum()*dVm):+8.3f} "
              f"{float(dref.abs().sum()*dVm):8.3f} "
              f"{float((dref*phi_m).sum()*dVm):+10.3f} | "
              f"{float(mtot.sum()*dVm):+8.3f} "
              f"{float(mtot.abs().sum()*dVm):8.3f} "
              f"{float((mtot*phi_m).sum()*dVm):+10.3f} | "
              f"{float(mbg.abs().sum()*dVm):8.3f} "
              f"{float(mdel.abs().sum()*dVm):9.3f}", flush=True)

    # ---------------- T2: bound split plane-average vs lateral ----------
    pm_d = db_dm.mean(dim=(0, 1), keepdim=True).expand(shm)
    lat_d = db_dm - pm_d
    print(f"\n[T2] BOUND charge split. The 1-D pipeline owns the plane "
          f"averages; the 3-D residual owns the lateral part and must not be "
          f"charged with the 1-D error.")
    print(f"  plane-average part:  DFT int|.| "
          f"{float(pm_d.abs().sum()*dVm):.4f} e  cross "
          f"{float((pm_d*phi_m).sum()*dVm):+.4f} eV   |   model int|.| "
          f"{float(bg_b.abs().sum()*dVm):.4f} e  cross "
          f"{float((bg_b*phi_m).sum()*dVm):+.4f} eV", flush=True)
    print(f"  lateral part:        DFT int|.| "
          f"{float(lat_d.abs().sum()*dVm):.4f} e  cross "
          f"{float((lat_d*phi_m).sum()*dVm):+.4f} eV   |   model int|.| "
          f"{float(db.abs().sum()*dVm):.4f} e  cross "
          f"{float((db*phi_m).sum()*dVm):+.4f} eV", flush=True)
    print(f"\n  {'dist (A)':>10} | {'DFT pm |q|':>10} {'DFT pm cr':>10} "
          f"{'mdl pm |q|':>10} {'mdl pm cr':>10} | {'DFT lat |q|':>11} "
          f"{'DFT lat cr':>11} {'mdl lat |q|':>11} {'mdl lat cr':>11}")
    for k, m in enumerate(mmd):
        if not bool(m.any()):
            continue
        print(f"  {lbl(DB,k):>10} | "
              f"{float((pm_d.abs()*m).sum()*dVm):10.3f} "
              f"{float((pm_d*phi_m*m).sum()*dVm):+10.3f} "
              f"{float((bg_b.abs()*m).sum()*dVm):10.3f} "
              f"{float((bg_b*phi_m*m).sum()*dVm):+10.3f} | "
              f"{float((lat_d.abs()*m).sum()*dVm):11.3f} "
              f"{float((lat_d*phi_m*m).sum()*dVm):+11.3f} "
              f"{float((db.abs()*m).sum()*dVm):11.3f} "
              f"{float((db*phi_m*m).sum()*dVm):+11.3f}", flush=True)

    # ---------------- T3: continuous cavity where the gap is ------------
    gapf = (mb_tot + mi_tot - db_dm - di_dm) * phi_m
    gtot = float(gapf.sum() * dVm)
    print(f"\n[T3] cross-energy gap (model total minus DFT total, model "
          f"potential): {gtot:+.4f} eV total. Continuous cavity value and "
          f"gradient magnitude where that gap sits.")
    print(f"  {'dist (A)':>10} {'gap (eV)':>9} {'gap%':>7} | "
          f"{'<s^m>':>7} {'<s^D>':>7} {'rms ds':>8} | "
          f"{'<|gs|^m>':>9} {'<|gs|^D>':>9} {'ratio':>7} {'rms dgs':>8}")
    for k, m in enumerate(mmd):
        if not bool(m.any()):
            continue
        gk = float((gapf * m).sum() * dVm)
        gsm = float(gm_m[m].mean()); gsd = float(gm_dm[m].mean())
        print(f"  {lbl(DB,k):>10} {gk:+9.3f} "
              f"{100.0*gk/max(abs(gtot),1e-12):7.1f} | "
              f"{float(s_diel_m[m].mean()):7.4f} "
              f"{float(s_diel_dm[m].mean()):7.4f} "
              f"{float(((s_diel_m-s_diel_dm)[m]**2).mean().sqrt()):8.4f} | "
              f"{gsm:9.4f} {gsd:9.4f} {gsm/max(gsd,1e-12):7.3f} "
              f"{float(((gm_m-gm_dm)[m]**2).mean().sqrt()):8.4f}", flush=True)
    msh = (s_diel_dm > 0.01) & (s_diel_dm < 0.9)
    for nm, x, y in (("s_diel", s_diel_m, s_diel_dm),
                     ("|grad s_diel|", gm_m, gm_dm)):
        xv = x[msh]; yv = y[msh]
        xc = xv - xv.mean(); yc = yv - yv.mean()
        cc = float((xc * yc).sum() / torch.clamp(
            xc.norm() * yc.norm(), min=1e-30))
        print(f"  in the DFT transition shell 0.01<s<0.9: {nm} correlation "
              f"{cc:+.4f}, model/DFT mean ratio "
              f"{float(xv.mean()/torch.clamp(yv.mean(),min=1e-30)):.3f}",
              flush=True)
    print(f"  (interpolation caveat: the DFT cavity is smoothed onto the "
          f"100x100x300 model grid here, which lowers |grad s|; the native "
          f"distribution is in T4 below)")

    # ---------------- T4: position diagnostic only ----------------------
    print(f"\n[T4] POSITION DIAGNOSTIC ONLY -- no pass/fail. Each charge type "
          f"against its own switch. The envelope is not a prediction of the "
          f"charge, so these shares have no reason to match.")
    mdd = bins_of(dd, DB)
    def sh(f, ms, dV):
        t = float(f.abs().sum() * dV)
        return [100.0 * float((f.abs() * m).sum() * dV) / max(t, 1e-30)
                for m in ms]
    q_b = sh(db_d, mdd, dVd); q_i = sh(di_d, mdd, dVd)
    e_b = sh(env_b, mmd, dVm); e_s = sh(env_sq, mmd, dVm)
    s_i = sh(s_ion_m, mmd, dVm)
    gnat = sh(gm_d, mdd, dVd)
    print(f"  {'dist (A)':>10} {'RHOB%':>7} {'env_b%':>7} {'env^.5%':>8} "
          f"{'|gs|nat%':>9} | {'RHOION%':>8} {'s_ion%':>7}")
    for k in range(len(DB) - 1):
        if not bool(mmd[k].any()):
            continue
        print(f"  {lbl(DB,k):>10} {q_b[k]:7.2f} {e_b[k]:7.2f} {e_s[k]:8.2f} "
              f"{gnat[k]:9.2f} | {q_i[k]:8.2f} {s_i[k]:7.2f}", flush=True)
    wm = lambda f, d, dV: float((f.abs() * d).sum()
                                / torch.clamp(f.abs().sum(), min=1e-30))
    print(f"  weighted mean distance: RHOB {wm(db_d,dd,dVd):.2f} A, "
          f"env_b {wm(env_b,dm,dVm):.2f} A, env^0.5 {wm(env_sq,dm,dVm):.2f} A, "
          f"|grad s| native {wm(gm_d,dd,dVd):.2f} A | "
          f"RHOION {wm(di_d,dd,dVd):.2f} A, s_ion {wm(s_ion_m,dm,dVm):.2f} A",
          flush=True)

    del (dm, dd, db_d, di_d, db_dm, di_dm, s_diel_d, s_ion_d, gm_d, gm_dm,
         s_diel_dm, s_diel_m, s_ion_m, gm_m, env_b, env_sq, phi_m,
         bg_b, bg_i, db, di, mb_tot, mi_tot, pm_d, lat_d, gapf, gd)
    import gc; gc.collect(); torch.cuda.empty_cache()
print("DONE")
