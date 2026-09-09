"""Joint-fit screening of the residual basis (user protocol 2026-09-09).

Question: can the residual basis represent the TRUE LATERAL solvent charge —
scored on all four quantities at once (point error, cross energy, self-energy,
charge amplitude) — and if not, does one extra channel fix it?

Target (user constraint: the 3-D part owns the lateral shortfall ONLY, it is
not asked to absorb the 1-D plane-averaged error):
    n_lat(r) = n_solv_DFT(r) - <n_solv_DFT>(z)          [plane mean removed]
Reference scores, same-potential convention (model potential phi = -(cvhar-mean)):
    cross_ref = int n_lat*phi dV ,  self_ref = 0.5 int n_lat*phi[n_lat] dV ,
    amp_ref   = int |n_lat| dV
Design columns, all put through the SAME per-plane conserving projection the
energy and the loss use, so every column is purely lateral:
    channel  env_b   = |grad s_diel|/max      (interface term of -div P; as built)
    channel  s_ion                            (ionic channel; as built)
    channel  s_diel                           (CANDIDATE: the bulk-polarisation
             term of -div P = s_diel*div E, which the interface envelope cannot
             express — the physically motivated near-solute shape)
    channel  env_b^0.5 (smooth)               (CANDIDATE: numerically softened
             interface shape, the earlier envelope-A idea, now ADDED rather
             than substituted so it can only help)
Bases compared:  A = {env_b, s_ion}  (what the model actually spans)
                 A + s_diel ,  A + env_b^0.5
Fit: ridge least squares against n_lat on a uniform random sample of grid
points (unbiased estimate of the grid L2 fit); normal equations accumulated in
float64. Frames: NiN44 charged, NiN88 charged, neutral solvated.
cwd = gate_bl.
"""
# Paths come from env vars with the LS6 values as defaults, so this
# runs unchanged on Lonestar6 and on a machine that holds the same
# payload elsewhere: KIT_PB_REPO, KIT_MACE_REPO, KIT_DFT,
# KIT_PB_CONFIG. Previously hardcoded, which made every audit
# script non-portable (found by the workstation session 2026-09-09).
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get("KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get("KIT_MACE_REPO", "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.pb1d_backend as PB
from mace.modules.solvent3d import (
    normalized_gradient_envelope, poisson_phi_periodic, _interp3_periodic)

NPTS_SMALL = 80000
NPTS_BIG = 45000
RIDGE = 1.0e-7
EPS_SQ = 1.0e-3
GCE44 = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single") + "/1-44_GCE"
GCE88 = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single") + "/2-88_GCE"
NEU = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single") + "/5-44_neutral_withsolv"
FRAMES = [(1, f"{GCE44}/cal_1", "NiN44 q=-1.00"),
          (201, f"{GCE88}/cal_1", "NiN88 q=-1.00"),
          (601, f"{NEU}/cal_1", "neutral")]

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
model = torch.load(f="checkpoints/s3d_gate_bl2_run-123.model", map_location=device).to(device)
ck = torch.load("checkpoints/s3d_gate_bl2_run-123_epoch-33.pt", map_location=device)
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
    cap["cvhar3"] = cv.detach().clone()
    cap["grid"] = grid
    return _clo(n_e, cv, grid, params, tp_)
PB.closure_from_fields = wrap_clo
_solve = PB.PB1DBackend.solve_graph
def wrap_solve(self, *a, **k):
    out = _solve(self, *a, **k)
    if out.get("s3d_obs") is not None:
        o = out["s3d_obs"]
        cap["delta"] = (o["d_sup_b"] + o["d_sup_i"]).detach().clone()
    return out
PB.PB1DBackend.solve_graph = wrap_solve

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
    lat = np.array([[float(x) for x in f.readline().split()] for _ in range(3)]) * s
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
        t[None, None], size=tuple(shape), mode="trilinear",
        align_corners=False)[0, 0].contiguous()


sigmas = [float(x) for x in (model.solvent3d_sigmas.tolist()
                             if torch.is_tensor(model.solvent3d_sigmas)
                             else model.solvent3d_sigmas)]
print(f"points {NPTS_SMALL}/{NPTS_BIG}, ridge {RIDGE}, sigmas {sigmas}", flush=True)
results = []
for sid, dftdir, tag in FRAMES:
    a = want[sid]
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    cap.clear()
    with torch.no_grad():
        model(b.to_dict(), compute_force=False, training=False)
    grid = cap["grid"]
    cell64 = torch.as_tensor(grid.cell, dtype=torch.float64, device=device)
    shape = tuple(grid.shape)
    V = grid.volume
    dV = V / float(np.prod(shape))
    phi = -(cap["cvhar3"] - cap["cvhar3"].mean())
    cavd = grid._solv3d_cavity
    env_b = normalized_gradient_envelope(cavd[1], cell64)
    s_ion = torch.clamp(cavd[0], 0.0, 1.0)
    s_diel = torch.clamp(cavd[1], 0.0, 1.0)
    env_sq = env_b / torch.sqrt(env_b + EPS_SQ)
    env_sq = env_sq / torch.clamp(env_sq.max(), min=1e-30)
    CH = [("env_b", env_b), ("s_ion", s_ion), ("s_diel", s_diel), ("env^0.5", env_sq)]
    CHm = [torch.clamp(w.mean(dim=(0, 1)), min=1e-12) for _, w in CH]

    # ---- target: DFT lateral solvent charge on the model grid ----
    lat, rb = read_grid(f"{dftdir}/RHOB")
    _, ri = read_grid(f"{dftdir}/RHOION")
    Vd = float(abs(np.linalg.det(lat)))
    n_solv = to_shape(-(rb + ri) / Vd, shape).to(device)
    n_lat = n_solv - n_solv.mean(dim=(0, 1), keepdim=True)
    del rb, ri, n_solv
    cross_ref = float((n_lat * phi).sum() * dV)
    self_ref = 0.5 * float((n_lat * poisson_phi_periodic(n_lat, cell64)).sum() * dV)
    amp_ref = float(n_lat.abs().sum() * dV)
    d_cross = float((cap["delta"].to(device) * phi).sum() * dV)
    d_amp = float(cap["delta"].abs().sum() * dV)
    print(f"\n== {tag} ==  reference lateral: cross {cross_ref:+.3f}  "
          f"self {self_ref:+.3f}  |q| {amp_ref:.3f} e   |   trained model: "
          f"cross {d_cross:+.3f}  |q| {d_amp:.3f}", flush=True)

    # ---- sample points, build the design matrix channel-blocked ----
    NPTS = NPTS_SMALL if len(a) < 250 else NPTS_BIG
    g = torch.Generator(device="cpu").manual_seed(11)
    idx = torch.randint(0, int(np.prod(shape)), (NPTS,), generator=g).to(device)
    nx, ny, nz = shape
    iz = idx % nz; iy = (idx // nz) % ny; ix = idx // (nz * ny)
    frac = torch.stack([(ix.to(torch.float64) + 0.5) / nx,
                        (iy.to(torch.float64) + 0.5) / ny,
                        (iz.to(torch.float64) + 0.5) / nz], dim=1)
    y = _interp3_periodic(n_lat, frac)
    nat = len(a)
    K1 = nat * len(sigmas) * 9
    A = torch.zeros(NPTS, 4 * K1, dtype=torch.float32, device=device)
    gvec = torch.zeros(4 * K1, dtype=torch.float64, device=device)
    pos_frac = torch.remainder(
        torch.as_tensor(a.get_positions(), device=device, dtype=torch.float64)
        @ torch.linalg.inv(cell64), 1.0)
    beobj = model._pb1d_backend
    cvec = torch.zeros(nat, len(sigmas), 9, dtype=torch.float64, device=device)
    col = 0
    for ia in range(nat):
        for isig in range(len(sigmas)):
            for lm in range(9):
                cvec.zero_(); cvec[ia, isig, lm] = 1.0
                Gk = grid.ifft_real(beobj._gto_net_density_g(
                    grid, pos_frac, cvec, sigmas)) / V
                for ci, ((_, w), wm) in enumerate(zip(CH, CHm)):
                    raw = w * Gk
                    r = raw.mean(dim=(0, 1)) / wm
                    prj = raw - r[None, None, :] * w
                    A[:, ci * K1 + col] = _interp3_periodic(
                        prj, frac).to(torch.float32)
                    gvec[ci * K1 + col] = (prj * phi).sum() * dV
                col += 1
        if ia % 60 == 0:
            print(f"   ... build atom {ia}/{nat}", flush=True)

    # ---- per-basis normal equations, chunked (never materialise the full H) ----
    y2 = float((y ** 2).sum())
    rms_ref = math.sqrt(y2 / NPTS)
    BASES = [("A = env_b + s_ion", [0, 1]),
             ("A + s_diel", [0, 1, 2]),
             ("A + env^0.5", [0, 1, 3])]
    coeffs = {}
    step = 4000
    for name, chans in BASES:
        sel = torch.cat([torch.arange(c * K1, (c + 1) * K1, device=device)
                         for c in chans])
        n = sel.numel()
        Hs = torch.zeros(n, n, dtype=torch.float64, device=device)
        Aty = torch.zeros(n, dtype=torch.float64, device=device)
        for i in range(0, NPTS, step):
            blk = A[i:i + step][:, sel].to(torch.float64)
            Hs += blk.T @ blk
            Aty += blk.T @ y[i:i + step]
            del blk
        H0 = Hs.clone()
        lam = RIDGE * float(torch.diagonal(Hs).mean())
        Hs += lam * torch.eye(n, dtype=torch.float64, device=device)
        L = torch.linalg.cholesky(Hs)
        c_unc = torch.cholesky_solve(Aty.unsqueeze(1), L).squeeze(1)
        gs = gvec[sel]
        d_dir = torch.cholesky_solve(gs.unsqueeze(1), L).squeeze(1)
        gHg = float(gs @ d_dir)
        cross_unc = float(gs @ c_unc)
        mu = (cross_unc - cross_ref) / gHg if gHg != 0 else 0.0
        c_con = c_unc - mu * d_dir

        def rms_of(cv):
            r2 = y2 - 2 * float(cv @ Aty) + float(cv @ (H0 @ cv))
            return math.sqrt(max(r2, 0.0) / NPTS)

        coeffs[name] = (chans, c_unc, c_con, rms_of(c_unc), rms_of(c_con),
                        float(c_unc.abs().max()), float(c_con.abs().max()), mu)
        del Hs, H0, L, Aty
        torch.cuda.empty_cache()
    del A
    torch.cuda.empty_cache()

    # ---- rebuild the fitted fields, one synthesis pass for all of them ----
    flds = {(name, k): torch.zeros(shape, dtype=torch.float64, device=device)
            for name, _ in BASES for k in ("unc", "con")}
    col = 0
    for ia in range(nat):
        for isig in range(len(sigmas)):
            for lm in range(9):
                needed = []
                for name, _ in BASES:
                    chans, cu, cc = coeffs[name][0], coeffs[name][1], coeffs[name][2]
                    for j, ch in enumerate(chans):
                        vu = float(cu[j * K1 + col]); vc = float(cc[j * K1 + col])
                        if abs(vu) > 1e-14 or abs(vc) > 1e-14:
                            needed.append((name, ch, vu, vc))
                if needed:
                    cvec.zero_(); cvec[ia, isig, lm] = 1.0
                    Gk = grid.ifft_real(beobj._gto_net_density_g(
                        grid, pos_frac, cvec, sigmas)) / V
                    projs = {}
                    for name, ch, vu, vc in needed:
                        if ch not in projs:
                            w = CH[ch][1]; wm = CHm[ch]
                            raw = w * Gk
                            r = raw.mean(dim=(0, 1)) / wm
                            projs[ch] = raw - r[None, None, :] * w
                        if abs(vu) > 1e-14:
                            flds[(name, "unc")] += vu * projs[ch]
                        if abs(vc) > 1e-14:
                            flds[(name, "con")] += vc * projs[ch]
                col += 1
        if ia % 60 == 0:
            print(f"   ... rebuild atom {ia}/{nat}", flush=True)

    print(f"  {'basis':>20} {'mode':>7} {'fit rms/ref':>12} {'cross':>9} "
          f"{'self':>9} {'|q| (e)':>9} {'|c|max':>10}")
    print(f"  {'reference':>20} {'—':>7} {'—':>12} {cross_ref:+9.3f} "
          f"{self_ref:+9.3f} {amp_ref:9.3f} {'—':>10}")
    for name, _ in BASES:
        chans, cu, cc, ru, rc, mu_max, mc_max, mu = coeffs[name]
        for k, rms, cmax in (("unc", ru, mu_max), ("con", rc, mc_max)):
            f = flds[(name, k)]
            cr = float((f * phi).sum() * dV)
            se = 0.5 * float((f * poisson_phi_periodic(f, cell64)).sum() * dV)
            am = float(f.abs().sum() * dV)
            results.append((tag, name, k, rms / rms_ref, cr, se, am, cmax,
                            cross_ref, self_ref, amp_ref))
            print(f"  {name:>20} {k:>7} {rms/rms_ref:12.3f} {cr:+9.3f} "
                  f"{se:+9.3f} {am:9.3f} {cmax:10.2e}", flush=True)

    del flds, n_lat, phi, env_b, s_ion, s_diel, env_sq
    import gc; gc.collect(); torch.cuda.empty_cache()

print("\n=== summary: fraction of the reference recovered ===")
print(f"{'frame':>14} {'basis':>20} {'mode':>5} {'rms ratio':>10} "
      f"{'cross/ref':>10} {'self/ref':>9} {'|q|/ref':>8}")
for r in results:
    print(f"{r[0]:>14} {r[1]:>20} {r[2]:>5} {r[3]:10.3f} {r[4]/r[8]:10.3f} "
          f"{r[5]/r[9]:9.3f} {r[6]/r[10]:8.3f}")
print("DONE  (unc = point-optimal; con = constrained to cross = reference."
      " basis PASSES if, at con, self/ref and |q|/ref stay near 1)")
