"""Feasible-solution search with ALL FOUR quantities as conditions, per basis.

Everything before this either (a) minimised point error and only measured the
energies afterwards, or (b) constrained the cross energy alone. This run asks
the actual question, separately for each basis so the answer is attributable:

    does the basis contain a field with  cross = cross_ref,  self = self_ref,
    |q| ~ |q|_ref,  at a point error not much worse than the plain fit?

Method. Inside each basis, build an 8-vector family from four weighted fits
and their four cross-energy pull directions:
    weightings w = 1 , |phi| , phi^2 , near-solute indicator (d < 2.5 A)
    v_(2i-1) = argmin_c  sum_p w_p (A c - y)_p^2  (+ridge)
    v_(2i)   = H_w^{-1} g       (the direction that moves the cross energy)
For alpha in R^8 and c = sum alpha_i v_i, all three functionals are exact and
tiny to evaluate:
    point error  = alpha' M alpha - 2 alpha' b + const ,  M_ij = (A v_i)·(A v_j)
    cross        = x · alpha ,                            x_i = g · v_i
    self         = 0.5 alpha' S alpha ,                   S_ij = int f_i phi[f_j] dV
(f_i are the eight fields, built explicitly; S needs 8 Poisson solves.)

Two things are then computed in closed form, per basis:
 1. min_self at cross = cross_ref  =  0.5 * cross_ref^2 / (x' S^{-1} x).
    If that minimum already exceeds self_ref, the family provably contains NO
    field with the reference cross energy at or below the reference
    self-energy — a decisive statement for this family.
 2. if it is below self_ref, the point-error-minimal alpha on the intersection
    {cross = cross_ref, self = self_ref}, found by a 1-D root search on the
    self multiplier, with |q| evaluated on the resulting field. |q| is a PASS
    CONDITION, not a footnote.

Caveats stated up front: a negative result speaks for this 8-vector family and
this basis only, not for the whole span; a positive result is constructive and
therefore conclusive for that basis.  cwd = gate_bl.
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

RIDGE = 1.0e-7
EPS_SQ = 1.0e-3
GCE44 = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single") + "/1-44_GCE"
GCE88 = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single") + "/2-88_GCE"
NEU = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single") + "/5-44_neutral_withsolv"
FRAMES = [(1, f"{GCE44}/cal_1", "NiN44 q=-1.00", 80000),
          (601, f"{NEU}/cal_1", "neutral", 80000),
          (201, f"{GCE88}/cal_1", "NiN88 q=-1.00", 45000)]
# BASES is set per frame inside the loop (the ion channel is gated there)
WTS = ["1", "|phi|", "phi^2", "near"]

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
want = {sid: None for sid, _, _, _ in FRAMES}
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
summary = []
for sid, dftdir, tag, NPTS in FRAMES:
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
    shape = tuple(grid.shape); nx, ny, nz = shape
    V = grid.volume; dV = V / float(np.prod(shape))
    phi = -(cap["cvhar3"] - cap["cvhar3"].mean())
    cavd = grid._solv3d_cavity
    env_b = normalized_gradient_envelope(cavd[1], cell64)
    s_ion = torch.clamp(cavd[0], 0.0, 1.0)
    s_diel = torch.clamp(cavd[1], 0.0, 1.0)
    env_sq = env_b / torch.sqrt(env_b + EPS_SQ)
    env_sq = env_sq / torch.clamp(env_sq.max(), min=1e-30)
    # FIX (user review 2026-09-09): the model's ion-channel coefficients are
    # multiplied by q_tot (solvent3d.py: c[:,1]*q_gate), so on a neutral frame
    # that channel is identically zero and must NOT be a free parameter here.
    q_tot = float(a.info.get("total_charge", 0.0))
    ion_live = abs(q_tot) > 1.0e-6
    CH = [env_b] + ([s_ion] if ion_live else []) + [s_diel, env_sq]
    CH_NAMES = ["env_b"] + (["s_ion"] if ion_live else []) + ["s_diel", "env^0.5"]
    NCH = len(CH)
    IDX_DIEL = CH_NAMES.index("s_diel")
    IDX_SQ = CH_NAMES.index("env^0.5")
    BASE_CH = [0] + ([1] if ion_live else [])
    BASES = [("A", BASE_CH), ("A+s_diel", BASE_CH + [IDX_DIEL]),
             ("A+env^0.5", BASE_CH + [IDX_SQ])]
    CHm = [torch.clamp(w.mean(dim=(0, 1)), min=1e-12) for w in CH]
    print(f"  channels in play: {CH_NAMES}  (q_tot {q_tot:+.3f}, "
          f"ion channel {'live' if ion_live else 'GATED OFF as in the model'})",
          flush=True)

    lat, rb = read_grid(f"{dftdir}/RHOB")
    _, ri = read_grid(f"{dftdir}/RHOION")
    Vd = float(abs(np.linalg.det(lat)))
    n_solv = to_shape(-(rb + ri) / Vd, shape).to(device)
    n_lat = n_solv - n_solv.mean(dim=(0, 1), keepdim=True)
    del rb, ri, n_solv
    X_ref = float((n_lat * phi).sum() * dV)
    S_ref = 0.5 * float((n_lat * poisson_phi_periodic(n_lat, cell64)).sum() * dV)
    Q_ref = float(n_lat.abs().sum() * dV)
    print(f"\n===== {tag} =====  reference lateral: cross {X_ref:+.4f}  "
          f"self {S_ref:+.4f}  |q| {Q_ref:.4f} e", flush=True)

    g = torch.Generator(device="cpu").manual_seed(11)
    idx = torch.randint(0, int(np.prod(shape)), (NPTS,), generator=g).to(device)
    iz = idx % nz; iy = (idx // nz) % ny; ix = idx // (nz * ny)
    frac = torch.stack([(ix.to(torch.float64) + 0.5) / nx,
                        (iy.to(torch.float64) + 0.5) / ny,
                        (iz.to(torch.float64) + 0.5) / nz], dim=1)
    y = _interp3_periodic(n_lat, frac)
    phi_p = _interp3_periodic(phi, frac)
    pos_frac = torch.remainder(
        torch.as_tensor(a.get_positions(), device=device, dtype=torch.float64)
        @ torch.linalg.inv(cell64), 1.0)
    dfp = frac[:, None, :] - pos_frac[None, :, :]
    dfp = dfp - torch.round(dfp)
    dmin_p = torch.einsum("pni,ij->pnj", dfp, cell64).norm(dim=-1).min(dim=1).values
    del dfp
    rms_all_ref = math.sqrt(float((y ** 2).mean()))
    deep = phi_p.abs() >= torch.quantile(phi_p.abs(), 0.90)
    rms_deep_ref = math.sqrt(float((y[deep] ** 2).mean()))

    nat = len(a); K1 = nat * len(sigmas) * 9
    A = torch.zeros(NPTS, NCH * K1, dtype=torch.float32, device=device)
    gvec = torch.zeros(NCH * K1, dtype=torch.float64, device=device)
    beobj = model._pb1d_backend
    cvec = torch.zeros(nat, len(sigmas), 9, dtype=torch.float64, device=device)
    col = 0
    for ia in range(nat):
        for isig in range(len(sigmas)):
            for lm in range(9):
                cvec.zero_(); cvec[ia, isig, lm] = 1.0
                Gk = grid.ifft_real(beobj._gto_net_density_g(
                    grid, pos_frac, cvec, sigmas)) / V
                for ci, (w, wm) in enumerate(zip(CH, CHm)):
                    raw = w * Gk
                    r = raw.mean(dim=(0, 1)) / wm
                    prj = raw - r[None, None, :] * w
                    A[:, ci * K1 + col] = _interp3_periodic(prj, frac).to(torch.float32)
                    gvec[ci * K1 + col] = (prj * phi).sum() * dV
                col += 1
        if ia % 80 == 0:
            print(f"   [{tag}] build {ia}/{nat}", flush=True)

    fam = {}          # (basis, i) -> coefficient vector in the FULL column space
    for bname, chans in BASES:
        sel = torch.cat([torch.arange(c * K1, (c + 1) * K1, device=device)
                         for c in chans])
        n = sel.numel()
        vs = []
        for wt in WTS:
            if wt == "1":
                wp = torch.ones_like(phi_p)
            elif wt == "|phi|":
                wp = phi_p.abs()
            elif wt == "phi^2":
                wp = phi_p ** 2
            else:
                wp = (dmin_p < 2.5).to(torch.float64) + 0.01
            wp = wp / wp.mean()
            Hw = torch.zeros(n, n, dtype=torch.float64, device=device)
            Awy = torch.zeros(n, dtype=torch.float64, device=device)
            for i in range(0, NPTS, 4000):
                blk = A[i:i + 4000][:, sel].to(torch.float64)
                wb = wp[i:i + 4000]
                Hw += blk.T @ (blk * wb[:, None])
                Awy += blk.T @ (y[i:i + 4000] * wb)
                del blk
            Hw += RIDGE * float(torch.diagonal(Hw).mean()) * torch.eye(
                n, dtype=torch.float64, device=device)
            L = torch.linalg.cholesky(Hw)
            c_sol = torch.cholesky_solve(Awy.unsqueeze(1), L).squeeze(1)
            c_dir = torch.cholesky_solve(gvec[sel].unsqueeze(1), L).squeeze(1)
            for cc in (c_sol, c_dir):
                full = torch.zeros(NCH * K1, dtype=torch.float64, device=device)
                full[sel] = cc
                vs.append(full)
            del Hw, Awy, L
            torch.cuda.empty_cache()
        fam[bname] = vs
        print(f"   [{tag}] {bname}: {len(vs)} family vectors", flush=True)

    # point-space Gram matrices while A is alive
    small = {}
    for bname, vs in fam.items():
        m = len(vs)
        P = torch.zeros(NPTS, m, dtype=torch.float64, device=device)
        for j, v in enumerate(vs):
            for i in range(0, NPTS, 4000):
                P[i:i + 4000, j] = A[i:i + 4000].to(torch.float64) @ v
        M = P.T @ P
        bb = P.T @ y
        Pd = P[deep]
        small[bname] = (M, bb, Pd.T @ Pd, Pd.T @ y[deep],
                        torch.stack([gvec @ v for v in vs]))
        del P, Pd
        torch.cuda.empty_cache()
    del A
    torch.cuda.empty_cache()

    # build the family fields (one synthesis pass for every basis at once)
    flds = {(bn, j): torch.zeros(shape, dtype=torch.float64, device=device)
            for bn, vs in fam.items() for j in range(len(vs))}
    col = 0
    for ia in range(nat):
        for isig in range(len(sigmas)):
            for lm in range(9):
                todo = []
                for bn, vs in fam.items():
                    for j, v in enumerate(vs):
                        for ci in range(NCH):
                            val = float(v[ci * K1 + col])
                            if abs(val) > 1e-16:
                                todo.append((bn, j, ci, val))
                if todo:
                    cvec.zero_(); cvec[ia, isig, lm] = 1.0
                    Gk = grid.ifft_real(beobj._gto_net_density_g(
                        grid, pos_frac, cvec, sigmas)) / V
                    projs = {}
                    for bn, j, ci, val in todo:
                        if ci not in projs:
                            w = CH[ci]; wm = CHm[ci]
                            raw = w * Gk
                            r = raw.mean(dim=(0, 1)) / wm
                            projs[ci] = raw - r[None, None, :] * w
                        flds[(bn, j)] += val * projs[ci]
                col += 1
        if ia % 80 == 0:
            print(f"   [{tag}] rebuild {ia}/{nat}", flush=True)

    y2 = float((y ** 2).sum()); y2d = float((y[deep] ** 2).sum())
    nd = int(deep.sum())
    for bname, vs in fam.items():
        m = len(vs)
        M, bb, Md, bbd, x = small[bname]
        phis = [poisson_phi_periodic(flds[(bname, j)], cell64) for j in range(m)]
        S = torch.zeros(m, m, dtype=torch.float64, device=device)
        for i in range(m):
            for j in range(i, m):
                val = float((flds[(bname, i)] * phis[j]).sum() * dV)
                S[i, j] = val; S[j, i] = val
        del phis
        Ssym = 0.5 * (S + S.T)
        ev = torch.linalg.eigvalsh(Ssym)
        Sinv = torch.linalg.pinv(Ssym, rtol=1e-10)
        xSx = float(x @ (Sinv @ x))
        min_self = 0.5 * X_ref ** 2 / xSx if xSx > 0 else float("inf")

        def solve_nu(nu):
            Kmat = 2 * M + nu * Ssym
            Ki = torch.linalg.pinv(Kmat, rtol=1e-12)
            Kx = Ki @ x; Kb = Ki @ (2 * bb)
            lam = (float(x @ Kb) - X_ref) / float(x @ Kx)
            return Kb - lam * Kx

        def self_of(al):
            return 0.5 * float(al @ (Ssym @ al))

        # FIX (user review): nu<0 REWARDS self-energy and is needed whenever the
        # point-optimal solution sits BELOW the reference self-energy. The valid
        # range is nu > -1/lambda_max of the pencil (Ssym, 2M), where
        # 2M + nu*Ssym stays positive definite.
        Mr = 2 * M + 1e-12 * float(torch.diagonal(2 * M).mean()) * torch.eye(
            m, dtype=torch.float64, device=device)
        Lm = torch.linalg.cholesky(Mr)
        Li = torch.linalg.inv(Lm)
        lam_max = float(torch.linalg.eigvalsh(Li @ Ssym @ Li.T).max())
        nu_min = -0.99 / lam_max if lam_max > 0 else -1e6
        feasible = min_self <= S_ref
        chosen, nu_used, note = None, None, ""
        if feasible:
            # self(nu) decreases monotonically in nu; bracket S_ref
            lo, hi = nu_min, 1.0
            for _ in range(80):
                if self_of(solve_nu(hi)) <= S_ref:
                    break
                hi *= 2.0
            if self_of(solve_nu(lo)) < S_ref:
                note = " (S_ref above the family's attainable range at nu_min)"
                chosen = solve_nu(lo); nu_used = lo
            else:
                for _ in range(120):
                    mid = 0.5 * (lo + hi)
                    if self_of(solve_nu(mid)) > S_ref:
                        lo = mid
                    else:
                        hi = mid
                nu_used = 0.5 * (lo + hi)
                chosen = solve_nu(nu_used)
        else:
            chosen = Sinv @ x * (X_ref / xSx)
            note = " (min-self point; S_ref unreachable from below)"
        f = torch.zeros(shape, dtype=torch.float64, device=device)
        for j in range(m):
            f += float(chosen[j]) * flds[(bname, j)]
        cr = float((f * phi).sum() * dV)
        se = 0.5 * float((f * poisson_phi_periodic(f, cell64)).sum() * dV)
        am = float(f.abs().sum() * dV)
        r2 = y2 - 2 * float(chosen @ bb) + float(chosen @ (M @ chosen))
        r2d = y2d - 2 * float(chosen @ bbd) + float(chosen @ (Md @ chosen))
        rms = math.sqrt(max(r2, 0.0) / NPTS) / rms_all_ref
        rmsd = math.sqrt(max(r2d, 0.0) / nd) / rms_deep_ref
        # plain unweighted L2 solution = family vector 0, the natural yardstick
        e0 = torch.zeros(m, dtype=torch.float64, device=device); e0[0] = 1.0
        r2b = y2 - 2 * float(e0 @ bb) + float(e0 @ (M @ e0))
        rms_base = math.sqrt(max(r2b, 0.0) / NPTS) / rms_all_ref
        ok = (abs(cr / X_ref - 1) < 0.05 and abs(se / S_ref - 1) < 0.15
              and abs(am / Q_ref - 1) < 0.20 and rms <= 1.5 * rms_base)
        np.savez(f"jss_{sid}_{bname.replace('+','p').replace('^','')}.npz",
                 M=M.cpu().numpy(), b=bb.cpu().numpy(), S=Ssym.cpu().numpy(),
                 x=x.cpu().numpy(), alpha=chosen.cpu().numpy(),
                 coeffs=torch.stack(fam[bname]).cpu().numpy(),
                 refs=np.array([X_ref, S_ref, Q_ref, rms_all_ref, rms_deep_ref]))
        print(f"\n  basis {bname}: family {m} vectors, self-Gram eigenvalues "
              f"{float(ev.min()):.2e} … {float(ev.max()):.2e}")
        print(f"    min attainable self at cross = ref : {min_self:+.4f} "
              f"({min_self/S_ref:.3f} x ref)  -> "
              f"{'feasible' if feasible else 'INFEASIBLE in this family'}")
        print(f"    chosen point: rms/ref {rms:.3f} (plain fit {rms_base:.3f}, "
              f"cap {1.5*rms_base:.3f})  rms_deep/ref {rmsd:.3f}  "
              f"cross/ref {cr/X_ref:.3f}  self/ref {se/S_ref:.3f}  "
              f"|q|/ref {am/Q_ref:.3f}   nu "
              f"{('%+.4e' % nu_used) if nu_used is not None else 'n/a'}{note}")
        print(f"    VERDICT for basis {bname} on {tag}: "
              f"{'PASS (all four conditions)' if ok else 'FAIL'}", flush=True)
        summary.append((tag, bname, min_self / S_ref, rms, rms_base, rmsd,
                        cr / X_ref, se / S_ref, am / Q_ref, ok))
        del f
    del flds, fam, small, n_lat, phi, env_b, s_ion, s_diel, env_sq, CH
    import gc; gc.collect(); torch.cuda.empty_cache()

print("\n=== summary (four conditions: cross within 5%, self within 15%, "
      "|q| within 20%, point rms not blown up) ===")
print(f"{'frame':>15} {'basis':>11} {'minself/ref':>12} {'rms/ref':>8} "
      f"{'plain':>7} {'deep':>7} {'cross':>7} {'self':>7} {'|q|':>7} {'verdict':>8}")
for r in summary:
    print(f"{r[0]:>15} {r[1]:>11} {r[2]:12.3f} {r[3]:8.3f} {r[4]:7.3f} "
          f"{r[5]:7.3f} {r[6]:7.3f} {r[7]:7.3f} {r[8]:7.3f} "
          f"{'PASS' if r[9] else 'FAIL':>8}")
print("DONE  four conditions: cross within 5%, self within 15%, |q| within "
      "20%, point rms <= 1.5x the plain fit. A PASS is constructive for that "
      "basis and frame; a FAIL speaks for this 8-vector family only. On "
      "neutral frames the ion channel is gated off, exactly as in the model.")
