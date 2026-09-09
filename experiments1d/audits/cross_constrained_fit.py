"""Cross-energy-constrained refit on the EXISTING basis (user protocol
2026-09-08): does the current basis+projection have to give a too-small
cross energy, or does the point loss simply not ask for it?

Setup (same basis, same per-z-plane conserving projection, same supervision
operator as the loss):
    delta(c) = sum_k c_k * proj[env * G_k]
    point objective  ||A c - y||^2 + lam |c|^2      (A = proj basis at points)
    cross(c)  = g^T c ,  g_k = int proj_k * (-(cvhar - mean)) dV   [LINEAR]
    self(c)   = 0.5 int delta(c) * phi[delta(c)] dV                [QUADRATIC]
Constrained solution family (exact, analytic in the multiplier mu):
    c(mu) = c_unc - mu * H^{-1} g ,  H = A^T A + lam I
    cross(mu) = g^T c_unc - mu * (g^T H^{-1} g)       -> hits any target
    ||A c - y||^2 , |c|^2 , self  are all quadratic in mu
so the whole trade-off curve costs 2 field builds + 3 Poisson solves.

POTENTIAL CONVENTION (the mistake this script fixes): the fitted cross
energy uses the MODEL potential -(cvhar3-mean), exactly as the energy path
does. The same-potential reference is therefore n_lat_DFT x phi_model
(-2.333 charged / -0.566 neutral, mixed_cross_audit 2026-09-08), NOT the
DFT-potential value (-2.659 / -1.211). Both are marked on the curve.

cwd = gate_bl. Usage: cross_constrained_fit.py
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
    normalized_gradient_envelope,
    poisson_phi_periodic,
    _interp3_periodic,
    load_solvent3d_targets,
    attach_solvent3d_samples_to_batch,
)

RIDGE = 1.0e-7
NPTS = 120000
# same-potential references (model potential), from the 2x2 matrix
REF_SAME_POT = {1: -2.333, 601: -0.566}
REF_DFT_POT = {1: -2.659, 601: -1.211}
TRAINED = {1: -1.196, 601: -1.015}

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
def wrap_clo(n_e, cv, grid, params, tp):
    cap["cvhar3"] = cv.detach().clone()
    cap["grid"] = grid
    return _clo(n_e, cv, grid, params, tp)
PB.closure_from_fields = wrap_clo

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff",
               "solvated": "solvated"},
    arrays_keys={"forces": "forces"},
)


class StubLoss:
    solvent3d_targets = load_solvent3d_targets("data/solvent3d_points_manifest.json")
    solvent3d_samples = NPTS
    solvent3d_rng = None


SIG_MS = StubLoss.solvent3d_targets.signal_ms
want = {1: None, 601: None}
for a in read("data/train.xyz", ":"):
    sid = int(a.info.get("sample_id", -1))
    if sid in want and want[sid] is None:
        want[sid] = a
    if all(v is not None for v in want.values()):
        break

sigmas = [float(s) for s in (model.solvent3d_sigmas.tolist()
                             if torch.is_tensor(model.solvent3d_sigmas)
                             else model.solvent3d_sigmas)]

for sid, a in want.items():
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    attach_solvent3d_samples_to_batch(b, StubLoss)
    cap.clear()
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
    grid = cap["grid"]
    cell64 = torch.as_tensor(grid.cell, dtype=torch.float64, device=device)
    shape = tuple(grid.shape)
    V = grid.volume
    dV = V / float(np.prod(shape))
    env = normalized_gradient_envelope(grid._solv3d_cavity[1], cell64)
    envm = torch.clamp(env.mean(dim=(0, 1)), min=1e-12)
    phi_m = -(cap["cvhar3"] - cap["cvhar3"].mean())   # model potential

    valid = pred["solv3d_valid"]
    pts = b.solv3d_points[valid].to(torch.float64)
    y = (b.solv3d_ref_b[valid].to(torch.float64)
         - pred["solv3d_base_b"][valid].to(torch.float64))
    P = pts.shape[0]
    frac = torch.remainder(pts @ torch.linalg.inv(cell64), 1.0)
    nat = len(a)
    pos_frac = torch.remainder(
        torch.as_tensor(a.get_positions(), device=device, dtype=torch.float64)
        @ torch.linalg.inv(cell64), 1.0)
    beobj = model._pb1d_backend
    K = nat * len(sigmas) * 9
    print(f"\n== sid {sid} (q={float(a.info['total_charge']):+.2f})  P={P}  K={K}",
          flush=True)

    A = torch.zeros(P, K, dtype=torch.float32, device=device)
    g = torch.zeros(K, dtype=torch.float64, device=device)
    col = 0
    cvec = torch.zeros(nat, len(sigmas), 9, dtype=torch.float64, device=device)
    for ia in range(nat):
        for isig in range(len(sigmas)):
            for lm in range(9):
                cvec.zero_()
                cvec[ia, isig, lm] = 1.0
                Gk = grid.ifft_real(beobj._gto_net_density_g(
                    grid, pos_frac, cvec, sigmas)) / V
                raw = env * Gk
                rk = raw.mean(dim=(0, 1)) / envm
                proj = raw - rk[None, None, :] * env
                A[:, col] = _interp3_periodic(proj, frac).to(torch.float32)
                g[col] = (proj * phi_m).sum() * dV
                col += 1
        if ia % 40 == 0:
            print(f"   ... atom {ia}/{nat}", flush=True)

    Af = A.to(torch.float64)
    del A
    H = Af.T @ Af
    lam = RIDGE * float(torch.diagonal(H).mean())
    H += lam * torch.eye(K, dtype=torch.float64, device=device)
    L = torch.linalg.cholesky(H)
    Aty = Af.T @ y
    c_unc = torch.cholesky_solve(Aty.unsqueeze(1), L).squeeze(1)
    d_dir = torch.cholesky_solve(g.unsqueeze(1), L).squeeze(1)   # H^{-1} g
    gHg = float(g @ d_dir)
    r_unc = Af @ c_unc - y
    Ad = Af @ d_dir
    del Af, H

    # fields for the two coefficient vectors (delta = du - mu*dd)
    def build(cv_all):
        out = torch.zeros(shape, dtype=torch.float64, device=device)
        col = 0
        buf = torch.zeros(nat, len(sigmas), 9, dtype=torch.float64, device=device)
        for ia in range(nat):
            for isig in range(len(sigmas)):
                buf.zero_()
                nz = False
                for lm in range(9):
                    v = float(cv_all[col]); col += 1
                    if abs(v) > 1e-14:
                        buf[ia, isig, lm] = v
                        nz = True
                if nz:
                    Gk = grid.ifft_real(beobj._gto_net_density_g(
                        grid, pos_frac, buf, sigmas)) / V
                    raw = env * Gk
                    rk = raw.mean(dim=(0, 1)) / envm
                    out += raw - rk[None, None, :] * env
        return out

    du = build(c_unc)
    dd = build(d_dir)
    phu = poisson_phi_periodic(du, cell64)
    phd = poisson_phi_periodic(dd, cell64)
    s_uu = float((du * phu).sum() * dV)
    s_ud = float((du * phd).sum() * dV)
    s_dd = float((dd * phd).sum() * dV)
    cross_unc = float(g @ c_unc)
    sig = math.sqrt(SIG_MS["b"])
    rms0 = float(torch.sqrt((y ** 2).mean()))

    print(f"  unconstrained: cross {cross_unc:+.3f}  point rms "
          f"{float(torch.sqrt((r_unc**2).mean())):.3e}  "
          f"self {0.5*s_uu:+.3f}  |c|max {float(c_unc.abs().max()):.2e}",
          flush=True)
    print(f"  markers: same-potential ref {REF_SAME_POT[sid]:+.3f}   "
          f"DFT-potential ref {REF_DFT_POT[sid]:+.3f}   trained model "
          f"{TRAINED[sid]:+.3f}   (g^T H^-1 g = {gHg:.3e})", flush=True)
    targets = sorted({round(cross_unc, 3), TRAINED[sid], REF_SAME_POT[sid],
                      REF_DFT_POT[sid], -1.5, -2.0, -3.0})
    print(f"  {'target':>8} {'mu':>10} {'point rms':>11} {'ratio':>7} "
          f"{'/signal':>8} {'self':>8} {'|c|max':>10} {'|c|rms':>10} "
          f"{'int|d|':>7}")
    for tgt in targets:
        mu = (cross_unc - tgt) / gHg if gHg != 0 else 0.0
        r2 = float((r_unc ** 2).sum()) - 2 * mu * float(r_unc @ Ad) \
            + mu * mu * float((Ad ** 2).sum())
        rms = math.sqrt(max(r2, 0.0) / P)
        selfe = 0.5 * (s_uu - 2 * mu * s_ud + mu * mu * s_dd)
        cvv = c_unc - mu * d_dir
        dmix = du - mu * dd
        print(f"  {tgt:+8.3f} {mu:+10.3e} {rms:11.3e} {rms/rms0:7.3f} "
              f"{rms/sig:8.3f} {selfe:+8.3f} {float(cvv.abs().max()):10.2e} "
              f"{float(cvv.pow(2).mean().sqrt()):10.2e} "
              f"{float(dmix.abs().sum()*dV):7.3f}", flush=True)
    del du, dd, phu, phd
print("DONE")
