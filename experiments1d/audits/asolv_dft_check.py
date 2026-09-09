"""Recompute VASPsol++ A_solv from a cal's own DFT fields; validate against
the log-printed A_corr - A_cav (the user-endorsed step before wiring any new
energy term). NLPB functional (solvation.F Update_NLPB, source-read):

  A_solv = q_sol*<phi_solv>_G0
         - 0.5*<phi_solv, L0(phi_solv)>           [= n_b+n_ion at convergence]
         + n_mol * int S_diel*lambda_diel dV
         + n_max * int S_ion *lambda_ion  dV

Fields used: PHI (total phi, eV), RHOB/RHOION (solvent charge * cell), CHGCAR
(electron density * cell, for the cavity chain). phi_solv(G/=0) is rebuilt by
Poisson from RHOB+RHOION; its G0 is taken as mean(PHI) (phi_sol G0 = 0
convention - checked on the neutral cal where the q_sol term vanishes).
E_loc = f_loc * grad(-w_b conv phi), w_b = normalized Gaussian a_k = 0.125 A,
f_loc solves the SIC fixed point (g_rot). Parameters parsed from the cal's
own log prints. Usage: asolv_dft_check.py <cal_dir> [...]  (cwd anywhere)
"""
import math
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "/work/08384/tg876840/ls6/repos/cep-dip-python-pb")
sys.path.insert(0, "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy")

from pure_python import torch_pb as tp
import mace.modules.pb1d_backend as PB
from mace.modules.solvent3d import poisson_phi_periodic

torch.set_default_dtype(torch.float64)
EDEPS = 4.0 * math.pi * 2.0 * 13.605826 * 0.529177249  # VASP constant.inc
N_MOL = 0.0335
A_K = 0.125
NELECT0 = {"NiN44": 660.0}  # neutral valence electron count (44-water cell)

backend = PB.PB1DBackend(
    config_path="/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500/cal1_train.json",
    repo_path="/work/08384/tg876840/ls6/repos/cep-dip-python-pb",
    baseline_cache=None,
    fixsol_steps=2,
)
p = backend.params


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
    return lat, torch.tensor(
        np.ascontiguousarray(np.array(vals[:need]).reshape(nz, ny, nx)
                             .transpose(2, 1, 0)))


def par(txt, name):
    m = re.findall(rf"{name}\s*=\s*([-\d.Ee+]+)", txt)
    assert m, f"param {name} not in log"
    return float(m[-1])


def g_rot(x):
    out = torch.ones_like(x)
    m = x >= 2e-4
    xm = x[m]
    out[m] = 3.0 * (xm - torch.tanh(xm)) / (xm ** 2 * torch.tanh(xm))
    return out


for cal in sys.argv[1:]:
    txt = open(f"{cal}/log.out").read()
    kT = par(txt, "invBETA"); PBETA = par(txt, "PBETA"); ZBETA = par(txt, "ZBETA")
    a0r = par(txt, "alpha0_rot"); a_pol = par(txt, "alpha_pol")
    sic = par(txt, "invalpha_sic"); n_max = par(txt, "n_max")
    theta_b = par(txt, "theta_b")
    a_corr = [float(x) for x in re.findall(r"A_corr=\s*([-\d.Ee+]+)", txt)][-1]
    a_cav = [float(x) for x in re.findall(r"A_cav:\s*([-\d.Ee+]+)", txt)][-1]
    target = a_corr - a_cav
    nelect = None
    with open(f"{cal}/OUTCAR") as fh:
        for ln in fh:
            m = re.search(r"NELECT\s*=\s*([-\d.Ee+]+)", ln)
            if m:
                nelect = float(m.group(1))
                break
    assert nelect is not None, "NELECT not found"
    q_sol = NELECT0["NiN44"] - nelect  # VASP: extra electrons -> negative q

    lat, phi = read_grid(f"{cal}/PHI")
    _, rb = read_grid(f"{cal}/RHOB")
    _, ri = read_grid(f"{cal}/RHOION")
    _, chg = read_grid(f"{cal}/CHGCAR")
    V = float(abs(np.linalg.det(lat)))
    cell64 = torch.tensor(lat)
    shape = tuple(phi.shape)
    dV = V / phi.numel()

    # shape functions from the DFT density (validated chain: B2 hit 433.82)
    ne = torch.clamp(chg / V, min=0.0)
    grid_l = tp.TorchGrid(lat, shape, device="cpu", dtype=torch.float64, rspec=True)
    s_ion3, s_diel3, _ = tp.create_cavity_torch(ne, grid_l, p)

    # electrostatic terms from the solvent charge
    n_solv = (rb + ri) / V  # e/A^3, VASP native sign
    q_ion_net = float(n_solv.sum() * dV)
    phi_solv = poisson_phi_periodic(n_solv, cell64)  # zero-mean by helper
    t_es = -0.5 * float(((phi_solv - phi_solv.mean())
                         * (n_solv - n_solv.mean())).sum() * dV)
    # PHI is stored in the electron-energy sign convention (= -physical
    # electrostatic potential): every other A_solv term is EVEN in that sign
    # (measured: neutral cal unaffected), only this G0 term is odd. Verified
    # on charged cal_1: +sign missed the target by exactly 2*t_q.
    phi_solv_g0 = -float(phi.mean())  # <phi_sol> = 0 convention
    t_q = q_sol * phi_solv_g0

    # E_loc: f_loc * grad(-(w_b conv phi))
    nx, ny, nz = shape
    B = 2.0 * math.pi * torch.linalg.inv(cell64).T
    fx = torch.fft.fftfreq(nx) * nx
    fy = torch.fft.fftfreq(ny) * ny
    fz = torch.fft.fftfreq(nz)[: nz // 2 + 1] * nz
    fz = torch.arange(nz // 2 + 1, dtype=torch.float64)
    G = (fx[:, None, None, None] * B[0] + fy[None, :, None, None] * B[1]
         + fz[None, None, :, None] * B[2])
    G2 = (G * G).sum(-1)
    phig = torch.fft.rfftn(phi)
    W = torch.exp(-0.5 * G2 * A_K ** 2)
    E0 = torch.stack([
        torch.fft.irfftn(1j * G[..., j] * (-W * phig), s=shape)
        for j in range(3)])
    E0m = torch.sqrt((E0 ** 2).sum(0) + 1e-300)
    x0 = PBETA * E0m
    a = 1.0 / (1.0 - a_pol * sic)
    b = 1.0 / (1.0 - (a_pol + a0r) * sic) * (1.0 + 4e-8)
    lo = torch.full_like(x0, a); hi = torch.full_like(x0, b)
    for _ in range(60):  # bisection on F(f) = 1/(1-(g_rot(f x0) a0r + a_pol) sic) - f
        mid = 0.5 * (lo + hi)
        F = 1.0 / (1.0 - (g_rot(mid * x0) * a0r + a_pol) * sic) - mid
        lo = torch.where(F > 0, mid, lo)
        hi = torch.where(F > 0, hi, mid)
    f_loc = 0.5 * (lo + hi)
    Em = f_loc * E0m
    x = PBETA * Em

    # lambda_diel: rotational + polarization + SIC parts
    lam_d = torch.where(
        x < 2e-4, -kT / 6.0 * x ** 2,
        torch.where(x > 100.0, -kT * (x - torch.log(2 * x.clamp(min=1e-30))),
                    -kT * torch.log(torch.sinh(x.clamp(max=100.0))
                                    / x.clamp(min=1e-30))))
    n_work = (a0r * g_rot(x) + a_pol) / EDEPS
    lam_d = lam_d - 0.5 * a_pol / EDEPS * Em ** 2 \
        + 0.5 * EDEPS * sic * (n_work * Em) ** 2
    t_diel = N_MOL * float((s_diel3 * lam_d).sum() * dV)

    # lambda_ion (theta_b hard-sphere branch)
    X = ZBETA * phi
    aX = X.abs()
    lam_i = torch.where(
        aX > 100.0, math.log(0.5 * theta_b) + aX,
        torch.where(aX < math.sqrt(theta_b) * 2e-4, 0.5 * theta_b * X ** 2,
                    torch.log(1.0 + theta_b * (torch.cosh(
                        X.clamp(-100, 100)) - 1.0))))
    lam_i = -kT * lam_i
    t_ion = n_max * float((s_ion3 * lam_i).sum() * dV)

    a_solv = t_q + t_es + t_diel + t_ion
    print(f"{cal.split('/')[-2]}/{cal.split('/')[-1]}: q_sol {q_sol:+.2f} "
          f"q_ion(files) {q_ion_net:+.3f}", flush=True)
    print(f"   terms: q*phi0 {t_q:+.4f}  elec {t_es:+.4f}  diel {t_diel:+.4f} "
          f" ion {t_ion:+.4f}", flush=True)
    print(f"   A_solv = {a_solv:+.4f} eV   TARGET (A_corr-A_cav) = {target:+.4f}"
          f"   diff {a_solv - target:+.4f}", flush=True)
    del phi, rb, ri, chg, ne, grid_l, s_ion3, s_diel3, n_solv, phi_solv
    del E0, E0m, x0, lo, hi, f_loc, Em, x, lam_d, lam_i, G, G2, phig, W
    import gc
    gc.collect()
print("DONE")
