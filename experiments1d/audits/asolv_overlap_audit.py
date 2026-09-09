"""Overlap audit (step 2 of the user plan): decompose the DFT-validated
A_solv into the pieces the MODEL already carries vs the net-missing rest.

Model's existing 1-D compensation energy (extensions.py, comp_profile path):
    E_comp = E_int[n_sol, n_solv] + 0.5*E_selfpair[n_solv]   (1-D, G/=0)
A_solv electrostatics (validated):
    A_elec = q_sol*phi0_solv - E_self[n_solv]
At exact linear response E_int = -2*E_self, so E_comp == A_elec - q*phi0.
This script MEASURES that identity's violation and the 1-D-vs-3-D loss:
  3-D (DFT fields): E_self3, E_int3 (n_sol = spectral n_tot - n_solv),
       ratio E_int/(-2 E_self), A_elec3, lambda_diel3, lambda_ion3
  1-D (plane-averaged same fields, same formulas): E_self1, E_int1,
       lambda_ion1 (from phi(z)), and the lateral losses per term.
Signs: PHI/RHOB/RHOION are stored in the electron-energy convention
(= -physical); physical fields used throughout here.
Usage: asolv_overlap_audit.py <cal_dir> [...]
"""
# Paths come from env vars with the LS6 values as defaults, so this
# runs unchanged on Lonestar6 and on a machine that holds the same
# payload elsewhere: KIT_PB_REPO, KIT_MACE_REPO, KIT_DFT,
# KIT_PB_CONFIG. Previously hardcoded, which made every audit
# script non-portable (found by the workstation session 2026-09-09).
import math
import re
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get("KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get("KIT_MACE_REPO", "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))

from pure_python import torch_pb as tp
import mace.modules.pb1d_backend as PB
from mace.modules.solvent3d import poisson_phi_periodic, FIELD_CONSTANT_EVA

torch.set_default_dtype(torch.float64)
N_MOL = 0.0335
NELECT0 = 660.0

backend = PB.PB1DBackend(
    config_path=os.environ.get("KIT_PB_CONFIG", "cal1_train.json"),
    repo_path=os.environ.get("KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"),
    baseline_cache=None, fixsol_steps=2)
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
    return lat, torch.tensor(np.ascontiguousarray(
        np.array(vals[:need]).reshape(nz, ny, nx).transpose(2, 1, 0)))


def par(txt, name):
    return float(re.findall(rf"{name}\s*=\s*([-\d.Ee+]+)", txt)[-1])


def phi1d(rho_z, lz, area):
    """1-D periodic Poisson (zero-mean), rho_z in e/A^3 on uniform z grid."""
    n = rho_z.shape[0]
    g = 2.0 * math.pi * torch.fft.fftfreq(n, d=lz / n)
    rg = torch.fft.fft(rho_z)
    pg = torch.zeros_like(rg)
    m = g.abs() > 1e-14
    pg[m] = 4.0 * math.pi * FIELD_CONSTANT_EVA * rg[m] / (g[m] ** 2)
    out = torch.fft.ifft(pg).real
    return out - out.mean()


for cal in sys.argv[1:]:
    txt = open(f"{cal}/log.out").read()
    kT = par(txt, "invBETA"); ZBETA = par(txt, "ZBETA")
    n_max = par(txt, "n_max"); theta_b = par(txt, "theta_b")
    nelect = None
    with open(f"{cal}/OUTCAR") as fh:
        for ln in fh:
            m = re.search(r"NELECT\s*=\s*([-\d.Ee+]+)", ln)
            if m:
                nelect = float(m.group(1)); break
    q_sol = NELECT0 - nelect

    lat, phi_f = read_grid(f"{cal}/PHI")
    _, rb = read_grid(f"{cal}/RHOB")
    _, ri = read_grid(f"{cal}/RHOION")
    V = float(abs(np.linalg.det(lat)))
    cell64 = torch.tensor(lat)
    shape = tuple(phi_f.shape)
    dV = V / phi_f.numel()
    lz = float(lat[2][2]); area = V / lz
    nz = shape[2]; dz = lz / nz

    phi = -phi_f                      # physical electrostatic potential, eV
    n_solv = -(rb + ri) / V           # physical solvent charge, e/A^3

    # total charge from the potential (spectral), then solute = total - solv
    import math as _m
    B = 2.0 * _m.pi * torch.linalg.inv(cell64).T
    fx = torch.fft.fftfreq(shape[0]) * shape[0]
    fy = torch.fft.fftfreq(shape[1]) * shape[1]
    fz = torch.arange(shape[2] // 2 + 1, dtype=torch.float64)
    G = (fx[:, None, None, None] * B[0] + fy[None, :, None, None] * B[1]
         + fz[None, None, :, None] * B[2])
    G2 = (G * G).sum(-1)
    n_tot = torch.fft.irfftn(
        torch.fft.rfftn(phi) * G2 / (4.0 * _m.pi * FIELD_CONSTANT_EVA), s=shape)
    n_sol = n_tot - n_solv
    print(f"\n== {cal.split('/')[-2]}/{cal.split('/')[-1]}  q_sol {q_sol:+.2f}  "
          f"(check: int n_sol = {float(n_sol.sum()*dV):+.3f}, "
          f"int n_solv = {float(n_solv.sum()*dV):+.3f})", flush=True)

    # ---- 3-D electrostatics (G/=0 pieces; G0 bookkeeping = q*phi0) ----
    phis = poisson_phi_periodic(n_solv, cell64)
    e_self3 = 0.5 * float((n_solv * phis).sum() * dV)
    e_int3 = float((n_sol * phis).sum() * dV)
    phi0 = float(phi.mean()) - 0.0    # <phi_sol>=0 convention (validated)
    a_elec3 = q_sol * phi0 - e_self3
    print(f"  3D: E_self {e_self3:+.4f}  E_int {e_int3:+.4f}  "
          f"ratio E_int/(-2E_self) = {e_int3 / (-2*e_self3):.4f}", flush=True)
    print(f"  3D: E_comp-analog (int+self) = {e_int3 + e_self3:+.4f}   "
          f"A_elec (q*phi0 - E_self) = {a_elec3:+.4f}  (q*phi0 {q_sol*phi0:+.4f})",
          flush=True)

    # ---- 1-D (plane-averaged fields, same integrals) ----
    ns1 = n_solv.mean(dim=(0, 1))
    nn1 = n_sol.mean(dim=(0, 1))
    ph1 = phi1d(ns1, lz, area)
    e_self1 = 0.5 * float((ns1 * ph1).sum() * area * dz)
    e_int1 = float((nn1 * ph1).sum() * area * dz)
    print(f"  1D: E_self {e_self1:+.4f}  E_int {e_int1:+.4f}  "
          f"(lateral loss: self {e_self3-e_self1:+.4f}, int {e_int3-e_int1:+.4f})",
          flush=True)

    # ---- ionic lambda: 3-D vs 1-D(plane-mean phi) ----
    ne = torch.clamp(read_grid(f"{cal}/CHGCAR")[1] / V, min=0.0)
    grid_l = tp.TorchGrid(lat, shape, device="cpu", dtype=torch.float64, rspec=True)
    s_ion3, s_diel3, _ = tp.create_cavity_torch(ne, grid_l, p)

    def lam_ion_of(phi_x):
        X = ZBETA * (-phi_x)   # their lambda used file-sign phi; even anyway
        aX = X.abs()
        lam = torch.where(
            aX > 100.0, math.log(0.5 * theta_b) + aX,
            torch.where(aX < math.sqrt(theta_b) * 2e-4, 0.5 * theta_b * X ** 2,
                        torch.log(1.0 + theta_b * (torch.cosh(
                            X.clamp(-100, 100)) - 1.0))))
        return -kT * lam

    t_ion3 = n_max * float((s_ion3 * lam_ion_of(phi)).sum() * dV)
    s_ion1 = s_ion3.mean(dim=(0, 1))
    phz = phi.mean(dim=(0, 1))
    t_ion1 = n_max * float((s_ion1 * lam_ion_of(phz)).sum() * area * dz)
    print(f"  ion lambda: 3D {t_ion3:+.4f}   1D(plane-mean fields) {t_ion1:+.4f}",
          flush=True)
    del phi_f, rb, ri, phi, n_solv, n_tot, n_sol, phis, ne, grid_l, s_ion3, s_diel3
    import gc; gc.collect()
print("DONE")
