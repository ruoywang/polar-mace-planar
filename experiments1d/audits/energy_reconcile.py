"""Unified energy reconciliation (user directive 2026-09-09), six frames.

PART A — model-side self-consistency, ONE Coulomb function for everything.
Synthesise the complete model solvent charge on the solver grid,
    rho_full(r) = rho_1d(z)  [plane profile, broadcast]  +  delta(r)  [residual]
and evaluate, in one call, in the model's own solute potential
(phi = -(cvhar3 - mean), i.e. G0 dropped exactly as every existing term does):
    coulomb(rho, phi) = int rho*phi dV  +  0.5*int rho*phi[rho] dV
Compare that single number against the sum of the terms the model actually
adds to its energy:  comp (1-D compensation) + E_bl (baseline coupling)
+ E_3d (e_xsol + e_self).  Then break the one-shot number into the five
pieces the identity predicts, so a mismatch is diagnosable rather than just
a pass/fail:
    (1) int rho_1d*phi        should equal  comp_cross + E_bl
    (2) 0.5 int rho_1d*phi[rho_1d]   should equal  comp_self
    (3) int delta*phi         should equal  e_xsol
    (4) 0.5 int delta*phi[delta]     should equal  e_self
    (5) int rho_1d*phi[delta]        should be 0 by the per-plane projection
Excluded on BOTH sides, by construction and on purpose: the G=0 (monopole)
piece, and the slab dipole correction — those are a separate bookkeeping
layer whose audit is still open.

PART B — DFT-side reproduction of the VASPsol++ nonlinear-PB free energy on
the SAME six frames (NiN88 has never been validated before), reported as its
separate pieces: q*phi0, electrostatic, dielectric lambda, ionic lambda, and
A_cav, against the values the DFT run printed. These are NOT comparable to
the Coulomb-only number of Part A and are never added on top of it.

cwd = gate_bl.  Usage: energy_reconcile.py
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
import mace.modules.extensions as ext
import mace.modules.pb1d_backend as PB
from pure_python import torch_pb as tp
from mace.modules.solvent3d import poisson_phi_periodic, FIELD_CONSTANT_EVA

GCE44 = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single") + "/1-44_GCE"
GCE88 = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single") + "/2-88_GCE"
NEU = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single") + "/5-44_neutral_withsolv"
FRAMES = [
    (61, f"{GCE44}/cal_61", "NiN44 q=-0.80", 660.0),
    (1, f"{GCE44}/cal_1", "NiN44 q=-1.00", 660.0),
    (28, f"{GCE44}/cal_28", "NiN44 q=-1.32", 660.0),
    (201, f"{GCE88}/cal_1", "NiN88 q=-1.00", 1012.0),
    (353, f"{GCE88}/cal_153", "NiN88 q=-1.31", 1012.0),
    (601, f"{NEU}/cal_1", "neutral", 660.0),
]
N_MOL = 0.0335
A_K = 0.125
EDEPS = 4.0 * math.pi * 2.0 * 13.605826 * 0.529177249

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
_comp = ext._slab_compensation_periodic_1d_energy_radial
ext._slab_compensation_periodic_1d_energy_radial = (
    lambda *a, **k: cap.__setitem__("comp", _comp(*a, **k).detach().clone())
    or cap["comp"])
_clo = PB.closure_from_fields
def wrap_clo(n_e, cv, grid, params, tp_):
    cap["cvhar3"] = cv.detach().clone()
    cap["grid"] = grid
    return _clo(n_e, cv, grid, params, tp_)
PB.closure_from_fields = wrap_clo
_solve = PB.PB1DBackend.solve_graph
def wrap_solve(self, *a, **k):
    out = _solve(self, *a, **k)
    cap["rho_layer_z"] = out["rho_layer_z"].detach().clone()
    if out.get("s3d_obs") is not None:
        o = out["s3d_obs"]
        cap["delta"] = (o["d_sup_b"] + o["d_sup_i"]).detach().clone()
        cap["e_xsol"] = o["e_xsol"]
        cap["e_self"] = o["e_self"]
    cap["params"] = self.params
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
want = {sid: None for sid, _, _, _ in FRAMES}
for split in ("train", "val"):
    for a in read(f"data/{split}.xyz", ":"):
        sid = int(a.info.get("sample_id", -1))
        if sid in want and want[sid] is None:
            want[sid] = a
    if all(v is not None for v in want.values()):
        break


# ---------------------------------------------------------------- helpers
def coulomb(rho, phi_ext, cell, dV):
    """THE Coulomb energy function: cross with an external potential plus the
    charge's own half self-energy, same G0-dropped periodic convention as
    every existing model term."""
    cross = float((rho * phi_ext).sum() * dV)
    selfe = 0.5 * float((rho * poisson_phi_periodic(rho, cell)).sum() * dV)
    return cross, selfe


def resample_fourier(prof, n_out):
    """Band-limited resample of a periodic profile, mean preserved."""
    n_in = prof.shape[0]
    if n_in == n_out:
        return prof
    g = torch.fft.fft(prof)
    out = torch.zeros(n_out, dtype=g.dtype, device=g.device)
    h = min(n_in, n_out) // 2
    out[:h] = g[:h]
    out[-h:] = g[-h:]
    return torch.fft.ifft(out).real * (n_out / n_in)


def phi1d(rho_z, dz):
    n = rho_z.shape[0]
    g = 2 * math.pi * torch.fft.fftfreq(n, d=dz).to(rho_z.device)
    rg = torch.fft.fft(rho_z)
    pg = torch.zeros_like(rg)
    m = g.abs() > 1e-14
    pg[m] = 4 * math.pi * FIELD_CONSTANT_EVA * rg[m] / (g[m] ** 2)
    out = torch.fft.ifft(pg).real
    return out - out.mean()


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
    import re
    return float(re.findall(rf"{name}\s*=\s*([-\d.Ee+]+)", txt)[-1])


def g_rot(x):
    out = torch.ones_like(x)
    m = x >= 2e-4
    xm = x[m]
    out[m] = 3.0 * (xm - torch.tanh(xm)) / (xm ** 2 * torch.tanh(xm))
    return out


# =============================================== PART A: model reconciliation
partA, partB = [], []
for sid, dftdir, tag, nelect0 in FRAMES:
    a = want[sid]
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    cap.clear()
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
    grid = cap["grid"]
    cell64 = torch.as_tensor(grid.cell, dtype=torch.float64, device=device)
    shape = tuple(grid.shape)
    V = grid.volume
    dV = V / float(np.prod(shape))
    nz = shape[2]; lz = float(cell64[2, 2]); dz = lz / nz; area = V / lz
    comp = float(cap["comp"].sum())
    ebl = float(pred["baseline_coupling_energy_g"].sum())
    e3d = float(pred["solvent3d_energy_g"].sum())
    exsol, eself = cap["e_xsol"], cap["e_self"]
    phi = -(cap["cvhar3"] - cap["cvhar3"].mean())          # solute potential
    delta = cap["delta"].to(device)
    rho1z = resample_fourier(cap["rho_layer_z"].to(device), nz)
    rho1 = rho1z[None, None, :].expand(shape).contiguous()
    rho_full = rho1 + delta

    # one call, complete charge
    c_all, s_all = coulomb(rho_full, phi, cell64, dV)
    e_one = c_all + s_all
    # the five predicted pieces, same function / same convention
    c_1d, s_1d = coulomb(rho1, phi, cell64, dV)
    c_dl, s_dl = coulomb(delta, phi, cell64, dV)
    x_1d_dl = float((rho1 * poisson_phi_periodic(delta, cell64)).sum() * dV)
    # comp's own 1-D-grid convention, for a discretisation cross-check
    s_1d_512 = 0.5 * float((resample_fourier(cap["rho_layer_z"].to(device), 512)
                            * phi1d(resample_fourier(cap["rho_layer_z"].to(device), 512),
                                    lz / 512)).sum() * area * (lz / 512))
    total_terms = comp + ebl + e3d
    partA.append((tag, e_one, comp, ebl, e3d, total_terms, e_one - total_terms,
                  c_1d, s_1d, c_dl, s_dl, x_1d_dl, exsol, eself, s_1d_512,
                  float(rho_full.sum() * dV), float(delta.sum() * dV)))
    del rho1, rho_full, delta, phi
    torch.cuda.empty_cache()

    # ========================================== PART B: DFT reproduction
    txt = open(f"{dftdir}/log.out").read()
    kT = par(txt, "invBETA"); PBETA = par(txt, "PBETA"); ZBETA = par(txt, "ZBETA")
    a0r = par(txt, "alpha0_rot"); a_pol = par(txt, "alpha_pol")
    sic = par(txt, "invalpha_sic"); n_max = par(txt, "n_max")
    theta_b = par(txt, "theta_b")
    import re as _re
    a_corr = [float(x) for x in _re.findall(r"A_corr=\s*([-\d.Ee+]+)", txt)][-1]
    a_cav_p = [float(x) for x in _re.findall(r"A_cav:\s*([-\d.Ee+]+)", txt)][-1]
    target = a_corr - a_cav_p
    latd, phi_f = read_grid(f"{dftdir}/PHI")
    _, rb = read_grid(f"{dftdir}/RHOB")
    _, ri = read_grid(f"{dftdir}/RHOION")
    _, chg = read_grid(f"{dftdir}/CHGCAR")
    Vd = float(abs(np.linalg.det(latd)))
    celld = torch.tensor(latd, device=device)
    shd = tuple(phi_f.shape)
    dVd = Vd / phi_f.numel()
    phi_f = phi_f.to(device); rb = rb.to(device); ri = ri.to(device)
    n_solv_file = (rb + ri) / Vd
    q_sol = float(n_solv_file.sum() * dVd)          # solute net charge
    ne = torch.clamp(chg.to(device) / Vd, min=0.0)
    grid_l = tp.TorchGrid(latd, shd, device=str(device), dtype=torch.float64,
                          rspec=True)
    s_ion3, s_diel3, _ = tp.create_cavity_torch(ne, grid_l, cap["params"])
    phis_d = poisson_phi_periodic(n_solv_file, celld)
    t_es = -0.5 * float(((phis_d - phis_d.mean())
                         * (n_solv_file - n_solv_file.mean())).sum() * dVd)
    t_q = q_sol * (-float(phi_f.mean()))
    # E_loc for the dielectric lambda
    B = 2 * math.pi * torch.linalg.inv(celld).T
    fx = torch.fft.fftfreq(shd[0], device=device) * shd[0]
    fy = torch.fft.fftfreq(shd[1], device=device) * shd[1]
    fz = torch.arange(shd[2] // 2 + 1, dtype=torch.float64, device=device)
    G = (fx[:, None, None, None] * B[0] + fy[None, :, None, None] * B[1]
         + fz[None, None, :, None] * B[2])
    G2 = (G * G).sum(-1)
    phig = torch.fft.rfftn(phi_f)
    W = torch.exp(-0.5 * G2 * A_K ** 2)
    E0 = torch.stack([torch.fft.irfftn(1j * G[..., j] * (-W * phig), s=shd)
                      for j in range(3)])
    E0m = torch.sqrt((E0 ** 2).sum(0) + 1e-300)
    del E0, G, phig, W
    x0 = PBETA * E0m
    lo = torch.full_like(x0, 1.0 / (1.0 - a_pol * sic))
    hi = torch.full_like(x0, 1.0 / (1.0 - (a_pol + a0r) * sic) * (1.0 + 4e-8))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        F = 1.0 / (1.0 - (g_rot(mid * x0) * a0r + a_pol) * sic) - mid
        lo = torch.where(F > 0, mid, lo)
        hi = torch.where(F > 0, hi, mid)
    Em = 0.5 * (lo + hi) * E0m
    del lo, hi, x0, E0m
    x = PBETA * Em
    lam_d = torch.where(
        x < 2e-4, -kT / 6.0 * x ** 2,
        torch.where(x > 100.0, -kT * (x - torch.log(2 * x.clamp(min=1e-30))),
                    -kT * torch.log(torch.sinh(x.clamp(max=100.0))
                                    / x.clamp(min=1e-30))))
    n_work = (a0r * g_rot(x) + a_pol) / EDEPS
    lam_d = lam_d - 0.5 * a_pol / EDEPS * Em ** 2 \
        + 0.5 * EDEPS * sic * (n_work * Em) ** 2
    t_diel = N_MOL * float((s_diel3 * lam_d).sum() * dVd)
    X = ZBETA * phi_f
    aX = X.abs()
    lam_i = torch.where(
        aX > 100.0, math.log(0.5 * theta_b) + aX,
        torch.where(aX < math.sqrt(theta_b) * 2e-4, 0.5 * theta_b * X ** 2,
                    torch.log(1.0 + theta_b * (torch.cosh(X.clamp(-100, 100)) - 1.0))))
    t_ion = n_max * float((s_ion3 * (-kT * lam_i)).sum() * dVd)
    a_solv = t_q + t_es + t_diel + t_ion
    nelect = None
    with open(f"{dftdir}/OUTCAR") as fh:
        for ln in fh:
            m = _re.search(r"NELECT\s*=\s*([-\d.Ee+]+)", ln)
            if m:
                nelect = float(m.group(1)); break
    partB.append((tag, q_sol, nelect0 - nelect if nelect else float("nan"),
                  t_q, t_es, t_diel, t_ion, a_solv, target, a_solv - target,
                  a_cav_p))
    del phi_f, rb, ri, chg, ne, grid_l, s_ion3, s_diel3, phis_d, n_solv_file
    del lam_d, lam_i, Em, n_work, x, X, aX
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"  [{tag}] done", flush=True)

# ---------------------------------------------------------------- output
print("\n=== PART A: model-side reconciliation (eV per frame, G0 dropped, "
      "slab dipole excluded on both sides) ===")
print(f"{'frame':>15} {'one-shot':>9} | {'comp':>8} {'E_bl':>7} {'E_3d':>7} "
      f"{'sum':>9} | {'one - sum':>10} | {'int rho_full':>12}")
for r in partA:
    print(f"{r[0]:>15} {r[1]:+9.4f} | {r[2]:+8.4f} {r[3]:+7.4f} {r[4]:+7.4f} "
          f"{r[5]:+9.4f} | {r[6]:+10.5f} | {r[15]:+12.2e}")
print("\n=== PART A detail: the five predicted pieces ===")
print(f"{'frame':>15} {'(1) rho1*phi':>13} {'(2) rho1 self':>13} "
      f"{'(3) d*phi':>10} {'e_xsol':>9} {'(4) d self':>10} {'e_self':>9} "
      f"{'(5) rho1*phi[d]':>16} {'comp_self(512)':>14}")
for r in partA:
    print(f"{r[0]:>15} {r[7]:+13.4f} {r[8]:+13.4f} {r[9]:+10.4f} {r[12]:+9.4f} "
          f"{r[10]:+10.4f} {r[13]:+9.4f} {r[11]:+16.2e} {r[14]:+14.4f}")
print("\n=== PART B: DFT free-energy reproduction, pieces listed separately "
      "(NOT comparable to Part A) ===")
print(f"{'frame':>15} {'q_sol':>7} {'q(NELECT)':>10} | {'q*phi0':>8} "
      f"{'elec':>8} {'diel':>8} {'ion':>8} | {'A_solv':>9} {'printed':>9} "
      f"{'diff':>9} | {'A_cav':>8}")
for r in partB:
    print(f"{r[0]:>15} {r[1]:+7.3f} {r[2]:+10.3f} | {r[3]:+8.3f} {r[4]:+8.3f} "
          f"{r[5]:+8.3f} {r[6]:+8.3f} | {r[7]:+9.4f} {r[8]:+9.4f} "
          f"{r[9]:+9.5f} | {r[10]:+8.3f}")
print("DONE")
