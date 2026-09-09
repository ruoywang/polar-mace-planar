"""Two user-ordered attribution measurements on sid 1 (charged) / 601
(neutral), pp2 ep52 weights:

A. 1-D baseline coupling (the confirmed missing term): with the MODEL's own
   fields, split the full 1-D solute x solvent cross energy
      T_full = sum rho_solv(z) * (-<cvhar3>(z)) * A dz
   into the net part already inside the compensation term and the BASELINE
   part that is missing:  E_bl = T_full - (comp - S1d_self).
   Expected if it explains the gaps: ~ -0.62 (neutral), ~ -0.23 (charged).

B. 2x2 mixed cross-energy factorization of the lateral 3-D shortfall:
      rows = {model residual delta, DFT lateral solvent charge}
      cols = {model solute potential (-cvhar3), DFT solute potential}
   Distinguishes charge-distribution error from potential error.
   DFT fields resampled to the model grid (trilinear; % level is enough).
cwd = gate_pp2_ext.
"""
# Paths come from env vars with the LS6 values as defaults, so this
# runs unchanged on Lonestar6 and on a machine that holds the same
# payload elsewhere: KIT_PB_REPO, KIT_MACE_REPO, KIT_DFT,
# KIT_PB_CONFIG. Previously hardcoded, which made every audit
# script non-portable (found by the workstation session 2026-09-09).
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
from mace.modules.solvent3d import poisson_phi_periodic, FIELD_CONSTANT_EVA
import math

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
model = torch.load(f="checkpoints/s3d_gate_pp22_run-123.model", map_location=device).to(device)
ck = torch.load("checkpoints/s3d_gate_pp22_run-123_epoch-52.pt", map_location=device)
model.load_state_dict(ck["model"] if "model" in ck else ck.model)
model.eval()
for p in model.parameters():
    p.requires_grad = False
model.solvent3d_energy = True
model.solvent_cavity_energy = True

cap = {}
_comp = ext._slab_compensation_periodic_1d_energy_radial
ext._slab_compensation_periodic_1d_energy_radial = (
    lambda *a, **k: cap.__setitem__("comp", _comp(*a, **k).detach().clone())
    or cap["comp"])
_clo = PB.closure_from_fields
def wrap_clo(n_e, cv, grid, params, tp):
    cap["cvhar3"] = cv.detach().clone()
    cap["grid"] = grid
    return _clo(n_e, cv, grid, params, tp)
PB.closure_from_fields = wrap_clo
_solve = PB.PB1DBackend.solve_graph
def wrap_solve(self, *a, **k):
    out = _solve(self, *a, **k)
    cap["rho_layer_z"] = out["rho_layer_z"].detach().clone()
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
DFT = {1: os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single") + "/1-44_GCE/cal_1",
       601: os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single") + "/5-44_neutral_withsolv/cal_1"}
want = {1: None, 601: None}
for a in read("data/train.xyz", ":"):
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


def to_model_grid(t, shape):
    if tuple(t.shape) == tuple(shape):
        return t
    return torch.nn.functional.interpolate(
        t[None, None], size=tuple(shape), mode="trilinear",
        align_corners=False)[0, 0].contiguous()


for sid, a in want.items():
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    cap.clear()
    with torch.no_grad():
        model(b.to_dict(), compute_force=False, training=False)
    grid = cap["grid"]
    cell64 = torch.as_tensor(grid.cell, dtype=torch.float64, device=device)
    V = grid.volume
    shape = tuple(grid.shape)
    dV = V / float(np.prod(shape))
    lz = float(cell64[2, 2]); nzm = shape[2]
    area = V / lz; dz = lz / nzm
    comp = float(cap["comp"].sum())
    cvz = cap["cvhar3"].mean(dim=(0, 1))
    cvz = cvz - cvz.mean()
    rl = cap["rho_layer_z"]
    # resample solver profile (upsampled length) to the grid z if needed
    if rl.shape[0] != nzm:
        rlg = torch.fft.fft(rl)
        keep = torch.zeros(nzm, dtype=torch.complex128, device=rl.device)
        h = min(nzm, rl.shape[0]) // 2
        keep[:h] = rlg[:h]; keep[-h:] = rlg[-h:]
        rl = torch.fft.ifft(keep).real * (nzm / rl.shape[0])
    # A. 1-D split
    t_full = float((rl * (-cvz)).sum() * area * dz)
    g = 2 * math.pi * torch.fft.fftfreq(nzm, d=dz).to(rl.device)
    rg = torch.fft.fft(rl)
    pg = torch.zeros_like(rg)
    m = g.abs() > 1e-14
    pg[m] = 4 * math.pi * FIELD_CONSTANT_EVA * rg[m] / g[m] ** 2
    ph1 = torch.fft.ifft(pg).real
    s1d = 0.5 * float((rl * (ph1 - ph1.mean())).sum() * area * dz)
    cross_net = comp - s1d
    e_bl = t_full - cross_net
    print(f"\n== sid {sid} (q={float(a.info['total_charge']):+.2f}) ==")
    print(f"[A] comp {comp:+.4f} = cross_net {cross_net:+.4f} + self1d {s1d:+.4f}")
    print(f"[A] T_full_1d {t_full:+.4f}  -> missing baseline coupling "
          f"E_bl = {e_bl:+.4f} eV  (expected ~ -0.62 neutral / -0.23 charged)",
          flush=True)
    # B. 2x2 factorization on the model grid
    lat, phi_f = read_grid(f"{DFT[sid]}/PHI")
    _, rb = read_grid(f"{DFT[sid]}/RHOB")
    _, ri = read_grid(f"{DFT[sid]}/RHOION")
    Vd = float(abs(np.linalg.det(lat)))
    n_solv = to_model_grid(-(rb + ri) / Vd, shape).to(device)
    phi_d = to_model_grid(-phi_f, shape).to(device)  # physical potential
    B = 2 * math.pi * torch.linalg.inv(cell64).T
    fx = torch.fft.fftfreq(shape[0], device=device) * shape[0]
    fy = torch.fft.fftfreq(shape[1], device=device) * shape[1]
    fz = torch.arange(shape[2] // 2 + 1, dtype=torch.float64, device=device)
    G2 = ((fx[:, None, None, None] * B[0] + fy[None, :, None, None] * B[1]
           + fz[None, None, :, None] * B[2]) ** 2).sum(-1)
    n_tot = torch.fft.irfftn(torch.fft.rfftn(phi_d) * G2
                             / (4 * math.pi * FIELD_CONSTANT_EVA), s=shape)
    n_sol = n_tot - n_solv
    phi_sol_d = poisson_phi_periodic(n_sol, cell64.to(device))
    n_lat = n_solv - n_solv.mean(dim=(0, 1), keepdim=True)
    delta = cap["delta"].to(device)
    phi_sol_m = -(cap["cvhar3"].to(device) - cap["cvhar3"].mean())
    rows = {"delta_model": delta, "n_lat_DFT": n_lat}
    cols = {"phi_model": phi_sol_m, "phi_DFT": phi_sol_d}
    print("[B] cross-energy matrix (eV):")
    for rn, rv in rows.items():
        line = f"      {rn:>11}: "
        for cn, cv in cols.items():
            line += f"{cn} {float((rv * cv).sum() * dV):+.4f}   "
        print(line, flush=True)
    print(f"      (charges: int|delta| {float(delta.abs().sum()*dV):.3f} e, "
          f"int|n_lat| {float(n_lat.abs().sum()*dV):.3f} e)", flush=True)
print("DONE")
