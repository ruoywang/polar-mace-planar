"""Extend the 1-D / lateral energy decomposition and the 2x2 cross-energy
matrix beyond the two frames used so far (user question 2026-09-09: does the
charged/neutral analysis cover ALL the systems?).

Per frame it prints, in one pass:
  reference (DFT fields): E_int and E_self, 3-D and plane-averaged (1-D),
      hence the lateral remainder;
  model: the 1-D block (compensation energy + E_bl) and the lateral block
      (E_3d = e_xsol + e_self);
  the 2x2 matrix {model delta, DFT lateral charge} x {model potential, DFT
      potential}.
Coverage note: vacuum frames (NiN44vac) carry no solvent at all, so every
solvent term is identically zero and the decomposition is undefined for them
by construction — they enter only through the pairing conflict.
Usage: cohort_audit_ext.py            (frame list is hardcoded below)
cwd = gate_bl.
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, "/work/08384/tg876840/ls6/repos/cep-dip-python-pb")
sys.path.insert(0, "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy")

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.extensions as ext
import mace.modules.pb1d_backend as PB
from mace.modules.solvent3d import poisson_phi_periodic, FIELD_CONSTANT_EVA

GCE44 = "/scratch/08384/tg876840/tmp/2-NiN_single/1-44_GCE"
GCE88 = "/scratch/08384/tg876840/tmp/2-NiN_single/2-88_GCE"
NEU = "/scratch/08384/tg876840/tmp/2-NiN_single/5-44_neutral_withsolv"
# (sid, DFT dir, cohort tag). NiN44: cal = sid. NiN88: cal = sid-200.
# neutral solvated: cal = sid-600.
FRAMES = [
    (61, f"{GCE44}/cal_61", "NiN44 charged  q=-0.80"),
    (1, f"{GCE44}/cal_1", "NiN44 charged  q=-1.00"),
    (28, f"{GCE44}/cal_28", "NiN44 charged  q=-1.32"),
    (201, f"{GCE88}/cal_1", "NiN88 charged  q=-1.00"),
    (353, f"{GCE88}/cal_153", "NiN88 charged  q=-1.31"),
    (601, f"{NEU}/cal_1", "neutral solvated"),
]

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
def wrap_clo(n_e, cv, grid, params, tp):
    cap["cvhar3"] = cv.detach().clone()
    cap["grid"] = grid
    return _clo(n_e, cv, grid, params, tp)
PB.closure_from_fields = wrap_clo
_solve = PB.PB1DBackend.solve_graph
def wrap_solve(self, *a, **k):
    out = _solve(self, *a, **k)
    if out.get("s3d_obs") is not None:
        o = out["s3d_obs"]
        cap["delta"] = (o["d_sup_b"] + o["d_sup_i"]).detach().clone()
        cap["e_xsol"] = o["e_xsol"]
        cap["e_self"] = o["e_self"]
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


def phi1d(rho_z, dz):
    n = rho_z.shape[0]
    g = 2 * math.pi * torch.fft.fftfreq(n, d=dz).to(rho_z.device)
    rg = torch.fft.fft(rho_z)
    pg = torch.zeros_like(rg)
    m = g.abs() > 1e-14
    pg[m] = 4 * math.pi * FIELD_CONSTANT_EVA * rg[m] / (g[m] ** 2)
    out = torch.fft.ifft(pg).real
    return out - out.mean()


print(f"{'frame':>26} | {'ref 1D':>8} {'mod 1D':>8} {'gap':>6} | "
      f"{'ref lat':>8} {'mod lat':>8} {'gap':>6} | "
      f"{'d*phiM':>8} {'d*phiD':>8} {'n*phiM':>8} {'n*phiD':>8}", flush=True)
for sid, dftdir, tag in FRAMES:
    a = want[sid]
    if a is None:
        print(f"{tag:>26} | frame not found"); continue
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
    lz = float(cell64[2, 2]); nz = shape[2]
    area = V / lz; dz = lz / nz
    comp = float(cap["comp"].sum())
    ebl = float(pred["baseline_coupling_energy_g"].sum())
    e3d = float(pred["solvent3d_energy_g"].sum())
    phi_m = -(cap["cvhar3"] - cap["cvhar3"].mean())
    delta = cap["delta"].to(device)

    lat, phi_f = read_grid(f"{dftdir}/PHI")
    _, rb = read_grid(f"{dftdir}/RHOB")
    _, ri = read_grid(f"{dftdir}/RHOION")
    Vd = float(abs(np.linalg.det(lat)))
    phi_d = to_shape(-phi_f, shape).to(device)
    n_solv = to_shape(-(rb + ri) / Vd, shape).to(device)
    B = 2 * math.pi * torch.linalg.inv(cell64).T
    fx = torch.fft.fftfreq(shape[0], device=device) * shape[0]
    fy = torch.fft.fftfreq(shape[1], device=device) * shape[1]
    fz = torch.arange(shape[2] // 2 + 1, dtype=torch.float64, device=device)
    G2 = ((fx[:, None, None, None] * B[0] + fy[None, :, None, None] * B[1]
           + fz[None, None, :, None] * B[2]) ** 2).sum(-1)
    n_tot = torch.fft.irfftn(torch.fft.rfftn(phi_d) * G2
                             / (4 * math.pi * FIELD_CONSTANT_EVA), s=shape)
    n_sol = n_tot - n_solv
    phis = poisson_phi_periodic(n_solv, cell64)
    e_self3 = 0.5 * float((n_solv * phis).sum() * dV)
    e_int3 = float((n_sol * phis).sum() * dV)
    ns1 = n_solv.mean(dim=(0, 1)); nn1 = n_sol.mean(dim=(0, 1))
    ph1 = phi1d(ns1, dz)
    e_self1 = 0.5 * float((ns1 * ph1).sum() * area * dz)
    e_int1 = float((nn1 * ph1).sum() * area * dz)
    ref_1d = e_int1 + e_self1
    ref_lat = (e_int3 + e_self3) - ref_1d
    mod_1d = comp + ebl
    n_lat = n_solv - n_solv.mean(dim=(0, 1), keepdim=True)
    phi_sol_d = poisson_phi_periodic(n_sol, cell64)
    m11 = float((delta * phi_m).sum() * dV)
    m12 = float((delta * phi_sol_d).sum() * dV)
    m21 = float((n_lat * phi_m).sum() * dV)
    m22 = float((n_lat * phi_sol_d).sum() * dV)
    print(f"{tag:>26} | {ref_1d:+8.3f} {mod_1d:+8.3f} {mod_1d-ref_1d:+6.2f} | "
          f"{ref_lat:+8.3f} {e3d:+8.3f} {e3d-ref_lat:+6.2f} | "
          f"{m11:+8.3f} {m12:+8.3f} {m21:+8.3f} {m22:+8.3f}", flush=True)
    del phi_f, rb, ri, phi_d, n_solv, n_tot, n_sol, phis, n_lat, phi_sol_d
    import gc; gc.collect(); torch.cuda.empty_cache()
print("DONE  (gap = model - reference; negative gap = model too weak)")
