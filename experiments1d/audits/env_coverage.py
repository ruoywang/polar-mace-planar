"""Envelope-coverage diagnostic for the charged-side 3-D fix (measure before
design): where does the TRUE lateral solvent charge (DFT n_lat) live relative
to the residual head's envelope support, binned by distance to the nearest
solute atom, and where does the model's delta live by contrast.
Frames: sid 1 (charged) + 601 (neutral). cwd = gate_bl (ep33 weights).
"""
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
from mace.modules.solvent3d import normalized_gradient_envelope

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
_solve = PB.PB1DBackend.solve_graph
def wrap(self, *a, **k):
    out = _solve(self, *a, **k)
    cap["grid"] = self._grids[list(self._grids)[-1]]
    if out.get("s3d_obs") is not None:
        o = out["s3d_obs"]
        cap["delta"] = (o["d_sup_b"] + o["d_sup_i"]).detach().clone()
    return out
PB.PB1DBackend.solve_graph = wrap

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff",
               "solvated": "solvated"},
    arrays_keys={"forces": "forces"},
)
DFT = {1: "/scratch/08384/tg876840/tmp/2-NiN_single/1-44_GCE/cal_1",
       601: "/scratch/08384/tg876840/tmp/2-NiN_single/5-44_neutral_withsolv/cal_1"}
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


BINS = [0.0, 2.5, 3.5, 5.0, 1e9]
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
    shape = tuple(grid.shape)
    nx, ny, nz = shape
    V = grid.volume
    dV = V / float(np.prod(shape))
    cavd = grid._solv3d_cavity
    env_b = normalized_gradient_envelope(cavd[1], cell64)
    env_i = torch.clamp(cavd[0], 0.0, 1.0)
    env = torch.maximum(env_b, env_i)
    delta = cap["delta"].to(device).abs()

    lat, rb = read_grid(f"{DFT[sid]}/RHOB")
    _, ri = read_grid(f"{DFT[sid]}/RHOION")
    Vd = float(abs(np.linalg.det(lat)))
    n_solv = -(rb + ri) / Vd
    if tuple(n_solv.shape) != shape:
        n_solv = torch.nn.functional.interpolate(
            n_solv[None, None], size=shape, mode="trilinear",
            align_corners=False)[0, 0]
    n_solv = n_solv.to(device)
    n_lat = (n_solv - n_solv.mean(dim=(0, 1), keepdim=True)).abs()

    # distance to nearest solute atom (periodic min-image, general cell)
    pos_frac = torch.remainder(
        torch.as_tensor(a.get_positions(), device=device, dtype=torch.float64)
        @ torch.linalg.inv(cell64), 1.0)
    fx = (torch.arange(nx, device=device, dtype=torch.float64) / nx)
    fy = (torch.arange(ny, device=device, dtype=torch.float64) / ny)
    fz = (torch.arange(nz, device=device, dtype=torch.float64) / nz)
    dmin = torch.empty(shape, device=device)
    for iz0 in range(0, nz, 10):
        iz1 = min(iz0 + 10, nz)
        gf = torch.stack(torch.meshgrid(fx, fy, fz[iz0:iz1], indexing="ij"),
                         dim=-1).reshape(-1, 3)
        df = gf[:, None, :] - pos_frac[None, :, :]
        df = df - torch.round(df)
        dc = torch.einsum("pni,ij->pnj", df, cell64)
        dmin[:, :, iz0:iz1] = dc.norm(dim=-1).min(dim=1).values.reshape(
            nx, ny, iz1 - iz0)
        del gf, df, dc
    tot_lat = float(n_lat.sum() * dV)
    tot_del = float(delta.sum() * dV)
    print(f"\n== sid {sid} (q={float(a.info['total_charge']):+.2f})  "
          f"int|n_lat| {tot_lat:.3f} e   int|delta| {tot_del:.3f} e", flush=True)
    for t in (0.02, 0.05, 0.2):
        cb = float((n_lat * (env_b > t)).sum() * dV) / tot_lat
        ca = float((n_lat * (env > t)).sum() * dV) / tot_lat
        print(f"  coverage of |n_lat| at env>{t}: bound-env {100*cb:5.1f}%  "
              f"max(b,ion) {100*ca:5.1f}%", flush=True)
    print(f"  {'bin (A)':>10} {'|n_lat| share':>13} {'covered@0.05':>13} "
          f"{'|delta| share':>13} {'<env_b>':>8}")
    for k in range(4):
        m = (dmin >= BINS[k]) & (dmin < BINS[k + 1])
        sl = float((n_lat * m).sum() * dV) / tot_lat
        cv = float((n_lat * m * (env > 0.05)).sum() * dV) / max(
            float((n_lat * m).sum() * dV), 1e-30)
        sd = float((delta * m).sum() * dV) / tot_del
        eb = float(env_b[m].mean()) if int(m.sum()) else 0.0
        print(f"  {BINS[k]:>4.1f}-{min(BINS[k+1],99):<4.1f} {100*sl:12.1f}% "
              f"{100*cv:12.1f}% {100*sd:12.1f}% {eb:8.3f}", flush=True)
print("DONE")
