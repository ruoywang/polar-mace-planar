"""Localise the un-representable part of the lateral solvent charge.

The joint-fit screen showed that on charged frames no envelope channel tried
so far can reach the reference cross energy without wrecking the self-energy,
while on the neutral frame the basis is capable. Before designing a near-shell
channel by guesswork, measure WHERE the missing cross energy actually sits.

For basis A (env_b + s_ion, as built) take the point-optimal fit of the true
lateral field, form the residual  r = n_lat - fit, and bin it by distance to
the nearest solute atom. Per bin report: the share of |charge| in the target
and in the residual, the residual's cross-energy contribution and its share of
the total missing cross energy, and the mean of |phi|, env_b, s_ion, s_diel
there — i.e. exactly what an added channel would have to reach.
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

RIDGE = 1.0e-7
GCE44 = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single") + "/1-44_GCE"
GCE88 = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single") + "/2-88_GCE"
NEU = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single") + "/5-44_neutral_withsolv"
FRAMES = [(1, f"{GCE44}/cal_1", "NiN44 q=-1.00", 80000),
          (201, f"{GCE88}/cal_1", "NiN88 q=-1.00", 60000),
          (601, f"{NEU}/cal_1", "neutral", 80000)]
BINS = [0.0, 1.5, 2.5, 3.5, 5.0, 1e9]

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
    CH = [env_b, s_ion]
    CHm = [torch.clamp(w.mean(dim=(0, 1)), min=1e-12) for w in CH]

    lat, rb = read_grid(f"{dftdir}/RHOB")
    _, ri = read_grid(f"{dftdir}/RHOION")
    Vd = float(abs(np.linalg.det(lat)))
    n_solv = to_shape(-(rb + ri) / Vd, shape).to(device)
    n_lat = n_solv - n_solv.mean(dim=(0, 1), keepdim=True)
    del rb, ri, n_solv

    g = torch.Generator(device="cpu").manual_seed(11)
    idx = torch.randint(0, int(np.prod(shape)), (NPTS,), generator=g).to(device)
    iz = idx % nz; iy = (idx // nz) % ny; ix = idx // (nz * ny)
    frac = torch.stack([(ix.to(torch.float64) + 0.5) / nx,
                        (iy.to(torch.float64) + 0.5) / ny,
                        (iz.to(torch.float64) + 0.5) / nz], dim=1)
    y = _interp3_periodic(n_lat, frac)
    nat = len(a); K1 = nat * len(sigmas) * 9
    A = torch.zeros(NPTS, 2 * K1, dtype=torch.float32, device=device)
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
                for ci, (w, wm) in enumerate(zip(CH, CHm)):
                    raw = w * Gk
                    r = raw.mean(dim=(0, 1)) / wm
                    A[:, ci * K1 + col] = _interp3_periodic(
                        raw - r[None, None, :] * w, frac).to(torch.float32)
                col += 1
        if ia % 80 == 0:
            print(f"   [{tag}] build {ia}/{nat}", flush=True)
    n = 2 * K1
    H = torch.zeros(n, n, dtype=torch.float64, device=device)
    Aty = torch.zeros(n, dtype=torch.float64, device=device)
    for i in range(0, NPTS, 4000):
        blk = A[i:i + 4000].to(torch.float64)
        H += blk.T @ blk
        Aty += blk.T @ y[i:i + 4000]
        del blk
    H += RIDGE * float(torch.diagonal(H).mean()) * torch.eye(
        n, dtype=torch.float64, device=device)
    c = torch.cholesky_solve(Aty.unsqueeze(1),
                             torch.linalg.cholesky(H)).squeeze(1)
    del A, H, Aty
    torch.cuda.empty_cache()

    fit = torch.zeros(shape, dtype=torch.float64, device=device)
    col = 0
    for ia in range(nat):
        for isig in range(len(sigmas)):
            for lm in range(9):
                vs = [float(c[ci * K1 + col]) for ci in range(2)]
                if max(abs(v) for v in vs) > 1e-14:
                    cvec.zero_(); cvec[ia, isig, lm] = 1.0
                    Gk = grid.ifft_real(beobj._gto_net_density_g(
                        grid, pos_frac, cvec, sigmas)) / V
                    for ci, (w, wm) in enumerate(zip(CH, CHm)):
                        if abs(vs[ci]) > 1e-14:
                            raw = w * Gk
                            r = raw.mean(dim=(0, 1)) / wm
                            fit += vs[ci] * (raw - r[None, None, :] * w)
                col += 1
        if ia % 80 == 0:
            print(f"   [{tag}] rebuild {ia}/{nat}", flush=True)
    resid = n_lat - fit

    # distance to the nearest solute atom
    fxs = torch.arange(nx, device=device, dtype=torch.float64) / nx
    fys = torch.arange(ny, device=device, dtype=torch.float64) / ny
    fzs = torch.arange(nz, device=device, dtype=torch.float64) / nz
    dmin = torch.empty(shape, device=device, dtype=torch.float64)
    for z0 in range(0, nz, 8):
        z1 = min(z0 + 8, nz)
        gf = torch.stack(torch.meshgrid(fxs, fys, fzs[z0:z1], indexing="ij"),
                         dim=-1).reshape(-1, 3)
        df = gf[:, None, :] - pos_frac[None, :, :]
        df = df - torch.round(df)
        dmin[:, :, z0:z1] = torch.einsum("pni,ij->pnj", df, cell64).norm(
            dim=-1).min(dim=1).values.reshape(nx, ny, z1 - z0)
        del gf, df
    cross_tot_ref = float((n_lat * phi).sum() * dV)
    cross_tot_fit = float((fit * phi).sum() * dV)
    miss = cross_tot_ref - cross_tot_fit
    amp_ref = float(n_lat.abs().sum() * dV)
    amp_res = float(resid.abs().sum() * dV)
    print(f"\n== {tag} ==  cross: ref {cross_tot_ref:+.3f}  fit {cross_tot_fit:+.3f}"
          f"  missing {miss:+.3f} eV   |   |q|: ref {amp_ref:.3f}  resid {amp_res:.3f}",
          flush=True)
    print(f"  {'bin (A)':>10} {'|target|%':>10} {'|resid|%':>10} "
          f"{'resid cross':>12} {'% of missing':>13} {'<|phi|>':>9} "
          f"{'<env_b>':>9} {'<s_ion>':>9} {'<s_diel>':>9}")
    for k in range(len(BINS) - 1):
        m = (dmin >= BINS[k]) & (dmin < BINS[k + 1])
        if not bool(m.any()):
            continue
        at = float((n_lat.abs() * m).sum() * dV) / amp_ref * 100
        ar = float((resid.abs() * m).sum() * dV) / amp_res * 100
        cr = float((resid * phi * m).sum() * dV)
        pc = cr / miss * 100 if abs(miss) > 1e-12 else float("nan")
        print(f"  {BINS[k]:>4.1f}-{min(BINS[k+1],99):<5.1f} {at:10.1f} {ar:10.1f} "
              f"{cr:+12.3f} {pc:13.1f} {float(phi[m].abs().mean()):9.3f} "
              f"{float(env_b[m].mean()):9.4f} {float(s_ion[m].mean()):9.4f} "
              f"{float(s_diel[m].mean()):9.4f}", flush=True)
    del fit, resid, n_lat, phi, env_b, s_ion, s_diel, dmin
    import gc; gc.collect(); torch.cuda.empty_cache()
print("DONE")
