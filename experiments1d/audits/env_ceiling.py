"""Frozen-model least-squares ceiling for envelope candidate A (user
protocol): per frame, refit the FULL bound-channel coefficient space under
the ACTUAL supervision operator (per-z-plane conserving projection on the
grid + trilinear interpolation to sampled points), for
    env0 = current bound envelope
    envS = smooth sqrt: env / sqrt(env + EPS)  (0 stays 0, bounded slope)
Report per variant: point-fit RMSE (raw + /signal), cross energy of the
fitted residual with the solute potential, self energy, |c| norms.
Ridge on the normal equations. Frames: sid 1 (charged) + 601 (neutral).
cwd = gate_bl. GPU.
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
import mace.modules.pb1d_backend as PB
from mace.modules.solvent3d import (
    normalized_gradient_envelope,
    poisson_phi_periodic,
    _interp3_periodic,
    load_solvent3d_targets,
    attach_solvent3d_samples_to_batch,
)

EPS = 1.0e-3
RIDGE = 1.0e-7
NPTS = 120000
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

sigmas = [float(s) for s in model.solvent3d_sigmas] \
    if not torch.is_tensor(model.solvent3d_sigmas) else \
    [float(s) for s in model.solvent3d_sigmas.tolist()]

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
    cavd = grid._solv3d_cavity
    env0 = normalized_gradient_envelope(cavd[1], cell64)
    envS = env0 / torch.sqrt(env0 + EPS)
    envS = envS / envS.max()
    cvz = cap["cvhar3"]

    valid = pred["solv3d_valid"]
    pts = b.solv3d_points[valid].to(torch.float64)
    ref_b = b.solv3d_ref_b[valid].to(torch.float64)
    base_b = pred["solv3d_base_b"][valid].to(torch.float64)
    y = (ref_b - base_b)
    P = pts.shape[0]
    frac = torch.remainder(
        pts @ torch.linalg.inv(cell64), 1.0)
    nat = len(a)
    pos_frac = torch.remainder(
        torch.as_tensor(a.get_positions(), device=device, dtype=torch.float64)
        @ torch.linalg.inv(cell64), 1.0)
    beobj = model._pb1d_backend
    assert beobj is not None, "backend not initialized"

    K = nat * len(sigmas) * 9
    print(f"\n== sid {sid} (q={float(a.info['total_charge']):+.2f})  P={P}  K={K}",
          flush=True)
    variants = {"env0": env0, "envS(sqrt-smooth)": envS}
    A = {name: torch.zeros(P, K, dtype=torch.float32, device=device)
         for name in variants}
    envm = {name: torch.clamp(e.mean(dim=(0, 1)), min=1e-12)
            for name, e in variants.items()}
    col = 0
    for ia in range(nat):
        for isig in range(len(sigmas)):
            cvec = torch.zeros(nat, len(sigmas), 9, dtype=torch.float64,
                               device=device)
            for lm in range(9):
                cvec.zero_()
                cvec[ia, isig, lm] = 1.0
                Gk = grid.ifft_real(beobj._gto_net_density_g(
                    grid, pos_frac, cvec, sigmas)) / V
                for name, e in variants.items():
                    raw = e * Gk
                    rk = raw.mean(dim=(0, 1)) / envm[name]
                    proj = raw - rk[None, None, :] * e
                    A[name][:, col] = _interp3_periodic(proj, frac).to(torch.float32)
                col += 1
        if ia % 40 == 0:
            print(f"   ... atom {ia}/{nat}", flush=True)
    sig = math.sqrt(SIG_MS["b"])
    for name, e in variants.items():
        Af = A[name].to(torch.float64)
        G = Af.T @ Af
        lam = RIDGE * float(torch.diagonal(G).mean())
        G += lam * torch.eye(K, dtype=torch.float64, device=device)
        c = torch.linalg.solve(G, Af.T @ y)
        resid = Af @ c - y
        rms = float(torch.sqrt((resid ** 2).mean()))
        rms0 = float(torch.sqrt((y ** 2).mean()))
        # rebuild fitted delta on the grid for energies (second pass)
        delta = torch.zeros(shape, dtype=torch.float64, device=device)
        col = 0
        for ia in range(nat):
            for isig in range(len(sigmas)):
                cvec = torch.zeros(nat, len(sigmas), 9, dtype=torch.float64,
                                   device=device)
                for lm in range(9):
                    if abs(float(c[col])) > 1e-14:
                        cvec[ia, isig, lm] = float(c[col])
                    col += 1
                if float(cvec.abs().sum()) > 0:
                    Gk = grid.ifft_real(beobj._gto_net_density_g(
                        grid, pos_frac, cvec, sigmas)) / V
                    raw = e * Gk
                    rk = raw.mean(dim=(0, 1)) / envm[name]
                    delta += raw - rk[None, None, :] * e
                    cvec.zero_()
        e_x = float((delta * (-(cvz - cvz.mean()))).sum() * dV)
        e_s = 0.5 * float((delta * poisson_phi_periodic(delta, cell64)).sum() * dV)
        print(f"  [{name}] fit rms {rms:.3e} (raw {rms0:.3e}, ratio "
              f"{rms/rms0:.3f}, /signal {rms/sig:.3f})  |c|max "
              f"{float(c.abs().max()):.2e} |c|rms {float(c.pow(2).mean().sqrt()):.2e}"
              f"  cross {e_x:+.3f} eV  self {e_s:+.3f} eV  "
              f"int|delta| {float(delta.abs().sum()*dV):.3f} e", flush=True)
        del Af, G
print("DONE  (reference lateral cross: charged -2.66, neutral -1.21 eV)")
