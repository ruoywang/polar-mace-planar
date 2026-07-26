"""Where does the pb1d step time go? Micro-benchmarks on a trained model.

Times (cuda-synced, after warmup):
  A. one full train-like step: forward / backward split, GRAD_PASSES=1 vs 0
  B. solver variants on the real stage-2 inputs: nograd 5-step / nograd
     converged / adjoint fwd / adjoint fwd+bwd / unrolled fwd+bwd
  C. closure sub-steps: cavity / kernel / response / fft-grad / plane stats

Usage: python bench_pb1d_speed.py <model> <xyz>
Run from a dir with ./cache and ./data mirroring training.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from ase.io import read

model_path, xyz_path = sys.argv[1], sys.argv[2]
torch.set_default_dtype(torch.float64)
device = "cuda"
model = torch.load(model_path, map_location=device).to(device)

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff"},
    arrays_keys={"forces": "forces", "charges": "REF_charges"},
)
atoms = read(xyz_path, "0")
cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
bdict = batch.to_dict()


def timed(fn, n=3):
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


# ---- A. full step fwd/bwd split ----------------------------------------
print("=== A. train-like step (fwd/bwd ms) ===")
model.train()
for gp in ("1", "0"):
    os.environ["MACE_PB1D_GRAD_PASSES"] = gp

    def fwd():
        out = model(dict(bdict), training=True, compute_force=True)
        loss = (out["energy"].sum() + out["forces"].pow(2).sum()
                + out["potential"].pow(2).sum() + out["fermi_level"].pow(2).sum())
        return loss

    t_f = timed(lambda: fwd().detach())

    def fb():
        model.zero_grad(set_to_none=True)
        fwd().backward()

    t_fb = timed(fb)
    print(f"  GRAD_PASSES={gp}: fwd {t_f:7.1f}  fwd+bwd {t_fb:7.1f}  (bwd ~{t_fb - t_f:7.1f})")
os.environ["MACE_PB1D_GRAD_PASSES"] = "1"

# ---- B. solver variants on real inputs ----------------------------------
print("=== B. solver variants (ms) ===")
backend = model._get_pb1d_backend()
holder = {}
orig = backend.solve_graph
def spy(**kw):
    holder.update(kw); return orig(**kw)
backend.solve_graph = spy
model(dict(bdict), training=True, compute_force=False)
backend.solve_graph = orig
kw = holder

cell_np = kw["cell"].detach().cpu().numpy().astype(float).reshape(3, 3)
shape = backend._grid_shape(cell_np)
grid = backend._grid_for(cell_np, shape, device)
nz = shape[2]
f = backend.solve_upsample
solver = backend._solver_for(cell_np, nz * f, device)

# rebuild solver inputs once (differentiable coeffs for grad variants)
from mace.modules.pb1d_backend import fourier_upsample
from mace.modules.pb1d_closure import closure_from_fields, solute_dipole_z
cell64 = torch.as_tensor(cell_np, device=device, dtype=torch.float64)
pos_frac = torch.remainder(kw["positions"].to(torch.float64) @ torch.linalg.inv(cell64), 1.0)
coeffs = kw["radial_coeffs"].detach().requires_grad_(True)
bl_row = backend._bl_index[int(bdict["sample_id"].view(-1)[0])]
fields = torch.as_tensor(np.ascontiguousarray(backend._bl_arr[bl_row][[0, 2]]), device=device).to(torch.float64)
net_g = backend._gto_net_density_g(grid, pos_frac.detach(), coeffs, kw["sigmas"])
net_values = grid.ifft_real(net_g)
n_e = torch.clamp((fields[0] - net_values) / grid.volume, min=0.0)
cvhar3 = fields[1] - grid.ifft_real(grid.l0_inv_op(net_g))
clo = closure_from_fields(n_e, cvhar3, grid, backend.params, backend._tp)
cv = fourier_upsample(cvhar3.mean(dim=(0, 1)), f)
si = torch.clamp(fourier_upsample(clo["S_ion_z"], f), min=0.0)
aa = torch.clamp(fourier_upsample(clo["A_scr"], f), min=0.0)
pp = fourier_upsample(clo["prior"], f)
vd = solute_dipole_z(net_values.mean(dim=(0, 1)), pos_frac, kw["z_valence"].to(torch.float64), cell64)
q_sol = float(-kw["total_charge"])
center_z = 0.5 * (cell_np[0, 2] + cell_np[1, 2] + cell_np[2, 2])
nzs = nz * f
indmin = int((nzs // 2 + int(0.5 * nzs) + 10 * nzs) % nzs + 1)

import inspect
_solve_params = set(inspect.signature(solver.solve).parameters)


def run_solve(gp, converge, grad):
    args = (cv if grad else cv.detach(), si.detach(), aa.detach(),
            pp if grad else pp.detach(), q_sol, vd.detach(),
            float(backend._c_unit(cell_np)), center_z, indmin)
    kwargs = dict(fixsol_steps=5, tol=backend.tol, max_outer=backend.max_outer)
    if "grad_passes" in _solve_params:
        kwargs["grad_passes"] = gp
    if "fixsol_converge" in _solve_params:
        kwargs["fixsol_converge"] = converge
    if grad:
        out = solver.solve(*args, **kwargs)
        s = out["phi"].sum() + out["n_b"].sum()
        g = torch.autograd.grad(s, cv, retain_graph=False, allow_unused=True)
        return g
    with torch.no_grad():
        return solver.solve(*args, **kwargs)

variants = [
    ("nograd 5-step        ", 0, False, False),
    ("nograd converged     ", 0, True, False),
    ("adjoint fwd+bwd      ", 1, True, True),
    ("unrolled fwd+bwd     ", 0, True, True),
]
if "fixsol_converge" not in _solve_params:  # legacy tree: only 5-step unrolled exists
    variants = [("nograd 5-step        ", 0, False, False),
                ("unrolled 5-step f+b  ", 0, False, True)]
for label, gp, conv, grad in variants:
    print(f"  {label}: {timed(lambda: run_solve(gp, conv, grad)):7.1f}")

# ---- C. closure sub-steps -----------------------------------------------
print("=== C. closure sub-steps (ms, nograd) ===")
tp = backend._tp
params = backend.params
with torch.no_grad():
    print(f"  full closure     : {timed(lambda: closure_from_fields(n_e, cvhar3, grid, params, tp)):7.1f}")
    print(f"  create_cavity    : {timed(lambda: tp.create_cavity_torch(n_e, grid, params)):7.1f}")
    sigma_b = float(params['R_B']) if float(params['R_B']) > 0.0 else float(params['A_K'])
    print(f"  wb kernel (fresh): {timed(lambda: tp._normalized_gaussian_kernel_g(grid, sigma_b)):7.1f}")
    s_ion3, s_diel3, _ = tp.create_cavity_torch(n_e, grid, params)
    w_b = tp._normalized_gaussian_kernel_g(grid, sigma_b)
    phi_g = grid.fft(cvhar3)
    print(f"  fft(phi)         : {timed(lambda: grid.fft(cvhar3)):7.1f}")
    print(f"  grad_from_recip  : {timed(lambda: grid.grad_from_recip(-torch.conj(w_b) * phi_g)):7.1f}")
    from mace.modules.pb1d_closure import response_a3
    ex_, ey_, ez_, emag = grid.grad_from_recip(-torch.conj(w_b) * phi_g)
    print(f"  response_a3 x2   : {timed(lambda: (response_a3(torch.zeros_like(s_diel3), s_diel3, params, tp), response_a3(emag, s_diel3, params, tp))):7.1f}")
print("done")
