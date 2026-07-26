"""Phase-level timing of the pb1d forward/backward on real structures.

Loads the smoke model, times (with cuda sync):
  A. full model forward, planar mode vs pb1d mode (same weights)
  B. pb1d backend phases: baseline load / assembly / poisson / closure /
     upsample+head / 1-D solve / moments
  C. backward pass cost with and without the pb1d observable gradients

Usage: python profile_pb1d.py <model_path> <xyz> [n_structs]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from ase.io import read

model_path, xyz_path = sys.argv[1], sys.argv[2]
n_structs = int(sys.argv[3]) if len(sys.argv) > 3 else 4

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(model_path, map_location=device)
model = model.to(device).eval()

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
r_max = float(model.r_max)
kspec = KeySpecification(
    info_keys={
        "energy": "energy", "total_charge": "total_charge",
        "total_spin": "total_spin", "sample_id": "sample_id",
        "fermi_level": "Fermi", "potential": "potential_diff",
    },
    arrays_keys={"forces": "forces", "charges": "REF_charges"},
)


def make_batch(ats):
    cfg = mace_data.config_from_atoms(ats, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=r_max)]
    loader = torch_geometric.dataloader.DataLoader(ds, batch_size=1, shuffle=False)
    return next(iter(loader)).to(device).to_dict()


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


def timed(fn, *a, **k):
    sync(); t0 = time.perf_counter()
    out = fn(*a, **k)
    sync()
    return out, time.perf_counter() - t0


structs = read(xyz_path, f":{n_structs}")
batches = [make_batch(a) for a in structs]

# ---------- A. whole-model forward+backward, pb1d vs planar ----------
print("=== A. whole-model timings (per structure, second call = warm) ===")
for mode in ("pb1d", "planar"):
    model.solvent_model = mode
    for tag, train_flag in (("eval fwd", False), ("train fwd+bwd", True)):
        ts = []
        for b in batches:
            def run():
                if train_flag:
                    model.train()
                    out = model(dict(b), training=True, compute_force=True)
                    loss = out["energy"].sum() + out["forces"].pow(2).sum()
                    for key in ("potential", "fermi_level"):
                        if out.get(key) is not None:
                            loss = loss + out[key].pow(2).sum()
                    model.zero_grad(set_to_none=True)
                    loss.backward()
                else:
                    model.eval()
                    with torch.no_grad():
                        model(dict(b), training=False, compute_force=False)
            run()  # warm (cache seeding, cudnn, grids)
            _, dt = timed(run)
            ts.append(dt)
        print(f"  {mode:6s} {tag:14s}: " +
              " ".join(f"{t*1e3:8.1f}" for t in ts) + "  ms")
model.solvent_model = "pb1d"
model.train()

# ---------- B. backend phase breakdown ----------
print("=== B. pb1d backend phase breakdown (one graph, grad on) ===")
backend = model._get_pb1d_backend()

b = batches[0]
sid = int(b["sample_id"].view(-1)[0].item())
# rebuild the stage-2 inputs by running one forward and stealing tensors
holder = {}
orig = backend.solve_graph
def spy(**kw):
    holder.update(kw)
    return orig(**kw)
backend.solve_graph = spy
out = model(dict(b), training=True, compute_force=False)
backend.solve_graph = orig
kw = holder

cell_np = kw["cell"].detach().cpu().numpy().astype(float).reshape(3, 3)
shape = backend._grid_shape(cell_np)
grid = backend._grid_for(cell_np, shape, device)
bl_row = backend._bl_index[sid]

def ph_baseline():
    return torch.as_tensor(
        np.ascontiguousarray(backend._bl_arr[bl_row]), device=device
    ).to(torch.float64)
fields, t_bl = timed(ph_baseline)

cell64 = torch.as_tensor(cell_np, device=device, dtype=torch.float64)
pos_frac = torch.remainder(
    kw["positions"].to(torch.float64) @ torch.linalg.inv(cell64), 1.0)
coeffs = kw["radial_coeffs"].detach().requires_grad_(True)

def ph_assemble():
    return backend._gto_net_density_g(grid, pos_frac.detach(), coeffs, kw["sigmas"])
net_g, t_asm = timed(ph_assemble)

def ph_fields():
    net_values = grid.ifft_real(net_g)
    n_e = torch.clamp((fields[0] - net_values) / grid.volume, min=0.0)
    cvhar3 = fields[2] - grid.ifft_real(grid.l0_inv_op(net_g))
    return n_e, cvhar3
(n_e, cvhar3), t_pois = timed(ph_fields)

from mace.modules.pb1d_closure import closure_from_fields
clo, t_clo = timed(lambda: closure_from_fields(n_e, cvhar3, grid, backend.params, backend._tp))

def ph_solve():
    return backend.solve_graph(**kw)
res, t_slv_total = timed(ph_solve)   # whole backend call for reference

def ph_backward():
    s = (res["q_ion_t"] + res["layer_mean_t"] + res["mu_bound_t"]
         + res["rho_layer_z"].sum())
    g = torch.autograd.grad(s, kw["radial_coeffs"], retain_graph=False,
                            allow_unused=True)
    return g
_, t_bwd = timed(ph_backward)

print(f"  baseline row load+H2D : {t_bl*1e3:8.1f} ms")
print(f"  spectral assembly     : {t_asm*1e3:8.1f} ms")
print(f"  ifft + poisson fields : {t_pois*1e3:8.1f} ms")
print(f"  closure               : {t_clo*1e3:8.1f} ms")
print(f"  FULL backend call     : {t_slv_total*1e3:8.1f} ms  (assembly..solve, ckpt fwd)")
print(f"  backend backward      : {t_bwd*1e3:8.1f} ms  (through ckpt recompute + unroll)")

# 1-D solve alone (fixed inputs, no grad)
with torch.no_grad():
    f = backend.solve_upsample
    nz = shape[2]
    from mace.modules.pb1d_backend import fourier_upsample
    cv = fourier_upsample(cvhar3.mean(dim=(0, 1)), f)
    si = torch.clamp(fourier_upsample(clo["S_ion_z"], f), min=0.0)
    aa = torch.clamp(fourier_upsample(clo["A_scr"], f), min=0.0)
    pp = fourier_upsample(clo["prior"], f)
    solver = backend._solver_for(cell_np, nz * f, device)
    center_z = 0.5 * (cell_np[0, 2] + cell_np[1, 2] + cell_np[2, 2])
    nzs = nz * f
    indmin = int((nzs // 2 + int(0.5 * nzs) + 10 * nzs) % nzs + 1)
    vdz = res["z"].new_zeros(())

    def ph_1d():
        return solver.solve(
            cvhar_z=cv, s_ion=si, a1=aa, p_off=pp,
            q_sol=float(-kw["total_charge"]),
            val_ion_dipole_z=vdz, c_unit=backend._c_unit(cell_np),
            center_z=center_z, indmin=indmin,
            fixsol_steps=backend.fixsol_steps, tol=backend.tol,
            max_outer=backend.max_outer,
        )
    ph_1d()
    _, t_1d = timed(ph_1d)
print(f"  bare 1-D solve (nograd): {t_1d*1e3:8.1f} ms")
print("done")
