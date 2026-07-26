"""Dump everything the single-structure report page needs (one val structure).

Contents (npz):
- per-atom predicted charges, elements, z positions (+ ref charges if any)
- predicted vs DFT 1-D potential (EXACT Phi1D-loss construction, align=upper)
- plane-averaged net charge density: model vs DFT grid reference
- solvent solve: rho_ion(z), rho_bound(z), P_off prior, head dP
- scalars: potential/fermi pred vs ref, q_ion, layer_mean, mu_bound

Usage: python dump_structure_report.py <model> <xyz> <index> <out_npz>
Run from a dir with ./data and ./cache mirroring training.
"""
from __future__ import annotations

import os
import sys

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"

import numpy as np
import torch
from ase.io import read

model_path, xyz_path, idx, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
torch.set_default_dtype(torch.float64)
device = "cuda"
model = torch.load(model_path, map_location=device).to(device)
model.eval()

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils
import mace.modules.loss as L

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff"},
    arrays_keys={"forces": "forces", "charges": "REF_charges"},
)
atoms = read(xyz_path, str(idx))
sid = int(atoms.info["sample_id"])
cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
bdict = batch.to_dict()

backend = model._get_pb1d_backend()
holder = {}
orig = backend.solve_graph
def spy(**kw):
    holder.update(kw)
    return orig(**kw)
backend.solve_graph = spy
with torch.no_grad():
    out = model(dict(bdict), training=False, compute_force=False)
backend.solve_graph = orig
diag = dict(backend.last_diagnostics)

# ---- per-atom charges ----------------------------------------------------
charges = out.get("charges")
charges = charges.detach().cpu().numpy().reshape(-1) if charges is not None else np.full(len(atoms), np.nan)
ref_charges = atoms.arrays.get("REF_charges", np.full(len(atoms), np.nan))

# ---- solvent profiles (stage-2 solve, full outputs) ---------------------
with torch.no_grad():
    res = backend.solve_graph(**holder)

# ---- model plane-avg net density vs DFT grid ----------------------------
cell_np = holder["cell"].detach().cpu().numpy().astype(float).reshape(3, 3)
shape = backend._grid_shape(cell_np)
grid = backend._grid_for(cell_np, shape, device)
cell64 = torch.as_tensor(cell_np, device=device, dtype=torch.float64)
pos_frac = torch.remainder(holder["positions"].to(torch.float64) @ torch.linalg.inv(cell64), 1.0)
with torch.no_grad():
    net_g = backend._gto_net_density_g(grid, pos_frac, holder["radial_coeffs"].to(torch.float64), holder["sigmas"])
    net_values = grid.ifft_real(net_g)
    # VASP values convention: rho(e/A^3) = values / V; plane average over xy
    nbar_model = (net_values.mean(dim=(0, 1)) / grid.volume * net_values.numel() / shape[2]).cpu().numpy()
lz = float(np.linalg.norm(cell_np[2]))
z_grid = np.arange(shape[2]) * lz / shape[2]

from mace.modules.loss import Density3DGridTargets
tg = Density3DGridTargets("data/density3d_net_grid_manifest_npy.json")
rho_ref, lattice, valid_iz = tg._load(sid)
rho_ref = np.asarray(rho_ref)
lattice = np.asarray(lattice)
nz_ref = rho_ref.shape[0]
lz_ref = float(lattice[2, 2])
valid_iz = np.asarray(valid_iz)
nbar_dft = rho_ref[valid_iz].mean(axis=(1, 2))
z_dft = valid_iz * lz_ref / nz_ref

# ---- predicted vs reference 1-D potential (loss construction verbatim) ---
targets = L._load_potential_1d_profiles_npz("data/potential1d_potcar_cache.npz")
t = targets[sid]
pred_dict = {
    "charge_density_radial_coefficients": out["charge_density_radial_coefficients"].detach()
        if out.get("charge_density_radial_coefficients") is not None else None,
    "solv_center": out["solv_center"].detach(),
    "dipole": out["dipole"].detach() if out.get("dipole") is not None else None,
    "solvent_profile_features": out["solvent_profile_features"].detach()
        if out.get("solvent_profile_features") is not None else None,
}
# 复用 loss 的逐图构造：直接调残差函数拿 residual，再叠回 phi_ref 得对齐后的 phi_pred
with torch.no_grad():
    residuals = L.potential_1d_profile_residuals(
        ref=batch, pred=pred_dict, potential_targets=targets,
        density_smearing_width=[0.25, 0.5, 1.0],  # = density_3d_sigma（训练同款）
        axis=2, solvent_sigma_g=0.85, align="upper", use_solvent_profile=True,
    )
z_phi = np.asarray(t["z_A"], dtype=float)
phi_ref = np.asarray(t["phi_eV"], dtype=float)
align_mask = np.asarray(t["align_mask"], dtype=float)
res_np = residuals.detach().cpu().numpy()
# align=upper: 两条曲线都减各自的 upper-mask 均值；residual = pred_cmp - ref_cmp
phi_ref_cmp = phi_ref - (phi_ref * align_mask).sum() / max(align_mask.sum(), 1.0)
phi_pred_cmp = phi_ref_cmp + res_np

np.savez(
    out_path,
    sid=sid,
    symbols=np.array(atoms.get_chemical_symbols()),
    z_atoms=atoms.positions[:, 2],
    charges=charges, ref_charges=np.asarray(ref_charges, dtype=float),
    total_charge=float(atoms.info["total_charge"]),
    pot_pred=float(out["potential"].detach().cpu()),
    pot_ref=float(atoms.info["potential_diff"]),
    fermi_pred=float(out["fermi_level_pred"].detach().cpu()) if out.get("fermi_level_pred") is not None else float(out["fermi_level"].detach().cpu()),
    fermi_ref=float(atoms.info["Fermi"]),
    z_solve=res["z"].cpu().numpy(),
    rho_ion=res["rho_ion_z"].cpu().numpy(),
    rho_bound=res["rho_bound_z"].cpu().numpy(),
    prior=res["prior_solve"].cpu().numpy(),
    delta_p=res["delta_p"].cpu().numpy(),
    q_ion=diag.get("q_ion", np.nan), layer_mean=diag.get("layer_mean", np.nan),
    mu_bound=diag.get("mu_bound", np.nan),
    z_grid=z_grid, nbar_model=nbar_model,
    z_dft=z_dft, nbar_dft=nbar_dft,
    z_phi=z_phi, phi_ref_cmp=phi_ref_cmp, phi_pred_cmp=phi_pred_cmp,
    phi_residual=res_np,
    lz=lz,
)
print(f"dumped sid={sid} atoms={len(atoms)} -> {out_path}")
print(f"phi residual rms = {float(np.sqrt((res_np**2).mean())):.4f} eV")
print(f"pot pred/ref = {float(out['potential']):.4f}/{atoms.info['potential_diff']:.4f}")
