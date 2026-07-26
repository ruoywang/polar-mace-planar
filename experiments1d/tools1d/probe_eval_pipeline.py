"""Isolate the test-table anomaly: run mace's own evaluate() on the SAVED
model with (a) the valid set as one loader, (b) test split by config_type —
exactly the final-table structure, but outside run_train (no checkpoint
machinery, single process).

Usage: python probe_eval_pipeline.py <model> <valid_xyz> <test_xyz>
Run from a dir with ./data and ./cache mirroring training.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from ase.io import read

model_path, valid_xyz, test_xyz = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_default_dtype(torch.float64)
device = "cuda"
model = torch.load(model_path, map_location=device).to(device)
model.eval()  # true-quality tables; train-mode reproduced the warmup-branch artifact

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils
from mace.tools.train import evaluate
from mace.modules import WeightedEnergyForcesElectrostaticsLoss

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff",
               "head": "head"},
    arrays_keys={"forces": "forces", "charges": "REF_charges"},
)


def loader_of(atoms_list):
    ds = [mace_data.AtomicData.from_config(
        mace_data.config_from_atoms(a, key_specification=kspec),
        z_table=z_table, cutoff=float(model.r_max)) for a in atoms_list]
    return torch_geometric.dataloader.DataLoader(ds, batch_size=1, shuffle=False)


# 与训练相同的损失配置（决定表里各指标的计算方式）
loss_fn = WeightedEnergyForcesElectrostaticsLoss(
    energy_weight=1.0, forces_weight=100.0, dipole_weight=0.0,
    charges_weight=0.0, atomic_dipole_weight=0.0,
    potential_weight=1e-5, fermi_level_weight=1.0,
    fermi_residual_reg_weight=0.1,
    density_3d_weight=1.0,
    density_3d_file="data/density3d_net_grid_manifest_npy.json",
    density_3d_sigma="[0.25, 0.5, 1.0]", density_3d_samples=1024, density_3d_seed=123,
    potential_1d_profile_weight=1.0,
    potential_1d_profile_file="data/potential1d_potcar_cache.npz",
    potential_1d_profile_align="upper",
    potential_1d_profile_use_solvent_profile=True,
    solvent_center_weight=0.0,
    potential_axis=2, potential_sign=-1,
    solvent_sigma_g=0.85,
).to(device)

output_args = {"forces": True, "virials": False, "stress": False}

valid = read(valid_xyz, ":")
test = read(test_xyz, ":")
groups = {"valid_all": valid}
for a in test:
    groups.setdefault("test_" + str(a.info.get("config_type", "?")), []).append(a)

for name, alist in groups.items():
    _, metrics = evaluate(model=model, loss_fn=loss_fn, data_loader=loader_of(alist),
                          output_args=output_args, device=device)
    print(f"{name:14s}: potential {metrics.get('rmse_potential'):.4f}  "
          f"fermi {metrics.get('rmse_fermi_level'):.4f}  "
          f"E {1e3*metrics.get('rmse_e_per_atom'):.2f} meV  "
          f"phi1d {metrics.get('rmse_potential_1d_profile'):.4f}  (n={len(alist)})")
