"""Byte-identity probe: new solvated-flag code on flag-less data == old code.

Runs one eval forward of the SAME model on the SAME solvated structure and
dumps key outputs. Called twice (old/new PYTHONPATH); caller diffs the npz.

Usage: python probe_solvated_identity.py <model> <xyz> <index> <out_npz>
Run from a dir with ./data and ./cache.
"""
from __future__ import annotations

import os
import sys

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"

import numpy as np
import torch
from ase.io import read

model_path, xyz_path, idx, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
torch.set_default_dtype(torch.float64)
model = torch.load(model_path, map_location="cuda")
model.eval()

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
atoms = read(xyz_path, str(idx))
cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to("cuda")
o = model(batch.to_dict(), training=False, compute_force=True)
np.savez(out,
         energy=o["energy"].detach().double().cpu().numpy(),
         forces=o["forces"].detach().double().cpu().numpy(),
         potential=o["potential"].detach().double().cpu().numpy(),
         fermi=o["fermi_level_pred"].detach().double().cpu().numpy(),
         charges=o["charges"].detach().double().cpu().numpy())
print("dumped", out)
