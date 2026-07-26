"""Numeric check: does the rho_b loss produce nonzero grads on the head/charge path?

Usage: python probe_rhob_grad.py <model> <xyz> <sid> <ref_npz>
Run from a dir with ./data and ./cache (enc counters past warmup).
"""
from __future__ import annotations

import os
import sys

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"

import numpy as np
import torch
from ase.io import read

model_path, xyz_path, sid_want, ref_npz = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
torch.set_default_dtype(torch.float64)
device = "cuda"
model = torch.load(model_path, map_location=device).to(device)
model.train()  # parameter grads on

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils
from mace.modules.loss import _load_solvent_rhob_1d_npz, _gaussian_smear_periodic_1d

kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff"},
    arrays_keys={"forces": "forces", "charges": "REF_charges"},
)
atoms = next(a for a in read(xyz_path, ":") if int(a.info["sample_id"]) == sid_want)
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)

refs = _load_solvent_rhob_1d_npz(ref_npz)
out = model(batch.to_dict(), training=True, compute_force=False)
prof = out["solvent_rho_bound_1d"]
mask = out["solvent_rho_bound_1d_mask"]
assert float(mask.view(-1)[0]) > 0.5, "no healthy fresh solve (check cache enc counters)"
r = prof.new_tensor(refs["targets"][sid_want]).unsqueeze(0)
lz = refs["lz_A"]
loss = torch.mean((_gaussian_smear_periodic_1d(prof, 0.25, lz)
                   - _gaussian_smear_periodic_1d(r, 0.25, lz)) ** 2)
model.zero_grad(set_to_none=True)
loss.backward()

head_sq = other_sq = 0.0
head_n = other_n = 0
for name, p in model.named_parameters():
    if p.grad is None:
        continue
    g2 = float((p.grad ** 2).sum())
    if "pb1d_head" in name:
        head_sq += g2
        head_n += p.numel()
    else:
        other_sq += g2
        other_n += p.numel()
print(f"rho_b loss = {float(loss):.6e}")
print(f"grad norm on pb1d_head params : {head_sq**0.5:.6e}  ({head_n} params)")
print(f"grad norm on all other params : {other_sq**0.5:.6e}  ({other_n} params)")
print("PATH-OK" if head_sq > 0 else "PATH-DEAD: zero gradient on head")
