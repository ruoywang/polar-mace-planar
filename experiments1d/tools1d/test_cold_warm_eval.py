"""Cold-vs-warm cache evaluation on the TEST set.

Scheme C: stage-1 SCF features use the structure's cached previous-encounter
solvent profile. Test structures are never seen in training, so the final
test evaluation runs them cache-COLD (stage-1 = prior-only first-encounter
solve). This script evaluates test.xyz twice with a shared writable cache:
pass 1 cold (seeds the cache), pass 2 warm. If potential/fermi recover in
pass 2, the test-table degradation is the cold-cache evaluation artifact,
not model quality.

Usage: python test_cold_warm_eval.py <model> <test_xyz> <fresh_cache_dir>
Run from a dir with ./data mirroring training.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from ase.io import read

model_path, xyz_path, cache_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(cache_dir, exist_ok=True)
torch.set_default_dtype(torch.float64)
device = "cuda"
model = torch.load(model_path, map_location=device).to(device).eval()
model.solvent_pb_phi_cache_dir = cache_dir  # fresh, isolated

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
structs = read(xyz_path, ":")
batches = []
refs = []
for a in structs:
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    batches.append(next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device).to_dict())
    refs.append((float(a.info["potential_diff"]), float(a.info["Fermi"]),
                 str(a.info.get("config_type", "?")), int(a.info["sample_id"])))


def eval_pass(tag):
    dp, df = {}, {}
    for b, (p_ref, f_ref, ctype, sid) in zip(batches, refs):
        with torch.no_grad():
            out = model(dict(b), training=False, compute_force=False)
        dp.setdefault(ctype, []).append(float(out["potential"].detach().cpu()) - p_ref)
        df.setdefault(ctype, []).append(float(out["fermi_level"].detach().cpu()) - f_ref)
    for ctype in sorted(dp):
        rp = float(np.sqrt(np.mean(np.square(dp[ctype]))))
        rf = float(np.sqrt(np.mean(np.square(df[ctype]))))
        print(f"  {tag} {ctype:6s}: RMSE potential {rp:.4f} eV  fermi {rf:.4f} eV  (n={len(dp[ctype])})")


print("PASS 1 (cache COLD — 复现终评条件):")
eval_pass("cold")
print("PASS 2 (cache WARM — 特征来自上一遍的成熟剖面):")
eval_pass("warm")
print("PASS 3 (再暖一遍 — 收敛性检查):")
eval_pass("warm2")
