"""Score a model's 1-D bound-charge profile against the DFT RHOB references.

Runs eval-mode forwards over a whole xyz split (no warmup branch, fresh
solves), collects pred["solvent_rho_bound_1d"] per healthy graph, and reports
the same smeared RMSE the training loss uses, plus the zero-model baseline.

Usage: python eval_rhob_metric.py <model> <xyz> <ref_npz> [sigma]
Run from a dir with ./data and ./cache mirroring training.
"""
from __future__ import annotations

import os
import sys

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"

import numpy as np
import torch
from ase.io import read

model_path, xyz_path, ref_npz = sys.argv[1], sys.argv[2], sys.argv[3]
sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
torch.set_default_dtype(torch.float64)
device = "cuda"
model = torch.load(model_path, map_location=device).to(device)
model.eval()

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils
from mace.modules.loss import _load_solvent_rhob_1d_npz, _gaussian_smear_periodic_1d

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff"},
    arrays_keys={"forces": "forces", "charges": "REF_charges"},
)
refs = _load_solvent_rhob_1d_npz(ref_npz)
tgt = refs["targets"]
lz = refs["lz_A"]

frames = read(xyz_path, ":")
sq_model, sq_base, n = 0.0, 0.0, 0
per = []
for atoms in frames:
    sid = int(atoms.info["sample_id"])
    if sid not in tgt:
        continue
    cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    with torch.no_grad():
        out = model(batch.to_dict(), training=False, compute_force=False)
    prof = out.get("solvent_rho_bound_1d")
    mask = out.get("solvent_rho_bound_1d_mask")
    if prof is None or float(mask.view(-1)[0].item()) < 0.5:
        continue
    m = prof[0]
    r = m.new_tensor(tgt[sid])
    ms = _gaussian_smear_periodic_1d(m.unsqueeze(0), sigma, lz)[0]
    rs = _gaussian_smear_periodic_1d(r.unsqueeze(0), sigma, lz)[0]
    e = float(torch.mean((ms - rs) ** 2).item())
    b = float(torch.mean(rs ** 2).item())
    sq_model += e * ms.numel()
    sq_base += b * rs.numel()
    n += ms.numel()
    per.append((sid, e ** 0.5, b ** 0.5))

rmse = (sq_model / n) ** 0.5
base = (sq_base / n) ** 0.5
print(f"model {os.path.basename(model_path)}  n={len(per)} structs")
print(f"  RMSE_rhob_1d = {rmse:.6f} e/A^3   zero-baseline = {base:.6f}   ratio = {rmse/base:.3f}")
