"""Per-structure: forward under no_grad vs grad-enabled (evaluate-style),
same model/weights/cache. Prints potential prediction + solvent diagnostics
from both paths to localize the value divergence.

Usage: python probe_grad_vs_nograd.py <model> <xyz> [n]
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from ase.io import read

model_path, xyz_path = sys.argv[1], sys.argv[2]
nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 999
torch.set_default_dtype(torch.float64)
device = "cuda"
model = torch.load(model_path, map_location=device).to(device).eval()

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
backend = model._get_pb1d_backend()
structs = read(xyz_path, f":{nmax}")
rows = []
for a in structs:
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    sid = int(a.info["sample_id"])
    p_ref = float(a.info["potential_diff"])

    backend.last_diagnostics = {}
    with torch.no_grad():
        out_ng = model(b.to_dict(), training=False, compute_force=False)
    p_ng = float(out_ng["potential"].detach().cpu())
    d_ng = dict(backend.last_diagnostics)

    backend.last_diagnostics = {}
    out_g = model(b.to_dict(), training=False, compute_force=True)  # grad enabled
    p_g = float(out_g["potential"].detach().cpu())
    d_g = dict(backend.last_diagnostics)

    rows.append((sid, p_ref, p_ng, p_g, d_ng, d_g))
    flag = "  <-- DIVERGE" if abs(p_g - p_ng) > 0.02 else ""
    print(f"sid={sid:4d} ref={p_ref:+.4f}  nograd={p_ng:+.4f}  grad={p_g:+.4f}  "
          f"d(g-ng)={p_g-p_ng:+.4f}  mu_ng={d_ng.get('mu_bound', float('nan')):+7.2f} "
          f"mu_g={d_g.get('mu_bound', float('nan')):+7.2f}{flag}")

dng = np.sqrt(np.mean([(r[2]-r[1])**2 for r in rows]))
dg = np.sqrt(np.mean([(r[3]-r[1])**2 for r in rows]))
print(f"\nRMSE vs ref: nograd {dng:.4f}  grad {dg:.4f}")
