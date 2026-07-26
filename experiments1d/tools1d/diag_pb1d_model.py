"""Post-training pb1d diagnostics: run the trained model over a structure
set and tabulate the per-structure solve diagnostics (health, layer_mean,
mu_bound, head coefficient magnitude, residual size, fallback incidence).

Usage: python diag_pb1d_model.py <model> <xyz> [max_structs]
Run from a dir whose ./cache and ./data mirror the training run.
"""
from __future__ import annotations

import os
import sys

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"

import numpy as np
import torch
from ase.io import read

model_path, xyz_path = sys.argv[1], sys.argv[2]
nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 999

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(model_path, map_location=device).to(device).eval()

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

structs = read(xyz_path, f":{nmax}")
backend = model._get_pb1d_backend()
rows = []
n_fallback = 0
for ats in structs:
    cfg = mace_data.config_from_atoms(ats, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=r_max)]
    loader = torch_geometric.dataloader.DataLoader(ds, batch_size=1, shuffle=False)
    b = next(iter(loader)).to(device).to_dict()
    sid = int(ats.info["sample_id"])
    lz = float(ats.cell[2][2])
    backend.last_diagnostics = {}
    with torch.no_grad():
        out = model(b, training=False, compute_force=False)
    d = dict(backend.last_diagnostics)
    if not d:
        n_fallback += 1
        print(f"sid={sid:4d}  NO-DIAG (planar fallback or non-slab)")
        continue
    healthy = (d["rms_last"] == d["rms_last"] and d["rms_last"] < 10 * model.solvent_pb_tol
               and 0 <= d["layer_mean"] <= lz and abs(d["mu_bound"]) <= 2 * lz)
    if not healthy:
        n_fallback += 1
    rows.append((sid, d, healthy))
    print(f"sid={sid:4d}  rms={d['rms_last']:.2e} n_outer={d['n_outer']:2d} "
          f"q_ion={d['q_ion']:+.3f} lm={d['layer_mean']:+7.2f} mu_b={d['mu_bound']:+8.2f} "
          f"|c|max={d['c_absmax']:.4f} dP_rms={d['dp_rms']:.5f} "
          f"{'ok' if healthy else 'UNHEALTHY'}")

if rows:
    lm = np.array([r[1]["layer_mean"] for r in rows])
    mb = np.array([r[1]["mu_bound"] for r in rows])
    ca = np.array([r[1]["c_absmax"] for r in rows])
    dp = np.array([r[1]["dp_rms"] for r in rows])
    print(f"\nSUMMARY over {len(rows)} solved structures "
          f"(+{n_fallback} fallback/unhealthy):")
    print(f"  layer_mean: mean {lm.mean():+.2f}  sd {lm.std():.2f}  "
          f"range [{lm.min():+.2f}, {lm.max():+.2f}]")
    print(f"  mu_bound  : mean {mb.mean():+.2f}  sd {mb.std():.2f}  "
          f"range [{mb.min():+.2f}, {mb.max():+.2f}]")
    print(f"  |c|max    : mean {ca.mean():.4f}  max {ca.max():.4f}  "
          f"(saturation bound c_max={float(model.solvent_pb1d_c_max):.2f})")
    print(f"  dP_rms    : mean {dp.mean():.5f}  max {dp.max():.5f}")
