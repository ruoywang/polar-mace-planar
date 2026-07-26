"""FD force check for the pb1d model: load the smoke run's model, compute
autograd forces on one training structure, compare selected components with
central finite differences of the energy. Cache frozen via
MACE_PB1D_CACHE_READONLY so repeated forwards are identical.

Usage: python test_pb1d_forces.py <model_path> <xyz> [n_components]
"""
from __future__ import annotations

import os
import sys

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"
os.environ["MACE_PB1D_FREEZE_SOLVE"] = "1"

import numpy as np
import torch
from ase.io import read

model_path, xyz_path = sys.argv[1], sys.argv[2]
n_comp = int(sys.argv[3]) if len(sys.argv) > 3 else 4

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(model_path, map_location=device)
model = model.to(device).eval()

from mace import data as mace_data
from mace.tools import torch_geometric, utils

atoms = read(xyz_path, "0")
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
r_max = float(model.r_max)

from mace.data.utils import KeySpecification

kwargs = {
    "key_specification": KeySpecification(
        info_keys={
            "energy": "energy", "total_charge": "total_charge",
            "total_spin": "total_spin", "sample_id": "sample_id",
            "fermi_level": "Fermi", "potential": "potential_diff",
        },
        arrays_keys={"forces": "forces", "charges": "REF_charges"},
    )
}


def make_batch(ats):
    cfg = mace_data.config_from_atoms(ats, **kwargs)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=r_max)]
    loader = torch_geometric.dataloader.DataLoader(ds, batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)
    return batch.to_dict()


def energy_of(ats):
    with torch.no_grad():
        out = model(make_batch(ats), training=False, compute_force=False)
    return float(out["energy"].detach().cpu())


b = make_batch(atoms)
out = model(b, training=False, compute_force=True)
forces = out["forces"].detach().cpu().numpy()
e0 = float(out["energy"].detach().cpu())
print(f"E0 = {e0:.10f} eV; |F| max {np.abs(forces).max():.4f} eV/A")

rng = np.random.default_rng(0)
n_at = len(atoms)
picks = [(int(rng.integers(n_at)), int(rng.integers(3))) for _ in range(n_comp)]
h = 1.0e-4
ok = True
for ia, ax in picks:
    ap = atoms.copy(); ap.positions[ia, ax] += h
    am = atoms.copy(); am.positions[ia, ax] -= h
    f_fd = -(energy_of(ap) - energy_of(am)) / (2 * h)
    f_ad = float(forces[ia, ax])
    rel = abs(f_ad - f_fd) / max(abs(f_fd), 1e-8)
    print(f"atom {ia:3d} axis {ax}: autograd {f_ad:+.8f}  FD {f_fd:+.8f}  rel {rel:.2e}")
    if rel > 5e-5 and abs(f_ad - f_fd) > 1e-6:
        ok = False
print("FORCES:", "PASS" if ok else "FAIL")
