"""Dump 1-D bound-charge profiles of several models + DFT reference for one sid.

Usage: python dump_rhob_compare.py <xyz> <sid> <ref_npz> <out_npz> <label=model> [...]
Run from a dir with ./data and ./cache mirroring training.
"""
from __future__ import annotations

import os
import sys

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"

import numpy as np
import torch
from ase.io import read

xyz_path, sid_want, ref_npz, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
model_specs = [a.split("=", 1) for a in sys.argv[5:]]
torch.set_default_dtype(torch.float64)
device = "cuda"

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

refs = _load_solvent_rhob_1d_npz(ref_npz)
lz = refs["lz_A"]
rb_ref = torch.tensor(refs["targets"][sid_want])
sigma = 0.25

out = {"z": np.arange(512) * lz / 512.0, "lz": lz, "sid": sid_want,
       "ref_raw": rb_ref.numpy(),
       "ref_smeared": _gaussian_smear_periodic_1d(rb_ref.unsqueeze(0), sigma, lz)[0].numpy()}

for label, mpath in model_specs:
    model = torch.load(mpath, map_location=device).to(device)
    model.eval()
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    with torch.no_grad():
        pred = model(batch.to_dict(), training=False, compute_force=False)
    prof = pred["solvent_rho_bound_1d"][0].detach().cpu()
    assert float(pred["solvent_rho_bound_1d_mask"].view(-1)[0]) > 0.5, f"{label}: no healthy solve"
    out[f"{label}_raw"] = prof.numpy()
    out[f"{label}_smeared"] = _gaussian_smear_periodic_1d(prof.unsqueeze(0), sigma, lz)[0].numpy()
    d = out[f"{label}_smeared"] - out["ref_smeared"]
    print(f"{label}: smeared rmse vs ref = {float(np.sqrt(np.mean(d**2))):.6e} e/A^3")
    del model
    torch.cuda.empty_cache()

if len(model_specs) == 2:
    a, b = (out[f"{l}_raw"] for l, _ in model_specs)
    print(f"max |{model_specs[0][0]} - {model_specs[1][0]}| (raw) = {float(np.abs(a - b).max()):.3e} e/A^3")
np.savez(out_path, **out)
print(f"wrote {out_path}")
