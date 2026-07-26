"""What do the rho_b-ceiling coefficients cost on the potential side? (one sid)

Runs the full forward twice — stock head vs head forced to the ceiling
coefficients — and reports potential/fermi/Phi1D/rho_b for both.

Usage: python eval_ceiling_cost.py <model> <xyz> <sid> <ref_npz> <ceiling_npz>
Run from a dir with ./data and ./cache.
"""
from __future__ import annotations

import os
import sys

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"

import numpy as np
import torch
from ase.io import read

model_path, xyz_path, sid_want, ref_npz, ceil_npz = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
torch.set_default_dtype(torch.float64)
device = "cuda"
model = torch.load(model_path, map_location=device).to(device)
model.eval()

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils
import mace.modules.loss as L
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
tgt_sm = _gaussian_smear_periodic_1d(
    torch.tensor(refs["targets"][sid_want], device=device).unsqueeze(0), 0.25, refs["lz_A"])[0]
c_ceiling = torch.tensor(np.load(ceil_npz)["coeff_ceiling"], device=device)

pt_targets = L._load_potential_1d_profiles_npz("./data/potential1d_potcar_cache.npz")
head = model.pb1d_head
orig_coeff = head.coefficients


def run(tag, force_c):
    if force_c is not None:
        head.coefficients = lambda *a, **k: force_c
    try:
        with torch.no_grad():
            out = model(batch.to_dict(), training=False, compute_force=False)
    finally:
        head.coefficients = orig_coeff
    rb = out["solvent_rho_bound_1d"][0]
    rb_rmse = float(torch.sqrt(torch.mean(
        (_gaussian_smear_periodic_1d(rb.unsqueeze(0), 0.25, refs["lz_A"])[0] - tgt_sm) ** 2)))
    pot_pred = float(out["potential"].view(-1)[0])
    pot_ref = float(batch.to_dict()["potential"].view(-1)[0])
    fermi_pred = float(out["fermi_level_pred"].view(-1)[0])
    fermi_ref = float(batch.to_dict()["fermi_level"].view(-1)[0])
    phi_res = L.potential_1d_profile_residuals(
        ref=batch, pred=out, potential_targets=pt_targets,
        density_smearing_width=[0.25, 0.5, 1.0], axis=2, solvent_sigma_g=0.85,
        align="upper", use_solvent_profile=True)
    phi_rms = float(torch.sqrt(torch.mean(phi_res ** 2))) if phi_res is not None else float("nan")
    print(f"{tag:9s} rho_b {rb_rmse:.3e}  |pot err| {abs(pot_pred-pot_ref):.4f} eV  "
          f"|fermi err| {abs(fermi_pred-fermi_ref):.4f} eV  Phi1D rms {phi_rms:.4f} eV")


run("trained", None)
run("ceiling8", c_ceiling)
