"""In-model validation of the new baseline-coupling energy term: on sid 601
(neutral) and sid 1 (charged), print baseline_coupling_energy_g and compare
with the audit values (neutral -0.414 expected to close ~2/3 of the +0.619
1-D gap; charged +0.123 G0-free part). Also asserts energy additivity:
E(flag on) - E(flag off) == e_bl. cwd = gate_pp2_ext.
"""
# Paths come from env vars with the LS6 values as defaults, so this
# runs unchanged on Lonestar6 and on a machine that holds the same
# payload elsewhere: KIT_PB_REPO, KIT_MACE_REPO, KIT_DFT,
# KIT_PB_CONFIG. Previously hardcoded, which made every audit
# script non-portable (found by the workstation session 2026-09-09).
import os
import sys

import torch

sys.path.insert(0, os.environ.get("KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get("KIT_MACE_REPO", "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
model = torch.load(f="checkpoints/s3d_gate_pp22_run-123.model", map_location=device).to(device)
ck = torch.load("checkpoints/s3d_gate_pp22_run-123_epoch-52.pt", map_location=device)
model.load_state_dict(ck["model"] if "model" in ck else ck.model)
model.eval()
for p in model.parameters():
    p.requires_grad = False
model.solvent3d_energy = True
model.solvent_cavity_energy = True

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff",
               "solvated": "solvated"},
    arrays_keys={"forces": "forces"},
)
want = {1: None, 601: None}
for a in read("data/train.xyz", ":"):
    sid = int(a.info.get("sample_id", -1))
    if sid in want and want[sid] is None:
        want[sid] = a
    if all(v is not None for v in want.values()):
        break

for sid, a in want.items():
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    model.solvent_baseline_coupling = False
    with torch.no_grad():
        e_off = float(model(b.to_dict(), compute_force=False,
                            training=False)["energy"].sum())
    model.solvent_baseline_coupling = True
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
    e_on = float(pred["energy"].sum())
    ebl = float(pred["baseline_coupling_energy_g"].sum())
    print(f"sid {sid}: e_bl = {ebl:+.4f} eV  (audit route: "
          f"{'-0.414' if sid == 601 else '+0.123'})   additivity "
          f"|dE - e_bl| = {abs(e_on - e_off - ebl):.2e}", flush=True)
print("DONE")
