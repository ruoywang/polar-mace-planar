"""Model-side energy decomposition on the SAME two frames as the DFT
reference audit (sid 1 = 1-44_GCE/cal_1 charged, sid 601 = neutral cal_1),
term-by-term, same potential conventions:
  - compensation_periodic_1d_energy  (model 1-D cross+self, its own fields)
  - slab dipole correction delta     (with the solvent dipole term)
  - E_cav, and E_3d split into e_xsol / e_self (the 3-D lateral recovery)
Reference numbers to align against (audit 2026-09-08):
  charged: E_int3 -5.963 E_self3 +2.211 (1D: -3.303/+1.524), lateral
           int -2.660 self +0.687; target A_solv -3.6421
  neutral: E_int3 -1.765 E_self3 +0.682 (1D: -0.553/+0.019), lateral
           int -1.211 self +0.664; target A_solv -1.0126
cwd = gate_pp2. Model: pp2 ep52 EMA weights (checkpoint).
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
import mace.modules.extensions as ext
import mace.modules.pb1d_backend as PB

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

cap = {}
_comp = ext._slab_compensation_periodic_1d_energy_radial
def wrap_comp(**kw):
    out = _comp(**kw)
    cap["comp"] = out.detach().clone()
    return out
ext._slab_compensation_periodic_1d_energy_radial = (
    lambda *a, **k: wrap_comp(**k) if not a else _comp(*a, **k))

_dip = ext._slab_dipole_correction_delta
def wrap_dip(**kw):
    out = _dip(**kw)
    cap.setdefault("dipcorr", []).append(out.detach().clone())
    return out
ext._slab_dipole_correction_delta = (
    lambda *a, **k: wrap_dip(**k) if not a else _dip(*a, **k))

_solve = PB.PB1DBackend.solve_graph
def wrap_solve(self, *a, **k):
    out = _solve(self, *a, **k)
    if out.get("s3d_obs") is not None:
        cap["obs"] = out["s3d_obs"]
    return out
PB.PB1DBackend.solve_graph = wrap_solve

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
    cap.clear()
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
    ecav = float(pred["cavity_energy_g"].sum())
    es3d = float(pred["solvent3d_energy_g"].sum())
    obs = cap.get("obs", {})
    exsol = obs.get("e_xsol", float("nan"))
    eself = obs.get("e_self", float("nan"))
    comp = float(cap["comp"].sum()) if "comp" in cap else float("nan")
    dips = [float(d.sum()) for d in cap.get("dipcorr", [])]
    print(f"\n== sid {sid} (q={float(a.info['total_charge']):+.2f}) ==")
    print(f"  compensation_1d (cross+self, model fields): {comp:+.4f} eV")
    print(f"  slab dipole corrections per call: "
          f"{' '.join(f'{d:+.4f}' for d in dips)}")
    print(f"  E_cav {ecav:+.4f}   E_3d {es3d:+.4f} "
          f"(e_xsol {exsol:+.4f}  e_self {eself:+.4f})")
    print(f"  model 'solvent Coulomb total' = comp + E_3d = {comp + es3d:+.4f}",
          flush=True)
print("DONE")
