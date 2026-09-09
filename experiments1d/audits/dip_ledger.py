"""Single-frame periodic-bookkeeping ledger, model side (user directive:
close ONE charged frame from a common Coulomb reference before any 20-frame
regression). Captures the dipole-correction inputs (explicit/total mu) of
BOTH calls (observable path, energy path), the implied base correction
2pi/V*mu_exp^2, electro_energy, and every solvent term. sid 1 + 601,
pp2 ep52 weights. cwd = gate_pp2_ext.
"""
import sys

import torch

sys.path.insert(0, "/work/08384/tg876840/ls6/repos/cep-dip-python-pb")
sys.path.insert(0, "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy")

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.extensions as ext

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
model.solvent_baseline_coupling = True

cap = {}
_dip = ext._slab_dipole_correction_delta
def wrap_dip(**kw):
    out = _dip(**kw)
    cap.setdefault("calls", []).append({
        "mu_e": float(kw["explicit_dipole"][0, 2]),
        "mu_t": float(kw["total_dipole"][0, 2]),
        "V": float(kw["volume"].view(-1)[0]),
        "dE": float(out.sum()),
    })
    return out
ext._slab_dipole_correction_delta = lambda *a, **k: wrap_dip(**k)

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

C = 90.47564011373334
for sid, a in want.items():
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    cap.clear()
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
    print(f"\n== sid {sid} (q={float(a.info['total_charge']):+.2f}) ==")
    for i, c in enumerate(cap.get("calls", [])):
        mu_s = c["mu_t"] - c["mu_e"]
        base = C * c["mu_e"] ** 2 / c["V"]
        tot = C * c["mu_t"] ** 2 / c["V"]
        print(f"  call {i}: mu_exp {c['mu_e']:+8.3f}  mu_solv {mu_s:+8.3f}  "
              f"mu_tot {c['mu_t']:+8.3f}  dE {c['dE']:+8.4f}  "
              f"[2pi/V*mu_e^2 {base:+7.4f}  2pi/V*mu_t^2 {tot:+7.4f}]")
    print(f"  electro_energy {float(pred['electrostatic_energy'].sum()):+.4f}"
          f"  E_total {float(pred['energy'].sum()):+.4f}"
          f"  E_bl {float(pred['baseline_coupling_energy_g'].sum()):+.4f}",
          flush=True)
print("DONE")
