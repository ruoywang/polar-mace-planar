"""3c gate: runtime baseline vs cache, three levels.

L1 field parity: RuntimeBaselineTables (torch/GPU path) vs cached rows.
L2 model parity: same solvated val structures evaluated (a) with sample_id
   (cache path) and (b) with sample_id stripped (runtime baseline + the
   multipole center fallback). Reports per-quantity diffs.
L3 dynamics: 50-step Langevin on a solvated frame with no sample_id — the
   first true "new-geometry solvated MD".

Run from a dir with ./data ./cache ./cal1_train.json.
"""
import json
import os
import sys
import time

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"

import numpy as np
import torch
from ase.io import read

torch.set_default_dtype(torch.float64)
MODEL = sys.argv[1]
XYZ = sys.argv[2]

model = torch.load(MODEL, map_location="cuda")
model.eval()

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.modules.pb1d_runtime_baseline import RuntimeBaselineTables
from mace.tools import torch_geometric, utils

# ---------- L1: field parity on GPU ----------
tables = RuntimeBaselineTables("./data/baseline_cache/runtime_baseline_tables.npz")
index = {int(k): int(v) for k, v in
         json.load(open("./data/baseline_cache/baseline_index.json")).items()}
bl = np.load("./data/baseline_cache/baseline_cache.npy", mmap_mode="r")
frames = read(XYZ, ":")
solv = [(i, a) for i, a in enumerate(frames) if int(a.info.get("solvated", 1)) == 1]
cell = torch.tensor(tables.cell)
print("== L1 field parity (torch path vs cached rows) ==")
for i, at in solv[:3]:
    sid = int(at.info["sample_id"])
    frac = torch.tensor(at.get_positions() @ np.linalg.inv(tables.cell) % 1.0)
    node_z = torch.tensor(at.get_atomic_numbers())
    neutral, phi = tables.fields(frac.cuda(), node_z.cuda(), torch.device("cuda"))
    truth = bl[index[sid]][[0, 2]].astype(np.float64)
    for name, rec, tr in (("neutral", neutral, truth[0]), ("phi_base", phi, truth[1])):
        r = rec.cpu().numpy()
        rel = np.sqrt(np.mean((r - tr) ** 2)) / max(np.abs(tr).max(), 1e-30)
        print(f"  sid {sid:3d} {name}: rel-rmse {rel:.2e}")

# ---------- L2: model parity ----------
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
r_max = float(model.r_max)


def batch_for(at, with_sid: bool):
    info = {"total_charge": "total_charge", "total_spin": "total_spin",
            "solvated": "solvated"}
    if with_sid:
        info["sample_id"] = "sample_id"
    kspec = KeySpecification(info_keys=info, arrays_keys={})
    cfg = mace_data.config_from_atoms(at, key_specification=kspec)
    ad = mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=r_max)
    dl = torch_geometric.dataloader.DataLoader([ad], batch_size=1, shuffle=False)
    return next(iter(dl)).to("cuda")


print("== L2 model parity: ablation (cache / cache+rt-center / runtime) ==")
keys = (("energy", "energy"), ("forces", "forces"),
        ("potential", "potential"), ("fermi", "fermi_level_pred"))
saved_baselines = model.center_density_baselines
for i, at in solv[:6]:
    sid = int(at.info["sample_id"])
    outs = {}
    for tag in ("cache", "ctr", "runtime"):
        model.center_density_baselines = {} if tag == "ctr" else saved_baselines
        r = model(batch_for(at, tag != "runtime").to_dict(), training=False,
                  compute_force=True)
        outs[tag] = {k: r[key].detach().cpu().numpy() for k, key in keys}
    def signed(a, b, k):
        d = outs[a][k] - outs[b][k]
        return float(d.ravel()[np.argmax(np.abs(d))])
    print(f"  sid {sid:3d}  center-effect dpot {signed('ctr','cache','potential'):+.2e}"
          f" dfermi {signed('ctr','cache','fermi'):+.2e}"
          f" | rest dpot {signed('runtime','ctr','potential'):+.2e}"
          f" dE {signed('runtime','ctr','energy'):+.2e}"
          f" dF {signed('runtime','ctr','forces'):+.2e}")
model.center_density_baselines = saved_baselines

# ---------- L3: solvated MD without sample_id ----------
print("== L3 solvated MD (50 steps, no sample_id) ==")
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from mace.calculators import MACECalculator

at = solv[0][1].copy()
at.info.pop("sample_id", None)
at.info.pop("energy", None)
at.arrays.pop("forces", None)
calc = MACECalculator(
    model_paths=MODEL, device="cuda", default_dtype="float64",
    model_type="PolarMACE",
    info_keys={"total_charge": "total_charge", "total_spin": "total_spin",
               "solvated": "solvated"},
)
at.calc = calc
MaxwellBoltzmannDistribution(at, temperature_K=300)
Stationary(at)
dyn = Langevin(at, timestep=1.0 * units.fs, temperature_K=300, friction=0.02)
times, mus = [], []
for chunk in range(5):
    t0 = time.perf_counter()
    dyn.run(10)
    times.append((time.perf_counter() - t0) / 10)
    epot = at.get_potential_energy()
    mu = calc.results.get("potential")
    ef = calc.results.get("fermi_level")
    temp = at.get_kinetic_energy() / (1.5 * units.kB * len(at))
    mus.append(mu)
    print(f"  step {dyn.nsteps:3d}  Epot {epot:11.3f}  T {temp:5.0f} K  "
          f"potential {mu:.4f}  fermi {ef:.4f}  {times[-1]*1e3:.0f} ms/step",
          flush=True)
    assert np.isfinite(epot)
print(f"solvated MD: median {np.median(times)*1e3:.0f} ms/step")
print("RUNTIME-BASELINE GATE DONE")
