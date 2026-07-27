"""jit verification: eager vs jit.compile on GPU — numeric identity + speed.

Loads the mix150 production model, evaluates N structures from the mixed
val split (solvated + neutral) twice per variant, and reports:
  1. run-to-run GPU jitter of the eager model (identity criterion base)
  2. eager vs compiled max diffs for E / F / potential / fermi
  3. forward(+force) wall time per structure for both variants
PASS if diff <= max(10 * jitter, 1e-8) for every quantity.
Run from a dir with ./data and ./cache (probe_solvated_identity.py contract).
"""
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
NSTRUCT = int(sys.argv[3]) if len(sys.argv) > 3 else 12

model = torch.load(MODEL, map_location="cuda")
model.eval()

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
r_max = float(model.r_max)
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "solvated": "solvated",
               "fermi_level": "Fermi", "potential": "potential_diff"},
    arrays_keys={"forces": "forces", "charges": "REF_charges"},
)

frames = read(XYZ, ":")[:NSTRUCT]
print(f"{len(frames)} frames; solvated flags:",
      [int(a.info.get("solvated", 1)) for a in frames])

batches = []
for at in frames:
    cfg = mace_data.config_from_atoms(at, key_specification=kspec)
    ad = mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=r_max)
    dl = torch_geometric.dataloader.DataLoader([ad], batch_size=1, shuffle=False)
    batches.append(next(iter(dl)).to("cuda"))


def run(m, tag):
    out = {"energy": [], "forces": [], "potential": [], "fermi": []}
    times = []
    for batch in batches:
        d = batch.to_dict()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = m(d, training=False, compute_force=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        out["energy"].append(r["energy"].detach().double().cpu().numpy())
        out["forces"].append(r["forces"].detach().double().cpu().numpy())
        out["potential"].append(r["potential"].detach().double().cpu().numpy())
        out["fermi"].append(r["fermi_level_pred"].detach().double().cpu().numpy())
    arrs = {k: np.concatenate([a.ravel() for a in v]) for k, v in out.items()}
    print(f"{tag}: median fwd {np.median(times)*1e3:.1f} ms/structure "
          f"(first {times[0]*1e3:.0f} ms)")
    return arrs, times


from copy import deepcopy
from e3nn.util import jit

def attr_snapshot(m):
    snap = {"": set(m.__dict__.keys())}
    for name, sub in m.named_modules():
        snap[name] = set(sub.__dict__.keys())
    return snap

snap0 = attr_snapshot(model)
compiled = jit.compile(deepcopy(model))
print("COMPILE-BEFORE-FORWARD OK")

eager1, t_e1 = run(model, "eager pass 1")
eager2, t_e2 = run(model, "eager pass 2")

snap1 = attr_snapshot(model)
for name in snap1:
    newk = snap1[name] - snap0.get(name, set())
    if newk:
        print(f"attrs added by forward on '{name or '<root>'}': {sorted(newk)}")
try:
    jit.compile(deepcopy(model))
    print("COMPILE-AFTER-FORWARD OK")
except Exception as e:
    print(f"COMPILE-AFTER-FORWARD FAIL: {type(e).__name__}: {str(e)[:300]}")

comp1, t_c1 = run(compiled, "compiled pass 1")
comp2, t_c2 = run(compiled, "compiled pass 2")

jitter = {k: float(np.abs(eager1[k] - eager2[k]).max()) for k in eager1}
diff = {k: float(np.abs(eager1[k] - comp1[k]).max()) for k in eager1}
print("gpu jitter (eager vs eager):", jitter)
print("eager vs compiled:", diff)
ok = all(diff[k] <= max(10.0 * jitter[k], 1e-8) for k in diff)
print("IDENTITY-OK" if ok else "IDENTITY-FAIL")
print(f"speed: eager {np.median(t_e1+t_e2)*1e3:.1f} ms vs compiled "
      f"{np.median(t_c1+t_c2)*1e3:.1f} ms per structure "
      f"-> x{np.median(t_e1+t_e2)/np.median(t_c1+t_c2):.2f}")
sys.exit(0 if ok else 1)
