"""Constant-potential MD: mix400 + adapted blue_moon NoseHoover (step 4).

Replicates the sg_converge example's grand-canonical machinery — ne as a
Nose-Hoover-thermostatted dynamical variable driven by (targetmu - mu) —
with the pb1d mixed model: mu = the model's fermi_level (same physical
scale as the old CEP 'potential', both referenced to the solvent zero),
charge coupling Q = zval_sum - ne (verified against NELECT). No bond
constraint in v1; runtime baseline supplies per-step solvent physics
(structure has no sample_id).

Usage: python run_gcmd.py <model> <xyz> <frame> <nsteps> <targetmu>
Run from a dir with ./data ./cache ./cal1_train.json.
"""
import os
import sys
import time

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"

import numpy as np
import torch
from ase import units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

torch.set_default_dtype(torch.float64)
MODEL, XYZ = sys.argv[1], sys.argv[2]
IDX, NSTEPS = int(sys.argv[3]), int(sys.argv[4])
TARGETMU = float(sys.argv[5])

from blue_moon_gc import NoseHoover
from mace.calculators import MACECalculator

ZVAL = {"H": 1, "C": 4, "N": 5, "O": 6, "Ni": 10}

atoms = read(XYZ, str(IDX))
assert int(atoms.info.get("solvated", 1)) == 1
q0 = float(atoms.info["total_charge"])
q_neutral = sum(ZVAL[s] for s in atoms.get_chemical_symbols())
ne0 = q_neutral - q0
print(f"frame {IDX}: {len(atoms)} atoms, Q0 {q0:+.4f}, zval_sum {q_neutral}, "
      f"ne0 {ne0:.4f}, targetmu {TARGETMU}")
atoms.info.pop("sample_id", None)
atoms.info.pop("energy", None)
atoms.arrays.pop("forces", None)

calc = MACECalculator(
    model_paths=MODEL, device="cuda", default_dtype="float64",
    model_type="PolarMACE",
    info_keys={"total_charge": "total_charge", "total_spin": "total_spin",
               "solvated": "solvated"},
)
atoms.calc = calc
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)

dyn = NoseHoover(
    atoms,
    timestep=1.0 * units.fs,
    constraints=[],
    shaketol=1e-5,
    shakemaxiter=1000,
    increm=0.0,
    temperature=300.0 * units.kB,
    ttime=40.0,
    Mne=660.74,
    eta_length=2,
    ne=ne0,
    targetmu=TARGETMU,
    q_neutral=q_neutral,
)

hist = []
t0 = time.perf_counter()
for i in range(NSTEPS):
    ts = time.perf_counter()
    dyn.step()
    dt_step = time.perf_counter() - ts
    mu = calc.results["fermi_level"]
    q = float(atoms.info["total_charge"])
    temp = atoms.get_kinetic_energy() / (1.5 * units.kB * len(atoms))
    hist.append((i, mu, q, temp, dt_step))
    if (i + 1) % 10 == 0:
        print(f"step {i+1:4d}  mu {mu:+.4f} (target {TARGETMU:+.2f})  "
              f"Q {q:+.4f}  T {temp:5.0f} K  {dt_step*1e3:.0f} ms", flush=True)
        write("gcmd_traj.xyz", atoms, append=(i > 10))
    assert np.isfinite(mu) and temp < 1500

arr = np.array([(h[1], h[2], h[3]) for h in hist])
n2 = len(arr) // 2
print(f"\n{NSTEPS} steps in {(time.perf_counter()-t0)/60:.1f} min, "
      f"median {np.median([h[4] for h in hist])*1e3:.0f} ms/step")
print(f"second half: mu {arr[n2:,0].mean():+.4f} +- {arr[n2:,0].std():.4f} "
      f"(target {TARGETMU:+.4f}), Q {arr[n2:,1].mean():+.4f} +- {arr[n2:,1].std():.4f}, "
      f"T {arr[n2:,2].mean():.0f} K")
np.savez("gcmd_hist.npz", mu=arr[:, 0], q=arr[:, 1], T=arr[:, 2])
print("GCMD DONE")
