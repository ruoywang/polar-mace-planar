"""MD smoke test: vacuum (solvated=0) Langevin MD with the mix pb1d model.

Purpose: prove the ASE MD pipeline end to end with the NEW model —
calculator load, potential/fermi_level properties, forces, stable dynamics.
Vacuum frame -> pb1d per-graph gating skips the solvent solve, so no
baseline cache rows are needed for moving geometries.

Run from a dir with ./data (bundle symlink) and ./cache.
Usage: python md_smoke.py <model> <xyz> <frame_index> <nsteps>
"""
import os
import sys
import time

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"

import numpy as np
import torch
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

torch.set_default_dtype(torch.float64)

MODEL, XYZ, IDX, NSTEPS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

from mace.calculators import MACECalculator

atoms = read(XYZ, str(IDX))
assert int(atoms.info.get("solvated", 1)) == 0, "smoke test expects a vacuum frame"
print(f"frame {IDX}: {len(atoms)} atoms, solvated={atoms.info['solvated']}, "
      f"charge={atoms.info.get('total_charge')}")
# MD starts from the DFT geometry but immediately moves off it: after step 1
# every geometry is new — exactly the deployment situation for vacuum frames.
atoms.info.pop("energy", None)
atoms.arrays.pop("forces", None)

calc = MACECalculator(
    model_paths=MODEL,
    device="cuda",
    default_dtype="float64",
    model_type="PolarMACE",
    info_keys={"total_charge": "total_charge", "total_spin": "total_spin",
               "solvated": "solvated"},
)
atoms.calc = calc

MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)

dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.02)

t_hist, log = [], []


def report():
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    temp = ekin / (1.5 * units.kB * len(atoms))
    mu = calc.results.get("potential")
    ef = calc.results.get("fermi_level")
    step = dyn.nsteps
    log.append((step, epot, ekin, temp, mu, ef))
    print(f"step {step:4d}  Epot {epot:12.4f} eV  T {temp:6.1f} K  "
          f"potential {mu if mu is None else f'{mu:.4f}'}  "
          f"fermi {ef if ef is None else f'{ef:.4f}'}", flush=True)
    assert np.isfinite(epot) and np.isfinite(ekin), "non-finite energy"
    write("smoke_traj.xyz", atoms, append=step > 0)


dyn.attach(report, interval=10)
t0 = time.perf_counter()
report()
for chunk in range(NSTEPS // 10):
    ts = time.perf_counter()
    dyn.run(10)
    t_hist.append((time.perf_counter() - ts) / 10)

print(f"\n{NSTEPS} steps in {time.perf_counter()-t0:.1f} s; "
      f"median {np.median(t_hist)*1e3:.0f} ms/step")
temps = [r[3] for r in log[2:]]
mus = [r[4] for r in log if r[4] is not None]
print(f"T range after equilibration: {min(temps):.0f}-{max(temps):.0f} K")
if mus:
    print(f"potential range: {min(mus):.4f} to {max(mus):.4f} eV")
drift = abs(log[-1][1] - log[2][1]) / len(atoms)
print(f"Epot drift (step 20 -> end): {drift*1e3:.2f} meV/atom")
print("SMOKE OK" if (mus and max(temps) < 1000) else "SMOKE FAIL")
