"""Extract DFT solvent reference profiles (RHOION/RHOB plane averages) for one sid.

sid 1..200   -> 2-NiN_single/1-44_GCE/cal_<sid>
sid 201..400 -> 2-NiN_single/2-88_GCE/cal_<sid-200>
Verifies the structure matches the given xyz frame before writing.

Usage: python extract_dft_solvent_ref.py <sid> <xyz> <index> <out_npz>
"""
from __future__ import annotations

import sys

import numpy as np
from ase.io import read


def read_chgcar_like(path):
    with open(path) as f:
        f.readline()
        scale = float(f.readline())
        lat = np.array([[float(x) for x in f.readline().split()] for _ in range(3)]) * scale
        line = f.readline().split()
        try:
            counts = [int(x) for x in line]
        except ValueError:
            counts = [int(x) for x in f.readline().split()]
        nat = sum(counts)
        f.readline()
        for _ in range(nat):
            f.readline()
        f.readline()
        nx, ny, nz = [int(x) for x in f.readline().split()]
        need = nx * ny * nz
        vals = []
        while len(vals) < need:
            vals.extend(float(x) for x in f.readline().split())
        return lat, np.array(vals[:need]).reshape(nz, ny, nx)


sid, xyz, idx, out = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
sub = f"1-44_GCE/cal_{sid}" if sid <= 200 else f"2-88_GCE/cal_{sid-200}"
base = f"/scratch/08384/tg876840/tmp/2-NiN_single/{sub}/"

a_xyz = read(xyz, str(idx))
assert int(a_xyz.info["sample_id"]) == sid, "sid mismatch with xyz frame"
a_dft = read(base + "CONTCAR")
dmax = float(abs(a_xyz.positions - a_dft.positions).max())
assert dmax < 1e-6, f"structure mismatch: {dmax}"

lat, rion = read_chgcar_like(base + "RHOION")
_, rb = read_chgcar_like(base + "RHOB")
V = abs(np.linalg.det(lat))
nz = rion.shape[0]
lz = float(lat[2, 2])
z = np.arange(nz) * lz / nz
ion_z = rion.mean(axis=(1, 2)) / V
rb_z = rb.mean(axis=(1, 2)) / V
A = V / lz
q_int = float(np.trapz(ion_z, z) * A)
print(f"sid {sid}: struct match {dmax:.1e} A; ion integral {q_int:+.4f} (VASP conv); "
      f"total_charge {a_xyz.info['total_charge']:+.4f}")
np.savez(out, z=z, ion_z=ion_z, rb_z=rb_z, lz=lz)
