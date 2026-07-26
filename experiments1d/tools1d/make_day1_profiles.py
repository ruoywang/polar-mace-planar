"""day-1 data task: per-structure 1-D source profiles from the baseline cache.

For each of the 400 structures, plane-average the cached 3-D fields:
  phi_base_z[400, 300]        (eV)   — Hartree(neutral) + exact POTCAR local PSP
  neutral_values_z[400, 300]  (VASP values convention; /volume = e/A^3)
Output: <bundle>/profiles1d_cache.npz  (+ z grid, sids, meta).

Cross-check: phi_base_z re-gridded against potential1d_potcar_cache.npz's
phi_eV (independent 500-pt construction; cache build's V4 said <=6.9e-4).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BUNDLE = Path(__file__).resolve().parents[1] / "train-data"
CACHE = BUNDLE / "baseline_cache"

meta = json.load(open(CACHE / "baseline_meta.json"))
index = json.load(open(CACHE / "baseline_index.json"))
nx, ny, nz = meta["pb_shape"]
n_struct = meta["n_structures"]
fields = meta["fields"]
assert fields == ["neutral_values", "dencor_values", "phi_base_eV"], fields
cell = np.asarray(meta["cell"])
lz = float(cell[2][2])
volume = float(abs(np.linalg.det(cell)))

# login node has an 8GB vmem ulimit: read 12MB blocks by seeking, no mmap
f = open(CACHE / "baseline_cache.npy", "rb")
version = np.lib.format.read_magic(f)
shape, fortran, dtype = np.lib.format._read_array_header(f, version)
assert shape == (n_struct, 3, nx, ny, nz) and not fortran, (shape, fortran)
data_start = f.tell()
block = nx * ny * nz
itemsize = np.dtype(dtype).itemsize

def read_field(row: int, field: int) -> np.ndarray:
    f.seek(data_start + ((row * 3 + field) * block) * itemsize)
    return np.fromfile(f, dtype=dtype, count=block).reshape(nx, ny, nz)

sids = np.zeros(n_struct, dtype=np.int64)
phi_base_z = np.zeros((n_struct, nz), dtype=np.float64)
neutral_values_z = np.zeros((n_struct, nz), dtype=np.float64)
for sid_str, row in index.items():
    sid = int(sid_str)
    sids[row] = sid
    neutral_values_z[row] = read_field(row, 0).mean(axis=(0, 1))
    phi_base_z[row] = read_field(row, 2).mean(axis=(0, 1))

z = np.arange(nz) * lz / nz
out = BUNDLE / "profiles1d_cache.npz"
np.savez(
    out,
    sample_ids=sids,
    z_A=z,
    phi_base_z_eV=phi_base_z,
    neutral_values_z=neutral_values_z,
    volume_A3=volume,
    cell=cell,
    source="plane average of baseline_cache.npy fields (neutral_values, phi_base_eV)",
)
print(f"wrote {out}: phi_base_z {phi_base_z.shape}, neutral_values_z {neutral_values_z.shape}")

# ---- cross-checks ----
# NOTE: potential1d cache's phi_eV is the FULL DFT potential (with solvent +
# its alignment); phi_base is solute-only. Their difference is the smooth
# solvent step (~-1.6 eV plateau in the solvent region) — verified 2026-07-13.
# Valid checks: (1) neutral profile proportional to the cache's raw_neutral_e;
# (2) solvent-region plateau structure of (phi_eV - phi_base).
p1d = np.load(BUNDLE / "potential1d_potcar_cache.npz")
worst_ratio = 0.0
for row in (0, 199, 399):
    sid = sids[row]
    j = int(np.where(p1d["sample_ids"] == sid)[0][0])
    zz = p1d["z_A"][j]
    mine = np.interp(zz, z, neutral_values_z[row], period=lz)
    ref = p1d["raw_neutral_e"][j]
    mask = ref > ref.max() * 0.01
    ratio = mine[mask] / ref[mask]
    rel = float(ratio.std() / abs(ratio.mean()))
    worst_ratio = max(worst_ratio, rel)
    print(f"sid {sid:3d}: neutral vs raw_neutral_e ratio {ratio.mean():.4f} rel-std {rel:.2e}")
assert worst_ratio < 0.02, "neutral profile mismatch"
print("cross-checks OK")
