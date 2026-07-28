"""Convert a 3-field baseline cache to the trimmed 2-field format.

Drops the never-read dencor plane (field 1): [N,3,nx,ny,nz] ->
[N,2,nx,ny,nz] with meta fields ["neutral_values", "phi_base_eV"].
Streams row by row (login-node safe). Writes <dir>/baseline_cache_2f.npy
next to the original, verifies a few rows bit-exactly, then swaps
atomically (original renamed to baseline_cache_3f.npy.bak unless --purge).
"""
import json
import os
import sys

import numpy as np

d = sys.argv[1]
purge = len(sys.argv) > 2 and sys.argv[2] == "--purge"
meta = json.load(open(f"{d}/baseline_meta.json"))
fields = meta["fields"]
assert fields == ["neutral_values", "dencor_values", "phi_base_eV"], fields
nx, ny, nz = meta["pb_shape"]
n = meta["n_structures"]

src = open(f"{d}/baseline_cache.npy", "rb")
version = np.lib.format.read_magic(src)
shape, fortran, dtype = np.lib.format._read_array_header(src, version)
assert shape == (n, 3, nx, ny, nz) and not fortran
data_start = src.tell()
block = nx * ny * nz
itemsize = np.dtype(dtype).itemsize

dst_path = f"{d}/baseline_cache_2f.npy"
dst = open(dst_path, "wb")
hdr = {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
       "fortran_order": False, "shape": (n, 2, nx, ny, nz)}
np.lib.format.write_array_header_2_0(dst, hdr)
for row in range(n):
    for fi in (0, 2):
        src.seek(data_start + (row * 3 + fi) * block * itemsize)
        dst.write(src.read(block * itemsize))
    if (row + 1) % 100 == 0:
        print(f"  {row+1}/{n}")
dst.close()

# verify a few rows
new = np.load(dst_path, mmap_mode="r")
old = np.load(f"{d}/baseline_cache.npy", mmap_mode="r")
for row in (0, n // 2, n - 1):
    assert np.array_equal(np.asarray(new[row][0]), np.asarray(old[row][0]))
    assert np.array_equal(np.asarray(new[row][1]), np.asarray(old[row][2]))
print("verification: 3 rows bit-exact")
del new, old

meta["fields"] = ["neutral_values", "phi_base_eV"]
json.dump(meta, open(f"{d}/baseline_meta.json", "w"))
if purge:
    os.remove(f"{d}/baseline_cache.npy")
else:
    os.rename(f"{d}/baseline_cache.npy", f"{d}/baseline_cache_3f.npy.bak")
os.rename(dst_path, f"{d}/baseline_cache.npy")
print(f"converted {d}: {n} rows, 3 -> 2 fields "
      f"({'old purged' if purge else 'old kept as .bak'})")
