"""Plane-averaged valence electron density n_e(z) from CHGCAR pairs (sid 122 charged / 722 neutral,
prod500_w1000_ref): DFT and fully-ML CHGCARs (build_chgcar_prod500.py), plus the DFT net targets.
Usage: python extract_electron_profiles_prod500.py <out npz>
Reader copied from work_w200page/extract_electron_profiles.py (streamed, CHGCAR stores rho*V, x fastest).
"""
import re
import sys
import time
from pathlib import Path

import numpy as np

NEG = re.compile(rb"(?<=[0-9])-")
CHUNK = 1 << 24
HERE = Path("/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/exp_pair122_prod500")
CASES = {
    "charged_dft": Path("/scratch/08384/tg876840/tmp/2-NiN_single/1-44_GCE/cal_122/CHGCAR"),
    "charged_ml": HERE / "ml_sol_prod500/CHGCAR",
    "neutral_dft": Path("/scratch/08384/tg876840/tmp/2-NiN_single/5-44_neutral_withsolv/cal_122/CHGCAR"),
    "neutral_ml": HERE / "ml_neutral_prod500/CHGCAR",
}
NPY = {"charged": "/scratch/08384/tg876840/tmp/c-MACEsol/data/NiN-mix800/grid_cache_npy/density3d_net_grid_122.npy",
       "neutral": "/scratch/08384/tg876840/tmp/c-MACEsol/data/NiN-mix800/grid_cache_npy/density3d_net_grid_722.npy"}
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "electron_profiles_prod500.npz")


def plane_average(path: Path):
    with open(path, "rb") as f:
        cell_rows, dims = [], None
        for ln in range(400):
            line = f.readline()
            if ln == 1:
                scale = float(line.split()[0])
            elif 2 <= ln <= 4:
                cell_rows.append([float(x) for x in line.split()[:3]])
            parts = line.split()
            if len(parts) == 3 and all(p.isdigit() for p in parts) and ln > 5:
                dims = tuple(int(p) for p in parts)
                break
        if dims is None:
            raise ValueError(f"{path}: grid dimensions line not found")
        nx, ny, nz = dims
        ngrid = nx * ny * nz
        cell = np.asarray(cell_rows) * scale
        vol = abs(float(np.linalg.det(cell)))
        vals = np.empty(ngrid, dtype=np.float64)
        pos, tail = 0, b""
        while pos < ngrid:
            blk = f.read(CHUNK)
            if not blk:
                break
            blk = tail + blk
            cut = blk.rfind(b"\n")
            blk, tail = blk[:cut], blk[cut + 1:]
            stop = blk.find(b"augmentation")
            if stop >= 0:
                blk = blk[:stop]
            v = np.fromstring(NEG.sub(b" -", blk).decode(), dtype=np.float64, sep=" ")
            take = min(v.size, ngrid - pos)
            vals[pos:pos + take] = v[:take]
            pos += take
            if stop >= 0:
                break
        if pos != ngrid:
            raise ValueError(f"{path}: read {pos} of {ngrid} grid values")
    grid = vals.reshape(nz, ny, nx)
    nbar = grid.mean(axis=(1, 2)) / vol
    z = np.arange(nz) * float(cell[2, 2]) / nz
    total = float(vals.sum() / ngrid)
    return z, nbar, dims, vol, total


store = {}
for tag, p in CASES.items():
    t0 = time.time()
    z, nbar, dims, vol, total = plane_average(p)
    store[f"z_{tag}"] = z; store[f"ne_{tag}"] = nbar; store[f"nelect_{tag}"] = total
    print(f"{tag:12s} dims={dims} vol={vol:.3f} A^3 electrons={total:.4f} nbar range [{nbar.min():.5f}, {nbar.max():.5f}] e/A^3 ({time.time() - t0:.1f} s)", flush=True)
for st, npy in NPY.items():
    tgt = np.load(npy).astype(np.float64)
    store[f"net_dft_{st}"] = tgt.mean(axis=(1, 2))
np.savez_compressed(OUT, **store)
print("wrote", OUT, flush=True)
