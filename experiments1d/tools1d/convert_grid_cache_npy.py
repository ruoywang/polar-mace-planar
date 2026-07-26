"""Convert the density_3d grid cache from compressed npz to raw .npy.

The training loader decompresses a 54 MB npz PER STEP (LRU=32 misses under
global shuffle) — measured ~110 ms/step. Raw .npy lets the loader mmap and
fancy-index only the 1024 sampled points. Values are byte-identical.

Writes: <grid_cache>_npy/density3d_net_grid_<sid>.npy       (rho, f32)
        <grid_cache>_npy/density3d_net_grid_<sid>_meta.npz  (lattice, valid_iz)
        <manifest dir>/density3d_net_grid_manifest_npy.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

manifest_path = Path(sys.argv[1]).resolve()
with open(manifest_path) as f:
    manifest = json.load(f)
entries = manifest["entries"]

out_dir = None
new_entries = {}
for sid, entry in sorted(entries.items(), key=lambda kv: int(kv[0])):
    src = Path(entry["path"])
    if not src.is_absolute():
        src = manifest_path.parent / src
    if out_dir is None:
        out_dir = src.parent.parent / (src.parent.name + "_npy")
        out_dir.mkdir(exist_ok=True)
    dst_rho = out_dir / (src.stem + ".npy")
    dst_meta = out_dir / (src.stem + "_meta.npz")
    if not (dst_rho.exists() and dst_meta.exists()):
        with np.load(src) as d:
            rho = np.asarray(d["rho"], dtype=np.float32)
            lattice = np.asarray(d["lattice"], dtype=np.float64)
            valid_iz = np.asarray(d["valid_iz"], dtype=np.int64)
        np.save(dst_rho, rho)
        np.savez(dst_meta, lattice=lattice, valid_iz=valid_iz)
    new_entries[sid] = dict(entry)
    new_entries[sid]["path"] = str(dst_rho)
    new_entries[sid]["meta_path"] = str(dst_meta)

new_manifest = dict(manifest)
new_manifest["format"] = "density3d_grid_npy_v1"
new_manifest["entries"] = new_entries
out_manifest = manifest_path.parent / (manifest_path.stem + "_npy.json")
with open(out_manifest, "w") as f:
    json.dump(new_manifest, f)
print(f"converted {len(new_entries)} entries -> {out_dir}")
print(f"manifest: {out_manifest}")
