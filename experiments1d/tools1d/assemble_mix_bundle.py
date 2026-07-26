"""Assemble the mixed 400-solvated + 199-neutral training bundle (full copies).

- xyz: NiN-1d splits get solvated=1; neutral_draft.xyz split 80/10/10
  (seeded shuffle) and merged in.
- potential1d cache: solvated 400 + neutral 199 entries in one npz.
- density grids: copy both grid sets into the bundle, regenerate manifest
  with bundle-local absolute paths.
- baseline_cache, profiles1d_cache, cal1_train.json: copied (solvated only).

Usage: python assemble_mix_bundle.py <bundle_dir>
"""
from __future__ import annotations

import json
import random
import shutil
import sys

import numpy as np
from ase.io import read
from ase.io.extxyz import write_extxyz

SRC = "/scratch/08384/tg876840/tmp/c-MACEsol/data/NiN-1d"
PREP = "/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_neutral_prep"


def main():
    bundle = sys.argv[1]
    import os
    os.makedirs(f"{bundle}/grid_cache_npy", exist_ok=True)

    # ---- xyz merge -----------------------------------------------------
    neutral = read(f"{PREP}/neutral_draft.xyz", ":")
    rng = random.Random(123)
    rng.shuffle(neutral)
    n = len(neutral)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    parts = {"train": neutral[:n_tr], "val": neutral[n_tr:n_tr + n_va],
             "test": neutral[n_tr + n_va:]}
    counts = {}
    for split in ("train", "val", "test"):
        frames = read(f"{SRC}/{split}.xyz", ":")
        for a in frames:
            a.info["solvated"] = 1
        frames.extend(parts[split])
        with open(f"{bundle}/{split}.xyz", "w") as f:
            write_extxyz(f, frames)
        counts[split] = (len(frames) - len(parts[split]), len(parts[split]))
    print("splits (solvated + neutral):", counts)

    # ---- potential1d cache merge ---------------------------------------
    a = np.load(f"{SRC}/potential1d_potcar_cache.npz")
    b = np.load(f"{PREP}/potential1d_neutral.npz")
    merged = {}
    for k in a.files:
        merged[k] = np.concatenate([a[k], b[k]], axis=0)
    np.savez(f"{bundle}/potential1d_potcar_cache.npz", **merged)
    print(f"potential1d cache: {len(a['sample_ids'])} + {len(b['sample_ids'])} entries")

    # ---- density grids + manifest ---------------------------------------
    man = {"format": "density3d_grid_npy_v1", "entries": {}}
    for src_man, src_dir in [(f"{SRC}/density3d_net_grid_manifest_npy.json", None),
                             (f"{PREP}/manifest_neutral.json", None)]:
        m = json.load(open(src_man))
        for sid, e in m["entries"].items():
            for key in ("path", "meta_path"):
                src = e[key]
                dst = f"{bundle}/grid_cache_npy/{src.rsplit('/', 1)[-1]}"
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                e[key] = os.path.abspath(dst)
            man["entries"][sid] = e
    with open(f"{bundle}/density3d_net_grid_manifest_npy.json", "w") as f:
        json.dump(man, f)
    print(f"grids: {len(man['entries'])} entries copied")

    # ---- solvated-only assets -------------------------------------------
    for item in ("baseline_cache", "profiles1d_cache.npz"):
        src = f"{SRC}/{item}"
        dst = f"{bundle}/{item}"
        if os.path.isdir(src):
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print("copied", item)
    shutil.copy2("/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/cal1_train.json",
                 f"{bundle}/cal1_train.json")
    print("bundle complete:", bundle)


if __name__ == "__main__":
    import os
    main()
