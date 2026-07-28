"""Backfill neutral sid 594 (cal_194, ALGO=All EDIFF=1e-5) into NiN-mix.

Preconditions (produced by extract + build in the same job):
  exp_neutral_prep/audit.tsv         cal_194 row good
  exp_neutral_prep/neutral_draft.xyz 200 frames
  exp_neutral_prep/potential1d_neutral.npz  has sid 594
  exp_neutral_prep/grid_cache_npy/density3d_net_grid_594.npy + meta
  exp_neutral_prep/manifest_neutral.json    entry "594"

Bundle actions (split policy: appended to TRAIN — val/test frozen so all
earlier runs stay comparable):
  train.xyz            += frame 594 (solvated=0)
  potential1d cache    += row 594
  grid_cache_npy/      += npy + meta, manifest entry
Baseline cache untouched: neutral frames never reach the 3-D baseline
(solvated gating), and the center profile comes from the potential1d row.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

BASE = Path("/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB")
PREP = BASE / "exp_neutral_prep"
BUNDLE = Path("/scratch/08384/tg876840/tmp/c-MACEsol/data/NiN-mix")
SID = 594

# 0) audit row must be good
audit = {int(l.split("\t")[0]): l.split("\t")
         for l in open(PREP / "audit.tsv").read().splitlines()[1:]}
row = audit[194]
assert row[1] == "1", f"cal_194 audit not good: {row}"
print("audit cal_194:", row[:6])

# 1) frame
frames = [a for a in read(PREP / "neutral_draft.xyz", ":")
          if int(a.info["sample_id"]) == SID]
assert len(frames) == 1, f"draft has {len(frames)} frames for sid {SID}"
at = frames[0]
at.info["solvated"] = 0
present = any(f"sample_id={SID}" in l for l in open(BUNDLE / "train.xyz"))
if present:
    all_frames = read(BUNDLE / "train.xyz", ":")
    n_before = len(all_frames)
    all_frames = [f for f in all_frames if int(f.info.get("sample_id", -1)) != SID]
    all_frames.append(at)
    write(BUNDLE / "train_new.xyz", all_frames, format="extxyz")
    (BUNDLE / "train_new.xyz").rename(BUNDLE / "train.xyz")
    print(f"train.xyz: replaced sid {SID} in place ({n_before} frames)")
else:
    n_before = sum(1 for l in open(BUNDLE / "train.xyz") if "Lattice" in l)
    write(BUNDLE / "train.xyz", at, append=True)
    print(f"train.xyz: appended sid {SID} ({n_before} -> {n_before + 1})")

# 2) potential1d row
neu = np.load(PREP / "potential1d_neutral.npz")
i = list(np.asarray(neu["sample_ids"]).tolist()).index(SID)
bun = np.load(BUNDLE / "potential1d_potcar_cache.npz")
bsids = np.asarray(bun["sample_ids"]).tolist()
merged = {}
if SID in bsids:
    j = bsids.index(SID)
    for k in bun.files:
        arr = bun[k].copy()
        arr[j] = neu[k][i]
        merged[k] = arr
    print(f"potential1d: replaced row for sid {SID}")
else:
    for k in bun.files:
        merged[k] = np.concatenate([bun[k], neu[k][i:i + 1]], axis=0)
tmp = BUNDLE / "potential1d_potcar_cache.npz.tmp.npz"
np.savez_compressed(tmp, **merged)
tmp.rename(BUNDLE / "potential1d_potcar_cache.npz")
print(f"potential1d: {len(bun['sample_ids'])} -> {len(merged['sample_ids'])} entries")

# 3) grid npy + manifest
src_man = json.load(open(PREP / "manifest_neutral.json"))
entry = src_man["entries"][str(SID)]
for key in ("path", "meta_path"):
    src = Path(entry[key])
    dst = BUNDLE / "grid_cache_npy" / src.name
    shutil.copy2(src, dst)
    entry[key] = str(dst)
man = json.load(open(BUNDLE / "density3d_net_grid_manifest_npy.json"))
man["entries"][str(SID)] = entry  # insert or replace
with open(BUNDLE / "density3d_net_grid_manifest_npy.json", "w") as f:
    json.dump(man, f)
print(f"manifest: {len(man['entries'])} entries")

# 4) verification
g = np.load(BUNDLE / "grid_cache_npy" / Path(entry["path"]).name, mmap_mode="r")
print(f"grid 594 shape {g.shape}; sanity mean {float(np.asarray(g[::8, ::8, ::8]).mean()):.3e}")
print("BACKFILL 594 DONE")
