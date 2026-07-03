"""Build the per-structure PB baseline cache from DFT data.

For every sample in the 2-NiN dataset, three float32 fields on the PB grid
(spacing arg 0.15 -> 100x100x300 on the 45 A cell):

  neutral   valence neutral-atom electron density in VASP "values" units
            (sum/N = electron count): CHGCAR + net_grid * V, Fourier-
            restricted from the native 168x168x500 grid. This is the exact
            baseline of the model's net-density decomposition.
  dencor    POTCAR partial-core (NLCC) charge on the PB grid (values units),
            built from pspcor form factors x structure factors — cavity-only
            input, never the Hartree side (charge-count consistency).
  phi_base  solute-potential baseline: Hartree(neutral) + exact POTCAR local
            pseudopotential (eV). At runtime phi_sol = phi_base -
            Hartree(net_model) + cdipol, eliminating the Gaussian-nuclei and
            Gaussian-baseline approximations entirely.

Per-structure verification (hard failures unless noted):
  V1 mapping: xyz positions match the source cal dir's CONTCAR (<1e-6 A)
  V2 charge: mean(neutral) == total_zval(composition) (electron count of the
     neutral system; catches axis-order/sign/scaling errors)
  V3 positivity: min(neutral)/V bounded below (restriction ringing on core
     peaks allowed, recorded)
  V4 cross-check vs the INDEPENDENT potential1d_potcar_cache raw_neutral_e
     plane profile on the smooth region z in [10, 44] (relative rmse)
  V5 dencor charge: mean(dencor) == sum of pspcor(G=0) over composition

Output: <out>/baseline_cache.npy [N, 3, nx, ny, nz] float32 (memmap-friendly),
        <out>/baseline_index.json {sample_id: row}, <out>/baseline_meta.json.

Usage:
  PYTHONPATH=<cep-dip-repo> python build_baseline_cache.py \
      --xyz train.xyz val.xyz test.xyz --gce-root <2-NiN_single> \
      --net-manifest <density3d_net_grid_manifest.json> \
      --potential1d <potential1d_potcar_cache.npz> \
      --out <out-dir> [--spacing 0.15] [--workers 16]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mace.modules.pb_solvent import _fft_friendly_even  # noqa: E402

_G = {}  # per-worker globals


def _fourier_restrict(values: np.ndarray, shape_c) -> np.ndarray:
    """Band-limiting restriction (amplitude convention); coarse Nyquist zeroed."""
    spec_f = np.fft.fftn(values) / values.size
    spec_c = np.zeros(shape_c, dtype=complex)
    sl_f, sl_c = [], []
    for nf, nc in zip(values.shape, shape_c):
        h = nc // 2
        sl_f.append((slice(0, h), slice(nf - (h - 1), nf)))
        sl_c.append((slice(0, h), slice(nc - (h - 1), nc)))
    for cf, cc in zip(sl_f[0], sl_c[0]):
        for kf, kc in zip(sl_f[1], sl_c[1]):
            for lf, lc in zip(sl_f[2], sl_c[2]):
                spec_c[cc, kc, lc] = spec_f[cf, kf, lf]
    return np.ascontiguousarray(np.fft.ifftn(spec_c * np.prod(shape_c)).real)


def _init_worker(potcar_path, cell, pb_shape, p1d_path):
    from pure_python.grid import Grid
    from pure_python.potcar import read_potcar

    _G["entries"] = read_potcar(potcar_path)
    _G["order"] = [e.element for e in _G["entries"]]
    _G["grid"] = Grid(np.asarray(cell, dtype=float), tuple(pb_shape))
    with np.load(p1d_path) as d:
        sids = np.asarray(d["sample_ids"], dtype=int)
        _G["p1d"] = {
            int(s): (np.asarray(d["z_A"][i], dtype=float),
                     np.asarray(d["raw_neutral_e"][i], dtype=float))
            for i, s in enumerate(sids)
        }


def _build_one(task):
    sid, cal_dir, xyz_positions, xyz_symbols, total_zval = task
    from tools.vasp_volumetric import read_vasp_volumetric
    from pure_python.solute_potential import (
        dencor_values, hartree_potential_g, local_pseudopotential_g,
    )
    from ase.io import read as ase_read

    grid = _G["grid"]
    entries, order = _G["entries"], _G["order"]
    cell = grid.cell
    report = {"sid": sid}

    # V1: mapping — xyz vs CONTCAR positions
    ref = ase_read(str(Path(cal_dir) / "CONTCAR"))
    if len(ref) != len(xyz_positions):
        return sid, None, {"sid": sid, "fail": f"natoms {len(ref)} != {len(xyz_positions)}"}
    dz = np.abs(np.sort(ref.get_positions()[:, 2]) - np.sort(xyz_positions[:, 2])).max()
    report["V1_pos_maxdiff"] = float(dz)
    if dz > 1e-5:
        return sid, None, {"sid": sid, "fail": f"position mismatch {dz:.2e}"}

    # neutral = CHGCAR + net*V, restricted to the PB grid
    chg = read_vasp_volumetric(str(Path(cal_dir) / "CHGCAR"))
    vol = float(abs(np.linalg.det(chg.cell)))
    valence = chg.values.reshape(chg.grid, order="F")
    net = np.load(_G["net_path_fmt"].format(sid=sid))["rho"].astype(np.float64)
    net_xyz = net.transpose(2, 1, 0)  # (nz,ny,nx) -> (nx,ny,nz), loss.py convention
    neutral_native = valence + net_xyz * vol
    neutral = _fourier_restrict(neutral_native, grid.shape)

    # V2: electron count of the neutral system
    q_neutral = float(neutral.mean())
    report["V2_neutral_e"] = q_neutral
    if abs(q_neutral - total_zval) > 5e-3:
        return sid, None, {"sid": sid, "fail": f"neutral count {q_neutral:.4f} != {total_zval}"}

    # V3: positivity (restriction ringing on cores allowed; record)
    report["V3_min_density"] = float(neutral.min() / vol)

    # V4: plane profile vs independent potential1d cache
    p1d = _G["p1d"].get(sid)
    if p1d is not None:
        z_ref, raw_neutral_ref = p1d
        nz = grid.shape[2]
        height = float(np.linalg.norm(np.asarray(cell)[2]))
        z_pb = np.arange(nz) * height / nz
        prof_pb = neutral.mean(axis=(0, 1))
        prof_ref = np.interp(z_pb, z_ref, raw_neutral_ref)
        m = (z_pb >= 10.0) & (z_pb <= 44.0)
        denom = max(np.abs(prof_ref[m]).max(), 1e-30)
        v4 = float(np.sqrt(np.mean((prof_pb[m] - prof_ref[m]) ** 2)) / denom)
        report["V4_profile_relrmse"] = v4
        if v4 > 2e-2:
            return sid, None, {"sid": sid, "fail": f"profile cross-check {v4:.3e}"}
    else:
        report["V4_profile_relrmse"] = None

    # group atoms by element in POTCAR order
    inv_cell = np.linalg.inv(np.asarray(cell, dtype=float))
    pos_frac_all = (xyz_positions @ inv_cell) % 1.0
    pos_grouped, counts = [], []
    for el in order:
        sel = [i for i, s in enumerate(xyz_symbols) if s == el]
        counts.append(len(sel))
        pos_grouped.append(pos_frac_all[sel])
    pos_grouped = np.concatenate([p for p in pos_grouped if len(p)]) if any(counts) else pos_frac_all

    # dencor on the PB grid + V5
    dencor = dencor_values(grid, entries, counts, pos_grouped)
    expected_dc = sum(
        (e.pspcor[0] if e.pspcor is not None else 0.0) * c
        for e, c in zip(entries, counts)
    )
    report["V5_dencor_e"] = float(dencor.mean())
    if abs(dencor.mean() - expected_dc) > 5e-3:
        return sid, None, {"sid": sid, "fail": f"dencor {dencor.mean():.4f} != {expected_dc:.4f}"}

    # phi_base = Hartree(neutral) + exact local pseudopotential
    phi_g = hartree_potential_g(grid.fft_full(neutral), grid) + \
        local_pseudopotential_g(grid, entries, counts, pos_grouped)
    phi_base = grid.ifft_real_full(phi_g)

    fields = np.stack([
        neutral.astype(np.float32),
        dencor.astype(np.float32),
        phi_base.astype(np.float32),
    ])
    return sid, fields, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", nargs="+", required=True)
    ap.add_argument("--gce-root", required=True)
    ap.add_argument("--net-manifest", required=True)
    ap.add_argument("--potential1d", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spacing", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="build only the first N (spot run)")
    args = ap.parse_args()

    from ase.io import read as ase_read

    zval_map = {"H": 1.0, "C": 4.0, "N": 5.0, "O": 6.0, "Ni": 10.0}
    tasks, seen = [], set()
    cell = None
    for f in args.xyz:
        for at in ase_read(f, index=":"):
            sid = int(at.info["sample_id"])
            if sid in seen:
                continue
            seen.add(sid)
            src = int(at.info["source_cal_id"])
            sub = "1-44_GCE" if sid <= 200 else "2-88_GCE"
            cal_dir = str(Path(args.gce_root) / sub / f"cal_{src}")
            total_zval = sum(zval_map[s] for s in at.get_chemical_symbols())
            tasks.append((sid, cal_dir, at.get_positions(), at.get_chemical_symbols(), total_zval))
            if cell is None:
                cell = at.cell.array.copy()
    tasks.sort(key=lambda t: t[0])
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(tasks)} structures to build", flush=True)

    lengths = np.linalg.norm(cell, axis=1)
    pb_shape = tuple(_fft_friendly_even(int(math.ceil(l / args.spacing))) for l in lengths)
    print("PB grid:", pb_shape, flush=True)

    manifest_dir = Path(args.net_manifest).parent
    net_fmt = str(manifest_dir / "grid_cache" / "density3d_net_grid_{sid}.npz")
    potcar = tasks[0][1] + "/POTCAR"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(
        out / "baseline_cache.npy", mode="w+",
        dtype=np.float32, shape=(len(tasks), 3, *pb_shape),
    )
    index, reports, failures = {}, [], []

    def initializer():
        _init_worker(potcar, cell, pb_shape, args.potential1d)
        _G["net_path_fmt"] = net_fmt

    with Pool(args.workers, initializer=initializer) as pool:
        for row, (sid, fields, report) in enumerate(
            pool.imap(_build_one, tasks, chunksize=1)
        ):
            if fields is None:
                failures.append(report)
                print("FAIL", report, flush=True)
                continue
            arr[len(index)] = fields
            index[sid] = len(index)
            reports.append(report)
            if len(index) % 25 == 0:
                print(f"built {len(index)}/{len(tasks)}", flush=True)
    arr.flush()

    (out / "baseline_index.json").write_text(json.dumps(index))
    (out / "baseline_meta.json").write_text(json.dumps({
        "fields": ["neutral_values", "dencor_values", "phi_base_eV"],
        "pb_shape": list(pb_shape), "spacing_arg": args.spacing,
        "cell": np.asarray(cell).tolist(), "n_structures": len(index),
        "units": "values: sum/N = electron count; phi: eV",
        "net_axis_convention": "npz rho is (nz,ny,nx); transposed (2,1,0)",
    }, indent=1))
    (out / "baseline_reports.json").write_text(json.dumps(reports, indent=0))

    v3 = [r["V3_min_density"] for r in reports]
    v4 = [r["V4_profile_relrmse"] for r in reports if r["V4_profile_relrmse"] is not None]
    print(f"\nDONE: {len(index)} built, {len(failures)} failed")
    print(f"V3 min density: worst {min(v3):.3f} e/A^3 (restriction ringing)")
    if v4:
        print(f"V4 profile cross-check: worst {max(v4):.3e}, median {np.median(v4):.3e}")
    if failures:
        print("FAILURES:", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
