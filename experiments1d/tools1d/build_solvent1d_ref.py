"""Build the 400-structure DFT solvent 1-D reference pack (RHOB/RHOION plane averages).

For each sample_id in the given xyz files:
  sid 1..200   -> 2-NiN_single/1-44_GCE/cal_<sid>
  sid 201..400 -> 2-NiN_single/2-88_GCE/cal_<sid-200>
Verifies CONTCAR positions against the xyz frame, plane-averages both grids,
and stores stacked arrays keyed by a sample_ids index (same layout style as
potential1d_potcar_cache.npz). Values are kept in the raw VASP sign convention
(electron-positive); consumers negate for the physics convention.

Usage: python build_solvent1d_ref.py <out_npz> <xyz1> [<xyz2> ...]
"""
from __future__ import annotations

import sys
from multiprocessing import Pool

import numpy as np
from ase.io import read

DFT_BASE = "/scratch/08384/tg876840/tmp/2-NiN_single"
N_WORKERS = 16


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


def extract_one(args):
    sid, positions, q_tot = args
    sub = f"1-44_GCE/cal_{sid}" if sid <= 200 else f"2-88_GCE/cal_{sid - 200}"
    base = f"{DFT_BASE}/{sub}/"
    a_dft = read(base + "CONTCAR")
    dmax = float(abs(positions - a_dft.positions).max())
    assert dmax < 1e-6, f"sid {sid}: structure mismatch {dmax}"

    lat, rion = read_chgcar_like(base + "RHOION")
    lat2, rb = read_chgcar_like(base + "RHOB")
    assert np.allclose(lat, lat2), f"sid {sid}: RHOION/RHOB lattice mismatch"
    V = abs(np.linalg.det(lat))
    nz = rion.shape[0]
    lz = float(lat[2, 2])
    z = np.arange(nz) * lz / nz

    ion_z = rion.mean(axis=(1, 2)) / V  # e/A^3, VASP sign
    rb_z = rb.mean(axis=(1, 2)) / V
    A = V / lz
    q_ion = float(np.trapz(ion_z, z) * A)
    q_rb = float(np.trapz(rb_z, z) * A)
    # in the VASP electron-positive convention RHOION integrates to +total_charge
    assert abs(q_ion - q_tot) < 0.02, f"sid {sid}: ion integral {q_ion:+.4f} vs total {q_tot:+.4f}"
    assert abs(q_rb) < 0.05, f"sid {sid}: bound integral {q_rb:+.5f} not ~0"
    return sid, z, lz, ion_z, rb_z, dmax, q_ion, q_rb


def main():
    out = sys.argv[1]
    frames = []
    for xyz in sys.argv[2:]:
        frames.extend(read(xyz, ":"))
    frames = {int(a.info["sample_id"]): a for a in frames}
    sids = sorted(frames)
    print(f"{len(sids)} unique sample_ids from {len(sys.argv) - 2} xyz files", flush=True)

    tasks = [(sid, frames[sid].positions.copy(), float(frames[sid].info["total_charge"]))
             for sid in sids]
    z_ref = None
    lz_ref = None
    rb_rows, ion_rows = [], []
    with Pool(N_WORKERS) as pool:
        for n, (sid, z, lz, ion_z, rb_z, dmax, q_ion, q_rb) in enumerate(
                pool.imap(extract_one, tasks), 1):
            if z_ref is None:
                z_ref, lz_ref = z, lz
            else:
                assert len(z) == len(z_ref) and abs(lz - lz_ref) < 1e-9, f"sid {sid}: grid mismatch"
            ion_rows.append(ion_z)
            rb_rows.append(rb_z)
            if n % 25 == 0 or n == len(sids):
                print(f"  [{n}/{len(sids)}] sid {sid}: dmax {dmax:.1e}  q_ion {q_ion:+.4f}  "
                      f"q_rb {q_rb:+.5f}", flush=True)

    np.savez(
        out,
        sample_ids=np.array(sids, dtype=np.int64),
        z_A=z_ref,
        lz_A=np.float64(lz_ref),
        rb_z_vasp=np.array(rb_rows),    # e/A^3, electron-positive (negate for physics)
        ion_z_vasp=np.array(ion_rows),  # e/A^3, electron-positive
    )
    print(f"wrote {out}: rb/ion {np.array(rb_rows).shape} on nz={len(z_ref)} grid, lz={lz_ref:.3f} A")


if __name__ == "__main__":
    main()
