"""Build the neutral set's potential1d entries and density_3d grids.

Per good cal_N (from the audit):
- potential1d entry (sid 400+N): phi_eV = LOCPOT plane average; z_A,
  raw_neutral_e, raw_ion_potcar_e, align_mask, cdipol_indmin REUSED verbatim
  from the solvated sid N entry (identical geometry -> identical baselines).
- density grid (sid 400+N): net_neu = net_solv + (CHGCAR_neu - CHGCAR_solv)/V
  on the native grid (e/A^3); meta (lattice, valid_iz) copied from sid N.

Validations printed per structure (first M) and aggregated:
- neutral net integral over the full grid ~ 0 e (vs solvated ~ -1..-1.4)
- 1-D closure: Hartree1D(net_prof + raw_neutral - raw_ion) upper-aligned vs
  LOCPOT profile (rms; this is exactly what the Phi1D loss will compute at
  the DFT density).

Usage: python build_neutral_caches.py <audit_tsv> <out_dir> [n_workers]
Writes <out_dir>/potential1d_neutral.npz, <out_dir>/grid_cache_npy/*,
<out_dir>/manifest_neutral.json
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

BASE = "/scratch/08384/tg876840/tmp/2-NiN_single"
RERUN = "/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_neutral_rerun"


def calc_dir(n):
    r = f"{RERUN}/cal_{n}"
    try:
        if "aborting loop because EDIFF is reached" in open(f"{r}/OUTCAR").read():
            return r
    except OSError:
        pass
    return f"{BASE}/0-44_neutral/cal_{n}"
CACHE = "/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/train-data/potential1d_potcar_cache.npz"
GRID_DIR = "/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/train-data/grid_cache_npy"
FELECT = 14.39964546866782
AREA = 14.802 * 12.818908  # lateral cell area (A^2)


def read_grid(path):
    with open(path) as f:
        f.readline(); scale = float(f.readline())
        lat = [[float(x) for x in f.readline().split()] for _ in range(3)]
        line = f.readline().split()
        try:
            counts = [int(x) for x in line]
        except ValueError:
            counts = [int(x) for x in f.readline().split()]
        nat = sum(counts); f.readline()
        for _ in range(nat):
            f.readline()
        f.readline()
        nx, ny, nz = [int(x) for x in f.readline().split()]
        need = nx * ny * nz; vals = []
        while len(vals) < need:
            vals.extend(float(x) for x in f.readline().split())
    return np.array(lat) * scale, np.array(vals[:need], dtype=np.float64).reshape(nz, ny, nx)


def hartree_1d(rho_line, dz):
    """phi(z) from a periodic line charge density (e/A per z), eV units."""
    n = len(rho_line)
    line = rho_line - rho_line.mean()
    g = np.fft.rfftfreq(n, d=dz) * 2 * np.pi
    pg = np.zeros(n // 2 + 1, complex)
    rg = np.fft.rfft(line)
    pg[1:] = 4 * np.pi * FELECT * rg[1:] / g[1:] ** 2 / AREA
    return np.fft.irfft(pg, n=n)


def one(args):
    n, cache_arrays, out_dir = args
    sid_old = n
    d_neu = calc_dir(n)
    d_sol = f"{BASE}/1-44_GCE/cal_{n}"
    idx = list(cache_arrays["sample_ids"]).index(sid_old)
    z = cache_arrays["z_A"][idx]
    raw_ne = cache_arrays["raw_neutral_e"][idx]
    raw_io = cache_arrays["raw_ion_potcar_e"][idx]
    dz = float(z[1] - z[0])
    nz1d = len(z)

    # --- phi_eV from LOCPOT ---
    lat, vloc = read_grid(f"{d_neu}/LOCPOT")
    phi = vloc.mean(axis=(1, 2))
    assert len(phi) == nz1d, f"cal_{n}: LOCPOT nz {len(phi)} != cache nz {nz1d}"

    # --- density grids ---
    _, chg_neu = read_grid(f"{d_neu}/CHGCAR")
    _, chg_sol = read_grid(f"{d_sol}/CHGCAR")
    V = float(abs(np.linalg.det(lat)))
    net_sol = np.load(f"{GRID_DIR}/density3d_net_grid_{sid_old}.npy").astype(np.float64)
    assert net_sol.shape == chg_neu.shape, f"cal_{n}: grid shape mismatch"
    # npy grids store the net charge in the physics convention (electrons
    # negative); CHGCAR is electron-positive -> subtract the difference.
    net_neu = net_sol - (chg_neu - chg_sol) / V
    npts = net_neu.size
    int_neu = float(net_neu.mean() * V)
    int_sol = float(net_sol.mean() * V)
    np.save(f"{out_dir}/grid_cache_npy/density3d_net_grid_{400+n}.npy",
            net_neu.astype(np.float32))
    meta = np.load(f"{GRID_DIR}/density3d_net_grid_{sid_old}_meta.npz")
    np.savez(f"{out_dir}/grid_cache_npy/density3d_net_grid_{400+n}_meta.npz",
             lattice=meta["lattice"], valid_iz=meta["valid_iz"])

    # --- 1-D closure validation (loss-style reconstruction at DFT density) ---
    chg_prof = chg_neu.mean(axis=(1, 2)) / chg_neu.shape[0]      # electrons per slice, sums to 660
    scale_e = raw_ne.sum() / 660.0                               # cache raw units -> per-slice
    scale_i = raw_io.sum() / (-660.0)
    E = chg_prof                                                 # total electrons per slice (+)
    I_phys = -raw_io / scale_i                                   # physical ion charge per slice (+660)
    lam = (I_phys - E) / dz                                      # physical line charge density, e/A
    phi_rec = hartree_1d(lam, dz)
    win = (z > 40.0) & (z < 44.0)
    rms_pos = rms_neg = None
    for sgn in (+1.0, -1.0):
        r = sgn * phi_rec - phi
        r = r - r[win].mean()
        rms = float(np.sqrt(np.mean(r ** 2)))
        if sgn > 0:
            rms_pos = rms
        else:
            rms_neg = rms
    return {"cal": n, "phi": phi, "int_neu": int_neu, "int_sol": int_sol,
            "rms_pos": rms_pos, "rms_neg": rms_neg}


def main():
    audit_tsv, out_dir = sys.argv[1], sys.argv[2]
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    os.makedirs(f"{out_dir}/grid_cache_npy", exist_ok=True)
    good = []
    for line in open(audit_tsv).read().splitlines()[1:]:
        f = line.split("\t")
        if f[1] == "1":
            n = int(f[0])
            if "aborting loop because EDIFF is reached" in open(f"{calc_dir(n)}/OUTCAR").read():
                good.append(n)
    print(f"{len(good)} structures to build")

    cache = np.load(CACHE)
    cache_arrays = {k: cache[k] for k in cache.files}
    with Pool(nw) as pool:
        rows = pool.map(one, [(n, cache_arrays, out_dir) for n in good])

    rows.sort(key=lambda r: r["cal"])
    ints_n = [r["int_neu"] for r in rows]
    ints_s = [r["int_sol"] for r in rows]
    rp = [r["rms_pos"] for r in rows]
    rn = [r["rms_neg"] for r in rows]
    print(f"net integral neutral: mean {np.mean(ints_n):+.3f} e, |max| {np.abs(ints_n).max():.3f}")
    print(f"net integral solvated (reference): mean {np.mean(ints_s):+.3f} e")
    print(f"1-D closure rms, sign(+): mean {np.mean(rp):.4f} eV | sign(-): mean {np.mean(rn):.4f} eV")

    # potential1d entries npz (same layout as the solvated cache)
    sids = np.array([400 + r["cal"] for r in rows], dtype=np.int64)
    idx_of = {int(s): i for i, s in enumerate(cache_arrays["sample_ids"])}
    take = [idx_of[r["cal"]] for r in rows]
    np.savez(f"{out_dir}/potential1d_neutral.npz",
             sample_ids=sids,
             z_A=cache_arrays["z_A"][take],
             phi_eV=np.array([r["phi"] for r in rows]),
             raw_neutral_e=cache_arrays["raw_neutral_e"][take],
             raw_ion_potcar_e=cache_arrays["raw_ion_potcar_e"][take],
             align_mask=cache_arrays["align_mask"][take],
             cdipol_indmin=cache_arrays["cdipol_indmin"][take])
    man = {"format": "density3d_grid_npy_v1", "entries": {}}
    for r in rows:
        sid = 400 + r["cal"]
        meta = np.load(f"{out_dir}/grid_cache_npy/density3d_net_grid_{sid}_meta.npz")
        viz = meta["valid_iz"]
        man["entries"][str(sid)] = {
            "path": os.path.abspath(f"{out_dir}/grid_cache_npy/density3d_net_grid_{sid}.npy"),
            "nx": 168, "ny": 168, "nz": 500,
            "valid_iz_count": int(len(viz)),
            "z_min": float(viz.min() * 45.0 / 500), "z_max": float(viz.max() * 45.0 / 500),
            "meta_path": os.path.abspath(f"{out_dir}/grid_cache_npy/density3d_net_grid_{sid}_meta.npz"),
        }
    with open(f"{out_dir}/manifest_neutral.json", "w") as f:
        json.dump(man, f)
    print(f"wrote potential1d_neutral.npz ({len(sids)} entries) + {len(rows)} grids + manifest")


if __name__ == "__main__":
    main()
