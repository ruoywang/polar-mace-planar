"""Extract the 200 solvent-free neutral calcs into a draft training xyz + audit.

Per cal_N under 0-44_neutral:
- completeness: OUTCAR finished, LOCPOT/CHGCAR present, NELECT, SCF not at NELM
- geometry identical to the solvated counterpart 1-44_GCE/cal_N (same frame)
- LOCPOT plane average -> right-side vacuum plateau (32-43 A): level + flatness
- frame: energy(sigma->0), forces, Fermi_shifted = E_fermi - V_vac (+ raw values
  kept as extra keys), total_charge 0, sample_id 400+N, solvated=0

potential_diff = V_vac(right) - V_vac(left), sign verified against the solvated
set (sid 1: PHI gives V_left-V_right = +0.3483, stored potential_diff = -0.3483).

Usage: python extract_neutral_set.py <out_xyz> <audit_tsv>
"""
from __future__ import annotations

import re
import sys
from multiprocessing import Pool

import numpy as np
from ase.io import read
from ase.io.extxyz import write_extxyz

BASE = "/scratch/08384/tg876840/tmp/2-NiN_single"
RERUN = "/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_neutral_rerun"
N_WORKERS = 16


def calc_dir(n):
    """Prefer the tightest converged rerun over the original calc."""
    import os
    for r in (f"{RERUN}/cal_{n}_e6", f"{RERUN}/cal_{n}"):
        try:
            if "aborting loop because EDIFF is reached" in open(f"{r}/OUTCAR").read():
                return r
        except OSError:
            pass
    return f"{BASE}/0-44_neutral/cal_{n}"


def vac_from_locpot(path):
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
    v = np.array(vals[:need]).reshape(nz, ny, nx).mean(axis=(1, 2))
    lz = lat[2][2] * scale
    z = np.arange(nz) * lz / nz
    wr = (z > 40.0) & (z < 44.0)
    wl = (z > 0.5) & (z < 3.5)
    return (float(v[wr].mean()), float(v[wr].max() - v[wr].min()),
            float(v[wl].mean()), float(v[wl].max() - v[wl].min()))


def one(n):
    d = calc_dir(n)
    out = {"cal": n}
    try:
        txt = open(f"{d}/OUTCAR").read()
        out["finished"] = "General timing and accounting" in txt
        out["converged"] = "aborting loop because EDIFF is reached" in txt
        m = re.findall(r"E-fermi :\s*([-\d.]+)", txt)
        out["efermi"] = float(m[-1])
        m = re.findall(r"energy\(sigma->0\) =\s*([-\d.]+)", txt)
        out["e_sigma0"] = float(m[-1])
        m = re.search(r"NELECT =\s*([\d.]+)", txt)
        out["nelect"] = float(m.group(1))
        # SCF steps in the (single) ionic step: count "Iteration" lines
        out["nscf"] = len(re.findall(r"Iteration\s+1\(", txt))
        a = read(f"{d}/OUTCAR")  # positions, forces
        out["natoms"] = len(a)
        out["fmax"] = float(np.abs(a.get_forces()).max())
        b = read(f"{BASE}/1-44_GCE/cal_{n}/CONTCAR")
        out["geom_match"] = float(np.abs(a.positions - b.positions).max())
        out["vvac"], out["vflat"], out["vleft"], out["vlflat"] = vac_from_locpot(f"{d}/LOCPOT")
        out["atoms"] = a
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001 - audit collects every failure mode
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main():
    out_xyz, audit_tsv = sys.argv[1], sys.argv[2]
    with Pool(N_WORKERS) as pool:
        rows = pool.map(one, range(1, 201))
    frames, bad = [], []
    with open(audit_tsv, "w") as f:
        f.write("cal\tok\tfinished\tnscf\tnelect\tefermi\tvvac\tvflat\tvleft\tvlflat\tpotential_diff\tfermi_shifted\tgeom_match\tfmax\terror\n")
        for r in sorted(rows, key=lambda x: x["cal"]):
            if r["ok"]:
                fs = r["efermi"] - r["vvac"]
                pd = r["vvac"] - r["vleft"]  # potential_diff = V_right - V_left
                f.write(f"{r['cal']}\t1\t{int(r['finished'])}\t{r['nscf']}\t{r['nelect']:.3f}\t"
                        f"{r['efermi']:.4f}\t{r['vvac']:.4f}\t{r['vflat']:.5f}\t{r['vleft']:.4f}\t{r['vlflat']:.5f}\t{pd:.4f}\t{fs:.4f}\t"
                        f"{r['geom_match']:.2e}\t{r['fmax']:.2f}\t\n")
                issues = []
                if not r["finished"]: issues.append("unfinished")
                if not r.get("converged"): issues.append("not-converged")
                if abs(r["nelect"] - 660.0) > 1e-6: issues.append(f"nelect={r['nelect']}")
                if r["geom_match"] > 1e-6: issues.append(f"geom{r['geom_match']:.1e}")
                if r["vflat"] > 0.01: issues.append(f"vac-not-flat{r['vflat']:.3f}")
                if r["vlflat"] > 0.01: issues.append(f"leftvac-not-flat{r['vlflat']:.3f}")
                if r["natoms"] != 207: issues.append(f"natoms{r['natoms']}")
                if issues:
                    bad.append((r["cal"], ",".join(issues)))
                    continue
                a = r["atoms"]
                a.info.clear()
                a.info.update({
                    "energy": r["e_sigma0"],
                    "Fermi": fs,
                    "Fermi_raw": r["efermi"],
                    "vacuum_level": r["vvac"],
                    "potential_diff": r["vvac"] - r["vleft"],
                    "total_charge": 0.0,
                    "total_spin": 1.0,
                    "sample_id": 400 + r["cal"],
                    "source_cal_id": r["cal"],
                    "config_type": "NiN44vac",
                    "solvated": 0,
                })
                frames.append(a)
            else:
                f.write(f"{r['cal']}\t0\t\t\t\t\t\t\t\t\t\t{r['error']}\n")
                bad.append((r["cal"], r["error"]))
    with open(out_xyz, "w") as f:
        write_extxyz(f, frames)
    print(f"frames written: {len(frames)}; excluded: {len(bad)}")
    for c, why in bad[:20]:
        print(f"  cal_{c}: {why}")
    if frames:
        fs = [a.info["Fermi"] for a in frames]
        vv = [a.info["vacuum_level"] for a in frames]
        print(f"Fermi_shifted: mean {np.mean(fs):.3f}, range [{min(fs):.3f}, {max(fs):.3f}]")
        print(f"vacuum_level:  mean {np.mean(vv):.3f}, range [{min(vv):.3f}, {max(vv):.3f}]")


if __name__ == "__main__":
    main()
