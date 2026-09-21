"""Charging density response dn(r) = n(charged) - n(neutral) of saved checkpoints against DFT,
from the per-pair arrays density_profile_eval.py stores (user 2026-09-21, diagnosis only).

  python charge_response_report.py LOGDIR RUNDATA OUT_PREFIX LABEL=TAG:FLAG [LABEL=TAG:FLAG ...]
LOGDIR   directory with density_profile_<TAG>.npz
RUNDATA  a run's data dir (train.xyz / val.xyz for geometry + split, density3d manifest for the grid)
FLAG     ON / OFF = solvent_pb1d_vsolv_input during TRAINING of that checkpoint

Definitions (all electron-POSITIVE: added electrons count positive; the grid's own convention
is net charge with electrons negative, so every stored dn is multiplied by -1):
  R(z)  = -A * <dn>_xy(z)            plane-integrated response, e/A   (A = cell area)
  N     = int R dz over the DFT window (z about 3.6-19.6 A)  -> must equal dN_e = -q(charged)
  E     = int |R_ML - R_DFT| dz  (e),  S = int |R_DFT| dz  (e),  ratio = E / S
  C(z)  = int_{z_min}^{z} R dz         cumulative added electrons
Regions from the charged frame's geometry (the same for its neutral twin):
  sheet = C atoms within 1 A of the median C height + Ni + N (the NiN-doped carbon sheet);
  electrode  [z_sheet_min - 1.5, z_sheet_max + 1.5]
  interface  (z_sheet_max + 1.5, z_sheet_max + 4.0]   (adsorbate + first water layer)
  water      (z_sheet_max + 4.0, z_water_max + 1.0]
  vacuum below / above = the rest of the window.
Region share = N_region / N (each side divided by ITS OWN window integral). A spatial partition,
not an atomic charge.
Writes OUT_PREFIX.json (tables + curves) and prints the tables.
"""
import json
import os
import sys

import numpy as np
from ase.io import read

LOG, RUNDATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ITEMS = []
for it in sys.argv[4:]:
    lab, rest = it.split("=", 1)
    tag, flag = rest.split(":") if ":" in rest else (rest, "?")
    ITEMS.append((lab, tag, flag))

REP_PAIRS = [28, 60, 128, 1, 167]   # representative: val q -1.32 / -0.82 / -1.12, train q -1.00 / -0.80

atoms, split = {}, {}
for sp in ("train", "val"):
    for a in read(os.path.join(RUNDATA, f"{sp}.xyz"), ":"):
        s = int(a.info["sample_id"]); atoms[s] = a; split[s] = sp
manifest = json.load(open(os.path.join(RUNDATA, "density3d_net_grid_manifest_npy.json")))["entries"]


def regions_of(k):
    a = atoms[k]; z = a.positions[:, 2]; sym = np.array(a.get_chemical_symbols())
    c = z[sym == "C"]; sheet = (sym == "C") & (z < np.median(c) + 1.0) | np.isin(sym, ["Ni", "N"])
    zs_min, zs_max = float(z[sheet].min()), float(z[sheet].max())
    zw_max = float(z[np.isin(sym, ["O", "H"])].max())
    return {"b1": zs_min - 1.5, "b2": zs_max + 1.5, "b3": zs_max + 4.0, "b4": zw_max + 1.0,
            "z_sheet_min": zs_min, "z_sheet_max": zs_max, "z_water_max": zw_max,
            "n_sheet": int(sheet.sum()), "z_adsorbate_C": [float(v) for v in z[(sym == "C") & ~sheet]]}


def verify_pair(k):
    a, b = atoms[k], atoms[k + 600]
    ma, mb = np.load(manifest[str(k)]["meta_path"]), np.load(manifest[str(k + 600)]["meta_path"])
    return {"pos_max_diff": float(np.abs(a.positions - b.positions).max()),
            "cell_equal": bool(np.allclose(a.cell[:], b.cell[:])),
            "lattice_equal": bool(np.allclose(ma["lattice"], mb["lattice"])),
            "lattice_is_cell": bool(np.allclose(ma["lattice"], a.cell[:], atol=1e-6)),
            "valid_iz_equal": bool(np.array_equal(ma["valid_iz"], mb["valid_iz"])),
            "grid_equal": all(manifest[str(k)][x] == manifest[str(k + 600)][x] for x in ("nx", "ny", "nz")),
            "q_charged": float(a.info["total_charge"]), "q_neutral": float(b.info.get("total_charge", 0.0))}


REG_NAMES = ["vacuum_below", "electrode", "interface", "water", "vacuum_above"]


def region_masks(z, r):
    return {"vacuum_below": z < r["b1"], "electrode": (z >= r["b1"]) & (z <= r["b2"]),
            "interface": (z > r["b2"]) & (z <= r["b3"]), "water": (z > r["b3"]) & (z <= r["b4"]),
            "vacuum_above": z > r["b4"]}


result = {"definitions": __doc__, "checkpoints": [], "verification": {}, "regions": {}}
first = True
for lab, tag, flag in ITEMS:
    p = os.path.join(LOG, f"density_profile_{tag}.npz")
    if not os.path.exists(p):
        print(f"[{lab}] missing {p}"); continue
    d = np.load(p, allow_pickle=True)
    frames = {int(f["sid"]): f for f in d["frames"]}; pairs = {int(r["k"]): r for r in d["pairs"]}
    ck = {"label": lab, "tag": tag, "flag_in_training": flag, "epoch": int(d["epoch"]), "pairs": {}}
    for k, r in sorted(pairs.items()):
        f = frames[k]; A, dz = float(f["area"]), float(f["dz"]); z = np.asarray(r["z"], float)
        R_ml = -A * np.asarray(r["dn_model"], float); R_dft = -A * np.asarray(r["dn_dft"], float)
        reg = regions_of(k); masks = region_masks(z, reg)
        if first:
            result["verification"][k] = verify_pair(k); result["regions"][k] = reg
        N_ml, N_dft = float(R_ml.sum() * dz), float(R_dft.sum() * dz)
        E = float(np.abs(R_ml - R_dft).sum() * dz); S = float(np.abs(R_dft).sum() * dz)
        rec = {"split": split[k], "q": float(r["q"]), "dNe": -float(r["q"]), "N_ml": N_ml, "N_dft": N_dft,
               "E": E, "S": S, "ratio": E / S, "zc_ml": float((R_ml * z).sum() / R_ml.sum()), "zc_dft": float((R_dft * z).sum() / R_dft.sum()),
               "regions": {}}
        for nm in REG_NAMES:
            m = masks[nm]
            rec["regions"][nm] = {"N_ml": float(R_ml[m].sum() * dz), "N_dft": float(R_dft[m].sum() * dz),
                                  "E": float(np.abs(R_ml - R_dft)[m].sum() * dz), "planes": int(m.sum())}
        if k in REP_PAIRS:
            rec["curve"] = {"z": z.tolist(), "R_ml": R_ml.tolist(), "R_dft": R_dft.tolist(),
                            "C_ml": (np.cumsum(R_ml) * dz).tolist(), "C_dft": (np.cumsum(R_dft) * dz).tolist()}
        rec["_R_ml"], rec["_R_dft"], rec["_z"] = R_ml, R_dft, z
        ck["pairs"][k] = rec
    # aggregate per-electron curves (each pair divided by its own N_dft), all pairs and per split
    zs = None
    for sp in ("all", "train", "val"):
        ks = [k for k, rec in ck["pairs"].items() if sp == "all" or rec["split"] == sp]
        if not ks:
            continue
        zref = ck["pairs"][ks[0]]["_z"]
        ok = [k for k in ks if ck["pairs"][k]["_z"].shape == zref.shape and np.allclose(ck["pairs"][k]["_z"], zref)]
        Rm = np.mean([ck["pairs"][k]["_R_ml"] / ck["pairs"][k]["N_dft"] for k in ok], axis=0)
        Rd = np.mean([ck["pairs"][k]["_R_dft"] / ck["pairs"][k]["N_dft"] for k in ok], axis=0)
        ck[f"aggregate_{sp}"] = {"n_pairs": len(ok), "n_pairs_other_grid": len(ks) - len(ok), "z": zref.tolist(),
                                 "R_ml_per_e": Rm.tolist(), "R_dft_per_e": Rd.tolist(),
                                 "C_ml_per_e": (np.cumsum(Rm) * dz).tolist(), "C_dft_per_e": (np.cumsum(Rd) * dz).tolist()}
    for rec in ck["pairs"].values():
        for kk in ("_R_ml", "_R_dft", "_z"):
            rec.pop(kk)
    result["checkpoints"].append(ck); first = False


def mean(x): x = np.asarray(x, float); return float(x.mean()) if x.size else float("nan")


print("=== verification of the 47 pairs (charged k, neutral k+600) ===")
V = result["verification"]
print(f"   positions identical: max |dr| over pairs {max(v['pos_max_diff'] for v in V.values()):.1e} A; cells equal {all(v['cell_equal'] for v in V.values())}; "
      f"grid lattice equal and = cell {all(v['lattice_equal'] and v['lattice_is_cell'] for v in V.values())}; valid planes identical {all(v['valid_iz_equal'] for v in V.values())}; "
      f"nx/ny/nz equal {all(v['grid_equal'] for v in V.values())}")
print(f"   q(charged) range {min(v['q_charged'] for v in V.values()):+.3f} .. {max(v['q_charged'] for v in V.values()):+.3f} e, q(neutral) all 0: {all(abs(v['q_neutral']) < 1e-9 for v in V.values())}")
rg = result["regions"]
print(f"   regions (A): electrode [{mean([r['b1'] for r in rg.values()]):.2f}, {mean([r['b2'] for r in rg.values()]):.2f}], interface to {mean([r['b3'] for r in rg.values()]):.2f}, "
      f"water to {mean([r['b4'] for r in rg.values()]):.2f} (means over pairs; sheet {mean([r['z_sheet_min'] for r in rg.values()]):.2f}-{mean([r['z_sheet_max'] for r in rg.values()]):.2f}, "
      f"{int(mean([r['n_sheet'] for r in rg.values()]))} sheet atoms, adsorbate C at {mean([r['z_adsorbate_C'][0] for r in rg.values() if r['z_adsorbate_C']]):.2f})")

for ck in result["checkpoints"]:
    print(f"\n=== {ck['label']}: epoch {ck['epoch']}, flag in training {ck['flag_in_training']}, {len(ck['pairs'])} pairs ===")
    print(f"   {'split':>5} {'n':>3} {'E=int|R_ML-R_DFT|':>18} {'S=int|R_DFT|':>13} {'E/S mean':>9} {'median':>7} {'max|N_ML-dNe|':>14} {'max|N_DFT-dNe|':>15} {'|zc err| A':>10}")
    for sp in ("train", "val", "all"):
        P = [r for r in ck["pairs"].values() if sp == "all" or r["split"] == sp]
        if not P:
            continue
        print(f"   {sp:>5} {len(P):>3} {mean([r['E'] for r in P]):18.4f} {mean([r['S'] for r in P]):13.4f} {mean([r['ratio'] for r in P]):9.3f} "
              f"{float(np.median([r['ratio'] for r in P])):7.3f} {max(abs(r['N_ml'] - r['dNe']) for r in P):14.4f} {max(abs(r['N_dft'] - r['dNe']) for r in P):15.4f} "
              f"{mean([abs(r['zc_ml'] - r['zc_dft']) for r in P]):10.3f}")
    print(f"   signed added electrons per region, mean over pairs (model / DFT), and share of the window integral:")
    print(f"   {'split':>5} " + " ".join(f"{nm:>26}" for nm in REG_NAMES))
    for sp in ("train", "val"):
        P = [r for r in ck["pairs"].values() if r["split"] == sp]
        cells = []
        for nm in REG_NAMES:
            nm_ml = mean([r["regions"][nm]["N_ml"] for r in P]); nm_d = mean([r["regions"][nm]["N_dft"] for r in P])
            sh_ml = mean([r["regions"][nm]["N_ml"] / r["N_ml"] for r in P]); sh_d = mean([r["regions"][nm]["N_dft"] / r["N_dft"] for r in P])
            cells.append(f"{nm_ml:+.3f}/{nm_d:+.3f} ({sh_ml:.2f}/{sh_d:.2f})")
        print(f"   {sp:>5} " + " ".join(f"{c:>26}" for c in cells))
    print(f"   where the error sits: share of E per region (mean over all pairs): " +
          ", ".join(f"{nm} {mean([r['regions'][nm]['E'] / r['E'] for r in ck['pairs'].values()]):.2f}" for nm in REG_NAMES))

with open(OUT + ".json", "w") as fh:
    json.dump(result, fh)
print(f"\nwritten {OUT}.json")
