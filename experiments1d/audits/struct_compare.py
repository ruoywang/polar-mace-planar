"""Side-by-side structural comparison of two checkpoints (gate vs production, same epoch) from
the npz files written by ebl_audit.py and density_profile_eval.py:

  python struct_compare.py LOGDIR TAG_A TAG_B [LABEL_A LABEL_B]

Per state (charged NiN44 / neutral NiN44 / charged NiN88): absolute energy error per atom (rmse,
bias), force rmse, solvent-profile L1 vs DFT (electrons) and solvent dipole error, E_bl with the
model field vs with the DFT field; density: window L1, 3-D rmse, tail electron errors; per pair
(charged - neutral): charging residual, E_bl pair difference, charge-response L1 / centroid / metal
share. The last column is B/A - 1 for error-like quantities.
"""
import os
import sys

import numpy as np

LOG, A, B = sys.argv[1], sys.argv[2], sys.argv[3]
LA, LB = (sys.argv[4], sys.argv[5]) if len(sys.argv) > 5 else (A, B)


def load(tag):
    out = {}
    p = os.path.join(LOG, f"ebl_audit_{tag}.npz")
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True); out["ebl"] = {int(r["sid"]): r for r in d["frames"]}; out["epoch"] = int(d["epoch"])
    p = os.path.join(LOG, f"density_profile_{tag}.npz")
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True); out["dens"] = {int(r["sid"]): r for r in d["frames"]}; out["pairs"] = {int(r["k"]): r for r in d["pairs"]}
    return out


def mean(x): x = np.asarray(x, float); x = x[np.isfinite(x)]; return float(x.mean()) if x.size else float("nan")
def rms(x): x = np.asarray(x, float); x = x[np.isfinite(x)]; return float(np.sqrt((x ** 2).mean())) if x.size else float("nan")


da, db = load(A), load(B)
print(f"A = {LA} ({A}, epoch {da.get('epoch','?')})   B = {LB} ({B}, epoch {db.get('epoch','?')})")


def row(name, va, vb, fmt="{:9.4f}", rel=True):
    r = f"{vb/va-1:+7.1%}" if (rel and va == va and va != 0 and vb == vb) else ""
    print(f"   {name:34s} {fmt.format(va):>11} {fmt.format(vb):>11} {r:>8}")


def header(title):
    print(f"\n=== {title} ===")
    print(f"   {'':34s} {LA[:11]:>11} {LB[:11]:>11} {'B/A-1':>8}")


if "ebl" in da and "ebl" in db:
    sids = sorted(set(da["ebl"]) & set(db["ebl"]))
    for st in sorted({da["ebl"][s]["state"] for s in sids}):
        S = [s for s in sids if da["ebl"][s]["state"] == st]
        header(f"{st}: energies, forces, solvent profile ({len(S)} frames)")
        for nm, f, fmt in (
            ("E err/atom rmse (meV/atom)", lambda r: 1e3 * (r["E_model"] - r["E_dft"]) / r["nat"], "{:9.3f}"),
            ("F rmse (meV/A)", lambda r: r.get("F_rms_err_meV", np.nan), "{:9.2f}"),
            ("solvent profile L1 vs DFT (e)", lambda r: r.get("rho_solv_L1_model_vs_dft", np.nan), "{:9.4f}"),
            ("solvent dipole err |m-d| (e*A)", lambda r: abs(r.get("mu_solv_model", np.nan) - r.get("mu_solv_dft", np.nan)), "{:9.4f}"),
        ):
            xa = [f(da["ebl"][s]) for s in S]; xb = [f(db["ebl"][s]) for s in S]
            row(nm if "err" not in nm else nm, rms(xa) if "rmse" in nm else mean(xa), rms(xb) if "rmse" in nm else mean(xb), fmt)
        row("E err/atom bias (meV/atom)", mean([1e3 * (da["ebl"][s]["E_model"] - da["ebl"][s]["E_dft"]) / da["ebl"][s]["nat"] for s in S]),
            mean([1e3 * (db["ebl"][s]["E_model"] - db["ebl"][s]["E_dft"]) / db["ebl"][s]["nat"] for s in S]), "{:+9.3f}", rel=False)
        row("E_bl model field (eV, mean)", mean([da["ebl"][s]["Ebl_recomp"] for s in S]), mean([db["ebl"][s]["Ebl_recomp"] for s in S]), "{:+9.4f}", rel=False)
        row("E_bl DFT field (eV, mean)", mean([da["ebl"][s].get("Ebl_dft_field", np.nan) for s in S]), mean([db["ebl"][s].get("Ebl_dft_field", np.nan) for s in S]), "{:+9.4f}", rel=False)
        row("|E_bl model - DFT field| (eV)", mean([abs(da["ebl"][s]["Ebl_recomp"] - da["ebl"][s].get("Ebl_dft_field", np.nan)) for s in S]),
            mean([abs(db["ebl"][s]["Ebl_recomp"] - db["ebl"][s].get("Ebl_dft_field", np.nan)) for s in S]), "{:9.4f}")
        row("q_solv model (e, mean)", mean([da["ebl"][s]["q_solv"] for s in S]), mean([db["ebl"][s]["q_solv"] for s in S]), "{:+9.4f}", rel=False)
    # pairs
    ks = sorted(k for k in sids if k < 600 and k + 600 in sids)
    header(f"paired charging dE = E(k) - E(k+600) vs DFT ({len(ks)} pairs)")
    for lab, d in (("A", da), ("B", db)):
        pass
    def pres(d, k): e = d["ebl"]; return (e[k]["E_model"] - e[k + 600]["E_model"]) - (e[k]["E_dft"] - e[k + 600]["E_dft"])
    ra = np.array([pres(da, k) for k in ks]); rb = np.array([pres(db, k) for k in ks])
    row("charging residual rmse (eV)", rms(ra), rms(rb), "{:9.4f}")
    row("charging residual bias (eV)", ra.mean(), rb.mean(), "{:+9.4f}", rel=False)
    row("charging residual std (eV)", ra.std(), rb.std(), "{:9.4f}")
    def debl(d, k): e = d["ebl"]; return e[k]["Ebl_recomp"] - e[k + 600]["Ebl_recomp"]
    row("dE_bl pair (eV, mean)", mean([debl(da, k) for k in ks]), mean([debl(db, k) for k in ks]), "{:+9.4f}", rel=False)
    row("dE_bl pair std (eV)", np.std([debl(da, k) for k in ks]), np.std([debl(db, k) for k in ks]), "{:9.4f}")
    for sub in ("NiN44", "NiN88"):
        kk = [k for k in ks if sub in da["ebl"][k]["state"]]
        if kk:
            row(f"  {sub} charging rmse ({len(kk)} pairs)", rms([pres(da, k) for k in kk]), rms([pres(db, k) for k in kk]), "{:9.4f}")

if "dens" in da and "dens" in db:
    sids = sorted(set(da["dens"]) & set(db["dens"]))
    for st in sorted({da["dens"][s]["state"] for s in sids}):
        S = [s for s in sids if da["dens"][s]["state"] == st]
        header(f"{st}: plane-averaged net density vs DFT grid ({len(S)} frames)")
        for nm, key, fmt in (("3-D rmse (e/A^3)", "rmse3d", "{:9.5f}"), ("window L1 (e)", "L1_window_e", "{:9.4f}"),
                             ("tail bottom L1 (e)", "tail_bottom_L1_e", "{:9.5f}"), ("tail top L1 (e)", "tail_top_L1_e", "{:9.5f}")):
            row(nm, mean([da["dens"][s][key] for s in S]), mean([db["dens"][s][key] for s in S]), fmt)
        for nm, key in (("tail bottom dN model-DFT (e)", "tail_bottom_dN_e"), ("tail top dN model-DFT (e)", "tail_top_dN_e")):
            row(nm, mean([da["dens"][s][key] for s in S]), mean([db["dens"][s][key] for s in S]), "{:+9.5f}", rel=False)
        row("tail top N_dft (e, scale)", mean([da["dens"][s]["tail_top_N_dft_e"] for s in S]), mean([db["dens"][s]["tail_top_N_dft_e"] for s in S]), "{:+9.5f}", rel=False)
    ks = sorted(set(da["pairs"]) & set(db["pairs"]))
    header(f"charge response dn(z) of the pairs vs DFT ({len(ks)} pairs)")
    row("L1(model-DFT) (e, mean)", mean([da["pairs"][k]["L1_e"] for k in ks]), mean([db["pairs"][k]["L1_e"] for k in ks]), "{:9.4f}")
    row("L1 of DFT dn (e, scale)", mean([da["pairs"][k]["L1_dft_e"] for k in ks]), mean([db["pairs"][k]["L1_dft_e"] for k in ks]), "{:9.4f}", rel=False)
    row("|dN model - dN dft| (e)", mean([abs(da["pairs"][k]["dN_model"] - da["pairs"][k]["dN_dft"]) for k in ks]),
        mean([abs(db["pairs"][k]["dN_model"] - db["pairs"][k]["dN_dft"]) for k in ks]), "{:9.4f}")
    row("|zc model - zc dft| (A)", mean([abs(da["pairs"][k]["zc_model"] - da["pairs"][k]["zc_dft"]) for k in ks]),
        mean([abs(db["pairs"][k]["zc_model"] - db["pairs"][k]["zc_dft"]) for k in ks]), "{:9.3f}")
    row("zc dft (A, mean)", mean([da["pairs"][k]["zc_dft"] for k in ks]), mean([db["pairs"][k]["zc_dft"] for k in ks]), "{:9.3f}", rel=False)
    row("|metal share model - dft|", mean([abs(da["pairs"][k]["metal_share_model"] - da["pairs"][k]["metal_share_dft"]) for k in ks]),
        mean([abs(db["pairs"][k]["metal_share_model"] - db["pairs"][k]["metal_share_dft"]) for k in ks]), "{:9.4f}")
    row("metal share dft (mean)", mean([da["pairs"][k]["metal_share_dft"] for k in ks]), mean([db["pairs"][k]["metal_share_dft"] for k in ks]), "{:9.4f}", rel=False)
    for sub in ("NiN44", "NiN88"):
        kk = [k for k in ks if da["pairs"][k]["state"] == sub]
        if kk:
            row(f"  {sub} dn L1 ({len(kk)} pairs)", mean([da["pairs"][k]["L1_e"] for k in kk]), mean([db["pairs"][k]["L1_e"] for k in kk]), "{:9.4f}")
