"""Side-by-side table of several ab_eval.py result files on the COMPLETE val set
(same 80 frames / 20 matched pairs in every file), plus the pair-by-pair paired
charging-energy residuals. Usage:
    python ab_table.py LABEL=path.npz LABEL=path.npz ...
Energies in meV/atom (rmse / bias), forces in meV/A (rmse over frames of the
per-frame rmse / largest component error), paired dE = E(k) - E(k+600) in eV
(rmse / bias / mean-removed = std over the 20 val pairs). Train columns are not
tabulated because the files use different train strides.
"""
import sys
import numpy as np
def rms(x): x = np.asarray(x, float); return float(np.sqrt((x**2).mean()))
files = [a.split("=", 1) for a in sys.argv[1:]]
data = {}
for lab, path in files:
    z = np.load(path, allow_pickle=True); rows = [r for r in z["rows"] if r[0] == "val"]
    data[lab] = {int(r[1]): r for r in rows}
labs = [l for l, _ in files]
print("VAL, per state: E rmse / bias (meV/atom) | F rmse / F max (meV/A)")
print(f"{'state':>14} " + " ".join(f"{l:>27}" for l in labs))
for st in ("charged NiN44", "charged NiN88", "neutral NiN44"):
    cells = []
    for l in labs:
        R = [r for r in data[l].values() if r[2] == st]
        e = [1e3 * (r[4] - r[5]) / r[3] for r in R]
        cells.append(f"{rms(e):6.2f}/{np.mean(e):+6.2f} | {rms([r[6] for r in R]):5.1f}/{max(r[7] for r in R):6.1f}")
    print(f"{st:>14} " + " ".join(f"{c:>27}" for c in cells))
print("\nVAL, all 80 frames: E rmse / bias (meV/atom) | F rmse (meV/A)")
for l in labs:
    R = list(data[l].values()); e = [1e3 * (r[4] - r[5]) / r[3] for r in R]
    print(f"{l:>14}: {rms(e):6.2f}/{np.mean(e):+6.2f} | {rms([r[6] for r in R]):5.1f}")
pairs = sorted(k for k in data[labs[0]] if k < 600 and k + 600 in data[labs[0]])
print(f"\nVAL paired charging energy residual (model dE - DFT dE, eV), {len(pairs)} pairs: rmse / bias / mean-removed")
res = {}
for l in labs:
    by = data[l]; r = np.array([(by[k][4] - by[k+600][4]) - (by[k][5] - by[k+600][5]) for k in pairs]); res[l] = r
    print(f"{l:>14}: {rms(r):.4f} / {r.mean():+.4f} / {r.std():.4f}")
print(f"\nPair by pair (eV): DFT dE, then residual per column; last column = change first->last")
print(f"{'k':>4} {'k+600':>5} {'n':>3} {'DFT dE':>8} " + " ".join(f"{l:>10}" for l in labs) + f" {'change':>8}")
by0 = data[labs[0]]
for i, k in enumerate(pairs):
    dft = by0[k][5] - by0[k+600][5]
    print(f"{k:>4} {k+600:>5} {by0[k][3]:>3} {dft:8.3f} " + " ".join(f"{res[l][i]:+10.4f}" for l in labs) + f" {res[labs[-1]][i]-res[labs[0]][i]:+8.4f}")
imp = np.sum(np.abs(res[labs[-1]]) < np.abs(res[labs[0]]))
print(f"\n|residual| smaller in the last column than in the first: {imp}/{len(pairs)} pairs")
