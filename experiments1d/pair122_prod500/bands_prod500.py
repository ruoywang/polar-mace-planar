"""sid 122 solvated band overlay: DFT (exp_band/sid122_bands/dft_sol, unchanged reference) vs the fully-ML
CHGCAR of prod500_w1000_ref (ml_sol_prod500). Convention of sid122_figs.py: each run minus its own E_F,
both Fermi levels marked at zero, 24 bands around the band edge, rigid treatment only.
Usage: python bands_prod500.py  -> bands_122_sol_prod500.png + bands_122_sol_prod500.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/exp_band")
from compare_bands3 import read_eigenval, read_fermi, read_cell, k_distances

HERE = Path("/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/exp_pair122_prod500")
DFT = Path("/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/exp_band/sid122_bands/dft_sol")
ML = HERE / "ml_sol_prod500"
NELECT = 660.942389
edge = round(NELECT / 2)
lo, hi = edge - 12, edge + 12

k, e1, nk, nb1 = read_eigenval(DFT / "EIGENVAL"); f1 = read_fermi(DFT / "OUTCAR")
_, e2, _, nb2 = read_eigenval(ML / "EIGENVAL"); f2 = read_fermi(ML / "OUTCAR")
n = min(nb1, nb2)
a1 = e1[:, :n] - f1
a2 = e2[:, :n] - f2
w = (a2 - a1)[:, lo:hi]
rmse_w = float(np.sqrt((w ** 2).mean())); rmse_all = float(np.sqrt(((a2 - a1) ** 2).mean()))
print(f"sol: fermi_diff={f2-f1:+.3f}  rmse_window={rmse_w:.3f}  rmse_all={rmse_all:.3f}  E_F dft {f1:.3f} / ml {f2:.3f}  nk {nk} nbands {n}")
kd = k_distances(k, read_cell(DFT / "POSCAR"))
seg = nk // 3
ticks = [kd[0], kd[seg], kd[2 * seg], kd[-1]]
plt.figure(figsize=(7.0, 4.5))
for b in range(lo, hi):
    plt.plot(kd, a1[:, b], color="black", lw=0.9, alpha=0.6)
    plt.plot(kd, a2[:, b], color="#d62728", lw=0.8, alpha=0.55)
plt.axhline(0.0, color="black", lw=1.1, ls="--")
plt.axhline(0.0, color="#d62728", lw=1.1, ls=(0, (3, 5)))
plt.text(kd[-1] * 0.995, 0.0, "E_F DFT ", color="black", fontsize=8, va="bottom", ha="right")
plt.text(kd[-1] * 0.995, 0.0, "E_F ML ", color="#d62728", fontsize=8, va="top", ha="right")
for x in ticks:
    plt.axvline(x, color="0.85", lw=0.7)
plt.xticks(ticks, ["Gamma", "M", "K", "Gamma"])
plt.xlim(kd[0], kd[-1])
plt.xlabel("k path"); plt.ylabel("E - E_F (eV)")
plt.title("sid 122, VASPsol solvent: DFT (black) vs fully-ML CHGCAR of prod500_w1000_ref (red)", fontsize=10)
plt.tight_layout()
plt.savefig(HERE / "bands_122_sol_prod500.png", dpi=220)
plt.close()
json.dump({"rmse_window": rmse_w, "rmse_all": rmse_all, "fermi_diff": float(f2 - f1), "ef_dft": float(f1), "ef_ml": float(f2),
           "nk": int(nk), "nbands": int(n), "nwin": hi - lo, "edge_band": edge}, open(HERE / "bands_122_sol_prod500.json", "w"))
print("wrote", HERE / "bands_122_sol_prod500.png")
