"""Matched-epoch validation table: a gate run against the production run, from the two run.logs.

  python gate_vs_prod_table.py GATE_RUN_LOG PROD_RUN_LOG [EPOCHS]
EPOCHS: comma list (default 20-33). Prints loss, E, F, potential, fermi, density_3d, Phi1D, rho_b,
occ_aug per epoch for both, the ratio gate/prod, and the mean ratio over the PB epochs listed.
"""
import re
import sys

KEYS = [("loss", r"loss=([0-9.]+)"), ("E", r"RMSE_E_per_atom=\s*([0-9.]+)"), ("F", r"RMSE_F=\s*([0-9.]+)"),
        ("pot", r"RMSE_potential=([0-9.]+)"), ("fermi", r"RMSE_fermi=([0-9.]+)"), ("dens3d", r"RMSE_density_3d=([0-9.]+)"),
        ("Phi1D", r"RMSE_potential_1d_profile=([0-9.]+)"), ("rho_b", r"RMSE_solvent3d_b=([0-9.]+)"),
        ("occ", r"RMSE_occ_aug=([0-9.]+)")]


def parse(path):
    out = {}
    for line in open(path, errors="ignore"):
        m = re.search(r"INFO: Epoch (\d+): ", line)
        if not m:
            continue
        ep = int(m.group(1)); row = {}
        for k, pat in KEYS:
            mm = re.search(pat, line)
            row[k] = float(mm.group(1)) if mm else float("nan")
        out[ep] = row
    return out


gate, prod = parse(sys.argv[1]), parse(sys.argv[2])
epochs = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(20, 34))
epochs = [e for e in epochs if e in gate and e in prod]
print(f"{'epoch':>5} " + " ".join(f"{k:>18}" for k, _ in KEYS))
print(f"{'':>5} " + " ".join(f"{'gate / prod':>18}" for _ in KEYS))
ratios = {k: [] for k, _ in KEYS}
for e in epochs:
    cells = []
    for k, _ in KEYS:
        g, p = gate[e][k], prod[e][k]
        cells.append(f"{g:8.4g}/{p:<8.4g}")
        if p == p and g == g and p != 0:
            ratios[k].append(g / p)
    print(f"{e:>5} " + " ".join(f"{c:>18}" for c in cells))
print(f"{'ratio':>5} " + " ".join(f"{(sum(v)/len(v) if v else float('nan')):>18.3f}" for k, _ in KEYS for v in [ratios[k]]))
last = epochs[-1] if epochs else None
if last is not None:
    print(f"last common epoch {last}: " + ", ".join(f"{k} {gate[last][k]:.4g} vs {prod[last][k]:.4g} ({gate[last][k]/prod[last][k]-1:+.1%})" if prod[last][k] else "" for k, _ in KEYS))
