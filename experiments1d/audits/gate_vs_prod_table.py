"""Matched-epoch validation table: a gate run against the production run, from the run.logs.

  python gate_vs_prod_table.py GATE_RUN_LOG PROD_RUN_LOG [EPOCHS] [REF_RUN_LOG]
EPOCHS: comma list (default 20-33). Prints loss, E, F, potential, fermi, density_3d, Phi1D, rho_b,
occ_aug per epoch for gate and prod, the mean ratio gate/prod over the listed epochs, and -- when a
third log is given (a same-config replicate of prod, e.g. 3-residual_3D/w1000_ref) -- the same
ratio ref/prod per epoch and its range: the run-to-run noise floor the gate ratio must exceed.
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


def ratios(a, b, epochs):
    """per-key list of (epoch, a/b)"""
    out = {k: [] for k, _ in KEYS}
    for e in epochs:
        for k, _ in KEYS:
            x, y = a[e][k], b[e][k]
            if x == x and y == y and y != 0:
                out[k].append((e, x / y))
    return out


gate, prod = parse(sys.argv[1]), parse(sys.argv[2])
epochs = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 and sys.argv[3] else list(range(20, 34))
ref = parse(sys.argv[4]) if len(sys.argv) > 4 else None
epochs = [e for e in epochs if e in gate and e in prod]
print(f"{'epoch':>5} " + " ".join(f"{k:>18}" for k, _ in KEYS))
print(f"{'':>5} " + " ".join(f"{'gate / prod':>18}" for _ in KEYS))
for e in epochs:
    print(f"{e:>5} " + " ".join(f"{gate[e][k]:8.4g}/{prod[e][k]:<8.4g}".rjust(18) for k, _ in KEYS))
rg = ratios(gate, prod, epochs)
print(f"{'g/p':>5} " + " ".join(f"{(sum(v for _, v in rg[k]) / len(rg[k]) if rg[k] else float('nan')):>18.3f}" for k, _ in KEYS) + "   mean ratio gate/prod")
print(f"{'':>5} " + " ".join(f"{(min(v for _, v in rg[k]) if rg[k] else float('nan')):>8.3f}-{(max(v for _, v in rg[k]) if rg[k] else float('nan')):<9.3f}" for k, _ in KEYS) + "   range")
if ref is not None:
    er = [e for e in epochs if e in ref]
    rr = ratios(ref, prod, er)
    print(f"{'r/p':>5} " + " ".join(f"{(sum(v for _, v in rr[k]) / len(rr[k]) if rr[k] else float('nan')):>18.3f}" for k, _ in KEYS) + f"   mean ratio replicate/prod ({len(er)} epochs) = noise floor")
    print(f"{'':>5} " + " ".join(f"{(min(v for _, v in rr[k]) if rr[k] else float('nan')):>8.3f}-{(max(v for _, v in rr[k]) if rr[k] else float('nan')):<9.3f}" for k, _ in KEYS) + "   range")
    # how many gate epochs lie outside the replicate's range
    print(f"{'':>5} " + " ".join(f"{sum(1 for _, v in rg[k] if rr[k] and (v > max(x for _, x in rr[k]) or v < min(x for _, x in rr[k]))):>9d}/{len(rg[k]):<8d}" for k, _ in KEYS) + "   gate epochs outside the replicate range")
last = epochs[-1] if epochs else None
if last is not None:
    print(f"last common epoch {last}: " + ", ".join(f"{k} {gate[last][k]:.4g} vs {prod[last][k]:.4g} ({gate[last][k]/prod[last][k]-1:+.1%})" for k, _ in KEYS if prod[last][k]))
