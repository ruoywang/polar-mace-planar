"""Extract per-element form-factor tables from the cached baseline fields.

Model: every cached field (neutral, dencor, phi_base) is a superposition of
identical per-element radial functions placed at the atomic positions, so in
reciprocal space  F_j(G) = sum_el f_el(|G|) * S_el_j(G)  with the structure
factor S from geometry alone. Per unique-|G| shell least squares across a fit
subset recovers f_el exactly (the cache was CONSTRUCTED as such a
superposition); held-out structures give a blind reconstruction test.

Login-node safe: rows are read by seeking inside baseline_cache.npy (no mmap
of the 14 GB file); normal equations accumulate via bincount per shell.

Output: runtime_baseline_tables.npz with
  g2_shells [nshell]           sorted |G|^2 values of THIS grid (1/A^2, no 2pi)
  f_neutral/f_dencor/f_phi [nshell, nelem]   form factors per element
  elements, cell, pb_shape, fit_sids, holdout report printed to stdout
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

BASE = Path("/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB")
CACHE = BASE / "train-data" / "baseline_cache"
XYZS = [BASE / "train-data" / f for f in ("train.xyz", "val.xyz", "test.xyz")]
N_FIT = int(sys.argv[1]) if len(sys.argv) > 1 else 48
N_VAL = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OUT = Path(__file__).resolve().parent / "runtime_baseline_tables.npz"

meta = json.load(open(CACHE / "baseline_meta.json"))
index = {int(k): int(v) for k, v in json.load(open(CACHE / "baseline_index.json")).items()}
nx, ny, nz = meta["pb_shape"]
fields = meta["fields"]
assert fields == ["neutral_values", "dencor_values", "phi_base_eV"], fields
cell = np.asarray(meta["cell"], dtype=float)

# ---- geometry per sample_id (all three splits) --------------------------
geoms = {}
for xyz in XYZS:
    for at in ase_read(str(xyz), ":"):
        sid = int(at.info["sample_id"])
        if sid in index and sid not in geoms:
            geoms[sid] = (at.get_chemical_symbols(), at.get_positions())
print(f"{len(geoms)} geometries with cache rows")

elements = sorted({s for sym, _ in geoms.values() for s in sym})
print("elements:", elements)
n_el = len(elements)

# ---- row reader (seek, no mmap) ------------------------------------------
f_cache = open(CACHE / "baseline_cache.npy", "rb")
version = np.lib.format.read_magic(f_cache)
shape, fortran, dtype = np.lib.format._read_array_header(f_cache, version)
assert not fortran and shape[1] == 3 and shape[2:] == (nx, ny, nz), shape
data_start = f_cache.tell()
block = nx * ny * nz
itemsize = np.dtype(dtype).itemsize


def read_fields(row: int) -> np.ndarray:
    f_cache.seek(data_start + row * 3 * block * itemsize)
    buf = np.frombuffer(f_cache.read(3 * block * itemsize), dtype=dtype)
    return buf.reshape(3, nx, ny, nz).astype(np.float64)


# ---- reciprocal mesh, shell grouping (fixed grid, shared by all) ---------
hx = np.fft.fftfreq(nx) * nx
hy = np.fft.fftfreq(ny) * ny
hz = np.arange(nz // 2 + 1)  # rfft last axis
H = np.stack(np.meshgrid(hx, hy, hz, indexing="ij"), axis=-1).reshape(-1, 3)
inv_cell = np.linalg.inv(cell)
gmat = inv_cell @ inv_cell.T  # metric for |G/2pi|^2 = h . gmat . h
g2 = np.einsum("ki,ij,kj->k", H, gmat, H)
g2_round = np.round(g2, 9)
g2_shells, shell_idx = np.unique(g2_round, return_inverse=True)
n_shell = len(g2_shells)
print(f"grid {nx}x{ny}x{nz}, rfft points {len(H)}, unique shells {n_shell}")

# ---- pick fit / holdout sets ---------------------------------------------
rng = np.random.default_rng(7)
sids = sorted(geoms)
picked = rng.permutation(sids)[: N_FIT + N_VAL]
fit_sids, val_sids = sorted(picked[:N_FIT].tolist()), sorted(picked[N_FIT:].tolist())
print(f"fit on {len(fit_sids)} sids, blind-validate on {val_sids}")


def structure_factors(sid: int) -> np.ndarray:
    sym, pos = geoms[sid]
    frac = (pos @ np.linalg.inv(cell)) % 1.0
    S = np.zeros((n_el, len(H)), dtype=np.complex128)
    for e_i, el in enumerate(elements):
        sel = frac[[i for i, s in enumerate(sym) if s == el]]
        if len(sel) == 0:
            continue
        for c0 in range(0, len(sel), 16):  # chunk atoms: login vmem limit
            phase = H @ sel[c0:c0 + 16].T  # [nk, <=16]
            S[e_i] += np.exp(-2j * np.pi * phase).sum(axis=1)
    return S


def field_ffts(row: int) -> np.ndarray:
    F = read_fields(row)
    return np.stack([np.fft.rfftn(F[i]).reshape(-1) for i in range(3)])


# ---- accumulate normal equations ----------------------------------------
M = np.zeros((n_shell, n_el, n_el), dtype=np.complex128)
B = np.zeros((3, n_shell, n_el), dtype=np.complex128)


def shell_sum(values: np.ndarray) -> np.ndarray:
    return (np.bincount(shell_idx, weights=values.real, minlength=n_shell)
            + 1j * np.bincount(shell_idx, weights=values.imag, minlength=n_shell))


for n, sid in enumerate(fit_sids):
    S = structure_factors(sid)
    Fk = field_ffts(index[sid])
    for a in range(n_el):
        for b in range(a, n_el):
            acc = shell_sum(np.conj(S[a]) * S[b])
            M[:, a, b] += acc
            if b != a:
                M[:, b, a] += np.conj(acc)
        for fi in range(3):
            B[fi, :, a] += shell_sum(np.conj(S[a]) * Fk[fi])
    if (n + 1) % 8 == 0:
        print(f"  accumulated {n+1}/{len(fit_sids)}")

# ---- solve per shell (Tikhonov for degenerate shells) --------------------
tr = np.einsum("sii->s", M).real
reg = np.maximum(tr, 1e-30)[:, None, None] * 1e-10 * np.eye(n_el)
Minv_ready = M + reg
f_tables = np.zeros((3, n_shell, n_el))
rhs = np.stack([B[fi] for fi in range(3)], axis=-1)  # [n_shell, n_el, 3]
sol = np.linalg.solve(Minv_ready, rhs)  # [n_shell, n_el, 3]
imag_frac = np.abs(sol.imag).max() / max(np.abs(sol.real).max(), 1e-30)
print(f"max imag/real of solution: {imag_frac:.2e} (should be ~0)")
for fi in range(3):
    f_tables[fi] = sol[:, :, fi].real

# ---- blind validation -----------------------------------------------------
def reconstruct(sid: int, fi: int) -> np.ndarray:
    S = structure_factors(sid)
    Fk = (f_tables[fi][shell_idx, :].T * S).sum(axis=0)
    return np.fft.irfftn(Fk.reshape(nx, ny, nz // 2 + 1), s=(nx, ny, nz))


names = ["neutral", "dencor", "phi_base"]
report = []
for sid in val_sids:
    truth = read_fields(index[sid])
    line = [f"sid {sid:3d}"]
    for fi in range(3):
        rec = reconstruct(sid, fi)
        denom = max(np.abs(truth[fi]).max(), 1e-30)
        rel = np.sqrt(np.mean((rec - truth[fi]) ** 2)) / denom
        line.append(f"{names[fi]} rel-rmse {rel:.2e}")
        if fi == 0:
            line.append(f"e-count rec {rec.mean():.4f} vs true {truth[0].mean():.4f}")
    report.append("  ".join(line))
    print(report[-1])

np.savez_compressed(
    OUT,
    g2_shells=g2_shells,
    f_neutral=f_tables[0],
    f_dencor=f_tables[1],
    f_phi=f_tables[2],
    elements=np.array(elements),
    cell=cell,
    pb_shape=np.array([nx, ny, nz]),
    fit_sids=np.array(fit_sids),
    val_sids=np.array(val_sids),
)
print(f"wrote {OUT}")
