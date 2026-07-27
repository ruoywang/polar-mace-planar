"""High-fidelity 1-D profile tables for the runtime baseline.

The 3-D form-factor fit is normalized by core amplitudes, so its plane-
averaged profiles carry ~0.05 eV absolute noise — too coarse for the PB
closure. Plane averaging suppresses the float32 cache noise by ~100x;
fitting the per-element 1-D kernels directly on the (0,0,gz) line with ALL
cached structures gives plateau-accurate profiles. Appends f1d_neutral and
f1d_phi [nhz, nelem] (complex stored as real pairs? no — radial => real)
to runtime_baseline_tables.npz; runtime corrects the 3-D reconstruction's
plane average to these.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

BASE = Path("/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB")
CACHE = BASE / "train-data" / "baseline_cache"
XYZS = [BASE / "train-data" / f for f in ("train.xyz", "val.xyz", "test.xyz")]
TABLES = Path(__file__).resolve().parent / "runtime_baseline_tables.npz"

meta = json.load(open(CACHE / "baseline_meta.json"))
index = {int(k): int(v) for k, v in json.load(open(CACHE / "baseline_index.json")).items()}
nx, ny, nz = meta["pb_shape"]
cell = np.asarray(meta["cell"], dtype=float)
nhz = nz // 2 + 1

geoms = {}
for xyz in XYZS:
    for at in ase_read(str(xyz), ":"):
        sid = int(at.info["sample_id"])
        if sid in index and sid not in geoms:
            geoms[sid] = (at.get_chemical_symbols(), at.get_positions())

d_old = dict(np.load(TABLES))
elements = [str(x) for x in d_old["elements"]]
n_el = len(elements)

f_cache = open(CACHE / "baseline_cache.npy", "rb")
version = np.lib.format.read_magic(f_cache)
shape, fortran, dtype = np.lib.format._read_array_header(f_cache, version)
data_start = f_cache.tell()
block = nx * ny * nz
itemsize = np.dtype(dtype).itemsize


def plane_profiles(row: int) -> np.ndarray:
    """[2, nz]: plane averages of neutral (field 0) and phi_base (field 2)."""
    out = np.zeros((2, nz))
    for oi, fi in enumerate((0, 2)):
        f_cache.seek(data_start + (row * 3 + fi) * block * itemsize)
        buf = np.frombuffer(f_cache.read(block * itemsize), dtype=dtype)
        out[oi] = buf.reshape(nx, ny, nz).astype(np.float64).mean(axis=(0, 1))
    return out


hz = np.arange(nhz)
inv_cell = np.linalg.inv(cell)

M = np.zeros((nhz, n_el, n_el), dtype=np.complex128)
B = np.zeros((2, nhz, n_el), dtype=np.complex128)
sids = sorted(geoms)
for n, sid in enumerate(sids):
    sym, pos = geoms[sid]
    frac_z = ((pos @ inv_cell) % 1.0)[:, 2]
    S = np.zeros((n_el, nhz), dtype=np.complex128)
    for e_i, el in enumerate(elements):
        uz = frac_z[[i for i, s in enumerate(sym) if s == el]]
        if len(uz):
            S[e_i] = np.exp(-2j * np.pi * hz[:, None] * uz[None, :]).sum(axis=1)
    prof = plane_profiles(index[sid])
    Pk = np.stack([np.fft.rfft(prof[0]), np.fft.rfft(prof[1])])
    for a in range(n_el):
        for b in range(a, n_el):
            acc = np.conj(S[a]) * S[b]
            M[:, a, b] += acc
            if b != a:
                M[:, b, a] += np.conj(acc)
        for fi in range(2):
            B[fi, :, a] += np.conj(S[a]) * Pk[fi]
    if (n + 1) % 50 == 0:
        print(f"  accumulated {n+1}/{len(sids)}")

tr = np.einsum("sii->s", M).real
reg = np.maximum(tr, 1e-30)[:, None, None] * 1e-12 * np.eye(n_el)
rhs = np.stack([B[0], B[1]], axis=-1)
sol = np.linalg.solve(M + reg, rhs)  # [nhz, n_el, 2] complex
print(f"max imag/real: {np.abs(sol.imag).max()/max(np.abs(sol.real).max(),1e-30):.2e}")
f1d_neutral = sol[:, :, 0]
f1d_phi = sol[:, :, 1]

# validation: reconstruct plane profiles for a held-out-style subset
rng = np.random.default_rng(3)
for sid in rng.permutation(sids)[:6]:
    sym, pos = geoms[sid]
    frac_z = ((pos @ inv_cell) % 1.0)[:, 2]
    S = np.zeros((n_el, nhz), dtype=np.complex128)
    for e_i, el in enumerate(elements):
        uz = frac_z[[i for i, s in enumerate(sym) if s == el]]
        if len(uz):
            S[e_i] = np.exp(-2j * np.pi * hz[:, None] * uz[None, :]).sum(axis=1)
    prof = plane_profiles(index[sid])
    rec_n = np.fft.irfft((f1d_neutral * S.T).sum(axis=1), n=nz)
    rec_p = np.fft.irfft((f1d_phi * S.T).sum(axis=1), n=nz)
    sol_reg = slice(int(nz * 30 / 45), int(nz * 43 / 45))  # solvent-side window
    print(f"sid {sid:3d}: neutral prof max|d| {np.abs(rec_n-prof[0]).max():.3e} "
          f"phi prof max|d| {np.abs(rec_p-prof[1]).max():.3e} eV "
          f"(solvent window {np.abs(rec_p-prof[1])[sol_reg].max():.3e})")

d_old["f1d_neutral_re"] = f1d_neutral.real
d_old["f1d_neutral_im"] = f1d_neutral.imag
d_old["f1d_phi_re"] = f1d_phi.real
d_old["f1d_phi_im"] = f1d_phi.imag
np.savez_compressed(TABLES, **d_old)
print(f"appended 1d tables to {TABLES}")
