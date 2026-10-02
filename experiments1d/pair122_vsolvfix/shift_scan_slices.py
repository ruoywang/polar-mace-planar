"""Rigid-shift test of the ML 3-D solvent charge against DFT on the stored full-resolution xz slices
(s3d_eval/slices.npz: plane through the Ni atom, periodic along a and z, 500 x 168 points, no smoothing).
Reviewer hypothesis 2026-10-01: a small displacement of a sharp interface peak produces a large point-wise error.
Test: shift the ML slice rigidly (Fourier phase shift = trigonometric interpolation, exact on the periodic grid, no
smoothing) by (dz, dx) and minimise the squared error against DFT; report the error before / after, the best shift,
the correlation, the norm ratio ||ML|| / ||DFT|| (invariant under a shift: a displaced field keeps its norm) and the
error in the DFT-positive and DFT-negative regions. Shifts along b (the third direction) are not visible on one slice.
Usage: python shift_scan_slices.py [tags...]   (default: all frames + pairs in slices.npz)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

W = Path(__file__).parent
S = np.load(W / "s3d_eval/slices.npz")
TAGS = sys.argv[1:] or sorted(set(k.split("|")[0] for k in S.files), key=lambda t: (t[0] == "d", int(t.lstrip("d"))))


def fshift(F, dz, dx, lz, lx):
    """periodic sub-grid shift of F[nz, nx] by (dz, dx) in A: F_shift(r) = F(r - d)."""
    nz, nx = F.shape
    kz = np.fft.fftfreq(nz, d=lz / nz)[:, None]; kx = np.fft.rfftfreq(nx, d=lx / nx)[None, :]
    return np.fft.irfft2(np.fft.rfft2(F) * np.exp(-2j * np.pi * (kz * dz + kx * dx)), s=F.shape)


def metrics(M, D):
    d = M - D
    pos, neg = D > 0, D < 0
    return dict(L1=float(np.abs(d).sum() / np.abs(D).sum()), L2=float((d ** 2).sum() / (D ** 2).sum()),
                corr=float((M * D).sum() / np.sqrt((M * M).sum() * (D * D).sum())),
                L1pos=float(np.abs(d[pos]).sum() / np.abs(D[pos]).sum()), L1neg=float(np.abs(d[neg]).sum() / np.abs(D[neg]).sum()))


def best_shift(M, D, lz, lx, zmax=3.0, xmax=2.0, x_free=True):
    nz, nx = D.shape
    # integer-shift cross-correlation (periodic), restricted window
    cc = np.fft.irfft2(np.fft.rfft2(D) * np.conj(np.fft.rfft2(M)), s=D.shape)      # cc[s] = sum M(r) D(r + s)
    iz = np.arange(nz); ix = np.arange(nx)
    sz = np.where(iz <= nz // 2, iz, iz - nz) * lz / nz; sx = np.where(ix <= nx // 2, ix, ix - nx) * lx / nx
    mask = (np.abs(sz)[:, None] <= zmax) & ((np.abs(sx)[None, :] <= xmax) if x_free else (ix[None, :] == 0))
    cc = np.where(mask, cc, -np.inf)
    k = np.unravel_index(np.argmax(cc), cc.shape)
    z0, x0 = float(sz[k[0]]), float(sx[k[1]])
    obj = (lambda p: float(((fshift(M, p[0], p[1], lz, lx) - D) ** 2).sum())) if x_free else \
          (lambda p: float(((fshift(M, p[0], 0.0, lz, lx) - D) ** 2).sum()))
    r = minimize(obj, [z0, x0] if x_free else [z0], method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-12, maxiter=400))
    p = r.x if x_free else [r.x[0], 0.0]
    return float(p[0]), float(p[1])


rows = []
for tag in TAGS:
    lat = S[f"{tag}|lat"]; lz, lx = float(lat[2, 2]), float(np.linalg.norm(lat[0]))
    for ch in ("b", "i"):
        D = S[f"{tag}|dft_{ch}|xz"].astype(np.float64); M = S[f"{tag}|ml_{ch}|xz"].astype(np.float64)
        if np.abs(D).sum() == 0 or np.abs(M).sum() == 0:
            continue
        m0 = metrics(M, D)
        nr = float(np.sqrt((M * M).sum() / (D * D).sum()))
        dz1, _ = best_shift(M, D, lz, lx, x_free=False); m1 = metrics(fshift(M, dz1, 0.0, lz, lx), D)
        dz2, dx2 = best_shift(M, D, lz, lx, x_free=True); Ms = fshift(M, dz2, dx2, lz, lx); m2 = metrics(Ms, D)
        a = float((Ms * D).sum() / (Ms * Ms).sum()); m3 = metrics(a * Ms, D)                     # shift + one scale
        # lateral part on the slice (each z row minus its own mean over x), shifted on its own
        Dl = D - D.mean(axis=1, keepdims=True); Ml = M - M.mean(axis=1, keepdims=True)
        ml0 = metrics(Ml, Dl); nrl = float(np.sqrt((Ml * Ml).sum() / (Dl * Dl).sum()))
        dzl, dxl = best_shift(Ml, Dl, lz, lx, x_free=True); mll = metrics(fshift(Ml, dzl, dxl, lz, lx), Dl)
        rows.append(dict(tag=tag, ch=ch, lat_norm_ratio=nrl, lat_corr0=ml0["corr"], lat_L1_0=ml0["L1"], lat_L2_0=ml0["L2"], lat_dz=dzl, lat_dx=dxl,
                         lat_L1_zx=mll["L1"], lat_L2_zx=mll["L2"], lat_corr_zx=mll["corr"], norm_ratio=nr, corr0=m0["corr"], L1_0=m0["L1"], L2_0=m0["L2"], L1pos0=m0["L1pos"], L1neg0=m0["L1neg"],
                         dz_z=dz1, L1_z=m1["L1"], L2_z=m1["L2"], dz=dz2, dx=dx2, L1_zx=m2["L1"], L2_zx=m2["L2"], corr_zx=m2["corr"],
                         L1pos_zx=m2["L1pos"], L1neg_zx=m2["L1neg"], scale=a, L1_zxs=m3["L1"], L2_zxs=m3["L2"]))
        r = rows[-1]
        print(f"{tag:5s} {ch}: |ML|/|DFT| {nr:.3f} corr {m0['corr']:.3f} | L1 {m0['L1']:.3f} L2 {m0['L2']:.3f} (DFT>0 {m0['L1pos']:.3f}, <0 {m0['L1neg']:.3f})"
              f" | z-shift {dz1:+.3f} A -> L1 {m1['L1']:.3f} L2 {m1['L2']:.3f} | zx-shift ({dz2:+.3f},{dx2:+.3f}) -> L1 {m2['L1']:.3f} L2 {m2['L2']:.3f} corr {m2['corr']:.3f}"
              f" (>0 {m2['L1pos']:.3f}, <0 {m2['L1neg']:.3f}) | +scale {a:.3f} -> L1 {m3['L1']:.3f} L2 {m3['L2']:.3f}"
              f" || LATERAL |ML|/|DFT| {nrl:.3f} corr {ml0['corr']:.3f} L1 {ml0['L1']:.3f} L2 {ml0['L2']:.3f} -> shift ({dzl:+.3f},{dxl:+.3f}) L1 {mll['L1']:.3f} L2 {mll['L2']:.3f} corr {mll['corr']:.3f}", flush=True)
np.save(W / "shift_scan_slices.npy", rows, allow_pickle=True)
for ch in ("b", "i"):
    R = [r for r in rows if r["ch"] == ch and not r["tag"].startswith("d")]
    if not R:
        continue
    f = lambda k: np.array([r[k] for r in R])
    print(f"\n== {ch}: {len(R)} frames; median [min, max]")
    for k in ("norm_ratio", "corr0", "L1_0", "L2_0", "dz_z", "L1_z", "L2_z", "dz", "dx", "L1_zx", "L2_zx", "corr_zx", "scale", "L1_zxs", "L2_zxs",
              "lat_norm_ratio", "lat_corr0", "lat_L1_0", "lat_L2_0", "lat_dz", "lat_dx", "lat_L1_zx", "lat_L2_zx", "lat_corr_zx"):
        v = f(k); print(f"   {k:10s} {np.median(v):+.3f} [{v.min():+.3f}, {v.max():+.3f}]")
