"""3-D solvent charge report page (user request 2026-09-30) from the outputs of solvent3d_full_eval.py.

Usage: PYTHONPATH=$WORK/apps/pylibs_skimage python gen_s3d_page.py <eval_dir> <out.html>
Figures are matplotlib PNGs embedded as base64, one figure per block, full width; in every slice / isosurface
figure the rows are DFT, ML, ML - DFT (DFT and ML share one colour scale, the error has its own, both
centred on zero). Formulas are rendered offline (mathtext -> inline SVG, currentColor).
"""
from __future__ import annotations

import base64
import html
import io
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import map_coordinates

EV, OUTP = Path(sys.argv[1]), Path(sys.argv[2])
M = json.load(open(EV / "metrics.json"))
PZ = np.load(EV / "perz.npz")
SL = np.load(EV / "slices.npz")
VO = np.load(EV / "volumes.npz")
ROWS, PAIRS, REP = M["rows"], M["pairs"], M["representative"]
SPLIT = M["split"]
GROUPS = ["NiN44 charged", "NiN88 charged", "NiN44 solvated neutral"]
GCOL = {"NiN44 charged": "#2a78d6", "NiN88 charged": "#eb6834", "NiN44 solvated neutral": "#1baf7a"}
CHN = {"b": "bound", "i": "ionic", "t": "total"}
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
ACOL = {"C": "#3a3a3a", "N": "#2a4fd6", "O": "#e34948", "H": "#f2f2f2", "Ni": "#008300"}
ASZ = {"C": 34, "N": 40, "O": 40, "H": 18, "Ni": 90}
DPI = 90


def png(fig, dpi=None):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi or DPI, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def tex(formula, size=13):
    fig = plt.figure(figsize=(0.1, 0.1)); fig.text(0, 0, formula, fontsize=size)
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.06, transparent=True); plt.close(fig)
    s = buf.getvalue().decode(); s = s[s.index("<svg"):]
    w = float(re.search(r'width="([0-9.]+)pt"', s).group(1)); vb = re.search(r'viewBox="([^"]+)"', s).group(1)
    head = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="' + vb + '" style="width:min(100%,' + f"{w * 1.35:.0f}"
            + 'px);height:auto;display:block" role="img" class="math">')
    s = re.sub(r'<svg[^>]*>', lambda m: head, s, count=1)
    s = re.sub(r'<metadata>.*?</metadata>', '', s, flags=re.S)
    for a, b in (('fill="#000000"', 'fill="currentColor"'), ('fill:#000000', 'fill:currentColor'),
                 ('stroke="#000000"', 'stroke="currentColor"'), ('stroke:#000000', 'stroke:currentColor')):
        s = s.replace(a, b)
    return f'<div class="eq">{s}</div>'


def nice_step(span, n=6):
    raw = span / max(n, 1); mag = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1.0
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def row(sid):
    return next(r for r in ROWS if r["sid"] == sid)


def pair(sid_c):
    return next(p for p in PAIRS if p["sid_c"] == sid_c)


def symlim(*arrs, q=99.9):
    v = float(np.percentile(np.abs(np.concatenate([a.ravel() for a in arrs])), q))
    return v if v > 0 else 1e-12


def zwindow(tag, frac_floor=0.01, pad=1.0, chans=("b", "i")):
    """z window where the DFT |rho| (plane-integrated, summed over chans) exceeds frac_floor of its maximum."""
    key = (lambda ch, k: f"{tag}|{ch}|{k}")
    rz = sum(PZ[key(ch, "ref_z")] for ch in chans)
    meta = SL[f"{tag}|meta"]; lat = SL[f"{tag}|lat"]; nz = int(meta[3]); dz = lat[2, 2] / nz
    idx = np.where(rz > frac_floor * rz.max())[0]
    return max(0.0, idx.min() * dz - pad), min(lat[2, 2], idx.max() * dz + pad)


def atoms_near_xz(tag, tol=1.0):
    lat, frac, sym = SL[f"{tag}|lat"], SL[f"{tag}|frac"], SL[f"{tag}|sym"]
    meta = SL[f"{tag}|meta"]; fy0 = int(meta[0]) / int(meta[4])
    dfy = (frac[:, 1] - fy0 + 0.5) % 1.0 - 0.5
    dist = np.abs(dfy) * lat[1, 1]
    s = ((frac[:, 0] + dfy * lat[1, 0] / lat[0, 0]) % 1.0) * np.linalg.norm(lat[0])
    z = frac[:, 2] * lat[2, 2]
    m = dist <= tol
    return s[m], z[m], sym[m]


def draw_atoms(ax, xs, ys, syms):
    for el in ("H", "O", "C", "N", "Ni"):
        m = syms == el
        if m.any():
            ax.scatter(xs[m], ys[m], s=ASZ[el], c=ACOL[el], edgecolors="black", linewidths=0.6, zorder=5)


def xz_figure(tag, ch, title_tag, delta=False):
    """rows DFT / ML / ML-DFT of the xz plane through the Ni atom; ch in b, i, t."""
    def get(kind):
        if ch == "t":
            return SL[f"{tag}|{kind}_b|xz"].astype(float) + SL[f"{tag}|{kind}_i|xz"].astype(float)
        return SL[f"{tag}|{kind}_{ch}|xz"].astype(float)
    D, Mx = get("dft"), get("ml"); E = Mx - D
    lat = SL[f"{tag}|lat"]; meta = SL[f"{tag}|meta"]; nz = int(meta[3]); a = np.linalg.norm(lat[0])
    z0, z1 = zwindow(tag)
    fr, sy = SL[f"{tag}|frac"], SL[f"{tag}|sym"]
    if (sy == "Ni").any():
        z0 = max(0.0, min(z0, float(fr[sy == "Ni", 2][0] * lat[2, 2]) - 2.0))
    iz0, iz1 = int(np.floor(z0 / lat[2, 2] * nz)), int(np.ceil(z1 / lat[2, 2] * nz))
    D, Mx, E = D[iz0:iz1], Mx[iz0:iz1], E[iz0:iz1]
    ext = [iz0 * lat[2, 2] / nz, iz1 * lat[2, 2] / nz, 0.0, a]
    v = symlim(D, Mx); ve = symlim(E)
    s_at, z_at, sy_at = atoms_near_xz(tag)
    ma = (z_at >= ext[0]) & (z_at <= ext[1])
    width = 12.0; row_h = 0.82 * width * a / (ext[1] - ext[0]) + 0.55
    fig, axs = plt.subplots(3, 1, figsize=(width, 3 * row_h), sharex=True, constrained_layout=True)
    lab = "Δρ" if delta else "ρ"
    names = [f"DFT {lab}", f"ML {lab}", f"ML − DFT"]
    for ax, F, nm, vv in zip(axs, (D, Mx, E), names, (v, v, ve)):
        im = ax.imshow(F.T, origin="lower", extent=ext, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -vv, vv), aspect="equal",
                       interpolation="nearest")
        draw_atoms(ax, z_at[ma], s_at[ma], sy_at[ma])
        ax.set_ylabel("x along a (Å)"); ax.set_title(nm, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("e/Å³")
    axs[-1].set_xlabel("z (Å)")
    return png(fig), dict(v=v, ve=ve, z0=ext[0], z1=ext[1], dmax=float(np.abs(D).max()), emax=float(np.abs(E).max()))


def rect_resample(F, lat, nrx=168):
    """periodic bilinear resampling of an (ny, nx) hexagonal-plane slice onto the rectangle [0,a) x [0,b_y)."""
    a, bx, by = lat[0, 0], lat[1, 0], lat[1, 1]
    nry = int(round(nrx * by / a))
    x = (np.arange(nrx) + 0.5) / nrx * a; y = (np.arange(nry) + 0.5) / nry * by
    X, Y = np.meshgrid(x, y)
    fy = Y / by; fx = (X - fy * bx) / a
    ny, nx = F.shape
    return map_coordinates(F, [(fy * ny) % ny, (fx * nx) % nx], order=1, mode="grid-wrap"), (0, a, 0, by)


def atoms_near_xy(tag, zplane, tol=1.0):
    lat, frac, sym = SL[f"{tag}|lat"], SL[f"{tag}|frac"], SL[f"{tag}|sym"]
    z = frac[:, 2] * lat[2, 2]
    m = np.abs(z - zplane) <= tol
    fy = frac[m, 1] % 1.0
    x = (frac[m, 0] * lat[0, 0] + fy * lat[1, 0]) % lat[0, 0]
    return x, fy * lat[1, 1], sym[m]


def xy_figure(tag, ch, plane, title_tag, lateral=False, delta=False):
    """rows DFT / ML / ML-DFT of the xy plane 'xyb' (bound peak) or 'xyi' (ionic peak); lateral subtracts each plane mean."""
    def get(kind):
        if ch == "t":
            return SL[f"{tag}|{kind}_b|{plane}"].astype(float) + SL[f"{tag}|{kind}_i|{plane}"].astype(float)
        return SL[f"{tag}|{kind}_{ch}|{plane}"].astype(float)
    D, Mx = get("dft"), get("ml")
    means = (float(D.mean()), float(Mx.mean()))
    if lateral:
        D, Mx = D - D.mean(), Mx - Mx.mean()
    E = Mx - D
    lat = SL[f"{tag}|lat"]; meta = SL[f"{tag}|meta"]; nz = int(meta[3])
    izp = int(meta[1] if plane == "xyb" else meta[2]); zp = izp * lat[2, 2] / nz
    Dr, ext = rect_resample(D, lat); Mr, _ = rect_resample(Mx, lat); Er, _ = rect_resample(E, lat)
    v = symlim(Dr, Mr); ve = symlim(Er)
    xa, ya, sa = atoms_near_xy(tag, zp)
    width = 8.6
    fig, axs = plt.subplots(3, 1, figsize=(width, 3 * (0.82 * width * ext[3] / ext[1] + 0.55)), constrained_layout=True)
    lab = ("Δρ" if delta else "ρ") + ("⊥" if lateral else "")
    for ax, F, nm, vv in zip(axs, (Dr, Mr, Er), (f"DFT {lab}", f"ML {lab}", "ML − DFT"), (v, v, ve)):
        im = ax.imshow(F, origin="lower", extent=ext, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -vv, vv), aspect="equal", interpolation="nearest")
        draw_atoms(ax, xa, ya, sa)
        ax.set_ylabel("y (Å)"); ax.set_title(nm, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("e/Å³")
    axs[-1].set_xlabel("x (Å)")
    return png(fig), dict(v=v, ve=ve, z=zp, mean_dft=means[0], mean_ml=means[1], emax=float(np.abs(Er).max()))


def iso_figure(sid, ch, level, level_err, title_tag):
    """three separate full-width renders: DFT, ML, ML - DFT; fixed levels, same camera."""
    from skimage.measure import marching_cubes
    if ch == "t":
        D = VO[f"{sid}|dft_b"].astype(float) + VO[f"{sid}|dft_i"].astype(float)
        Mx = VO[f"{sid}|ml_b"].astype(float) + VO[f"{sid}|ml_i"].astype(float)
    else:
        D, Mx = VO[f"{sid}|dft_{ch}"].astype(float), VO[f"{sid}|ml_{ch}"].astype(float)
    E = Mx - D
    lat, frac, sym = VO[f"{sid}|lat"], VO[f"{sid}|frac"], VO[f"{sid}|sym"]
    nz2, ny2, nx2 = D.shape
    z0, z1 = zwindow(str(sid), frac_floor=0.12, pad=1.2, chans=(("b", "i") if ch == "t" else (ch,)))
    k0, k1 = int(np.floor(z0 / lat[2, 2] * nz2)), int(np.ceil(z1 / lat[2, 2] * nz2))
    zc = frac[:, 2] * lat[2, 2]; ma = (zc >= z0) & (zc <= z1)
    cart_at = frac[ma] @ lat
    counts, imgs = {}, []
    xs = np.array([0, lat[0, 0], lat[0, 0] + lat[1, 0], lat[1, 0]])
    for F, nm, L in ((D, "DFT", level), (Mx, "ML", level), (E, "ML − DFT", level_err)):
        fig = plt.figure(figsize=(12, 6.4))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        sub = F[k0:k1]
        for sign, col in ((1, "#d62728"), (-1, "#1f5fbf")):
            try:
                verts, faces, _, _ = marching_cubes(sign * sub, L)
            except (ValueError, RuntimeError):
                counts[(nm, sign)] = 0
                continue
            fr = np.stack([verts[:, 2] / nx2, verts[:, 1] / ny2, (verts[:, 0] + k0) / nz2], axis=1)
            cart = fr @ lat
            ax.add_collection3d(Poly3DCollection(cart[faces], facecolor=col, edgecolor="none", alpha=0.55))
            counts[(nm, sign)] = int(faces.shape[0])
        for el, colr in (("O", "#606060"), ("C", "#3a3a3a"), ("N", "#3a3a3a"), ("Ni", "#008300")):
            m = sym[ma] == el
            if m.any():
                ax.scatter(cart_at[m, 0], cart_at[m, 1], cart_at[m, 2], s=ASZ[el] * 0.6, c=colr, edgecolors="black", linewidths=0.4, depthshade=False)
        ax.set_xlim(xs.min(), xs.max()); ax.set_ylim(0, lat[1, 1]); ax.set_zlim(z0, z1)
        ax.set_box_aspect((xs.max() - xs.min(), lat[1, 1], max(z1 - z0, 4.0)), zoom=1.25)
        ax.view_init(elev=34, azim=-62)
        ax.set_xlabel("x (Å)", labelpad=10); ax.set_ylabel("y (Å)", labelpad=10); ax.set_zlabel("z (Å)", labelpad=8)
        ax.set_title(f"{nm}: surfaces at +{L:.2e} (red) and −{L:.2e} (blue) e/Å³", loc="left", fontsize=13)
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.97)
        imgs.append(png(fig, dpi=180))
    return imgs, dict(counts=counts, z0=z0, z1=z1)


def strip_figure(metric, ylabel, cats, signed=False, title=""):
    """per-frame dots; cats = [(label, group, ch)]"""
    fig, ax = plt.subplots(figsize=(12, 5.4))
    rng = np.random.default_rng(0)
    allv = []
    cats = [c for c in cats if any(r["group"] == c[1] and r["split"] == SPLIT for r in ROWS)]
    for k, (lab, g, ch) in enumerate(cats):
        vals = np.array([r[ch][metric] for r in ROWS if r["group"] == g and r["split"] == SPLIT])
        allv.append(vals)
        ax.scatter(k + rng.uniform(-0.18, 0.18, vals.size), vals, s=34, c=GCOL[g], edgecolors="#fcfcfb", linewidths=1.2, zorder=3)
        ax.plot([k - 0.3, k + 0.3], [np.median(vals)] * 2, color="#0b0b0b", lw=2, zorder=4)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([c[0] for c in cats], rotation=0)
    vmax = max(float(np.max(np.abs(v))) for v in allv)
    if signed:
        st = nice_step(2 * vmax); lim = np.ceil(vmax / st) * st
        ax.set_ylim(-lim, lim); ax.set_yticks(np.arange(-lim, lim + st / 2, st)); ax.axhline(0, color="#898781", lw=1)
    else:
        st = nice_step(vmax); top = np.ceil(vmax / st) * st
        ax.set_ylim(0, top); ax.set_yticks(np.arange(0, top + st / 2, st))
    ax.set_ylabel(ylabel); ax.grid(axis="y", color="#e1e0d9", lw=0.8); ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left")
    return png(fig)


def perz_figure(ch):
    fig, ax = plt.subplots(figsize=(12, 5.6))
    top = 0.0
    for g in GROUPS:
        sids = [r["sid"] for r in ROWS if r["group"] == g and r["split"] == SPLIT]
        if not sids:
            continue
        r0 = row(sids[0]); lz = 45.0; nz = r0["grid"][0]; z = np.arange(nz) * lz / nz
        err = np.mean([PZ[f"{s}|{ch}|err_z"] for s in sids], axis=0); ref = np.mean([PZ[f"{s}|{ch}|ref_z"] for s in sids], axis=0)
        ax.plot(z, ref, color=GCOL[g], lw=1.6, ls=(0, (4, 3)), label=f"{g}: DFT |ρ| (n = {len(sids)})")
        ax.plot(z, err, color=GCOL[g], lw=2.2, label=f"{g}: |ML − DFT|")
        top = max(top, float(ref.max()), float(err.max()))
    st = nice_step(top); ax.set_ylim(0, np.ceil(top / st) * st)
    ax.set_xlim(0, 45); ax.set_xticks(np.arange(0, 46, 5))
    ax.set_xlabel("z (Å)"); ax.set_ylabel("plane integral (e/Å)")
    ax.grid(color="#e1e0d9", lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(f"{CHN[ch]} charge: plane-integrated absolute error per z plane, mean over {SPLIT} frames", loc="left")
    return png(fig)


def pair_strip():
    fig, ax = plt.subplots(figsize=(12, 5.0))
    rng = np.random.default_rng(1)
    prs = [p for p in PAIRS if p["split"] == SPLIT]
    ex = [p for p in PAIRS if p["split"] != SPLIT]
    vmax = 0.0
    for k, ch in enumerate(("b", "i", "t")):
        v = np.array([p[ch]["eps_3d"] for p in prs]); vmax = max(vmax, float(v.max()))
        ax.scatter(k + rng.uniform(-0.18, 0.18, v.size), v, s=36, c="#2a78d6", edgecolors="#fcfcfb", linewidths=1.2, zorder=3,
                   label=f"{SPLIT} pairs (n = {len(prs)})" if k == 0 else None)
        ax.plot([k - 0.3, k + 0.3], [np.median(v)] * 2, color="#0b0b0b", lw=2, zorder=4)
        for p in ex:
            ax.scatter([k + 0.3], [p[ch]["eps_3d"]], marker="D", s=60, c="#eb6834", edgecolors="black", linewidths=0.6, zorder=5,
                       label=f"sid {p['sid_c']} / {p['sid_n']} (test)" if k == 0 else None)
            vmax = max(vmax, p[ch]["eps_3d"])
    st = nice_step(vmax); top = np.ceil(vmax / st) * st
    ax.set_ylim(0, top); ax.set_yticks(np.arange(0, top + st / 2, st))
    ax.set_xticks(range(3)); ax.set_xticklabels(["bound Δρ", "ionic Δρ", "total Δρ"])
    ax.set_ylabel("ε₃D of Δρ = ρ(charged) − ρ(neutral)"); ax.grid(axis="y", color="#e1e0d9", lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)
    return png(fig)


# ------------------------------------------------------------------ tables
def fmt_range(v):
    return f"{np.median(v):.3f} [{np.min(v):.3f}–{np.max(v):.3f}]"


def table(head, rows_, cls=""):
    return (f'<div class="tw"><table class="{cls}"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows_) + "</tbody></table></div>")


stat_rows = []
for g in GROUPS:
    rs = [r for r in ROWS if r["group"] == g and r["split"] == SPLIT]
    if not rs:
        continue
    for ch in ("b", "i", "t"):
        e3 = np.array([r[ch]["eps_3d"] for r in rs]); el = np.array([r[ch]["eps_lat"] for r in rs]); ep = np.array([r[ch]["eps_pa"] for r in rs])
        ae = np.array([r[ch]["abs_err_e"] for r in rs]); re_ = np.array([r[ch]["ref_abs_e"] for r in rs]); ne = np.array([r[ch]["net_err_e"] for r in rs])
        rm = np.array([r[ch]["rmse"] for r in rs])
        weak = False
        stat_rows.append([f"{g} · {CHN[ch]}", len(rs), "— (reference ≈ 0)" if weak else fmt_range(e3), "—" if weak else f"{np.median(el):.3f}",
                          "—" if weak else f"{np.median(ep):.3f}", f"{np.median(ae):.4f}", f"{np.median(re_):.4f}",
                          f"{ne.mean():+.4f} / {np.sqrt((ne ** 2).mean()):.4f}", f"{np.median(rm):.2e}"])
tbl_stats = table(["frames · channel", "n", "ε₃D median [min–max]", "ε⊥ median", "ε plane-avg median", "∫|ML−DFT| median (e)",
                   "∫|DFT| median (e)", "net error mean / rms (e)", "RMSE median (e/Å³)"], stat_rows)

prs = [p for p in PAIRS if p["split"] == SPLIT]
pair_rows = []
for ch in ("b", "i", "t"):
    e3 = np.array([p[ch]["eps_3d"] for p in prs]); el = np.array([p[ch]["eps_lat"] for p in prs]); ne = np.array([p[ch]["net_err_e"] for p in prs])
    nd = np.array([p[ch]["net_dft_e"] for p in prs])
    pair_rows.append([f"{SPLIT} pairs · {CHN[ch]} Δρ", len(prs), fmt_range(e3), f"{np.median(el):.3f}", f"{np.median([p[ch]['abs_err_e'] for p in prs]):.4f}",
                      f"{np.median([p[ch]['ref_abs_e'] for p in prs]):.4f}", f"{nd.mean():+.4f}", f"{ne.mean():+.4f} / {np.sqrt((ne ** 2).mean()):.4f}"])
for p in PAIRS:
    if p["split"] == SPLIT:
        continue
    for ch in ("b", "i", "t"):
        pair_rows.append([f"sid {p['sid_c']} / {p['sid_n']} ({p['split']}) · {CHN[ch]} Δρ", 1, f"{p[ch]['eps_3d']:.3f}", f"{p[ch]['eps_lat']:.3f}",
                          f"{p[ch]['abs_err_e']:.4f}", f"{p[ch]['ref_abs_e']:.4f}", f"{p[ch]['net_dft_e']:+.4f}", f"{p[ch]['net_err_e']:+.4f}"])
tbl_pairs = table(["pairs · channel", "n", "ε₃D median [min–max]", "ε⊥ median", "∫|ΔML−ΔDFT| median (e)", "∫|ΔDFT| median (e)",
                   "net ΔDFT mean (e)", "net error mean / rms (e)"], pair_rows)

rep_rows = []
for role, sid in REP.items():
    r = row(sid)
    rep_rows.append([{"median": "median-error frame", "max": "highest-error frame", "pair_page": "pair-page frame"}[role], sid, r["group"], r["split"],
                     f"{r['b']['eps_3d']:.3f}", f"{r['i']['eps_3d']:.3f}", f"{r['t']['eps_3d']:.3f}", f"{r['b']['eps_lat']:.3f}",
                     f"{r['zb']:.2f}", f"{r['zi']:.2f}"])
tbl_rep = table(["role", "sid", "frames", "split", "ε₃D bound", "ε₃D ionic", "ε₃D total", "ε⊥ bound", "bound peak plane z (Å)", "ionic peak plane z (Å)"], rep_rows)

# ------------------------------------------------------------------ figures
FIGS = {}
cats3 = [("NiN44 ch.\nbound", "NiN44 charged", "b"), ("NiN88 ch.\nbound", "NiN88 charged", "b"), ("neutral\nbound", "NiN44 solvated neutral", "b"),
         ("NiN44 ch.\nionic", "NiN44 charged", "i"), ("NiN88 ch.\nionic", "NiN88 charged", "i"), ("neutral\nionic", "NiN44 solvated neutral", "i"),
         ("NiN44 ch.\ntotal", "NiN44 charged", "t"), ("NiN88 ch.\ntotal", "NiN88 charged", "t"), ("neutral\ntotal", "NiN44 solvated neutral", "t")]
FIGS["eps3d"] = strip_figure("eps_3d", "ε₃D = ∫|ML−DFT| / ∫|DFT|", cats3)
FIGS["epslat"] = strip_figure("eps_lat", "ε⊥ (lateral part)", cats3)
catsn = cats3
FIGS["neterr"] = strip_figure("net_err_e", "net charge error ∫(ML−DFT) dV (e)", catsn, signed=True)
FIGS["perz_b"] = perz_figure("b")
FIGS["perz_i"] = perz_figure("i")
FIGS["pairs"] = pair_strip()
INFO = {}
med, mx = REP.get("median"), REP.get("max")
if med is not None:
    t = f"sid {med} (median-error frame)"
    for ch in ("b", "i", "t"):
        FIGS[f"med_xz_{ch}"], INFO[f"med_xz_{ch}"] = xz_figure(str(med), ch, t)
    FIGS["med_xy_b"], INFO["med_xy_b"] = xy_figure(str(med), "b", "xyb", t)
    FIGS["med_xy_i"], INFO["med_xy_i"] = xy_figure(str(med), "i", "xyi", t)
    FIGS["med_lat_b"], INFO["med_lat_b"] = xy_figure(str(med), "b", "xyb", t, lateral=True)
    vb = VO[f"{med}|dft_b"]
    LEV = float(nice_step(0.3 * float(np.abs(vb).max()), 1))
    FIGS["med_iso_b"], INFO["med_iso_b"] = iso_figure(med, "b", LEV, LEV / 2, t)
    INFO["iso_levels"] = (LEV, LEV / 2)
if mx is not None:
    t = f"sid {mx} (highest-error frame)"
    for ch in ("b", "i", "t"):
        FIGS[f"max_xz_{ch}"], INFO[f"max_xz_{ch}"] = xz_figure(str(mx), ch, t)
if any(p["split"] != SPLIT for p in PAIRS):
    pe = next(p for p in PAIRS if p["split"] != SPLIT); tg = f"d{pe['sid_c']}"
    if f"{tg}|meta" in SL.files:
        t = f"sid {pe['sid_c']} − {pe['sid_n']}"
        for ch in ("b", "i", "t"):
            FIGS[f"pair_xz_{ch}"], INFO[f"pair_xz_{ch}"] = xz_figure(tg, ch, t, delta=True)
        FIGS["pair_xy_i"], INFO["pair_xy_i"] = xy_figure(tg, "i", "xyi", t, delta=True)

EQ = {
    "field": tex(r"$\rho_{ch}^{ML}(\mathbf{r})=\bar\rho_{ch}(z)+\delta_{ch}(\mathbf{r}),\qquad \langle\delta_{ch}\rangle_{xy}(z)=0,\qquad ch\in\{b,\ ion\},\qquad \rho_{tot}=\rho_b+\rho_{ion}$"),
    "eps": tex(r"$\varepsilon_{3D}=\frac{\int|\rho_{ML}-\rho_{DFT}|\,dV}{\int|\rho_{DFT}|\,dV},\qquad \Delta Q=\int(\rho_{ML}-\rho_{DFT})\,dV$"),
    "lat": tex(r"$\rho_{\perp}(x,y,z)=\rho(x,y,z)-\langle\rho\rangle_{xy}(z),\qquad \varepsilon_{\perp}=\frac{\int|\rho_{\perp}^{ML}-\rho_{\perp}^{DFT}|\,dV}{\int|\rho_{\perp}^{DFT}|\,dV}$"),
    "perz": tex(r"$e(z)=\iint|\rho_{ML}(x,y,z)-\rho_{DFT}(x,y,z)|\,dx\,dy\qquad(\mathrm{e/Å})$"),
    "delta": tex(r"$\Delta\rho=\rho^{charged}-\rho^{neutral}\qquad(\mathrm{same\ geometry,\ both\ models\ and\ DFT})$"),
}


def fig(key, caption, note=""):
    if key not in FIGS:
        return ""
    ims = FIGS[key] if isinstance(FIGS[key], list) else [FIGS[key]]
    body = "".join(f'<img src="data:image/png;base64,{b}" alt="{html.escape(caption)}">' for b in ims)
    return (f'<figure><figcaption>{html.escape(caption)}</figcaption>{body}'
            + (f'<p class="note">{html.escape(note)}</p>' if note else "") + "</figure>")


def xz_note(k):
    i = INFO[k]
    return (f"Shown z window {i['z0']:.1f}–{i['z1']:.1f} Å (where the DFT charge is above 1 % of its peak). Colour scale of the DFT and ML "
            f"rows: ±{i['v']:.2e} e/Å³ (99.9th percentile of |ρ| on this slice, both rows); error row ±{i['ve']:.2e} e/Å³. "
            f"Largest |DFT| on the slice {i['dmax']:.2e}, largest |error| {i['emax']:.2e} e/Å³. Atoms within 1 Å of the plane are drawn.")


def xy_note(k, lateral=False):
    i = INFO[k]
    s = (f"Plane z = {i['z']:.2f} Å, shown on one rectangular periodic cell. Plane means: DFT {i['mean_dft']:+.2e}, ML {i['mean_ml']:+.2e} e/Å³"
         + (" (removed in both rows)." if lateral else ".")
         + f" Colour scale DFT/ML ±{i['v']:.2e}, error ±{i['ve']:.2e} e/Å³. Atoms within 1 Å of the plane are drawn.")
    return s


nval = len([r for r in ROWS if r["split"] == SPLIT])
gsum = {g: len([r for r in ROWS if r["group"] == g and r["split"] == SPLIT]) for g in GROUPS}
LEVTXT = (f"+{INFO['iso_levels'][0]:.2e} / −{INFO['iso_levels'][0]:.2e} e/Å³ for DFT and ML, ±{INFO['iso_levels'][1]:.2e} e/Å³ for the error"
          if "iso_levels" in INFO else "")

page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>3-D Solvent Charge</title>
<style>
:root{{--bg:#f9f9f7;--surf:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mute:#898781;--grid:#e1e0d9;--rule:rgba(11,11,11,.10)}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0d0d0d;--surf:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--mute:#898781;--grid:#2c2c2a;--rule:rgba(255,255,255,.10)}}}}
:root[data-theme="dark"]{{--bg:#0d0d0d;--surf:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--mute:#898781;--grid:#2c2c2a;--rule:rgba(255,255,255,.10)}}
html,body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:32px 16px 72px}}
h1{{font-size:1.75rem;margin:0 0 6px;text-wrap:balance}} h2{{font-size:1.2rem;margin:44px 0 12px;padding-top:14px;border-top:1px solid var(--rule)}}
p,li{{max-width:80ch}} .sub{{color:var(--ink2);margin:0 0 18px}} .note{{color:var(--ink2);font-size:.9rem;margin:6px 0 0}}
.eq{{margin:10px 0 12px;overflow-x:auto;padding:2px 0}} .eq svg{{color:var(--ink)}}
.tw{{overflow-x:auto;margin:10px 0 16px}} table{{border-collapse:collapse;width:100%;font-size:.88rem;font-variant-numeric:tabular-nums;background:var(--surf)}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap;vertical-align:top}} th{{color:var(--ink2);font-weight:600}}
td:first-child,th:first-child{{text-align:left;white-space:normal}}
figure{{margin:28px 0}} figcaption{{font-size:.95rem;color:var(--ink2);margin:0 0 6px}}
figure img{{display:block;width:100%;height:auto;border:1px solid var(--grid);border-radius:4px;background:#fcfcfb;margin:0 0 8px}}
</style></head><body><main>
<h1>3-D solvent charge of prod500_vsolv_fix against DFT</h1>
<p class="sub">Model {html.escape(M['model'])} (final weights of the 500-epoch run). {nval} {SPLIT}-split frames with DFT solvent labels
({gsum['NiN44 charged']} charged NiN44, {gsum['NiN88 charged']} charged NiN88, {gsum['NiN44 solvated neutral']} solvated neutral NiN44), every point of the
500 × 168 × 168 DFT grid, bound charge ρ_b and ionic charge ρ_ion separately and their sum. The vacuum neutral frames carry no solvent and are not included.</p>

<h2>What is compared</h2>
<p>The model's 3-D solvent charge is the field the training loss scores: the 1-D profile of the final solvent solve, broadcast over each z plane,
plus a lateral residual whose mean over every plane is zero by construction. It is rebuilt here on the DFT grid with the same interpolation
the loss uses. The DFT reference is the VASPsol charge (RHOB, RHOION) in the same sign convention, e/Å³.</p>
{EQ['field']}
<p>Per frame and channel. The denominator uses the absolute value because the solvent charge changes sign; the absolute numerator (e) is reported next to it.</p>
{EQ['eps']}
<p>Lateral part: each z plane minus its own mean, taken separately for ML and DFT. This separates the charge distribution within a plane from the
plane-averaged profile, which can agree even when the in-plane distribution is wrong.</p>
{EQ['lat']}
<p>Where the error sits along z: absolute value first, then the plane integral.</p>
{EQ['perz']}
<p>Charging response on twin pairs (charged frame and its solvated neutral twin, same geometry):</p>
{EQ['delta']}

<h2>1 · {SPLIT.capitalize()}-set statistics</h2>
{tbl_stats}
<p class="note">ε plane-avg is the same ratio for the plane-averaged part alone. On the neutral frames the model's ionic channel is zero by construction (its residual is gated by the total charge and the 1-D ionic profile carries no net charge), while the DFT ionic charge has separated positive and negative parts; the ratio there is therefore close to 1. Net error of the ionic channel on charged frames tests whether the ionic layer carries the right total.</p>
{fig('eps3d', f'ε₃D per frame, {SPLIT} split (black bar = median)')}
{fig('epslat', f'ε⊥ per frame, lateral part only, {SPLIT} split')}
{fig('neterr', f'Net charge error per frame, {SPLIT} split')}
{fig('perz_b', 'Bound charge: where the error sits along z', 'Solid: plane-integrated |ML − DFT|; dashed: plane-integrated |DFT| of the same frames, for scale. Both averaged over the frames of each group.')}
{fig('perz_i', 'Ionic charge: where the error sits along z', 'Solid: plane-integrated |ML − DFT|; dashed: plane-integrated |DFT|.')}

<h2>2 · Representative frames</h2>
{tbl_rep}
<p class="note">Median-error frame: the charged {SPLIT} frame whose ε₃D of the total charge is closest to the median of the charged {SPLIT} frames.
Highest-error frame: the largest ε₃D of the total charge among them. The xz plane passes through the Ni atom (fixed lattice row along b);
the xy planes are the DFT bound-charge and ionic-charge peak planes of that frame.</p>
{fig('med_xz_b', 'Median-error frame: bound charge, xz slice', xz_note('med_xz_b') if 'med_xz_b' in INFO else '')}
{fig('med_xz_i', 'Median-error frame: ionic charge, xz slice', xz_note('med_xz_i') if 'med_xz_i' in INFO else '')}
{fig('med_xz_t', 'Median-error frame: total solvent charge, xz slice', xz_note('med_xz_t') if 'med_xz_t' in INFO else '')}
{fig('med_xy_b', 'Median-error frame: bound charge, xy slice at the bound-charge peak plane', xy_note('med_xy_b') if 'med_xy_b' in INFO else '')}
{fig('med_xy_i', 'Median-error frame: ionic charge, xy slice at the ionic-charge peak plane', xy_note('med_xy_i') if 'med_xy_i' in INFO else '')}
{fig('med_lat_b', 'Median-error frame: lateral part ρ⊥ of the bound charge at the interface plane', xy_note('med_lat_b', True) if 'med_lat_b' in INFO else '')}
{fig('med_iso_b', 'Median-error frame: bound-charge isosurfaces', ('Fixed levels, the same for DFT and ML: ' + LEVTXT + f'. Same viewing angle in all three rows; z window {INFO["med_iso_b"]["z0"]:.1f}–{INFO["med_iso_b"]["z1"]:.1f} Å; grey dots are the oxygen atoms of the explicit water in that window; grid stride 2. An isosurface shows the shape at one threshold only; the numbers above are the accuracy measure.') if LEVTXT else '')}
{fig('max_xz_b', 'Highest-error frame: bound charge, xz slice', xz_note('max_xz_b') if 'max_xz_b' in INFO else '')}
{fig('max_xz_i', 'Highest-error frame: ionic charge, xz slice', xz_note('max_xz_i') if 'max_xz_i' in INFO else '')}
{fig('max_xz_t', 'Highest-error frame: total solvent charge, xz slice', xz_note('max_xz_t') if 'max_xz_t' in INFO else '')}

<h2>3 · Charging response: charged minus neutral</h2>
{tbl_pairs}
{fig('pairs', 'ε₃D of the charging response Δρ per twin pair')}
{fig('pair_xz_b', 'Pair-page structure: bound-charge response Δρ_b, xz slice', xz_note('pair_xz_b') if 'pair_xz_b' in INFO else '')}
{fig('pair_xz_i', 'Pair-page structure: ionic-charge response Δρ_ion, xz slice', xz_note('pair_xz_i') if 'pair_xz_i' in INFO else '')}
{fig('pair_xz_t', 'Pair-page structure: total solvent-charge response, xz slice', xz_note('pair_xz_t') if 'pair_xz_t' in INFO else '')}
{fig('pair_xy_i', 'Pair-page structure: ionic-charge response, xy slice at the ionic-charge peak plane', xy_note('pair_xy_i') if 'pair_xy_i' in INFO else '')}
</main></body></html>
"""
OUTP.write_text(page)
print(f"wrote {OUTP} ({OUTP.stat().st_size / 1e6:.2f} MB), {len(FIGS)} figures")
for k, v in INFO.items():
    print(k, v)
