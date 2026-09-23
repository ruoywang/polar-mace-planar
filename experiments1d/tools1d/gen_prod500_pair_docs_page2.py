"""Docs page for the prod500_w1000_ref model: training curves + twin-pair Fermi scatter + pair report (sid 122 / 722).
Adapted 2026-09-21 from gen_mix400_pair_docs_page.py for the mix800 data set (solvated neutral twins, sid k+600).

Section 0: training record (electrode potential / Fermi RMSE vs epoch, mix400
against the planar 4-grid baseline, log scale). Section 0b: per-structure
charged-vs-neutral Fermi scatter over the 200 NiN44 twin pairs (model and DFT).
The rest is the pair-report content from the structure_pair npz.

Usage: python gen_prod500_pair_docs_page2.py <pair_npz> <train_txt> <energy_compare_train_txt|none> <fermi_pairs_npz> <out_html> [MODEL_LABEL] [MODEL_DESC] [CPMACE_TXT|none] [ECOMP_LABEL] [ECOMP_DESC] [SPLIT_NOTE]
Section 0 (2026-09-23): Fermi-level curve = model vs FermiMACE (CPMACE_TXT, rmse_p); energy curve = model vs the
earlier 500-epoch model (energy_compare_train_txt, rmse_e_per_atom); no Phi1D curve.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from pathlib import Path

npz_path, mix_txt, grid4_txt, pairs_npz, out_path = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
MODEL_LABEL = sys.argv[6] if len(sys.argv) > 6 else "train_all"
MODEL_DESC = sys.argv[7] if len(sys.argv) > 7 else "600-structure training (400 charged + 200 neutral twins)"
CPMACE_TXT = (sys.argv[8] if len(sys.argv) > 8 and sys.argv[8].lower() not in ("none", "-") else None)
COMPARE_LABEL = sys.argv[9] if len(sys.argv) > 9 else "s3d_prod500 (before the local-electron energy)"
COMPARE_DESC = sys.argv[10] if len(sys.argv) > 10 else "same package, energy weight 1, no local-electron energy"
SPLIT_NOTE = sys.argv[11] if len(sys.argv) > 11 else "both frames are in the training split"
d = np.load(npz_path, allow_pickle=True)

sid = int(d["sid"]); sid_n = int(d["sid_n"])
symbols = d["symbols"].tolist()
z_atoms = d["z_atoms"].astype(float)
charges = d["charges"].astype(float)
charges_n = d["charges_n"].astype(float)
lz = float(d["lz"])
total_q = float(d["total_charge"])

palette = {
    "s1": ("#2a78d6", "#3987e5"),
    "s2": ("#008300", "#008300"),
    "s3": ("#e87ba4", "#d55181"),
    "s4": ("#eda100", "#c98500"),
    "s5": ("#1baf7a", "#199e70"),
    "s6": ("#eb6834", "#d95926"),
}
ELEM_SLOTS = {"O": "s1", "H": "s2", "C": "s3", "Ni": "s4", "N": "s5"}

W, H_MAIN, H_SUB, ML, MR, MT, MB = 860, 300, 130, 62, 16, 14, 40


def scale(v, lo, hi, a, b):
    return a + (v - lo) / (hi - lo) * (b - a)


def path_of(xs, ys, xlo, xhi, ylo, yhi, w, h):
    pts = []
    for x, y in zip(xs, ys):
        px = scale(x, xlo, xhi, ML, w - MR)
        py = scale(y, ylo, yhi, h - MB, MT)
        pts.append(f"{px:.1f},{py:.1f}")
    return "M" + " L".join(pts)


def ticks(lo, hi, n=6):
    """Round tick values inside [lo, hi] (steps 1, 2, 2.5, 5 x 10^k), never the raw data bounds."""
    span = hi - lo
    if span <= 0:
        return np.array([lo])
    mag = 10.0 ** np.floor(np.log10(span / n))
    step = mag
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = m * mag
        if span / step <= n:
            break
    first = np.ceil(lo / step - 1e-9) * step
    t = np.round(np.arange(first, hi + 1e-9 * step, step), 10) + 0.0   # + 0.0 turns -0.0 into 0.0
    return t


def axis_svg(xlo, xhi, ylo, yhi, w, h, xlab, ylab, xfmt="{:g}", yfmt="{:g}", ylog=False):
    out = []
    for tv in ticks(xlo, xhi):
        px = scale(tv, xlo, xhi, ML, w - MR)
        out.append(f'<line x1="{px:.1f}" y1="{MT}" x2="{px:.1f}" y2="{h-MB}" class="grid"/>')
        out.append(f'<text x="{px:.1f}" y="{h-MB+16}" class="tick" text-anchor="middle">{xfmt.format(tv)}</text>')
    if ylog:
        lo_d, hi_d = int(np.floor(ylo)), int(np.ceil(yhi))
        for dec in range(lo_d, hi_d + 1):
            if not (ylo <= dec <= yhi):
                continue
            py = scale(dec, ylo, yhi, h - MB, MT)
            out.append(f'<line x1="{ML}" y1="{py:.1f}" x2="{w-MR}" y2="{py:.1f}" class="grid"/>')
            v = 10.0 ** dec
            lab = f"{v:g}"
            out.append(f'<text x="{ML-6}" y="{py+4:.1f}" class="tick" text-anchor="end">{lab}</text>')
    else:
        for tv in ticks(ylo, yhi, 6):
            py = scale(tv, ylo, yhi, h - MB, MT)
            out.append(f'<line x1="{ML}" y1="{py:.1f}" x2="{w-MR}" y2="{py:.1f}" class="grid"/>')
            out.append(f'<text x="{ML-6}" y="{py+4:.1f}" class="tick" text-anchor="end">{yfmt.format(tv)}</text>')
    out.append(f'<text x="{(ML+w-MR)/2}" y="{h-6}" class="axis" text-anchor="middle">{xlab}</text>')
    out.append(f'<text x="14" y="{(MT+h-MB)/2}" class="axis" text-anchor="middle" transform="rotate(-90 14 {(MT+h-MB)/2})">{ylab}</text>')
    return "".join(out)


charts_js = {}


def line_chart(cid, series, xlab, ylab, h=H_MAIN, ypad=0.06, yclip=None, legend_xy=None,
               unit=None, ylog=False):
    """series: list of (name, xs, ys, slot, dash)"""
    xlo = min(min(s[1]) for s in series); xhi = max(max(s[1]) for s in series)
    raw = [np.asarray(s[2], dtype=float) for s in series]
    if ylog:
        raw = [np.log10(np.clip(y, 1e-12, None)) for y in raw]
    ys_all = np.concatenate(raw)
    if yclip:
        ys_all = ys_all[(ys_all >= yclip[0]) & (ys_all <= yclip[1])]
    ylo, yhi = float(ys_all.min()), float(ys_all.max())
    pad = (yhi - ylo) * ypad or 1e-6
    ylo, yhi = ylo - pad, yhi + pad
    if yclip:
        ylo, yhi = float(yclip[0]), float(yclip[1])   # the requested window exactly (axis from 0 when asked)
    parts = [f'<svg id="{cid}" viewBox="0 0 {W} {h}" data-xlo="{xlo}" data-xhi="{xhi}" data-ylo="{ylo}" data-yhi="{yhi}" data-h="{h}">']
    parts.append(axis_svg(xlo, xhi, ylo, yhi, W, h, xlab, ylab, ylog=ylog))
    for (name, xs, ys, slot, dash), yv in zip(series, raw):
        dash_attr = ' stroke-dasharray="6 4"' if dash else ""
        xs_a, yv_a = np.asarray(xs, dtype=float), np.asarray(yv, dtype=float)
        if yclip:   # points outside the window are left out instead of being flattened onto the frame
            keep = (yv_a >= ylo) & (yv_a <= yhi); xs_a, yv_a = xs_a[keep], yv_a[keep]
        parts.append(f'<path d="{path_of(xs_a, yv_a, xlo, xhi, ylo, yhi, W, h)}" class="ln {slot}"{dash_attr} fill="none"/>')
    if legend_xy:
        lx, ly = legend_xy
        for i, (name, *_rest) in enumerate(series):
            slot = series[i][3]
            dd = ' stroke-dasharray="6 4"' if series[i][4] else ""
            parts.append(f'<line x1="{lx}" y1="{ly+i*18}" x2="{lx+22}" y2="{ly+i*18}" class="ln {slot}"{dd}/>')
            parts.append(f'<text x="{lx+28}" y="{ly+i*18+4}" class="lg">{name}</text>')
    parts.append(f'<line class="xh" x1="0" y1="{MT}" x2="0" y2="{h-MB}" style="opacity:0"/>')
    parts.append("</svg>")
    if unit:
        xs0 = np.asarray(series[0][1], dtype=float)
        charts_js[cid] = {"x": xs0.round(3).tolist(),
                          "series": [{"n": s[0],
                                      "y": np.round(np.interp(xs0, np.asarray(s[1], float),
                                                              np.asarray(s[2], float)), 6).tolist()}
                                     for s in series],
                          "unit": unit}
    return "".join(parts)


# ==== section 0: training curves ============================================
def load_eval(path):
    seen = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("mode") != "eval" or r.get("epoch") is None:
            continue
        seen[int(r["epoch"])] = (r.get("rmse_potential", float("nan")), r.get("rmse_fermi_level", r.get("rmse_p", float("nan"))),
                                 r.get("rmse_e_per_atom", float("nan")), r.get("rmse_f", float("nan")))
    ep = np.array(sorted(seen)); a = np.array([seen[e] for e in ep], dtype=float)
    return ep, a[:, 0], a[:, 1], a[:, 2] * 1e3, a[:, 3] * 1e3   # E in meV/atom, F in meV/A


ep_m, pot_m, fer_m, e_m, f_m = load_eval(mix_txt)
HAS_ECOMP = str(grid4_txt).lower() not in ("none", "-")
if HAS_ECOMP:
    ep_p, pot_p, fer_p, e_p, f_p = load_eval(grid4_txt)
HAS_GRID4 = False
fer_series = [(MODEL_LABEL, ep_m, fer_m, "s1", False)]
if CPMACE_TXT:
    ep_c, pot_c, fer_c, e_c, f_c = load_eval(CPMACE_TXT)
    fer_series.append(("FermiMACE (11-cpmace_800)", ep_c, fer_c, "s2", True))
curve_fer = line_chart("tfer", fer_series, "epoch", "validation RMSE Fermi level (eV)", legend_xy=(ML + 200, MT + 12), unit="eV", ylog=False, yclip=(0.0, 0.6))
e_series = [(MODEL_LABEL, ep_m, e_m, "s1", False)]
if HAS_ECOMP:
    e_series.append((COMPARE_LABEL, ep_p, e_p, "s6", True))
curve_E = line_chart("tene", e_series, "epoch", "validation RMSE energy (meV/atom)", legend_xy=(ML + 200, MT + 12), unit="meV/atom", ylog=False, yclip=(0.0, 25.0))

train_table = f"""
<table class="kv">
<tr><th>run (val split = the same 80 frames)</th><th>epochs</th><th>E (meV/atom)</th><th>Fermi (eV)</th><th>F (meV/Å)</th></tr>
<tr><td>{MODEL_LABEL} — {MODEL_DESC}</td><td>{int(ep_m[-1])+1}</td><td>{e_m[-1]:.2f}</td><td>{fer_m[-1]:.3f}</td><td>{f_m[-1]:.1f}</td></tr>
{(f'<tr><td>{COMPARE_LABEL} — {COMPARE_DESC}</td><td>{int(ep_p[-1])+1}</td><td>{e_p[-1]:.2f}</td><td>{fer_p[-1]:.3f}</td><td>{f_p[-1]:.1f}</td></tr>') if HAS_ECOMP else ''}
{(f'<tr><td>FermiMACE (11-cpmace_800) — same package, plain MACE with a Fermi-level head</td><td>{int(ep_c[-1])+1}</td><td>{e_c[-1]:.2f}</td><td>{fer_c[-1]:.3f}</td><td>{f_c[-1]:.1f}</td></tr>') if CPMACE_TXT else ''}
</table>
"""
NOTE0 = ('<p class="note">Linear scale, validation RMSE per epoch (the first warm-up epochs lie above the plotted range); all three runs validate on the same 80 frames of the mix800 package. '
         'The energy comparison run is the last 500-epoch production before the native local-electron energy head and the energy weight 1000 were introduced (s3d_prod500: epochs 0-475 in the 30 h production job, epochs 457-499 completed from the epoch-457 checkpoint on the dev queue; the curve joins the two logs at epoch 457).</p>')

# ---- section 0b: per-structure Fermi, charged and neutral (DFT) -----------
fp = np.load(pairs_npz)
sid_arr, dft_c, ml_c, dft_n, ml_n = (fp["sid"].astype(int), fp["dft_c"], fp["ml_c"],
                                     fp["dft_n"], fp["ml_n"])
sx_lo, sx_hi = float(sid_arr.min()) - 3, float(sid_arr.max()) + 3
sy_lo = min(dft_c.min(), dft_n.min()); sy_hi = max(dft_c.max(), dft_n.max())
py_ = (sy_hi - sy_lo) * 0.08 or 0.1
sy_lo, sy_hi = sy_lo - py_, sy_hi + py_
H_SC = 440
partsS = [f'<svg viewBox="0 0 {W} {H_SC}">']
partsS.append(axis_svg(sx_lo, sx_hi, sy_lo, sy_hi, W, H_SC,
                       "structure id", "Fermi level (eV, DFT)",
                       xfmt="{:.0f}", yfmt="{:.1f}"))
for (name, ys, slot) in [("charged", dft_c, "s6"), ("neutral twin", dft_n, "s1")]:
    for s_, yy in zip(sid_arr, ys):
        px = scale(float(s_), sx_lo, sx_hi, ML, W - MR)
        py = scale(yy, sy_lo, sy_hi, H_SC - MB, MT)
        partsS.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.8" class="pt {slot}" fill-opacity="0.8">'
                      f'<title>sid {s_} · {name}  {yy:+.3f} eV</title></circle>')
lx, ly = ML + 16, MT + 12
for i, (name, slot) in enumerate([("charged (q ≈ −1 e)", "s6"), ("neutral twin (q = 0)", "s1")]):
    partsS.append(f'<circle cx="{lx}" cy="{ly+i*18}" r="4" class="pt {slot}"/>')
    partsS.append(f'<text x="{lx+12}" y="{ly+i*18+4}" class="lg">{name}</text>')
partsS.append("</svg>")
chartS = "".join(partsS)
rms_fc = float(np.sqrt(np.mean((ml_c - dft_c) ** 2)))
rms_fn = float(np.sqrt(np.mean((ml_n - dft_n) ** 2)))
shift_dft = float(np.mean(dft_c - dft_n)); shift_ml = float(np.mean(ml_c - ml_n))

# ==== pair-report sections (same construction as the review page) ===========
def ls_factor(z_m, v_m, z_r, v_r):
    m_on_ref = np.interp(z_r, z_m, v_m)
    return float(np.dot(m_on_ref, v_r) / np.dot(m_on_ref, m_on_ref))


z_grid = d["z_grid"].astype(float)
nbar_c = d["nbar_model"].astype(float)
nbar_n = d["nbar_model_n"].astype(float)
z_dft_c = d["z_dft"].astype(float); nbar_dft_c = d["nbar_dft"].astype(float)
z_dft_n = d["z_dft_n"].astype(float); nbar_dft_n = d["nbar_dft_n"].astype(float)
ls_c = ls_factor(z_grid, nbar_c, z_dft_c, nbar_dft_c)
ls_n = ls_factor(z_grid, nbar_n, z_dft_n, nbar_dft_n)

mask_c = (z_grid >= z_dft_c.min()) & (z_grid <= z_dft_c.max())
mask_n = (z_grid >= z_dft_n.min()) & (z_grid <= z_dft_n.max())

q_lo, q_hi = charges.min(), charges.max()
pad = (q_hi - q_lo) * 0.10
q_lo, q_hi = q_lo - pad, q_hi + pad
z_lo, z_hi = z_atoms.min() - 1, z_atoms.max() + 1
partsA = [f'<svg viewBox="0 0 {W} {H_MAIN}">']
partsA.append(axis_svg(z_lo, z_hi, q_lo, q_hi, W, H_MAIN, "z (Å)", "predicted atomic charge q (e)", yfmt="{:+.2f}"))
el_stats = {}
for el in ["O", "H", "C", "Ni", "N"]:
    m = np.array(symbols) == el
    if not m.any():
        continue
    el_stats[el] = (int(m.sum()), float(charges[m].mean()), float(charges_n[m].mean()),
                    float((charges - charges_n)[m].sum()))
    slot = ELEM_SLOTS[el]
    for zz, qq in zip(z_atoms[m], charges[m]):
        px = scale(zz, z_lo, z_hi, ML, W - MR)
        py = scale(qq, q_lo, q_hi, H_MAIN - MB, MT)
        partsA.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" class="pt {slot}"><title>{el}  z={zz:.2f} Å  q={qq:+.4f} e</title></circle>')
    zz_m = float(np.median(z_atoms[m])); qq_m = float(np.median(charges[m]))
    px = scale(zz_m, z_lo, z_hi, ML, W - MR)
    py = scale(qq_m, q_lo, q_hi, H_MAIN - MB, MT)
    dy = -10 if el not in ("O",) else 16
    partsA.append(f'<text x="{px:.1f}" y="{py+dy:.1f}" class="ellab {slot}" text-anchor="middle">{el}</text>')
partsA.append("</svg>")
chartA = "".join(partsA)

dq = charges - charges_n
dq_lo, dq_hi = dq.min(), dq.max()
pad = (dq_hi - dq_lo) * 0.10 or 1e-4
dq_lo, dq_hi = dq_lo - pad, dq_hi + pad
partsAd = [f'<svg viewBox="0 0 {W} {H_SUB+60}">']
partsAd.append(axis_svg(z_lo, z_hi, dq_lo, dq_hi, W, H_SUB + 60, "z (Å)", "Δq = q(charged) − q(neutral) (e)", yfmt="{:+.3f}"))
for el in ["O", "H", "C", "Ni", "N"]:
    m = np.array(symbols) == el
    if not m.any():
        continue
    slot = ELEM_SLOTS[el]
    for zz, qq in zip(z_atoms[m], dq[m]):
        px = scale(zz, z_lo, z_hi, ML, W - MR)
        py = scale(qq, dq_lo, dq_hi, H_SUB + 60 - MB, MT)
        partsAd.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" class="pt {slot}"><title>{el}  z={zz:.2f} Å  Δq={qq:+.4f} e</title></circle>')
partsAd.append("</svg>")
chartAd = "".join(partsAd)

z_phi = d["z_phi"].astype(float)
chartB = line_chart("phi", [
    ("DFT reference", z_phi, d["phi_ref_cmp"].astype(float), "s6", False),
    ("model prediction", z_phi, d["phi_pred_cmp"].astype(float), "s1", True),
], "z (Å)", "φ̄(z) − φ̄(align) (eV)", legend_xy=(ML + 16, MT + 12), unit="eV")
chartB_res = line_chart("phires", [
    ("residual model−DFT", z_phi, d["phi_residual"].astype(float), "s1", False),
], "z (Å)", "Δφ (eV)", h=H_SUB, unit="eV")

z_phi_n = d["z_phi_n"].astype(float)
chartBn = line_chart("phin", [
    ("DFT reference", z_phi_n, d["phi_ref_cmp_n"].astype(float), "s6", False),
    ("model prediction", z_phi_n, d["phi_pred_cmp_n"].astype(float), "s1", True),
], "z (Å)", "φ̄(z) − φ̄(align) (eV)", legend_xy=(ML + 16, MT + 12), unit="eV")
chartBn_res = line_chart("phinres", [
    ("residual model−DFT", z_phi_n, d["phi_residual_n"].astype(float), "s1", False),
], "z (Å)", "Δφ (eV)", h=H_SUB, unit="eV")

mzc, mvc = z_grid[mask_c], (nbar_c * ls_c)[mask_c]
refc = np.interp(mzc, z_dft_c, nbar_dft_c)
chartC = line_chart("den", [
    ("DFT reference", z_dft_c, nbar_dft_c, "s6", False),
    ("model (common LS scale)", mzc, mvc, "s1", True),
], "z (Å)", "n̄(z) (e/Å³)", legend_xy=(ML + 16, MT + 12), unit="e/Å³")
chartC_res = line_chart("denres", [
    ("difference model−DFT", mzc, mvc - refc, "s1", False),
], "z (Å)", "Δn̄ (e/Å³)", h=H_SUB, unit="e/Å³")

mzn, mvn = z_grid[mask_n], (nbar_n * ls_c)[mask_n]
refn = np.interp(mzn, z_dft_n, nbar_dft_n)
chartCn = line_chart("denn", [
    ("DFT reference", z_dft_n, nbar_dft_n, "s6", False),
    ("model (common LS scale)", mzn, mvn, "s1", True),
], "z (Å)", "n̄(z) (e/Å³)", legend_xy=(ML + 16, MT + 12), unit="e/Å³")
chartCn_res = line_chart("dennres", [
    ("difference model−DFT", mzn, mvn - refn, "s1", False),
], "z (Å)", "Δn̄ (e/Å³)", h=H_SUB, unit="e/Å³")

z_dd = d["z_dft_diff"].astype(float)
dn_dft = d["dn_dft"].astype(float)
mask_d = (z_grid >= z_dd.min()) & (z_grid <= z_dd.max())
mzd = z_grid[mask_d]
dn_m = (d["dn_model"].astype(float) * ls_c)[mask_d]
dn_ref_on_m = np.interp(mzd, z_dd, dn_dft)
# cell area from the pair xyz next to the npz (plane-averaged density x area x dz = electrons)
try:
    from ase.io import read as _read
    _cell = _read(str(Path(npz_path).parent / f"pair_{sid}.xyz"), 0).cell[:]
    A_CELL = float(abs(np.cross(_cell[0], _cell[1])[2]))
except Exception:
    A_CELL = 1.0
q_model = float(np.trapz(d["dn_model"].astype(float) * ls_c, z_grid)) * A_CELL
q_dft = float(np.trapz(dn_dft, z_dd)) * A_CELL
chartDD = line_chart("dd", [
    ("DFT: charged − neutral", z_dd, dn_dft, "s6", False),
    ("model: charged − neutral", mzd, dn_m, "s1", True),
], "z (Å)", "Δn̄ = n̄(charged) − n̄(neutral) (e/Å³)", legend_xy=(ML + 16, MT + 12), unit="e/Å³")
chartDD_res = line_chart("ddres", [
    ("difference model−DFT", mzd, dn_m - dn_ref_on_m, "s1", False),
], "z (Å)", "model−DFT (e/Å³)", h=H_SUB, unit="e/Å³")
rms_dd = float(np.sqrt(np.mean((dn_m - dn_ref_on_m) ** 2)))

ref = np.load(Path(npz_path).parent / f"dft_solvent_ref_sid{sid}.npz")
z_ref_sol = ref["z"].astype(float)
ion_dft = -ref["ion_z"].astype(float)
rb_dft = -ref["rb_z"].astype(float)
z_s = d["z_solve"].astype(float)


def _smear_periodic(v, z, sigma_A=0.15):
    """Match the training convention (solvent_rhob_1d_sigma = 0.15 A, model
    side only; the DFT reference stays raw)."""
    n = len(v)
    lz_p = float(z[-1] - z[0]) * n / (n - 1)
    k = np.fft.rfftfreq(n, d=lz_p / n) * 2.0 * np.pi
    return np.fft.irfft(np.fft.rfft(v) * np.exp(-0.5 * (k * sigma_A) ** 2), n=n)


rho_bound_disp = _smear_periodic(d["rho_bound"].astype(float), z_s)
chartD = line_chart("sol", [
    ("ionic charge ρ_ion", z_s, d["rho_ion"].astype(float), "s1", False),
    ("bound charge ρ_bound (smeared 0.15 Å)", z_s, rho_bound_disp, "s4", False),
], "z (Å)", "ρ(z) (e/Å³)", legend_xy=(ML + 16, MT + 12), unit="e/Å³")
chartD_ion = line_chart("solion", [
    ("DFT (VASPsol RHOION)", z_ref_sol, ion_dft, "s6", False),
    ("model 1-D PB", z_s, d["rho_ion"].astype(float), "s1", True),
], "z (Å)", "ρ_ion(z) (e/Å³)", legend_xy=(ML + 16, MT + 12), unit="e/Å³")
chartD_rb = line_chart("solrb", [
    ("DFT (VASPsol RHOB, raw)", z_ref_sol, rb_dft, "s6", False),
    ("model 1-D PB (smeared 0.15 Å, training convention)", z_s, rho_bound_disp, "s1", True),
], "z (Å)", "ρ_bound(z) (e/Å³)", legend_xy=(ML + 16, MT + 12), unit="e/Å³")
chartE = line_chart("poff", [
    ("prior P_off (screened vacuum)", z_s, d["prior"].astype(float), "s6", False),
    ("head correction ΔP", z_s, d["delta_p"].astype(float), "s1", False),
], "z (Å)", "P_off (e/Å²)", legend_xy=(ML + 16, MT + 12), unit="e/Å²")

el_rows = "".join(
    f"<tr><td><span class='chip {ELEM_SLOTS[el]}'></span>{el}</td><td>{n}</td>"
    f"<td>{muc:+.4f}</td><td>{mun:+.4f}</td><td>{sq:+.4f}</td></tr>"
    for el, (n, muc, mun, sq) in el_stats.items())

scalars = f"""
<table class="kv">
<tr><th></th><th colspan=3>charged (sid {sid}, q = {total_q:+.3f} e)</th><th colspan=3>neutral (sid {sid_n}, q = 0)</th></tr>
<tr><th></th><th>model</th><th>DFT</th><th>diff</th><th>model</th><th>DFT</th><th>diff</th></tr>
<tr><td>electrode potential (eV)</td>
<td>{float(d['pot_pred']):+.4f}</td><td>{float(d['pot_ref']):+.4f}</td><td>{float(d['pot_pred'])-float(d['pot_ref']):+.4f}</td>
<td>{float(d['pot_pred_n']):+.4f}</td><td>{float(d['pot_ref_n']):+.4f}</td><td>{float(d['pot_pred_n'])-float(d['pot_ref_n']):+.4f}</td></tr>
<tr><td>Fermi level (eV)</td>
<td>{float(d['fermi_pred']):+.4f}</td><td>{float(d['fermi_ref']):+.4f}</td><td>{float(d['fermi_pred'])-float(d['fermi_ref']):+.4f}</td>
<td>{float(d['fermi_pred_n']):+.4f}</td><td>{float(d['fermi_ref_n']):+.4f}</td><td>{float(d['fermi_pred_n'])-float(d['fermi_ref_n']):+.4f}</td></tr>
</table>
<table class="kv">
<tr><th>solvent quantity (charged state)</th><th>value</th></tr>
<tr><td>ionic layer charge q_ion (e)</td><td>{float(d['q_ion']):+.4f} (solute {total_q:+.4f})</td></tr>
<tr><td>ionic layer center (Å)</td><td>{float(d['layer_mean']):.2f}</td></tr>
<tr><td>bound-charge dipole μ_bound (e·Å)</td><td>{float(d['mu_bound']):+.2f}</td></tr>
<tr><td>Σ atomic charges: charged / neutral (e)</td><td>{charges.sum():+.4f} / {charges_n.sum():+.4f}</td></tr>
<tr><td>neutral twin: q_ion / μ_bound</td><td>{float(d['q_ion_n']):+.4f} e / {float(d['mu_bound_n']):+.2f} e·Å</td></tr>
<tr><td>total energy error charged / neutral (meV/atom)</td><td>{1e3*(float(d['energy_pred'])-float(d['energy_ref']))/len(symbols):+.2f} / {1e3*(float(d['energy_pred_n'])-float(d['energy_ref_n']))/len(symbols):+.2f}</td></tr>
<tr><td>charging energy E(charged) − E(neutral): model / DFT (eV)</td><td>{float(d['energy_pred'])-float(d['energy_pred_n']):+.4f} / {float(d['energy_ref'])-float(d['energy_ref_n']):+.4f}</td></tr>
</table>
"""

light_css = "".join(f"--{k}:{v[0]};" for k, v in palette.items())
dark_css = "".join(f"--{k}:{v[1]};" for k, v in palette.items())

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pair {sid} {MODEL_LABEL}</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ margin: 0; }}
.pg {{ max-width: 920px; margin: 0 auto; padding: 24px 16px 60px;
  font: 15px/1.65 -apple-system, "Segoe UI", "Noto Sans SC", sans-serif;
  --ink:#171512; --ink2:#5c574f; --grid:#e6e2da; --card:#faf9f6; {light_css} }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) .pg {{
  --ink:#ece9e2; --ink2:#a8a294; --grid:#37342e; --card:#232019; {dark_css} }} }}
:root[data-theme=dark] .pg {{ --ink:#ece9e2; --ink2:#a8a294; --grid:#37342e; --card:#232019; {dark_css} }}
.pg {{ color: var(--ink); }}
h1 {{ font-size: 24px; margin: 0 0 4px; }} h2 {{ font-size: 17px; margin: 34px 0 6px; }}
.sub {{ color: var(--ink2); margin: 0 0 20px; }}
.note {{ color: var(--ink2); font-size: 13.5px; margin: 4px 0 0; }}
.card {{ background: var(--card); border: 1px solid var(--grid); border-radius: 10px;
  padding: 14px 14px 6px; margin: 10px 0; }}
svg {{ width: 100%; height: auto; display: block; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.tick {{ font-size: 11px; fill: var(--ink2); }}
.axis {{ font-size: 12.5px; fill: var(--ink2); }}
.lg, .ellab {{ font-size: 12.5px; }} .lg {{ fill: var(--ink); }}
.ln.s1 {{ stroke: var(--s1); stroke-width: 2; }} .ln.s2 {{ stroke: var(--s2); stroke-width: 2; }}
.ln.s3 {{ stroke: var(--s3); stroke-width: 2; }} .ln.s4 {{ stroke: var(--s4); stroke-width: 2; }}
.ln.s6 {{ stroke: var(--s6); stroke-width: 2; }} .chip.s6 {{ background: var(--s6); }}
.pt.s1 {{ fill: var(--s1); }} .pt.s2 {{ fill: var(--s2); }} .pt.s3 {{ fill: var(--s3); }}
.pt.s4 {{ fill: var(--s4); }} .pt.s5 {{ fill: var(--s5); }}
.ellab.s1 {{ fill: var(--s1); }} .ellab.s2 {{ fill: var(--s2); }} .ellab.s3 {{ fill: var(--s3); }}
.ellab.s4 {{ fill: var(--s4); }} .ellab.s5 {{ fill: var(--s5); }}
.chip {{ display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:7px;
  background: var(--ink2); }}
.chip.s1 {{ background: var(--s1); }} .chip.s2 {{ background: var(--s2); }}
.chip.s3 {{ background: var(--s3); }} .chip.s4 {{ background: var(--s4); }} .chip.s5 {{ background: var(--s5); }}
table.kv {{ border-collapse: collapse; margin: 10px 24px 10px 0; display: inline-table;
  font-variant-numeric: tabular-nums; }}
table.kv td, table.kv th {{ border: 1px solid var(--grid); padding: 5px 12px; font-size: 13.5px; text-align: right; }}
table.kv th {{ color: var(--ink2); font-weight: 600; }}
table.kv td:first-child, table.kv th:first-child {{ text-align: left; }}
.xh {{ stroke: var(--ink2); stroke-width: 1; }}
#tip {{ position: fixed; pointer-events: none; background: var(--card); border: 1px solid var(--grid);
  border-radius: 6px; padding: 5px 9px; font-size: 12.5px; opacity: 0; z-index: 9;
  font-variant-numeric: tabular-nums; box-shadow: 0 2px 8px rgba(0,0,0,.12); }}
</style></head><body>
<div class="pg">
<h1>pb1d {MODEL_LABEL} — training record + pair report (sid {sid} charged / sid {sid_n} neutral)</h1>
<p class="sub">{MODEL_DESC} · model {MODEL_LABEL} · pair frames share one geometry, {len(symbols)} atoms, box z = {lz:.1f} Å, charged q = {total_q:+.4f} e · {SPLIT_NOTE}</p>

<h2>0 · Training curves: Fermi level vs FermiMACE, energy vs the model before the local-electron energy</h2>
<div class="card">{curve_fer}</div>
<div class="card">{curve_E}</div>
{train_table}
{NOTE0}

<h2>0b · Fermi level across the 200 NiN44 twin pairs (DFT): charged vs neutral twin (sid + 600, solvated), per structure</h2>
<div class="card">{chartS}</div>
<p class="note">Two DFT points per structure id (hover for values): the charging shift is nearly
uniform, mean {shift_dft:+.2f} eV across the 200 pairs. For reference (not plotted), the model
reproduces these values with RMSE {rms_fc:.3f} eV (charged) / {rms_fn:.3f} eV (neutral) and mean
shift {shift_ml:+.2f} eV.</p>

<h2>1 · Scalar observables</h2>
{scalars}

<h2>2 · Per-atom charge states (charged frame) and the charged−neutral shift</h2>
<div class="card">{chartA}</div>
<div class="card">{chartAd}</div>
<table class="kv"><tr><th>element</th><th>count</th><th>mean q charged (e)</th><th>mean q neutral (e)</th><th>Σ Δq (e)</th></tr>{el_rows}</table>
<p class="note">Upper: charged-frame per-atom charges (hover for values). Lower: per-atom difference charged − neutral — where the model places the extra {total_q:+.3f} e. The grand total equals the charge difference exactly.</p>

<h2>3 · Charged: predicted vs DFT 1-D potential (Phi1D, training-loss construction)</h2>
<div class="card">{chartB}</div>
<div class="card">{chartB_res}</div>
<p class="note">Residual rms = {float(np.sqrt((d['phi_residual']**2).mean())):.4f} eV.</p>

<h2>4 · Neutral: predicted vs DFT 1-D potential</h2>
<div class="card">{chartBn}</div>
<div class="card">{chartBn_res}</div>
<p class="note">Residual rms = {float(np.sqrt((d['phi_residual_n']**2).mean())):.4f} eV. Neutral twin is solvated in this data set (solvated = 1): the 1-D PB solve runs with zero net charge (q_ion = {float(d["q_ion_n"]):+.4f} e).</p>

<h2>5 · Charged: plane-averaged net charge density, model vs DFT</h2>
<div class="card">{chartC}</div>
<div class="card">{chartC_res}</div>

<h2>6 · Neutral: plane-averaged net charge density, model vs DFT</h2>
<div class="card">{chartCn}</div>
<div class="card">{chartCn_res}</div>
<p class="note">Both density sections use ONE common unit factor fitted on the charged state ({ls_c:.3e}); fitting the neutral state alone gives {ls_n:.3e} (ratio {ls_n/ls_c:.4f}) — the cross-check that the normalization is state-independent.</p>

<h2>7 · Charged − neutral density difference: model vs DFT</h2>
<div class="card">{chartDD}</div>
<div class="card">{chartDD_res}</div>
<p class="note">The response of the electron density to charging. Integrated charge (plane average × cell area {A_CELL:.1f} Å² × dz): model {q_model:+.4f} e over the whole cell vs DFT {q_dft:+.4f} e over its grid window z = {z_dd.min():.1f}–{z_dd.max():.1f} Å (solute {total_q:+.4f} e). Point-wise rms difference = {rms_dd:.5f} e/Å³ against a DFT signal peak of {np.abs(dn_dft).max():.4f} e/Å³ — the shape is captured, fine detail is at the few-×10⁻³ level.</p>

<h2>8 · Solvent: ionic and bound charge profiles (charged state, 1-D PB solve)</h2>
<div class="card">{chartD}</div>
<p class="note">Ionic layer integrates to {float(d['q_ion']):+.3f} e against the solute {total_q:+.3f} e; bound charge nets ≈0 with dipole {float(d['mu_bound']):+.2f} e·Å.</p>

<h2>8a · Ionic charge vs DFT (VASPsol RHOION, physics sign)</h2>
<div class="card">{chartD_ion}</div>

<h2>8b · Bound charge vs DFT (VASPsol RHOB)</h2>
<div class="card">{chartD_rb}</div>
<p class="note">DFT reference profiles are plane averages of the raw VASPsol solvent grids of this very calculation.</p>

<h2>9 · P_off: physics prior and learned head correction</h2>
<div class="card">{chartE}</div>
<p class="note">Head correction peaks at {abs(d['delta_p']).max()/abs(d['prior']).max()*100:.0f}% of the prior peak.</p>
</div>
<div id="tip"></div>
<script>
const DATA = {json.dumps(charts_js, ensure_ascii=False)};
const tip = document.getElementById('tip');
for (const [cid, cfg] of Object.entries(DATA)) {{
  const svg = document.getElementById(cid);
  if (!svg) continue;
  const xlo = +svg.dataset.xlo, xhi = +svg.dataset.xhi, h = +svg.dataset.h;
  const ML = {ML}, MR = {MR}, W = {W};
  const xh = svg.querySelector('.xh');
  svg.addEventListener('mousemove', ev => {{
    const r = svg.getBoundingClientRect();
    const fx = (ev.clientX - r.left) / r.width * W;
    if (fx < ML || fx > W - MR) {{ tip.style.opacity = 0; xh.style.opacity = 0; return; }}
    const xv = xlo + (fx - ML) / (W - MR - ML) * (xhi - xlo);
    let i = 0, best = 1e18;
    cfg.x.forEach((x, k) => {{ const dd = Math.abs(x - xv); if (dd < best) {{ best = dd; i = k; }} }});
    const px = ML + (cfg.x[i] - xlo) / (xhi - xlo) * (W - MR - ML);
    xh.setAttribute('x1', px); xh.setAttribute('x2', px); xh.style.opacity = 0.5;
    tip.innerHTML = `x = ${{cfg.x[i].toFixed(2)}}<br>` +
      cfg.series.map(s => `${{s.n}}: ${{s.y[i]}} ${{cfg.unit}}`).join('<br>');
    tip.style.left = (ev.clientX + 14) + 'px'; tip.style.top = (ev.clientY + 12) + 'px';
    tip.style.opacity = 1;
  }});
  svg.addEventListener('mouseleave', () => {{ tip.style.opacity = 0; xh.style.opacity = 0; }});
}}
</script>
</body></html>
"""
with open(out_path, "w") as f:
    f.write(html)
print(f"page -> {out_path} ({len(html)/1024:.0f} KB)")
print(f"final-epoch val RMSE: model E {e_m[-1]:.2f} meV/atom fermi {fer_m[-1]:.3f} eV" + (f" | energy-compare E {e_p[-1]:.2f} fermi {fer_p[-1]:.3f}" if HAS_ECOMP else "") + (f" | FermiMACE fermi {fer_c[-1]:.3f}" if CPMACE_TXT else ""))
print(f"pairs scatter: fermi RMSE charged {rms_fc:.3f} neutral {rms_fn:.3f} eV"
      f" | shift DFT {shift_dft:+.2f} model {shift_ml:+.2f} eV")
