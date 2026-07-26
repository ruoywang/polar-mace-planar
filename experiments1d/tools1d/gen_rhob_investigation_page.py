"""Summary page for the rho_b supervision investigation (draft for review).

Usage: python gen_rhob_investigation_page.py <out_html>
Run from 2-1D_PB root.
"""
from __future__ import annotations

import re
import sys

import numpy as np

out_path = sys.argv[1]

# ---------- palette / chart helpers ----------------------------------------
S1, S2, S3, S4, S5, S6 = ("var(--s1)", "var(--s2)", "var(--s3)",
                          "var(--s4)", "var(--s5)", "var(--s6)")
W, H, ML, MR, MT, MB = 880, 360, 70, 130, 16, 42


def ema(vals, span):
    out = []; a = 2.0 / (span + 1.0); m = None
    for v in vals:
        m = v if m is None else a * v + (1 - a) * m
        out.append(m)
    return out


def chart(series, xlab, ylab, logy=False, refs=(), band=None, xmax=None,
          height=H, yscale=1.0, right_pad=MR, smooth=None, ylim=None, sid_prefix=None):
    xs0 = min(p[0][0] for _, p, _, _ in series)
    xs1 = xmax if xmax is not None else max(p[-1][0] for _, p, _, _ in series)
    vals = [v * yscale for _, p, _, _ in series for z, v in p if xs0 <= z <= xs1]
    vals += [r * yscale for r, _ in refs]
    ymin, ymax = min(vals), max(vals)
    if ylim is not None:
        ymin, ymax = ylim
        Yv = lambda v: MT + (ymax - min(max(v, ymin), ymax)) / (ymax - ymin) * (height - MT - MB)
    elif logy:
        ymin *= 0.85; ymax *= 1.2
        Yv = lambda v: MT + (np.log10(ymax) - np.log10(v)) / (np.log10(ymax) - np.log10(ymin)) * (height - MT - MB)
    else:
        yr = ymax - ymin; ymin -= 0.07 * yr; ymax += 0.07 * yr
        Yv = lambda v: MT + (ymax - v) / (ymax - ymin) * (height - MT - MB)
    if ylim is not None:
        pass
    Xv = lambda v: ML + (v - xs0) / (xs1 - xs0) * (W - ML - right_pad)
    p = [f'<svg viewBox="0 0 {W} {height}" style="width:100%;height:auto">']
    if band:
        p.append(f'<rect x="{Xv(band[0]):.1f}" y="{MT}" width="{Xv(band[1])-Xv(band[0]):.1f}" height="{height-MT-MB}" fill="var(--ink2)" opacity="0.06"/>')
        p.append(f'<text x="{(Xv(band[0])+Xv(band[1]))/2:.1f}" y="{MT+13}" text-anchor="middle" font-size="10.5" fill="var(--ink2)">{band[2]}</text>')
    if logy:
        k0 = int(np.floor(np.log10(ymin)))
        ticks = [m * 10.0 ** k for k in range(k0, k0 + 8) for m in (1, 2, 5)
                 if ymin <= m * 10.0 ** k <= ymax]
    else:
        step = 10 ** np.floor(np.log10((ymax - ymin) / 4))
        for m in (1, 2, 5, 10):
            if (ymax - ymin) / (step * m) <= 6:
                step *= m
                break
        ticks = np.arange(np.ceil(ymin / step) * step, ymax, step)
    for t in ticks:
        p.append(f'<line x1="{ML}" y1="{Yv(t):.1f}" x2="{W-right_pad}" y2="{Yv(t):.1f}" stroke="var(--grid)"/>')
        lab = f"{t*1e4:g}" if logy else f"{t:g}"
        p.append(f'<text x="{ML-8}" y="{Yv(t)+4:.1f}" text-anchor="end" font-size="11" fill="var(--ink2)">{lab}</text>')
    xstep = 5 if xs1 - xs0 < 60 else (30 if xs1 - xs0 < 200 else 100)
    for xv in np.arange(np.ceil(xs0 / xstep) * xstep, xs1 + 0.1, xstep):
        p.append(f'<line x1="{Xv(xv):.1f}" y1="{MT}" x2="{Xv(xv):.1f}" y2="{height-MB}" stroke="var(--grid)"/>')
        p.append(f'<text x="{Xv(xv):.1f}" y="{height-MB+16}" text-anchor="middle" font-size="11" fill="var(--ink2)">{xv:g}</text>')
    p.append(f'<text x="{(ML+W-right_pad)/2:.0f}" y="{height-6}" text-anchor="middle" font-size="11.5" fill="var(--ink2)">{xlab}</text>')
    p.append(f'<text x="16" y="{(MT+height-MB)/2:.0f}" text-anchor="middle" font-size="11.5" fill="var(--ink2)" transform="rotate(-90 16 {(MT+height-MB)/2:.0f})">{ylab}</text>')
    if not logy and ymin < 0 < ymax:
        p.append(f'<line x1="{ML}" y1="{Yv(0):.1f}" x2="{W-right_pad}" y2="{Yv(0):.1f}" stroke="var(--ink2)" opacity="0.5"/>')
    labels = []  # (y, x, text, color) -> collision-resolved right-edge labels
    for r, lab in refs:
        p.append(f'<line x1="{ML}" y1="{Yv(r*yscale):.1f}" x2="{W-right_pad}" y2="{Yv(r*yscale):.1f}" stroke="var(--ink2)" stroke-dasharray="3 5" opacity="0.6"/>')
        labels.append([Yv(r * yscale) + 4, W - right_pad + 6, lab, "var(--ink2)"])
    for si, (lab, pts_, color, dash) in enumerate(series):
        inw = [(z, v) for z, v in pts_ if xs0 <= z <= xs1 and (not logy or v * yscale > 0)]
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        gid = f' id="{sid_prefix}-{si}"' if sid_prefix else ""
        if smooth:
            pl_raw = " ".join(f"{Xv(z):.1f},{Yv(v*yscale):.1f}" for z, v in inw)
            p.append(f'<polyline points="{pl_raw}" fill="none" stroke="{color}" stroke-width="1.3" opacity="0.22"{dd}/>')
            sm_v = ema([v for _, v in inw], smooth)
            inw = list(zip([z for z, _ in inw], sm_v))
            p.append('<polyline points="' + " ".join(f"{Xv(z):.1f},{Yv(v*yscale):.1f}" for z, v in inw)
                     + f'" fill="none" stroke="{color}" stroke-width="2.5"{dd}/>')
        else:
            pl = " ".join(f"{Xv(z):.1f},{Yv(v*yscale):.1f}" for z, v in inw)
            p.append(f'<polyline{gid} points="{pl}" fill="none" stroke="{color}" stroke-width="2"{dd}/>')
        ze, ve = inw[-1]
        if ze >= xs0 + 0.93 * (xs1 - xs0):
            labels.append([Yv(ve * yscale) + 4, W - right_pad + 6, lab, color])
        else:
            labels.append([Yv(ve * yscale) + 4, Xv(ze) + 5, lab, color])
    # resolve vertical collisions among right-edge labels
    edge = sorted((l for l in labels if l[1] == W - right_pad + 6), key=lambda l: l[0])
    for i in range(1, len(edge)):
        if edge[i][0] < edge[i - 1][0] + 13:
            edge[i][0] = edge[i - 1][0] + 13
    for y, x, lab, color in labels:
        p.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="10.5" fill="{color}">{lab}</text>')
    p.append("</svg>")
    return "".join(p)


def chips(items, toggle_prefix=None):
    o = []
    for i, (lab, color, dash) in enumerate(items):
        sw = f'<span style="display:inline-block;width:22px;height:0;border-top:3px {"dashed" if dash else "solid"} {color};vertical-align:middle;margin-right:6px"></span>'
        attr = (f' style="margin-right:16px;white-space:nowrap;cursor:pointer" '
                f'onclick="tgl(\'{toggle_prefix}-{i}\', this)" title="click to show/hide"'
                if toggle_prefix else ' style="margin-right:16px;white-space:nowrap"')
        o.append(f'<span{attr}>{sw}{lab}</span>')
    hint = ' &middot; <span style="opacity:0.7">click a legend entry to show/hide its curve</span>' if toggle_prefix else ''
    return '<div style="font-size:12.5px;color:var(--ink2);margin:2px 0 6px">' + "".join(o) + hint + "</div>"


def traj(logf):
    out = []
    for m in re.finditer(r"Epoch (\d+):.*?RMSE_rhob_1d=([0-9.]+)", open(logf).read()):
        out.append((int(m.group(1)), float(m.group(2))))
    return sorted(set(out))


def traj_metric(logfiles, column):
    vals = {}
    for lf in logfiles:
        for m in re.finditer(r"Epoch (\d+):.*?" + column + r"=([0-9.]+)", open(lf).read()):
            vals[int(m.group(1))] = float(m.group(2))
    return sorted(vals.items())


def smear(x, dz, s=0.25):
    n = len(x); g = np.fft.rfftfreq(n, d=dz) * 2 * np.pi
    return np.fft.irfft(np.fft.rfft(np.asarray(x, float)) * np.exp(-0.5 * (s * g) ** 2), n=n)


pts = lambda z, v: list(zip(np.asarray(z, float), np.asarray(v, float)))

# ---------- chart 1: rho_b vs epoch across weights --------------------------
runs = [("w50", "exp_pb1d_rhob80/run.log", S3), ("w1e4", "exp_pb1d_rhobw/run.log", S4),
        ("w1e5", "exp_pb1d_rhobw2/run.log", S2), ("w3e5", "exp_pb1d_rhob3e5/run.log", S1),
        ("w1e6", "exp_pb1d_rhob1e6/run.log", S5)]
c1_series = [(lab, traj(f), col, "") for lab, f, col in runs]
c1 = chart(c1_series, "epoch", "rho_b RMSE (10^-4 e/A^3, log)", logy=True,
           refs=[(5.69e-4, "zero-model baseline"), (3.55e-4, "unsupervised"),
                 (8.26e-5, "final 400-ep model")],
           band=(0, 30, "planar warmup"), xmax=150)

# ---------- chart 3: giant cancellation (FINAL model, sid 152) ---------------
d = np.load("exp_pb1d_rhob400a/structure_report_sid152_400a.npz")
zs = d["z_solve"]; dzs = float(zs[1] - zs[0])
poff = np.asarray(d["prior"], float) + np.asarray(d["delta_p"], float)
rb_off = smear(-np.gradient(poff, dzs), dzs, s=0.15)
rb_tot = smear(np.asarray(d["rho_bound"], float), dzs, s=0.15)
rb_resp = rb_tot - rb_off
_refs3 = np.load("train-data/dft_solvent1d_ref.npz")
_row3 = list(_refs3["sample_ids"]).index(152)
_zr3 = _refs3["z_A"]
_rdft3 = -_refs3["rb_z_vasp"][_row3]  # raw, as in the training construction
_m3 = np.abs(_rdft3) > 0.01 * np.abs(_rdft3).max()
_j0, _j1 = np.where(_m3)[0][[0, -1]]
w3 = (max(0.0, _zr3[_j0] - 2.5), _zr3[_j1] + 2.5)
win = (zs >= w3[0]) & (zs <= w3[1])
c3a = chart([("P_off term", pts(zs[win], rb_off[win]), S2, ""),
             ("A*<E> response term", pts(zs[win], rb_resp[win]), S3, "")],
            "", "each term (10^-3 e/A^3)", yscale=1e3, height=260, right_pad=170)
wref = (_zr3 >= w3[0]) & (_zr3 <= w3[1])
c3b = chart([("sum (model rho_b)", pts(zs[win], rb_tot[win]), S1, "7 4"),
             ("DFT (raw)", pts(_zr3[wref], _rdft3[wref]), S6, "")],
            "z (A)", "total (10^-3 e/A^3)", yscale=1e3, height=260, right_pad=170)

# ---------- chart 4: final model vs DFT, sid 152 ----------------------------
r152 = np.load("exp_pb1d_rhob400a/structure_report_sid152_400a.npz")
refs_pack = np.load("train-data/dft_solvent1d_ref.npz")
row = list(refs_pack["sample_ids"]).index(152)
rb_dft = -refs_pack["rb_z_vasp"][row]
zd = refs_pack["z_A"]; dzd = float(zd[1] - zd[0])
zs2 = r152["z_solve"]; dzs2 = float(zs2[1] - zs2[0])
mm = np.abs(rb_dft) > 0.01 * np.abs(rb_dft).max()
j0, j1 = np.where(mm)[0][[0, -1]]
w4 = (max(0.0, zd[j0] - 2.5), zd[j1] + 2.5)
sdft = smear(rb_dft, dzd); smod = smear(np.asarray(r152["rho_bound"], float), dzs2)
smod15 = smear(np.asarray(r152["rho_bound"], float), dzs2, s=0.15)
r152u = np.load("exp_pb1d_prod400/structure_report_sid152.npz")
zs2u = r152u["z_solve"]; dzs2u = float(zs2u[1] - zs2u[0])
smod15u = smear(np.asarray(r152u["rho_bound"], float), dzs2u, s=0.15)
c4t = chart([("DFT (raw, as in the loss)", [(z, v) for z, v in pts(zd, rb_dft) if w4[0] <= z <= w4[1]], S6, ""),
             ("supervised final model", [(z, v) for z, v in pts(zs2, smod15) if w4[0] <= z <= w4[1]], S1, "7 4"),
             ("unsupervised", [(z, v) for z, v in pts(zs2u, smod15u) if w4[0] <= z <= w4[1]], S2, "7 4")],
            "z (A)", "rho_b (10^-3 e/A^3)", yscale=1e3, right_pad=190, sid_prefix="rbc")
c4 = chart([("DFT", [(z, v) for z, v in pts(zd, sdft) if w4[0] <= z <= w4[1]], S6, ""),
            ("final model", [(z, v) for z, v in pts(zs2, smod) if w4[0] <= z <= w4[1]], S1, "2 4")],
           "z (A)", "smeared rho_b (10^-3 e/A^3)", yscale=1e3, right_pad=150)
c5 = chart([("DFT", pts(r152["z_phi"], r152["phi_ref_cmp"]), S6, ""),
            ("final model", pts(r152["z_phi"], r152["phi_pred_cmp"]), S1, "7 4")],
           "z (A)", "phi (eV)", right_pad=150)

phi_rms_152 = float(np.sqrt(np.mean(np.asarray(r152["phi_residual"]) ** 2)))

# ---------- charts 6/7: potential and fermi vs epoch ------------------------
LOGS_W30 = ["exp_pb1d_rhob400a/run.log"]
LOGS_NW = ["exp_pb1d_rhobnw/run_150.log", "exp_pb1d_rhobnw/run.log"]
LOGS_UN = ["exp_pb1d_prod400/run.log"]
c6 = chart([("warmup 30 (supervised)", traj_metric(LOGS_W30, "RMSE_potential"), S1, ""),
            ("no warmup (supervised)", traj_metric(LOGS_NW, "RMSE_potential"), S3, ""),
            ("unsupervised", traj_metric(LOGS_UN, "RMSE_potential"), S2, "")],
           "epoch", "potential RMSE (eV)", xmax=400, right_pad=185, ylim=(0.02, 0.20))
c7 = chart([("warmup 30 (supervised)", traj_metric(LOGS_W30, "RMSE_fermi"), S1, ""),
            ("no warmup (supervised)", traj_metric(LOGS_NW, "RMSE_fermi"), S3, ""),
            ("unsupervised", traj_metric(LOGS_UN, "RMSE_fermi"), S2, "")],
           "epoch", "fermi RMSE (eV)", xmax=400, right_pad=185, ylim=(0.02, 0.20))

html = f"""<meta charset="utf-8">
<title>Supervising the 1-D bound charge: what moved, what it cost, and why</title>
<style>
:root {{ color-scheme: light dark;
  --ink:#171512; --ink2:#5c574f; --grid:#e6e2da; --card:#faf9f6;
  --s1:#2a78d6; --s2:#008300; --s3:#c3339c; --s4:#a07400; --s5:#0e8f74; --s6:#eb6834; }}
@media (prefers-color-scheme: dark) {{ :root {{
  --ink:#f2f0ea; --ink2:#b3aea3; --grid:#33312c; --card:#1e1d1a;
  --s1:#3987e5; --s2:#3ba33b; --s3:#e05ab8; --s4:#c9a227; --s5:#1bc394; --s6:#d95926; }} }}
:root[data-theme="dark"] {{ --ink:#f2f0ea; --ink2:#b3aea3; --grid:#33312c; --card:#1e1d1a;
  --s1:#3987e5; --s2:#3ba33b; --s3:#e05ab8; --s4:#c9a227; --s5:#1bc394; --s6:#d95926; }}
:root[data-theme="light"] {{ --ink:#171512; --ink2:#5c574f; --grid:#e6e2da; --card:#faf9f6;
  --s1:#2a78d6; --s2:#008300; --s3:#c3339c; --s4:#a07400; --s5:#0e8f74; --s6:#eb6834; }}
body {{ margin:0; color:var(--ink); font:15px/1.6 -apple-system,"Segoe UI",sans-serif; }}
.pg {{ max-width:960px; margin:0 auto; padding:26px 18px 60px; }}
h1 {{ font-size:21px; margin:0 0 4px; }}
.sub {{ color:var(--ink2); font-size:13.5px; margin:0 0 16px; max-width:72ch; }}
h2 {{ font-size:16px; margin:26px 0 8px; }}
.card {{ background:var(--card); border:1px solid var(--grid); border-radius:8px;
  padding:14px 16px 8px; margin:12px 0; }}
.tiles {{ display:flex; gap:12px; flex-wrap:wrap; margin:14px 0; }}
.tile {{ background:var(--card); border:1px solid var(--grid); border-radius:8px; padding:10px 16px; }}
.tile .v {{ font-size:21px; font-weight:600; }}
.tile .l {{ font-size:12px; color:var(--ink2); max-width:230px; }}
.note {{ font-size:12.5px; color:var(--ink2); margin:6px 0 4px; }}
table {{ border-collapse:collapse; font-size:13.5px; margin:8px 0;
  font-variant-numeric: tabular-nums; }}
th, td {{ border:1px solid var(--grid); padding:5px 12px; text-align:right; }}
th:first-child, td:first-child {{ text-align:left; }}
thead th {{ background:var(--card); }}
.good {{ color:var(--s2); font-weight:600; }}
.bad {{ color:var(--s6); font-weight:600; }}
.wrap {{ overflow-x:auto; }}
</style>
<div class="pg">
<h1>Supervising the 1-D bound charge against DFT</h1>
<p class="sub">A loss term compares the model's solvent bound-charge profile rho_b(z) (from the
1-D PB solve, smeared sigma 0.15 A as part of the prediction) with the raw plane-averaged
VASPsol++ RHOB. Weight scan, mechanism, and the final 400-epoch model.</p>

<div class="tiles">
<div class="tile"><div class="v">4.3x</div><div class="l">rho_b improvement vs unsupervised
(RMSE 0.000355 &rarr; 0.0000826 e/A^3, 40 val structures, common metric)</div></div>
<div class="tile"><div class="v">3e5</div><div class="l">best supervision weight = the value
that makes the term's loss contribution match its siblings</div></div>
<div class="tile"><div class="v">+9% / +16%</div><div class="l">cost at 400 ep: Phi1D 0.041&rarr;0.045,
potential 0.052&rarr;0.060 eV; fermi and forces unchanged</div></div>
</div>

<h2>1 &middot; Weight scan: the error vs epoch, five weights</h2>
<div class="card">
{chips([(l, c, False) for l, _, c in runs])}
{c1}
<p class="note">Validation RMSE, log scale. This chart and all cross-model numbers on this page
use the symmetric sigma-0.25 construction — the one metric every run logged during training —
so eight models can be ranked on one scale; the final model's own training construction
(sigma 0.15, raw reference) appears in section 4. All runs share
the warmup segment (the loss is dormant there; the drop is driven by the other losses). The fan-out
after the ep-30 handover is the supervision at work: w50/w1e4 plateau, heavier weights keep
descending. The dashed marker is the final 400-epoch model's aggregate.</p>
</div>
<div class="wrap"><table>
<thead><tr><th>ep 149</th><th>unsup.</th><th>w1e5</th><th>w3e5</th><th>w1e6</th></tr></thead>
<tbody>
<tr><td>rho_b (e/A^3)</td><td>0.000355</td><td>0.000175</td><td>0.000130</td><td class="good">0.000093</td></tr>
<tr><td>potential (eV)</td><td>0.0615</td><td>0.0694</td><td class="good">0.0636</td><td>0.0669</td></tr>
<tr><td>fermi (eV)</td><td>0.0555</td><td>0.0665</td><td class="good">0.0513</td><td>0.0586</td></tr>
<tr><td>Phi1D (eV)</td><td class="good">0.0535</td><td>0.0577</td><td>0.0565</td><td class="bad">0.0733</td></tr>
<tr><td>density 3d (e/A^3)</td><td>0.0437</td><td class="good">0.0418</td><td>0.0445</td><td class="bad">0.0477</td></tr>
</tbody></table></div>
<p class="note">Benefit is monotone up to 3e5; at 1e6 the potential profile and density start paying.
The knee has a clean reading: at w3e5 the rho_b term's contribution ((1.3e-4)^2 x 3e5 ~ 5e-3)
matches the Phi1D ((0.057)^2) and density ((0.044)^2) terms — same units as the 3-D density loss,
but the plane-averaged bound charge is ~300x smaller in amplitude, hence the large weight.</p>

<h2>2 &middot; Why it is expensive: two giant terms cancel to a 2% residue</h2>
<div class="card">
{chips([("P_off term", S2, False), ("A*<E> response term", S3, False)])}
{c3a}
{chips([("sum = model rho_b", S1, True), ("DFT", S6, False)])}
{c3b}
<p class="note">Final model, sid 152, drawn in the same construction as section 4 (model terms
smeared sigma 0.15; DFT raw). The bound charge is the residue of a +-0.19 e/A^3-scale
P_off term and dielectric response term — a 50:1 cancellation. Supervision's real work was
coordinating a sub-percent alignment of two objects fifty times larger than the target: at
intermediate training the misalignment showed as spurious side peaks and a systematic
0.2-0.35 A metal-ward shift of every peak (the product of averages weights the response by
the unscreened mean field; the covariance content P_off must supply the correction). The final
model has driven the peak shift to +0.03 A and the side lobe from 25% to 16% of the main peak —
this precision requirement is what the contribution-parity weight (3e5) and 400 epochs paid for.</p>
</div>

<h2>3 &middot; Final model vs DFT (sid 152, training NiN44)</h2>
<div class="card">
{chips([("DFT (raw)", S6, False), ("supervised final model", S1, True), ("unsupervised", S2, True)], toggle_prefix="rbc")}
{c4t}
<p class="note">The training's own construction: the prediction is the sigma-0.15-smeared model
profile; the DFT reference enters raw (its remaining jitter is grid texture, worth &lt; 4 meV of
potential). Remaining deviation: the inner-edge lobe, where the metal-side potential leverage is
~100x the solvent side and the density-error screen lives.</p>
{chips([("DFT", S6, False), ("final model", S1, True)])}
{c5}
<p class="note">1-D potential, loss-verbatim construction, upper-aligned; residual rms
{phi_rms_152:.3f} eV on this structure.</p>
</div>

<h2>4 &middot; Is the planar warmup still needed?</h2>
<div class="wrap"><table>
<thead><tr><th>ep 399, full config</th><th>no warmup</th><th>warmup 30</th><th>unsupervised</th></tr></thead>
<tbody>
<tr><td>rho_b (this run's metric)</td><td>0.000111</td><td>0.000111</td><td>&mdash;</td></tr>
<tr><td>potential (eV)</td><td>0.0671</td><td class="good">0.0602</td><td>0.0518</td></tr>
<tr><td>fermi (eV)</td><td>0.0497</td><td class="good">0.0401</td><td>0.0388</td></tr>
<tr><td>E (meV/atom)</td><td>2.84</td><td class="good">2.32</td><td>2.04</td></tr>
<tr><td>F (meV/A)</td><td>17.9</td><td>17.4</td><td>17.4</td></tr>
<tr><td>density 3d</td><td class="good">0.0404</td><td>0.0423</td><td>0.0397</td></tr>
</tbody></table></div>
<div class="card">
{chips([("warmup 30 (supervised)", S1, False), ("no warmup (supervised)", S3, False), ("unsupervised", S2, False)])}
{c6}
{c7}
<p class="note">Validation potential and fermi errors vs epoch (y clipped to 0.02-0.20 eV;
the early warmup-phase values ride the top edge). The warmup band of the
supervised runs and the cold-start lag of the no-warmup run are both visible; the supervised
warmup-30 run tracks the unsupervised baseline to within its small endpoint offset.</p>
</div>
<p class="note">With strong rho_b supervision the cold start no longer diverges (the old
catastrophic failure mode is gone; rho_b and density even match or beat the warmup run) — but at
an equal epoch budget the potential family still lags 10-25%. Verdict unchanged: keep warmup 30;
it is free accuracy, no longer a stability requirement.</p>

<script>
function tgl(id, chip) {{
  const el = document.getElementById(id);
  if (!el) return;
  const off = el.style.display === 'none';
  el.style.display = off ? '' : 'none';
  chip.style.opacity = off ? 1 : 0.35;
}}
</script>
<h2>Method notes</h2>
<p class="note">Loss: model profile smeared (sigma 0.15, part of the prediction) vs raw DFT
reference — matching the raw reference denoised the model profile (its sub-grid texture dropped
1.62e-4 &rarr; 1.39e-4 e/A^3, toward DFT's 0.87e-4). Weight calibration rule: set a new term's weight
from its equilibrium residual so its loss contribution matches the sibling terms, then verify
after a few epochs. Distributed caveat: a loss whose scoreable-row count is data-dependent per
rank must join the reduction collective unconditionally (a rank-local early return deadlocked
3 ranks at the warmup handover; caught by the 5-epoch gate).</p>
</div>
"""
open(out_path, "w").write(html)
print(f"wrote {out_path} ({len(html)} bytes)")
