"""Temporary review page: final model vs DFT rho_b on four fresh structures.

Training construction throughout: model profile smeared sigma 0.15 (the
prediction); DFT reference raw. No 0.25 smoothing anywhere.

Usage: python gen_rhob_4sid_page.py <out_html> <npz1> <npz2> <npz3> <npz4>
"""
from __future__ import annotations

import sys

import numpy as np

out_path = sys.argv[1]
npzs = sys.argv[2:]

S1, S6 = "var(--s1)", "var(--s6)"
W, H, ML, MR, MT, MB = 860, 300, 64, 150, 14, 40


def smear(x, dz, s=0.15):
    n = len(x); g = np.fft.rfftfreq(n, d=dz) * 2 * np.pi
    return np.fft.irfft(np.fft.rfft(np.asarray(x, float)) * np.exp(-0.5 * (s * g) ** 2), n=n)


def chart(series, xlab, ylab, yscale=1e3):
    xs0 = min(p[0][0] for _, p, _, _ in series)
    xs1 = max(p[-1][0] for _, p, _, _ in series)
    vals = [v * yscale for _, p, _, _ in series for z, v in p]
    ymin, ymax = min(vals), max(vals)
    yr = ymax - ymin; ymin -= 0.07 * yr; ymax += 0.07 * yr
    X = lambda v: ML + (v - xs0) / (xs1 - xs0) * (W - ML - MR)
    Y = lambda v: MT + (ymax - v) / (ymax - ymin) * (H - MT - MB)
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto">']
    step = 10 ** np.floor(np.log10(yr / 4))
    for m in (1, 2, 5, 10):
        if yr / (step * m) <= 6:
            step *= m
            break
    t = np.ceil(ymin / step) * step
    while t <= ymax:
        o.append(f'<line x1="{ML}" y1="{Y(t):.1f}" x2="{W-MR}" y2="{Y(t):.1f}" stroke="var(--grid)"/>')
        o.append(f'<text x="{ML-8}" y="{Y(t)+4:.1f}" text-anchor="end" font-size="11" fill="var(--ink2)">{t:g}</text>')
        t += step
    for xv in range(int(np.ceil(xs0 / 5) * 5), int(xs1) + 1, 5):
        o.append(f'<line x1="{X(xv):.1f}" y1="{MT}" x2="{X(xv):.1f}" y2="{H-MB}" stroke="var(--grid)"/>')
        o.append(f'<text x="{X(xv):.1f}" y="{H-MB+16}" text-anchor="middle" font-size="11" fill="var(--ink2)">{xv}</text>')
    o.append(f'<text x="{(ML+W-MR)/2:.0f}" y="{H-4}" text-anchor="middle" font-size="11.5" fill="var(--ink2)">{xlab}</text>')
    o.append(f'<text x="14" y="{(MT+H-MB)/2:.0f}" text-anchor="middle" font-size="11.5" fill="var(--ink2)" transform="rotate(-90 14 {(MT+H-MB)/2:.0f})">{ylab}</text>')
    if ymin < 0 < ymax:
        o.append(f'<line x1="{ML}" y1="{Y(0):.1f}" x2="{W-MR}" y2="{Y(0):.1f}" stroke="var(--ink2)" opacity="0.5"/>')
    labels = []
    for lab, p, color, dash in series:
        pl = " ".join(f"{X(z):.1f},{Y(v*yscale):.1f}" for z, v in p)
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        o.append(f'<polyline points="{pl}" fill="none" stroke="{color}" stroke-width="2"{dd}/>')
        labels.append([Y(p[-1][1] * yscale) + 4, lab, color])
    labels.sort(key=lambda l: l[0])
    for i in range(1, len(labels)):
        if labels[i][0] < labels[i - 1][0] + 13:
            labels[i][0] = labels[i - 1][0] + 13
    for y, lab, color in labels:
        o.append(f'<text x="{W-MR+6}" y="{y:.1f}" font-size="10.5" fill="{color}">{lab}</text>')
    o.append("</svg>")
    return "".join(o)


cards = []
for f in npzs:
    d = np.load(f)
    sid = int(d["sid"])
    z = d["z"]; lz = float(d["lz"]); dz = z[1] - z[0]
    ref = d["ref_raw"]
    mod = smear(d["model_raw"], dz, 0.15)
    m = np.abs(ref) > 0.01 * np.abs(ref).max()
    i0, i1 = np.where(m)[0][[0, -1]]
    lo, hi = max(0, z[i0] - 2.5), min(z[-1], z[i1] + 2.5)
    sel = (z >= lo) & (z <= hi)
    pts_ref = list(zip(z[sel], ref[sel]))
    pts_mod = list(zip(z[sel], mod[sel]))
    rmse = float(np.sqrt(np.mean((mod - ref) ** 2)))
    hyd = "NiN44" if sid <= 200 else "NiN88"
    c = chart([("DFT (raw)", pts_ref, S6, ""), ("model (sigma 0.15)", pts_mod, S1, "7 4")],
              "z (A)", "rho_b (10^-3 e/A^3)")
    cards.append(f'''<div class="card"><h2>sid {sid} (validation, {hyd})</h2>
{c}
<p class="note">loss-construction RMSE {rmse*1e3:.3f}e-3 e/A^3 (full window, includes DFT grid texture)</p>
</div>''')

html = f"""<meta charset="utf-8">
<title>rhob400a vs DFT: four fresh validation structures</title>
<style>
:root {{ color-scheme: light dark;
  --ink:#171512; --ink2:#5c574f; --grid:#e6e2da; --card:#faf9f6;
  --s1:#2a78d6; --s6:#eb6834; }}
@media (prefers-color-scheme: dark) {{ :root {{
  --ink:#f2f0ea; --ink2:#b3aea3; --grid:#33312c; --card:#1e1d1a;
  --s1:#3987e5; --s6:#d95926; }} }}
:root[data-theme="dark"] {{ --ink:#f2f0ea; --ink2:#b3aea3; --grid:#33312c; --card:#1e1d1a;
  --s1:#3987e5; --s6:#d95926; }}
:root[data-theme="light"] {{ --ink:#171512; --ink2:#5c574f; --grid:#e6e2da; --card:#faf9f6;
  --s1:#2a78d6; --s6:#eb6834; }}
body {{ margin:0; color:var(--ink); font:15px/1.6 -apple-system,"Segoe UI",sans-serif; }}
.pg {{ max-width:920px; margin:0 auto; padding:24px 16px 56px; }}
h1 {{ font-size:20px; margin:0 0 4px; }}
.sub {{ color:var(--ink2); font-size:13.5px; margin:0 0 16px; }}
.card {{ background:var(--card); border:1px solid var(--grid); border-radius:8px;
  padding:14px 16px 8px; margin:14px 0; }}
h2 {{ font-size:15px; margin:0 0 8px; }}
.note {{ font-size:12.5px; color:var(--ink2); margin:6px 0 4px; }}
</style>
<div class="pg">
<h1>Final supervised model vs DFT: four fresh structures</h1>
<p class="sub">rhob400a (400 ep, weight 3e5), bound charge in the training construction —
model profile smeared sigma 0.15 (the prediction), DFT reference raw. Orange solid = DFT,
blue dashed = model. Structures never used in earlier reports.</p>
{"".join(cards)}
</div>
"""
open(out_path, "w").write(html)
print(f"wrote {out_path} ({len(html)} bytes)")
