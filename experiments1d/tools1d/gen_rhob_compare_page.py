"""Render the two-model vs DFT 1-D bound-charge comparison page (one sid).

Usage: python gen_rhob_compare_page.py <npz> <out_html>
"""
from __future__ import annotations

import sys

import numpy as np

npz_path, out_path = sys.argv[1], sys.argv[2]
d = np.load(npz_path)
z = d["z"]
sid = int(d["sid"])

S1 = "var(--s1)"   # blue
S2 = "var(--s2)"   # green
S5 = "var(--s5)"   # aqua
S6 = "var(--s6)"   # orange: DFT reference

# model curves discovered from npz keys (order = insertion order in dump)
model_labels = [k[:-4] for k in d.files if k.endswith("_raw") and k != "ref_raw"]
MODEL_STYLE = [(S1, "2 4"), (S2, "7 4"), (S5, "10 3 2 3")]

# window: where the reference has support, padded
ref = d["ref_raw"]
mask = np.abs(ref) > 0.01 * np.abs(ref).max()
i0, i1 = np.where(mask)[0][[0, -1]]
pad = 30
i0, i1 = max(0, i0 - pad), min(len(z) - 1, i1 + pad)
sl = slice(i0, i1 + 1)

W, H, ML, MR, MT, MB = 860, 340, 64, 16, 14, 40


def svg_chart(series, ylab):
    zz = z[sl]
    ys = [s[sl] for _, s, _, _ in series]
    ymin = min(s.min() for s in ys)
    ymax = max(s.max() for s in ys)
    yr = ymax - ymin
    ymin -= 0.06 * yr
    ymax += 0.06 * yr
    x0, x1 = zz[0], zz[-1]

    def X(v):
        return ML + (v - x0) / (x1 - x0) * (W - ML - MR)

    def Y(v):
        return MT + (ymax - v) / (ymax - ymin) * (H - MT - MB)

    parts = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto" data-x0="{x0:.3f}" data-x1="{x1:.3f}">']
    # grid + y ticks (1-2-5 style, ~5 lines)
    step = 10 ** np.floor(np.log10(yr / 4))
    for m in (1, 2, 5, 10):
        if yr / (step * m) <= 6:
            step *= m
            break
    t = np.ceil(ymin / step) * step
    while t <= ymax:
        yy = Y(t)
        parts.append(f'<line x1="{ML}" y1="{yy:.1f}" x2="{W-MR}" y2="{yy:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{ML-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="var(--ink2)">{t*1e3:.1f}</text>')
        t += step
    for xv in range(int(np.ceil(x0 / 5) * 5), int(x1) + 1, 5):
        xx = X(xv)
        parts.append(f'<line x1="{xx:.1f}" y1="{MT}" x2="{xx:.1f}" y2="{H-MB}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{xx:.1f}" y="{H-MB+16}" text-anchor="middle" font-size="11" fill="var(--ink2)">{xv}</text>')
    parts.append(f'<text x="{(ML+W-MR)/2:.0f}" y="{H-4}" text-anchor="middle" font-size="11.5" fill="var(--ink2)">z (A)</text>')
    parts.append(f'<text x="14" y="{(MT+H-MB)/2:.0f}" text-anchor="middle" font-size="11.5" fill="var(--ink2)" transform="rotate(-90 14 {(MT+H-MB)/2:.0f})">{ylab}</text>')
    zero_y = Y(0.0)
    parts.append(f'<line x1="{ML}" y1="{zero_y:.1f}" x2="{W-MR}" y2="{zero_y:.1f}" stroke="var(--ink2)" stroke-width="1" opacity="0.5"/>')
    for _, s, color, dash in series:
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(zz, s[sl]))
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
            f'{"stroke-dasharray=" + chr(34) + dash + chr(34) if dash else ""}/>')
    parts.append(f'<line class="xh" x1="0" y1="{MT}" x2="0" y2="{H-MB}" stroke="var(--ink2)" stroke-width="1" opacity="0"/>')
    parts.append("</svg>")
    return "".join(parts)


def chips(items):
    cs = []
    for label, color, dash in items:
        line = (f'<span style="display:inline-block;width:22px;height:0;border-top:3px '
                f'{"dashed" if dash else "solid"} {color};vertical-align:middle;margin-right:6px"></span>')
        cs.append(f'<span style="margin-right:18px;white-space:nowrap">{line}{label}</span>')
    return ('<div style="font-size:12.5px;color:var(--ink2);margin:2px 0 6px">' + "".join(cs) + "</div>")


raw_series = [("DFT (RHOB)", ref, S6, "")] + [
    (lab, d[f"{lab}_raw"], *MODEL_STYLE[i % 3]) for i, lab in enumerate(model_labels)
]
sm_series = [("DFT smeared", d["ref_smeared"], S6, "")] + [
    (lab + " smeared", d[f"{lab}_smeared"], *MODEL_STYLE[i % 3])
    for i, lab in enumerate(model_labels)
]

rmses = {lab: float(np.sqrt(np.mean((d[f"{lab}_smeared"] - d["ref_smeared"]) ** 2)))
         for lab in model_labels}
base = float(np.sqrt(np.mean(d["ref_smeared"] ** 2)))
rmse_txt = " / ".join(f"{v*1e3:.3f}" for v in rmses.values())
rmse_lbl = " / ".join(rmses)

html = f"""<meta charset="utf-8">
<title>rho_b comparison, sid {sid}: supervised vs unsupervised vs DFT</title>
<style>
:root {{ color-scheme: light dark;
  --ink:#171512; --ink2:#5c574f; --grid:#e6e2da; --card:#faf9f6;
  --s1:#2a78d6; --s2:#008300; --s5:#0e8f74; --s6:#eb6834; }}
@media (prefers-color-scheme: dark) {{ :root {{
  --ink:#f2f0ea; --ink2:#b3aea3; --grid:#33312c; --card:#1e1d1a;
  --s1:#3987e5; --s2:#3ba33b; --s5:#1bc394; --s6:#d95926; }} }}
:root[data-theme="dark"] {{
  --ink:#f2f0ea; --ink2:#b3aea3; --grid:#33312c; --card:#1e1d1a;
  --s1:#3987e5; --s2:#3ba33b; --s5:#1bc394; --s6:#d95926; }}
:root[data-theme="light"] {{
  --ink:#171512; --ink2:#5c574f; --grid:#e6e2da; --card:#faf9f6;
  --s1:#2a78d6; --s2:#008300; --s5:#0e8f74; --s6:#eb6834; }}
body {{ margin:0; color:var(--ink);
  font:15px/1.6 -apple-system,"Segoe UI",sans-serif; }}
.pg {{ max-width:920px; margin:0 auto; padding:24px 16px 56px; }}
h1 {{ font-size:20px; margin:0 0 4px; }}
.sub {{ color:var(--ink2); font-size:13.5px; margin:0 0 18px; }}
.card {{ background:var(--card); border:1px solid var(--grid); border-radius:8px;
  padding:14px 16px 8px; margin:14px 0; }}
h2 {{ font-size:15px; margin:0 0 8px; }}
.tiles {{ display:flex; gap:12px; flex-wrap:wrap; margin:14px 0; }}
.tile {{ background:var(--card); border:1px solid var(--grid); border-radius:8px;
  padding:10px 16px; }}
.tile .v {{ font-size:20px; font-weight:600; }}
.tile .l {{ font-size:12px; color:var(--ink2); }}
.note {{ font-size:12.5px; color:var(--ink2); margin:6px 0 4px; }}
.tip {{ position:fixed; pointer-events:none; background:var(--card);
  border:1px solid var(--grid); border-radius:6px; padding:4px 8px;
  font-size:12px; opacity:0; z-index:9; }}
</style>
<div class="pg">
<h1>1-D bound charge: supervised vs unsupervised vs DFT</h1>
<p class="sub">sid {sid} (validation, NiN44) &middot; both models evaluated with a fresh 1-D solve;
DFT reference = plane-averaged VASPsol++ RHOB (physics sign). y-axis in 10<sup>-3</sup> e/A<sup>3</sup>.</p>

<div class="tiles">
<div class="tile"><div class="v">{rmse_txt}</div><div class="l">smeared RMSE vs DFT: {rmse_lbl} (10^-3 e/A^3)</div></div>
<div class="tile"><div class="v">{base*1e3:.3f}</div><div class="l">zero-model baseline (10^-3 e/A^3)</div></div>
</div>

<div class="card">
<h2>Raw profiles (solver output vs DFT grid average)</h2>
{chips([(l, c, bool(dsh)) for l, _, c, dsh in raw_series])}
{svg_chart(raw_series, "rho_b (10^-3 e/A^3)")}
<p class="note">The model-DFT gap sits in the inner lobe; compare how far each supervision
strength moves the curve toward DFT there.</p>
</div>

<div class="card">
<h2>Loss view (both sides smeared, sigma 0.25 A)</h2>
{chips([(l, c, bool(dsh)) for l, _, c, dsh in sm_series])}
{svg_chart(sm_series, "smeared rho_b (10^-3 e/A^3)")}
<p class="note">This is exactly what the training loss compares. Same picture: smearing removes
sub-grid jaggedness but the lobe-shape difference stays.</p>
</div>
</div>
<div class="tip" id="tip"></div>
<script>
const tip = document.getElementById('tip');
for (const svg of document.querySelectorAll('svg')) {{
  const x0 = parseFloat(svg.dataset.x0), x1 = parseFloat(svg.dataset.x1);
  const xh = svg.querySelector('.xh');
  svg.addEventListener('mousemove', ev => {{
    const r = svg.getBoundingClientRect();
    const fx = (ev.clientX - r.left) / r.width * {W};
    if (fx < {ML} || fx > {W - MR}) {{ tip.style.opacity = 0; xh.style.opacity = 0; return; }}
    const zv = x0 + (fx - {ML}) / ({W - ML - MR}) * (x1 - x0);
    xh.setAttribute('x1', fx); xh.setAttribute('x2', fx); xh.style.opacity = 0.5;
    tip.textContent = 'z = ' + zv.toFixed(2) + ' A';
    tip.style.left = (ev.clientX + 14) + 'px'; tip.style.top = (ev.clientY + 12) + 'px';
    tip.style.opacity = 1;
  }});
  svg.addEventListener('mouseleave', () => {{ tip.style.opacity = 0; xh.style.opacity = 0; }});
}}
</script>
"""
open(out_path, "w").write(html)
print(f"wrote {out_path} ({len(html)} bytes)")
