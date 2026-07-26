"""Three-panel round-2 verdict page: rho_b, plane density, phi residual (one sid).

Usage: python gen_rhob_round2_page.py <rhob_cmp_npz> <report_new_npz> <report_unsup_npz> <out_html>
"""
from __future__ import annotations

import sys

import numpy as np

rb_npz, rep_new, rep_old, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
rb = np.load(rb_npz)
rn = np.load(rep_new)
ro = np.load(rep_old)
sid = int(rb["sid"])
A = 14.8 ** 2

S1, S2, S6 = "var(--s1)", "var(--s2)", "var(--s6)"
W, H, ML, MR, MT, MB = 860, 320, 64, 16, 14, 40


def svg_chart(x, series, xlab, ylab, yscale=1.0, xwin=None):
    if xwin is not None:
        m = (x >= xwin[0]) & (x <= xwin[1])
    else:
        m = np.ones_like(x, bool)
    xx = x[m]
    ys = [s[m] * yscale for _, s, _, _ in series]
    ymin = min(s.min() for s in ys); ymax = max(s.max() for s in ys)
    yr = ymax - ymin; ymin -= 0.06 * yr; ymax += 0.06 * yr
    x0, x1 = xx[0], xx[-1]
    X = lambda v: ML + (v - x0) / (x1 - x0) * (W - ML - MR)
    Y = lambda v: MT + (ymax - v) / (ymax - ymin) * (H - MT - MB)
    p = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto" data-x0="{x0:.3f}" data-x1="{x1:.3f}">']
    step = 10 ** np.floor(np.log10(yr / 4))
    for mlt in (1, 2, 5, 10):
        if yr / (step * mlt) <= 6:
            step *= mlt
            break
    t = np.ceil(ymin / step) * step
    while t <= ymax:
        p.append(f'<line x1="{ML}" y1="{Y(t):.1f}" x2="{W-MR}" y2="{Y(t):.1f}" stroke="var(--grid)"/>')
        p.append(f'<text x="{ML-8}" y="{Y(t)+4:.1f}" text-anchor="end" font-size="11" fill="var(--ink2)">{t:g}</text>')
        t += step
    for xv in range(int(np.ceil(x0 / 5) * 5), int(x1) + 1, 5):
        p.append(f'<line x1="{X(xv):.1f}" y1="{MT}" x2="{X(xv):.1f}" y2="{H-MB}" stroke="var(--grid)"/>')
        p.append(f'<text x="{X(xv):.1f}" y="{H-MB+16}" text-anchor="middle" font-size="11" fill="var(--ink2)">{xv}</text>')
    p.append(f'<text x="{(ML+W-MR)/2:.0f}" y="{H-4}" text-anchor="middle" font-size="11.5" fill="var(--ink2)">{xlab}</text>')
    p.append(f'<text x="14" y="{(MT+H-MB)/2:.0f}" text-anchor="middle" font-size="11.5" fill="var(--ink2)" transform="rotate(-90 14 {(MT+H-MB)/2:.0f})">{ylab}</text>')
    if ymin < 0 < ymax:
        p.append(f'<line x1="{ML}" y1="{Y(0):.1f}" x2="{W-MR}" y2="{Y(0):.1f}" stroke="var(--ink2)" opacity="0.5"/>')
    for _, s, color, dash in series:
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(xx, s[m] * yscale))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"{d}/>')
    p.append(f'<line class="xh" x1="0" y1="{MT}" x2="0" y2="{H-MB}" stroke="var(--ink2)" opacity="0"/>')
    p.append("</svg>")
    return "".join(p)


def chips(items):
    out = []
    for label, color, dash in items:
        sw = f'<span style="display:inline-block;width:22px;height:0;border-top:3px {"dashed" if dash else "solid"} {color};vertical-align:middle;margin-right:6px"></span>'
        out.append(f'<span style="margin-right:18px;white-space:nowrap">{sw}{label}</span>')
    return '<div style="font-size:12.5px;color:var(--ink2);margin:2px 0 6px">' + "".join(out) + "</div>"


# panel 1: rho_b (raw), window around support
z512 = rb["z"]
ref = rb["ref_raw"]
mask = np.abs(ref) > 0.01 * np.abs(ref).max()
i0, i1 = np.where(mask)[0][[0, -1]]
xwin = (max(0, z512[i0] - 2.5), min(z512[-1], z512[i1] + 2.5))
p1_series = [("DFT (RHOB)", ref, S6, ""),
             ("w1e5 supervised", rb["w1e5_raw"], S1, "2 4"),
             ("unsupervised", rb["unsupervised_raw"], S2, "7 4")]
p1 = svg_chart(z512, p1_series, "z (A)", "rho_b (10^-3 e/A^3)", 1e3, xwin)

# panel 2: plane-averaged net density (model raw grid units -> e/A^3)
zg, zd = rn["z_grid"], rn["z_dft"]
V = A * float(rn["lz"])
nd = rn["nbar_dft"]
nn = np.interp(zd, zg, rn["nbar_model"] / V)
no = np.interp(zd, ro["z_grid"], ro["nbar_model"] / V)
p2_series = [("DFT", nd, S6, ""),
             ("w1e5 supervised", nn, S1, "2 4"),
             ("unsupervised", no, S2, "7 4")]
p2 = svg_chart(zd, p2_series, "z (A)", "net density (10^-3 e/A^3)", 1e3)

# panel 3: phi residual (pred - ref, aligned as in the loss)
zp = rn["z_phi"]
r_new = rn["phi_pred_cmp"] - rn["phi_ref_cmp"]
r_old_full = ro["phi_pred_cmp"] - ro["phi_ref_cmp"]
r_old = np.interp(zp, ro["z_phi"], r_old_full)
p3_series = [("w1e5 supervised", r_new, S1, "2 4"),
             ("unsupervised", r_old, S2, "7 4")]
p3 = svg_chart(zp, p3_series, "z (A)", "phi residual (eV)", 1.0)

rms = lambda v: float(np.sqrt(np.mean(np.asarray(v, float) ** 2)))
tiles = f"""
<div class="tiles">
<div class="tile"><div class="v">{rms(rb['w1e5_smeared']-rb['ref_smeared'])*1e3:.3f} / {rms(rb['unsupervised_smeared']-rb['ref_smeared'])*1e3:.3f}</div>
<div class="l">rho_b RMSE vs DFT: w1e5 / unsupervised (10^-3 e/A^3, this sid)</div></div>
<div class="tile"><div class="v">{rms(nn-nd)*1e3:.2f} / {rms(no-nd)*1e3:.2f}</div>
<div class="l">plane density RMSE vs DFT: w1e5 / unsupervised (10^-3 e/A^3)</div></div>
<div class="tile"><div class="v">{rms(r_new):.4f} / {rms(r_old):.4f}</div>
<div class="l">phi residual rms: w1e5 / unsupervised (eV)</div></div>
</div>"""

html = f"""<meta charset="utf-8">
<title>round 2 verdict, sid {sid}: rho_b + density + potential together</title>
<style>
:root {{ color-scheme: light dark;
  --ink:#171512; --ink2:#5c574f; --grid:#e6e2da; --card:#faf9f6;
  --s1:#2a78d6; --s2:#008300; --s6:#eb6834; }}
@media (prefers-color-scheme: dark) {{ :root {{
  --ink:#f2f0ea; --ink2:#b3aea3; --grid:#33312c; --card:#1e1d1a;
  --s1:#3987e5; --s2:#3ba33b; --s6:#d95926; }} }}
:root[data-theme="dark"] {{
  --ink:#f2f0ea; --ink2:#b3aea3; --grid:#33312c; --card:#1e1d1a;
  --s1:#3987e5; --s2:#3ba33b; --s6:#d95926; }}
:root[data-theme="light"] {{
  --ink:#171512; --ink2:#5c574f; --grid:#e6e2da; --card:#faf9f6;
  --s1:#2a78d6; --s2:#008300; --s6:#eb6834; }}
body {{ margin:0; color:var(--ink); font:15px/1.6 -apple-system,"Segoe UI",sans-serif; }}
.pg {{ max-width:920px; margin:0 auto; padding:24px 16px 56px; }}
h1 {{ font-size:20px; margin:0 0 4px; }}
.sub {{ color:var(--ink2); font-size:13.5px; margin:0 0 18px; }}
.card {{ background:var(--card); border:1px solid var(--grid); border-radius:8px;
  padding:14px 16px 8px; margin:14px 0; }}
h2 {{ font-size:15px; margin:0 0 8px; }}
.tiles {{ display:flex; gap:12px; flex-wrap:wrap; margin:14px 0; }}
.tile {{ background:var(--card); border:1px solid var(--grid); border-radius:8px; padding:10px 16px; }}
.tile .v {{ font-size:19px; font-weight:600; }}
.tile .l {{ font-size:12px; color:var(--ink2); }}
.note {{ font-size:12.5px; color:var(--ink2); margin:6px 0 4px; }}
.tip {{ position:fixed; pointer-events:none; background:var(--card); border:1px solid var(--grid);
  border-radius:6px; padding:4px 8px; font-size:12px; opacity:0; z-index:9; }}
</style>
<div class="pg">
<h1>Round 2 (rho_b weight 1e5, 150 ep): all three curves, sid {sid}</h1>
<p class="sub">w1e5 supervised (exp_pb1d_rhobw2) vs unsupervised (prod400s, 400 ep) vs DFT.
The three quantities are potential-locked: rho_b can only approach DFT together with the density.</p>
{tiles}
<div class="card"><h2>1 &middot; Bound charge rho_b(z)</h2>
{chips([(l,c,bool(d)) for l,_,c,d in p1_series])}{p1}
<p class="note">Raw solver output vs DFT grid average.</p></div>
<div class="card"><h2>2 &middot; Plane-averaged net charge density</h2>
{chips([(l,c,bool(d)) for l,_,c,d in p2_series])}{p2}
<p class="note">The quantity whose error the solvent must screen; supervision reaches it only through the solve.</p></div>
<div class="card"><h2>3 &middot; 1-D potential residual (model - DFT, loss alignment)</h2>
{chips([(l,c,bool(d)) for l,_,c,d in p3_series])}{p3}
<p class="note">Zero line = perfect potential.</p></div>
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
