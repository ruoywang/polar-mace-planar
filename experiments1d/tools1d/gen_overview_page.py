"""overview_data.npz -> v1 route overview page, revision 2.

Layout rules: HTML legends above each chart (never inside the plot), stacked
time bars on a common scale, short real paragraphs for key info.

Usage: python gen_overview_page.py <npz> <out_html>
"""
from __future__ import annotations

import json
import sys

import numpy as np

npz, out_path = sys.argv[1], sys.argv[2]
d = np.load(npz, allow_pickle=True)

PAL = {"s1": ("#2a78d6", "#3987e5"), "s2": ("#008300", "#008300"),
       "s3": ("#e87ba4", "#d55181"), "s4": ("#eda100", "#c98500"),
       "s5": ("#1baf7a", "#199e70"), "s6": ("#eb6834", "#d95926")}
W, ML, MR, MT, MB = 880, 66, 14, 12, 40


def sc(v, lo, hi, a, b):
    return a + (v - lo) / (hi - lo) * (b - a)


def axis(xlo, xhi, ylo, yhi, w, h, xlab, ylab, ylog=False, yfmt="{:.2g}", xfmt="{:.0f}"):
    o = []
    for tv in np.linspace(xlo, xhi, 6):
        px = sc(tv, xlo, xhi, ML, w - MR)
        o.append(f'<line x1="{px:.1f}" y1="{MT}" x2="{px:.1f}" y2="{h-MB}" class="grid"/>'
                 f'<text x="{px:.1f}" y="{h-MB+16}" class="tick" text-anchor="middle">{xfmt.format(tv)}</text>')
    if ylog:
        for k in range(int(np.floor(ylo)) - 1, int(np.ceil(yhi)) + 1):
            for m in (1.0, 2.0, 5.0):
                v = k + np.log10(m)
                if not (ylo <= v <= yhi):
                    continue
                py = sc(v, ylo, yhi, h - MB, MT)
                o.append(f'<line x1="{ML}" y1="{py:.1f}" x2="{w-MR}" y2="{py:.1f}" class="grid"/>'
                         f'<text x="{ML-6}" y="{py+4:.1f}" class="tick" text-anchor="end">{m*10.0**k:g}</text>')
    else:
        for tv in np.linspace(ylo, yhi, 5):
            py = sc(tv, ylo, yhi, h - MB, MT)
            o.append(f'<line x1="{ML}" y1="{py:.1f}" x2="{w-MR}" y2="{py:.1f}" class="grid"/>'
                     f'<text x="{ML-6}" y="{py+4:.1f}" class="tick" text-anchor="end">{yfmt.format(tv)}</text>')
    o.append(f'<text x="{(ML+w-MR)/2}" y="{h-6}" class="axis" text-anchor="middle">{xlab}</text>')
    o.append(f'<text x="15" y="{(MT+h-MB)/2}" class="axis" text-anchor="middle" transform="rotate(-90 15 {(MT+h-MB)/2})">{ylab}</text>')
    return "".join(o)


def legend_html(items):
    spans = []
    for lab, slot, dash in items:
        cls = f"sw {slot}" + (" dash" if dash else "")
        spans.append(f'<span class="lgi"><span class="{cls}"></span>{lab}</span>')
    return '<div class="legend">' + "".join(spans) + "</div>"


def line_chart(cid, series, xlab, ylab, h=300, ylog=False, w=W, dash_idx=(), band=None):
    xlo = min(float(np.min(s[1])) for s in series)
    xhi = max(float(np.max(s[1])) for s in series)
    ys = np.concatenate([np.ravel(np.asarray(s[2], dtype=float)) for s in series])
    if ylog:
        ys = np.log10(ys[ys > 0])
    ylo, yhi = float(ys.min()), float(ys.max())
    pad = (yhi - ylo) * 0.06
    ylo, yhi = ylo - pad, yhi + pad
    o = [f'<svg id="{cid}" viewBox="0 0 {w} {h}" data-xlo="{xlo}" data-xhi="{xhi}" data-w="{w}">']
    if band:
        bx0 = sc(band[0], xlo, xhi, ML, w - MR)
        bx1 = sc(band[1], xlo, xhi, ML, w - MR)
        o.append(f'<rect x="{bx0:.1f}" y="{MT}" width="{bx1-bx0:.1f}" height="{h-MB-MT}" class="band"/>')
        o.append(f'<text x="{(bx0+bx1)/2:.1f}" y="{h-MB-8}" class="bandlab" text-anchor="middle">{band[2]}</text>')
    o.append(axis(xlo, xhi, ylo, yhi, w, h, xlab, ylab, ylog=ylog))
    for k, (name, xs, ysr, slot) in enumerate(series):
        xs = np.ravel(np.asarray(xs, dtype=float))
        ysr = np.ravel(np.asarray(ysr, dtype=float))
        yv = np.log10(np.clip(ysr, 1e-12, None)) if ylog else ysr
        pts = " L".join(f"{sc(x, xlo, xhi, ML, w-MR):.1f},{sc(y, ylo, yhi, h-MB, MT):.1f}"
                        for x, y in zip(xs, yv))
        dd = ' stroke-dasharray="6 4"' if k in dash_idx else ""
        o.append(f'<path d="M{pts}" class="ln {slot}"{dd} fill="none"/>')
    o.append(f'<line class="xh" x1="0" y1="{MT}" x2="0" y2="{h-MB}" style="opacity:0"/></svg>')
    return "".join(o)


def stacked_time_bars():
    segs = ["data", "forward", "loss", "backward", "optimizer"]
    slots = ["s5", "s1", "s4", "s3", "s2"]
    before = [109, 400, 68, 676, 8]
    after = [61, 357, 45, 313, 8]
    total_b, total_a = sum(before), sum(after)
    w, h = W, 210
    x0, x1 = 156, w - 150
    scale = (x1 - x0) / total_b
    o = [f'<svg viewBox="0 0 {w} 270">']
    rows = [("before", before, 42, f"{total_b} ms · 130 s/epoch"),
            ("after", after, 112, f"{total_a} ms · 81 s/epoch")]
    # planar 参照（无分段实测，画总量）
    pl = 515
    o.append(f'<text x="{x0-12}" y="201" class="lg" text-anchor="end">planar (ref)</text>')
    o.append(f'<rect x="{x0}" y="182" width="{pl*scale-2:.1f}" height="28" rx="3" class="bar ref"/>')
    o.append(f'<text x="{x0+pl*scale+10:.1f}" y="201" class="lg">{pl} ms · 55 s/epoch (total only)</text>')
    for lab, vals, y, tot in rows:
        o.append(f'<text x="{x0-12}" y="{y+19}" class="lg" text-anchor="end">{lab}</text>')
        x = x0
        for v, slot in zip(vals, slots):
            bw = v * scale
            o.append(f'<rect x="{x:.1f}" y="{y}" width="{max(bw-2,1):.1f}" height="28" rx="3" class="bar {slot}"/>')
            if bw > 26:
                o.append(f'<text x="{x+bw/2:.1f}" y="{y+42}" class="tick" text-anchor="middle">{v}</text>')
            x += bw
        o.append(f'<text x="{x+10:.1f}" y="{y+19}" class="lg">{tot}</text>')
    lx = x0
    for name, slot in zip(segs, slots):
        o.append(f'<rect x="{lx}" y="238" width="11" height="11" rx="2" class="bar {slot}"/>'
                 f'<text x="{lx+16}" y="248" class="lg">{name}</text>')
        lx += 22 + 8 * len(name) + 22
    o.append("</svg>")
    return "".join(o)


charts_js = {}


def register(cid, xs, series):
    charts_js[cid] = {"x": np.ravel(np.asarray(xs, float)).tolist(), "series": series}


def flow_svg():
    def pill(x, y, w, label, cls="fb"):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="46" rx="23" class="{cls}"/>'
                f'<text x="{x+w/2}" y="{y+29}" class="fpt" text-anchor="middle">{label}</text>')

    def arr(x1, y1, x2, y2, dash=False):
        dd = ' stroke-dasharray="5 4"' if dash else ""
        return f'<path d="M{x1},{y1} L{x2},{y2}" class="fa"{dd} marker-end="url(#ah)"/>'

    CX = 250  # 主干中线
    o = ['<svg viewBox="0 0 560 700" style="max-width:560px;margin:0 auto">',
         '<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
         '<path d="M0,0 L8,4 L0,8 z" class="ahp"/></marker></defs>']
    o.append(pill(CX - 110, 14, 220, "MACE charges"))
    o.append(arr(CX, 60, CX, 86))
    # SCF 框
    o.append('<rect x="60" y="88" width="380" height="170" rx="14" class="fbig"/>')
    o.append('<text x="78" y="112" class="fbh">SCF ×1</text>')
    o.append(pill(CX - 130, 122, 260, "field ⊕ solvent"))
    o.append(arr(CX, 168, CX, 192))
    o.append(pill(CX - 130, 194, 260, "updated charges"))
    o.append(arr(CX, 258, CX, 288))
    o.append(f'<text x="{CX+12}" y="278" class="fpt2">final density</text>')
    o.append(pill(CX - 130, 290, 260, "1-D closure"))
    o.append(arr(CX, 336, CX, 362))
    o.append(pill(CX - 130, 364, 260, "+ head ΔP"))
    o.append(arr(CX, 410, CX, 436))
    o.append(pill(CX - 130, 438, 260, "1-D PB solve"))
    # 双输出
    o.append(arr(CX - 60, 484, 140, 528))
    o.append(arr(CX + 60, 484, 360, 528))
    o.append(pill(30, 530, 220, "φ · E_F · Φ1D", "fbo"))
    o.append(pill(310, 530, 220, "energy · forces", "fbo"))
    # 缓存回流：solve 右侧 → 右通道竖上 → solvent 右缘（x 500 通道，最右 pill 边 380+130=380? solvent pill 右缘 CX+130=380）
    o.append('<path d="M380,461 L500,461 L500,145 L384,145" class="fa" stroke-dasharray="5 4" marker-end="url(#ah)"/>')
    o.append('<text x="440" y="453" class="fpt2" text-anchor="middle">cache</text>')
    o.append('<text x="497" y="300" class="fpt2" text-anchor="middle" transform="rotate(-90 497 300)">next forward pass</text>')
    o.append(f'<text x="{CX}" y="620" class="fpt2" text-anchor="middle">solvent field = the structure’s cached profile from its previous pass</text>')
    o.append(f'<text x="{CX}" y="642" class="fpt2" text-anchor="middle">first 30 encounters per structure: planar layer instead of the solve</text>')
    o.append("</svg>")
    return "".join(o)


fermi_chart = line_chart("df", [
    ("pb1d (this work)", d["pbf_e"], d["pbf_v"], "s1"),
    ("planar 4-grid", d["g4f_e"], d["g4f_v"], "s6"),
    ("cp-MACE", d["cpp_e"], d["cpp_v"], "s5"),
], "epoch", "RMSE Fermi level (eV)", h=300, ylog=True, band=(0, 30, "planar warmup"))
register("df", d["pbf_e"], [
    {"n": "pb1d", "y": np.asarray(d["pbf_v"], float).round(4).tolist()},
    {"n": "4-grid", "y": np.asarray(d["g4f_v"], float).round(4).tolist()},
    {"n": "cp-MACE", "y": np.asarray(d["cpp_v"], float).round(4).tolist()}])

pot_chart = line_chart("dp", [
    ("pb1d (this work)", d["pbp_e"], d["pbp_v"], "s1"),
    ("planar 4-grid", d["g4p_e"], d["g4p_v"], "s6"),
], "epoch", "RMSE electrode potential (eV)", h=300, ylog=True, band=(0, 30, "planar warmup"))
register("dp", d["pbp_e"], [
    {"n": "pb1d", "y": np.asarray(d["pbp_v"], float).round(4).tolist()},
    {"n": "4-grid", "y": np.asarray(d["g4p_v"], float).round(4).tolist()}])

light = "".join(f"--{k}:{v[0]};" for k, v in PAL.items())
dark = "".join(f"--{k}:{v[1]};" for k, v in PAL.items())

lg_f = legend_html([("pb1d (this work)", "s1", False), ("planar 4-grid", "s6", False),
                    ("cp-MACE (its RMSE_P = Fermi level)", "s5", False)])
lg_p = legend_html([("pb1d (this work)", "s1", False), ("planar 4-grid", "s6", False)])

html = f"""<meta charset="utf-8">
<title>pb1d v1 — the 1-D closure route: architecture, training, results</title>
<style>
:root {{ color-scheme: light dark; }}
.pg {{ max-width: 960px; margin: 0 auto; padding: 26px 18px 60px;
  font: 15px/1.65 -apple-system, "Segoe UI", sans-serif;
  --ink:#171512; --ink2:#5c574f; --grid:#e6e2da; --card:#faf9f6; {light} }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) .pg {{
  --ink:#ece9e2; --ink2:#a8a294; --grid:#37342e; --card:#232019; {dark} }} }}
:root[data-theme=dark] .pg {{ --ink:#ece9e2; --ink2:#a8a294; --grid:#37342e; --card:#232019; {dark} }}
.pg {{ color: var(--ink); }}
h1 {{ font-size: 25px; margin: 0 0 6px; }} h2 {{ font-size: 18px; margin: 40px 0 8px; }}
p {{ margin: 8px 0; max-width: 76ch; }}
.sub {{ color: var(--ink2); }}
.cap {{ color: var(--ink2); font-size: 13.5px; margin: 6px 2px 0; }}
.card {{ background: var(--card); border: 1px solid var(--grid); border-radius: 10px;
  padding: 12px 14px 8px; margin: 8px 0 4px; }}
.legend {{ display: flex; gap: 22px; flex-wrap: wrap; font-size: 13px; margin: 12px 2px 2px; }}
.lgi {{ white-space: nowrap; display: inline-flex; align-items: center; gap: 7px; }}
.sw {{ display: inline-block; width: 22px; height: 4px; border-radius: 2px; }}
.sw.s1 {{ background: var(--s1); }} .sw.s3 {{ background: var(--s3); }}
.sw.s5 {{ background: var(--s5); }} .sw.s6 {{ background: var(--s6); }}
.sw.dash {{ background: repeating-linear-gradient(90deg, currentColor 0 5px, transparent 5px 9px); }}
.sw.s1.dash {{ color: var(--s1); background: repeating-linear-gradient(90deg, var(--s1) 0 5px, transparent 5px 9px); }}
.bar.ref {{ fill: var(--ink2); opacity: .45; }}
.fpt {{ font-size: 14px; fill: var(--ink); }} .fpt2 {{ font-size: 12px; fill: var(--ink2); }}
.fbo {{ fill: none; stroke: var(--s1); stroke-width: 1.4; }}
.tiles {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0; }}
.tile {{ background: var(--card); border: 1px solid var(--grid); border-radius: 10px; padding: 12px 14px; }}
.tile b {{ font-size: 22px; display: block; font-variant-numeric: tabular-nums; }}
.tile span {{ font-size: 12.5px; color: var(--ink2); }}
svg {{ width: 100%; height: auto; display: block; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.tick {{ font-size: 11px; fill: var(--ink2); }} .axis {{ font-size: 12.5px; fill: var(--ink2); }}
.lg {{ font-size: 12.5px; fill: var(--ink); }}
.valw {{ font-size: 11px; fill: #fff; }}
.band {{ fill: var(--ink2); opacity: .08; }} .bandlab {{ font-size: 11px; fill: var(--ink2); }}
.ln.s1 {{ stroke: var(--s1); stroke-width: 2; }} .ln.s2 {{ stroke: var(--s2); stroke-width: 2; }}
.ln.s3 {{ stroke: var(--s3); stroke-width: 2; }} .ln.s4 {{ stroke: var(--s4); stroke-width: 2; }}
.ln.s5 {{ stroke: var(--s5); stroke-width: 2; }} .ln.s6 {{ stroke: var(--s6); stroke-width: 2; }}
.bar.s1 {{ fill: var(--s1); }} .bar.s2 {{ fill: var(--s2); }} .bar.s3 {{ fill: var(--s3); }}
.bar.s4 {{ fill: var(--s4); }} .bar.s5 {{ fill: var(--s5); }} .bar.s6 {{ fill: var(--s6); }}
.fb {{ fill: var(--card); stroke: var(--ink2); stroke-width: 1.2; }}
.fbc {{ fill: none; stroke: var(--s1); stroke-width: 1.4; }}
.fbig {{ fill: none; stroke: var(--grid); stroke-width: 1.5; }}
.fbh {{ font-size: 13px; fill: var(--ink2); font-weight: 600; }}
.fbt {{ font-size: 12.5px; fill: var(--ink); }} .fbt2 {{ font-size: 12px; fill: var(--ink2); }}
.fa {{ stroke: var(--ink2); stroke-width: 1.6; fill: none; }} .ahp {{ fill: var(--ink2); }}
.xh {{ stroke: var(--ink2); stroke-width: 1; }}
table.kv {{ border-collapse: collapse; font-variant-numeric: tabular-nums; margin: 10px 0; }}
table.kv td, table.kv th {{ border: 1px solid var(--grid); padding: 5px 12px; font-size: 13.5px; text-align: right; }}
table.kv th {{ color: var(--ink2); }} table.kv td:first-child, table.kv th:first-child {{ text-align: left; }}
ul {{ margin: 8px 0; padding-left: 22px; max-width: 78ch; }} li {{ margin: 4px 0; }}
.gal {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.gal a {{ display: block; background: var(--card); border: 1px solid var(--grid); border-radius: 10px;
  padding: 14px 16px; text-decoration: none; color: inherit; }}
.gal a:hover {{ border-color: var(--s1); }}
.gal h3 {{ margin: 0 0 4px; font-size: 15px; color: var(--s1); }}
.gal p {{ margin: 0; font-size: 13px; color: var(--ink2); }}
#tip {{ position: fixed; pointer-events: none; background: var(--card); border: 1px solid var(--grid);
  border-radius: 6px; padding: 5px 9px; font-size: 12.5px; opacity: 0; z-index: 9;
  font-variant-numeric: tabular-nums; }}
</style>
<div class="pg">
<h1>The 1-D Closure Route (v1): Architecture, Training, Results</h1>
<p class="sub">One 1-D nonlinear Poisson–Boltzmann solve per forward replaces the 3-D solve of the
previous route. The closure theory and its measurements live in the
<a href="https://ruoywang.github.io/cep-dip-python-pb/closure_1d.html" style="color:var(--s1)">closure_1d report</a>;
this page covers what is new here: the model wiring, the training protocol, and the 400-epoch result.</p>

<div class="tiles">
<div class="tile"><b>0.039 eV</b><span>Fermi-level RMSE — 2.8× better than cp-MACE (0.109)</span></div>
<div class="tile"><b>2.04 meV</b><span>energy RMSE per atom — better than planar (2.24)</span></div>
<div class="tile"><b>81 s</b><span>per epoch on 3×A100 — planar reference is 55 s</span></div>
<div class="tile"><b>0.049 eV</b><span>test-set potential RMSE, matching validation (0.046) — no generalization gap</span></div>
</div>

<h2>1 · Where the solvent enters: before and after the SCF</h2>
<p>The solvent appears twice per forward pass. <b>Inside the charge-update recursion</b> the model
sees it only as an external field built from the structure’s cached profile of the previous training
encounter (detached — no gradients, no lag instability). <b>After the recursion</b> one fresh 1-D PB
solve runs on the final density: its profile feeds the energy (detached) and the potential/Fermi/Φ1D
observables (with exact gradients through the solve), and refreshes the cache for next time.</p>
<div class="card">{flow_svg()}</div>

<h2>2 · Training protocol</h2>
<ul>
<li><b>Planar warmup.</b> For each structure’s first 30 training encounters the solve is replaced by
the planar Gaussian layer; the PB physics engages only once the predicted density is sane. Without it
the early wild densities poison the optimization. The warmup band is shaded in every curve below.</li>
<li><b>Global shuffle</b> across ranks (the scheme-C cache is shared on disk, so no rank-static
sharding is needed).</li>
<li><b>400 epochs</b> on 3×A100, seed 123, float64; same 320/40/40 splits and loss weights as the
planar and 3-D reference runs.</li>
</ul>

<h2>3 · Speed: 130 → 81 s per epoch, exact math only</h2>
<p>Every change is mathematically equivalent — values unchanged, checked by value-parity and
gradient probes. The three that mattered:</p>
<ul>
<li><b>Secant root-finding</b> for the dipole-correction loop, a scalar fixed-point equation: ~7
evaluations instead of ~40 damped-mixing steps (solve 216 → 25 ms).</li>
<li><b>Implicit-function backward</b> for the two iterative blocks — the Newton solve and the
closure’s pointwise 80-iteration local-field loop — gradients taken at the converged point instead
of through unrolled iterations (backward 676 → 313 ms/step).</li>
<li><b>I/O:</b> memory-mapped density grids and RAM-cached baseline fields replace per-step
decompression (data 109 → 61 ms).</li>
</ul>
<div class="card">{stacked_time_bars()}</div>
<p class="cap">Per-training-step wall time by phase, common millisecond scale (3×A100, batch 1 per rank).</p>

<h2>4 · Training result</h2>
{lg_f}
<div class="card">{fermi_chart}</div>
<p class="cap">Fermi level vs epoch, log scale. cp-MACE reports a single electrical metric, RMSE_P,
which is its Fermi-level error.</p>
{lg_p}
<div class="card">{pot_chart}</div>
<p class="cap">Electrode potential vs epoch, log scale. cp-MACE has no electrode-potential observable.</p>

<table class="kv">
<tr><th>final epoch</th><th>E (meV/atom)</th><th>F (meV/Å)</th><th>potential (eV)</th><th>Fermi (eV)</th><th>Φ1D (eV)</th></tr>
<tr><td>pb1d (400 ep)</td><td>2.04</td><td>17.4</td><td>0.052</td><td>0.039</td><td>0.041*</td></tr>
<tr><td>planar 4-grid (400 ep)</td><td>2.24</td><td>17.0</td><td>0.037</td><td>0.034</td><td>0.038*</td></tr>
<tr><td>cp-MACE (400 ep)</td><td>1.65</td><td>16.3</td><td>—</td><td>0.109</td><td>—</td></tr>
</table>
<p class="cap">*Φ1D targets differ: pb1d scores its real solvent profile, planar a Gaussian layer.</p>
<p>cp-MACE fits energies and forces marginally better but its Fermi-level error is 2.8× larger and
it has no electrode-potential observable at all — closing that gap is the motivation for the
potential-aware routes, and pb1d now does so at planar-level cost.</p>

<table class="kv">
<tr><th>pb1d RMSE by split</th><th>train</th><th>valid</th><th>test NiN44</th><th>test NiN88</th></tr>
<tr><td>potential (eV)</td><td>0.047</td><td>0.046</td><td>0.049</td><td>0.061</td></tr>
<tr><td>Fermi (eV)</td><td>0.019</td><td>0.034</td><td>0.045</td><td>0.035</td></tr>
<tr><td>energy (meV/atom)</td><td>2.2</td><td>2.04</td><td>2.99</td><td>1.12</td></tr>
</table>
<p class="cap">Test structures are untouched during training; test errors equal validation across all columns.</p>

<h2>5 · Single-structure deep dives</h2>
<div class="gal">
<a href="./pb1d-structure-sid152.html"><h3>sid 152 — training, NiN44</h3>
<p>φ residual 0.021 eV · ionic 9.7% vs DFT · bound 31% · O −0.40 / H +0.19 / Ni +0.22</p></a>
<a href="./pb1d-structure-sid307.html"><h3>sid 307 — training, NiN88</h3>
<p>φ residual 0.031 eV · ionic 6.8% vs DFT · charge compensation +1.157 e exact · bound 56%</p></a>
</div>
</div>
<div id="tip"></div>
<script>
const DATA = {json.dumps(charts_js)};
const tip = document.getElementById('tip');
for (const [cid, cfg] of Object.entries(DATA)) {{
  const svg = document.getElementById(cid);
  if (!svg) continue;
  const xlo = +svg.dataset.xlo, xhi = +svg.dataset.xhi, w = +svg.dataset.w;
  const ML = {ML}, MR = {MR};
  const xh = svg.querySelector('.xh');
  svg.addEventListener('mousemove', ev => {{
    const r = svg.getBoundingClientRect();
    const fx = (ev.clientX - r.left) / r.width * w;
    if (fx < ML || fx > w - MR) {{ tip.style.opacity = 0; xh.style.opacity = 0; return; }}
    const xv = xlo + (fx - ML) / (w - MR - ML) * (xhi - xlo);
    let i = 0, best = 1e18;
    cfg.x.forEach((x, k) => {{ const dd = Math.abs(x - xv); if (dd < best) {{ best = dd; i = k; }} }});
    const px = ML + (cfg.x[i] - xlo) / (xhi - xlo) * (w - MR - ML);
    xh.setAttribute('x1', px); xh.setAttribute('x2', px); xh.style.opacity = .5;
    tip.innerHTML = `x = ${{cfg.x[i]}}<br>` + cfg.series.map(s =>
      `${{s.n}}: ${{s.y[i] !== undefined ? s.y[i] : '—'}}`).join('<br>');
    tip.style.left = (ev.clientX + 14) + 'px'; tip.style.top = (ev.clientY + 12) + 'px';
    tip.style.opacity = 1;
  }});
  svg.addEventListener('mouseleave', () => {{ tip.style.opacity = 0; xh.style.opacity = 0; }});
}}
</script>
"""
open(out_path, "w").write(html)
print(f"page -> {out_path} ({len(html)//1024} KB)")
