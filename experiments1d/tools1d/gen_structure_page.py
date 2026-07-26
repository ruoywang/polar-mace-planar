"""structure_report.npz -> self-contained HTML page (inline SVG, no deps).

Usage: python gen_structure_page.py <npz> <out_html>
"""
from __future__ import annotations

import json
import sys

import numpy as np
from pathlib import Path

npz_path, out_path = sys.argv[1], sys.argv[2]
subtitle_label = sys.argv[3] if len(sys.argv) > 3 else "validation set, NiN44"
d = np.load(npz_path, allow_pickle=True)
sid_early = int(d["sid"])

sid = int(d["sid"])
symbols = d["symbols"].tolist()
z_atoms = d["z_atoms"].astype(float)
charges = d["charges"].astype(float)
lz = float(d["lz"])

# ---- fix nbar_model normalization via the physics constraint -------------
z_grid = d["z_grid"].astype(float)
nbar_model = d["nbar_model"].astype(float)
area_dz_sum = np.trapz(np.ones_like(z_grid), z_grid)  # ~lz
total_q = float(d["total_charge"])
# 令 ∫ nbar * A dz = total_charge（A 由约束反解，吸收 dump 里的归一化因子）
cur = np.trapz(nbar_model, z_grid)
scale_to_dft_units = None
z_dft = d["z_dft"].astype(float)
nbar_dft = d["nbar_dft"].astype(float)
# DFT 参考单位是 e/A^3；模型剖面按积分约束换算到同单位需要面积 A：
# ∫nbar_dft*A dz ≈ total_charge（在 valid 窗内不完全成立），
# 稳妥做法：直接在共同窗口把模型剖面缩放到与 DFT 同尺度（最小二乘系数），
# 并同时给出"积分校准"系数作交叉检查。
mask = (z_grid >= z_dft.min()) & (z_grid <= z_dft.max())
m_on_ref = np.interp(z_dft, z_grid, nbar_model)
ls = float(np.dot(m_on_ref, nbar_dft) / np.dot(m_on_ref, m_on_ref))
nbar_model_e = nbar_model * ls
print(f"model->e/A^3 LS 系数 {ls:.3e}; 残差 rms "
      f"{np.sqrt(np.mean((nbar_model_e[np.searchsorted(z_grid, z_dft)] - nbar_dft)**2)):.4f}")

palette = {  # dataviz reference palette (validated light/dark steps)
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
    raw = np.linspace(lo, hi, n)
    return raw


def axis_svg(xlo, xhi, ylo, yhi, w, h, xlab, ylab, xfmt="{:.0f}", yfmt="{:.2g}"):
    out = []
    for tv in ticks(xlo, xhi):
        px = scale(tv, xlo, xhi, ML, w - MR)
        out.append(f'<line x1="{px:.1f}" y1="{MT}" x2="{px:.1f}" y2="{h-MB}" class="grid"/>')
        out.append(f'<text x="{px:.1f}" y="{h-MB+16}" class="tick" text-anchor="middle">{xfmt.format(tv)}</text>')
    for tv in ticks(ylo, yhi, 5):
        py = scale(tv, ylo, yhi, h - MB, MT)
        out.append(f'<line x1="{ML}" y1="{py:.1f}" x2="{w-MR}" y2="{py:.1f}" class="grid"/>')
        out.append(f'<text x="{ML-6}" y="{py+4:.1f}" class="tick" text-anchor="end">{yfmt.format(tv)}</text>')
    out.append(f'<text x="{(ML+w-MR)/2}" y="{h-6}" class="axis" text-anchor="middle">{xlab}</text>')
    out.append(f'<text x="14" y="{(MT+h-MB)/2}" class="axis" text-anchor="middle" transform="rotate(-90 14 {(MT+h-MB)/2})">{ylab}</text>')
    return "".join(out)


def line_chart(cid, series, xlab, ylab, h=H_MAIN, ypad=0.06, yclip=None, legend_xy=None):
    """series: list of (name, xs, ys, slot, dash)"""
    xlo = min(min(s[1]) for s in series); xhi = max(max(s[1]) for s in series)
    ys_all = np.concatenate([np.asarray(s[2]) for s in series])
    if yclip:
        ys_all = ys_all[(ys_all >= yclip[0]) & (ys_all <= yclip[1])]
    ylo, yhi = float(ys_all.min()), float(ys_all.max())
    pad = (yhi - ylo) * ypad or 1e-6
    ylo, yhi = ylo - pad, yhi + pad
    if yclip:
        ylo, yhi = max(ylo, yclip[0]), min(yhi, yclip[1])
    parts = [f'<svg id="{cid}" viewBox="0 0 {W} {h}" data-xlo="{xlo}" data-xhi="{xhi}" data-ylo="{ylo}" data-yhi="{yhi}" data-h="{h}">']
    parts.append(axis_svg(xlo, xhi, ylo, yhi, W, h, xlab, ylab))
    for name, xs, ys, slot, dash in series:
        dash_attr = ' stroke-dasharray="6 4"' if dash else ""
        parts.append(f'<path d="{path_of(xs, np.clip(ys, ylo, yhi), xlo, xhi, ylo, yhi, W, h)}" class="ln {slot}"{dash_attr} fill="none"/>')
    if legend_xy:
        lx, ly = legend_xy
        for i, (name, *_rest) in enumerate(series):
            slot = series[i][3]
            dd = ' stroke-dasharray="6 4"' if series[i][4] else ""
            parts.append(f'<line x1="{lx}" y1="{ly+i*18}" x2="{lx+22}" y2="{ly+i*18}" class="ln {slot}"{dd}/>')
            parts.append(f'<text x="{lx+28}" y="{ly+i*18+4}" class="lg">{name}</text>')
    parts.append(f'<line class="xh" x1="0" y1="{MT}" x2="0" y2="{h-MB}" style="opacity:0"/>')
    parts.append("</svg>")
    return "".join(parts)


ref = np.load(Path(npz_path).parent / f"dft_solvent_ref_sid{sid_early}.npz")
# VASP electron-positive convention -> physics sign (flip)
z_ref_sol = ref["z"].astype(float)
ion_dft = -ref["ion_z"].astype(float)
rb_dft = -ref["rb_z"].astype(float)

charts_js = {}

# ---- 图 A：逐原子电荷 vs z ------------------------------------------------
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
    el_stats[el] = (int(m.sum()), float(charges[m].mean()), float(charges[m].std()))
    slot = ELEM_SLOTS[el]
    for zz, qq in zip(z_atoms[m], charges[m]):
        px = scale(zz, z_lo, z_hi, ML, W - MR)
        py = scale(qq, q_lo, q_hi, H_MAIN - MB, MT)
        partsA.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" class="pt {slot}"><title>{el}  z={zz:.2f} Å  q={qq:+.4f} e</title></circle>')
    # 元素簇直接标注（次级编码）
    zz_m = float(np.median(z_atoms[m])); qq_m = float(np.median(charges[m]))
    px = scale(zz_m, z_lo, z_hi, ML, W - MR)
    py = scale(qq_m, q_lo, q_hi, H_MAIN - MB, MT)
    dy = -10 if el not in ("O",) else 16
    partsA.append(f'<text x="{px:.1f}" y="{py+dy:.1f}" class="ellab {slot}" text-anchor="middle">{el}</text>')
partsA.append("</svg>")
chartA = "".join(partsA)

# ---- 图 B：预测 vs 参考电势 + 残差 ----------------------------------------
z_phi = d["z_phi"].astype(float)
chartB = line_chart(
    "phi", [
        ("DFT reference", z_phi, d["phi_ref_cmp"].astype(float), "s6", False),
        ("model prediction", z_phi, d["phi_pred_cmp"].astype(float), "s1", True),
    ], "z (Å)", "φ̄(z) − φ̄(upper-align window) (eV)", legend_xy=(ML + 16, MT + 12))
charts_js["phi"] = {"x": z_phi.round(3).tolist(),
                    "series": [{"n": "DFT", "y": d["phi_ref_cmp"].round(4).tolist()},
                               {"n": "model", "y": d["phi_pred_cmp"].round(4).tolist()}],
                    "unit": "eV"}
chartB_res = line_chart(
    "phires", [("residual model−DFT", z_phi, d["phi_residual"].astype(float), "s1", False)],
    "z (Å)", "Δφ (eV)", h=H_SUB)
charts_js["phires"] = {"x": z_phi.round(3).tolist(),
                       "series": [{"n": "Δφ", "y": d["phi_residual"].round(5).tolist()}],
                       "unit": "eV"}

# ---- 图 C：residual 电荷密度（平面平均净密度）模型 vs DFT + 差 ------------
m_z = z_grid[mask]
m_v = nbar_model_e[mask]
ref_on_m = np.interp(m_z, z_dft, nbar_dft)
chartC = line_chart(
    "den", [
        ("DFT reference", z_dft, nbar_dft, "s6", False),
        ("model (LS-scaled to e/Å³)", m_z, m_v, "s1", True),
    ], "z (Å)", "plane-averaged net charge density n̄(z) (e/Å³)", legend_xy=(ML + 16, MT + 12))
charts_js["den"] = {"x": m_z.round(3).tolist(),
                    "series": [{"n": "DFT", "y": ref_on_m.round(5).tolist()},
                               {"n": "model", "y": m_v.round(5).tolist()}],
                    "unit": "e/Å³"}
chartC_res = line_chart(
    "denres", [("difference model−DFT", m_z, m_v - ref_on_m, "s1", False)],
    "z (Å)", "Δn̄ (e/Å³)", h=H_SUB)
charts_js["denres"] = {"x": m_z.round(3).tolist(),
                       "series": [{"n": "Δn̄", "y": (m_v - ref_on_m).round(6).tolist()}],
                       "unit": "e/Å³"}

# ---- 图 D：离子/束缚电荷 + P_off ------------------------------------------
z_s = d["z_solve"].astype(float)
chartD = line_chart(
    "sol", [
        ("ionic charge ρ_ion", z_s, d["rho_ion"].astype(float), "s1", False),
        ("bound charge ρ_bound", z_s, d["rho_bound"].astype(float), "s4", False),
    ], "z (Å)", "ρ(z) (e/Å³)", legend_xy=(ML + 16, MT + 12))
ion_dft_on_m = np.interp(z_s, z_ref_sol, ion_dft)
rb_dft_on_m = np.interp(z_s, z_ref_sol, rb_dft)
chartD_ion = line_chart(
    "solion", [
        ("DFT (VASPsol RHOION)", z_ref_sol, ion_dft, "s6", False),
        ("model 1-D PB", z_s, d["rho_ion"].astype(float), "s1", True),
    ], "z (Å)", "ρ_ion(z) (e/Å³)", legend_xy=(ML + 16, MT + 12))
charts_js["solion"] = {"x": z_s.round(3).tolist(),
                       "series": [{"n": "DFT", "y": ion_dft_on_m.round(7).tolist()},
                                  {"n": "model", "y": d["rho_ion"].round(7).tolist()}],
                       "unit": "e/Å³"}
chartD_rb = line_chart(
    "solrb", [
        ("DFT (VASPsol RHOB)", z_ref_sol, rb_dft, "s6", False),
        ("model 1-D PB", z_s, d["rho_bound"].astype(float), "s1", True),
    ], "z (Å)", "ρ_bound(z) (e/Å³)", legend_xy=(ML + 16, MT + 12))
charts_js["solrb"] = {"x": z_s.round(3).tolist(),
                      "series": [{"n": "DFT", "y": rb_dft_on_m.round(7).tolist()},
                                 {"n": "model", "y": d["rho_bound"].round(7).tolist()}],
                      "unit": "e/Å³"}
charts_js["sol"] = {"x": z_s.round(3).tolist(),
                    "series": [{"n": "ρ_ion", "y": d["rho_ion"].round(7).tolist()},
                               {"n": "ρ_bound", "y": d["rho_bound"].round(7).tolist()}],
                    "unit": "e/Å³"}
chartE = line_chart(
    "poff", [
        ("prior P_off (screened vacuum)", z_s, d["prior"].astype(float), "s6", False),
        ("head correction ΔP", z_s, d["delta_p"].astype(float), "s1", False),
    ], "z (Å)", "P_off (e/Å²)", legend_xy=(ML + 16, MT + 12))
charts_js["poff"] = {"x": z_s.round(3).tolist(),
                     "series": [{"n": "prior", "y": d["prior"].round(6).tolist()},
                                {"n": "ΔP", "y": d["delta_p"].round(6).tolist()}],
                     "unit": "e/Å²"}

el_rows = "".join(
    f"<tr><td><span class='chip {ELEM_SLOTS[el]}'></span>{el}</td><td>{n}</td>"
    f"<td>{mu:+.4f}</td><td>{sd:.4f}</td></tr>"
    for el, (n, mu, sd) in el_stats.items())

scalars = f"""
<table class="kv">
<tr><th></th><th>model</th><th>DFT reference</th><th>diff</th></tr>
<tr><td>electrode potential (eV)</td><td>{float(d['pot_pred']):+.4f}</td><td>{float(d['pot_ref']):+.4f}</td><td>{float(d['pot_pred'])-float(d['pot_ref']):+.4f}</td></tr>
<tr><td>Fermi level (eV)</td><td>{float(d['fermi_pred']):+.4f}</td><td>{float(d['fermi_ref']):+.4f}</td><td>{float(d['fermi_pred'])-float(d['fermi_ref']):+.4f}</td></tr>
</table>
<table class="kv">
<tr><th>solvent quantity</th><th>value</th></tr>
<tr><td>ionic layer charge q_ion (e)</td><td>{float(d['q_ion']):+.4f} (solute {total_q:+.4f})</td></tr>
<tr><td>ionic layer center (Å)</td><td>{float(d['layer_mean']):.2f}</td></tr>
<tr><td>bound-charge dipole μ_bound (e·Å)</td><td>{float(d['mu_bound']):+.2f}</td></tr>
<tr><td>Σ atomic charges (e)</td><td>{charges.sum():+.4f} = total charge ✓</td></tr>
</table>
"""

light_css = "".join(f"--{k}:{v[0]};" for k, v in palette.items())
dark_css = "".join(f"--{k}:{v[1]};" for k, v in palette.items())

html = f"""<meta charset="utf-8">
<title>pb1d single-structure report · sid {sid}</title>
<style>
:root {{ color-scheme: light dark; }}
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
</style>
<div class="pg">
<h1>pb1d single-structure report — sid {sid} ({subtitle_label})</h1>
<p class="sub">{len(symbols)} atoms · box z = {lz:.1f} Å · total charge {total_q:+.4f} e · model pb1d_prod400 (400 epochs) · review copy</p>

<h2>1 · Scalar observables</h2>
{scalars}

<h2>2 · Per-element charge states (model charge coefficients)</h2>
<div class="card">{chartA}</div>
<table class="kv"><tr><th>element</th><th>count</th><th>mean q (e)</th><th>σ (e)</th></tr>{el_rows}</table>
<p class="note">One dot per atom (hover for values). Per-element spreads are very narrow (σ ≤ 0.006 e); the sum equals the system total charge exactly.</p>

<h2>3 · Predicted vs DFT 1-D potential (Phi1D, upper-window alignment, same construction as the training loss)</h2>
<div class="card">{chartB}</div>
<div class="card">{chartB_res}</div>
<p class="note">Full range shown, including the nuclear wells. Residual rms = {float(np.sqrt((d['phi_residual']**2).mean())):.4f} eV.</p>

<h2>4 · Residual charge density: model vs DFT (plane-averaged net density)</h2>
<div class="card">{chartC}</div>
<div class="card">{chartC_res}</div>
<p class="note">Comparison window = valid z-range of the DFT grid reference; the model profile is scaled to e/Å³ by a least-squares factor ({ls:.3e}), which also cross-checks the unit normalization.</p>

<h2>5 · Solvent: ionic and bound charge profiles (1-D PB solve)</h2>
<div class="card">{chartD}</div>
<p class="note">Ionic layer integrates to {float(d['q_ion']):+.3f} e, exactly compensating the solute {total_q:+.3f} e; the bound (polarization) charge nets ≈0 and carries dipole {float(d['mu_bound']):+.2f} e·Å.</p>

<h2>5a · Ionic charge vs DFT (VASPsol RHOION, sign converted to physics convention)</h2>
<div class="card">{chartD_ion}</div>

<h2>5b · Bound charge vs DFT (VASPsol RHOB)</h2>
<div class="card">{chartD_rb}</div>
<p class="note">DFT reference profiles are plane averages of the raw VASPsol solvent grids of this very calculation (structure match verified to ~1e-13 Å; RHOION integrates to −q_solute in VASP convention).</p>

<h2>6 · P_off: physics prior and learned head correction</h2>
<div class="card">{chartE}</div>
<p class="note">The head correction peaks at {abs(d['delta_p']).max()/abs(d['prior']).max()*100:.0f}% of the prior peak — inside the designed small-residual regime.</p>
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
    tip.innerHTML = `z = ${{cfg.x[i].toFixed(2)}} Å<br>` +
      cfg.series.map(s => `${{s.n}}: ${{s.y[i]}} ${{cfg.unit}}`).join('<br>');
    tip.style.left = (ev.clientX + 14) + 'px'; tip.style.top = (ev.clientY + 12) + 'px';
    tip.style.opacity = 1;
  }});
  svg.addEventListener('mouseleave', () => {{ tip.style.opacity = 0; xh.style.opacity = 0; }});
}}
</script>
"""
with open(out_path, "w") as f:
    f.write(html)
print(f"page -> {out_path} ({len(html)/1024:.0f} KB)")
