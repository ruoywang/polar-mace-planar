"""Report page: the solvent effective-potential input (v_new = d(A_cav + A_diel + A_ion)/dn_e at the
stage-1 field) -- theory, implementation, every pre-training check, the training gate, the charge
response diagnosis, the Phi1D construction fix, and the final changes. Static HTML with hand-drawn
SVG figures; the only script is the hover read-out.

  python vsolv_report_page.py OUT.html
Reads: 3-residual_3D/{gate_vsolv,prod500_w1000_ref,w1000_ref}/run.log, claude/2-1D_PB/exp_vsolv/logs/
{density_profile,ebl_audit}_{gate_vsolv,prod500_w1000_ref}_e{25,30,33}.npz and cr_all.json,
claude/2-1D_PB/exp_pair122_prod500/structure_pair_sid122_722{,_fix}.npz. Numbers that live only in
RUNS.md (acceptance rounds, probes) are transcribed below with their job ids.
"""
import html
import json
import math
import re
import sys

import numpy as np

B = "/scratch/08384/tg876840/tmp/c-MACEsol"
L = f"{B}/claude/2-1D_PB/exp_vsolv/logs"
OUT = sys.argv[1]

# ---------------------------------------------------------------- data: validation logs
KEYS = [("loss", r"loss=([0-9.]+)"), ("E", r"RMSE_E_per_atom=\s*([0-9.]+)"), ("F", r"RMSE_F=\s*([0-9.]+)"),
        ("pot", r"RMSE_potential=([0-9.]+)"), ("fermi", r"RMSE_fermi=([0-9.]+)"), ("dens3d", r"RMSE_density_3d=([0-9.]+)"),
        ("Phi1D", r"RMSE_potential_1d_profile=([0-9.]+)"), ("rho_b", r"RMSE_solvent3d_b=([0-9.]+)"), ("occ", r"RMSE_occ_aug=([0-9.]+)")]
KEY_CN = {"loss": "loss", "E": "E (meV/atom)", "F": "F (meV/Å)", "pot": "电极电势", "fermi": "费米能级", "dens3d": "3D 密度",
          "Phi1D": "Φ1D", "rho_b": "ρ_b", "occ": "占据数 occ_aug"}


def parse_log(path):
    out = {}
    for line in open(path, errors="ignore"):
        m = re.search(r"INFO: Epoch (\d+): ", line)
        if not m:
            continue
        row = {}
        for k, pat in KEYS:
            mm = re.search(pat, line)
            row[k] = float(mm.group(1)) if mm else float("nan")
        out[int(m.group(1))] = row
    return out


gate = parse_log(f"{B}/3-residual_3D/gate_vsolv/run.log")
prod = parse_log(f"{B}/3-residual_3D/prod500_w1000_ref/run.log")
ref = parse_log(f"{B}/3-residual_3D/w1000_ref/run.log")
EP = [e for e in range(20, 34) if e in gate and e in prod and e in ref]
ratio_g = {k: [gate[e][k] / prod[e][k] for e in EP] for k, _ in KEYS}
ratio_r = {k: [ref[e][k] / prod[e][k] for e in EP] for k, _ in KEYS}
outside = {k: sum(1 for v in ratio_g[k] if v > max(ratio_r[k]) or v < min(ratio_r[k])) for k, _ in KEYS}

# ---------------------------------------------------------------- data: structural npz
def mean(x):
    x = [v for v in x if v == v]
    return sum(x) / len(x) if x else float("nan")


def rms(x):
    x = [v for v in x if v == v]
    return math.sqrt(sum(v * v for v in x) / len(x)) if x else float("nan")


def struct_numbers(tag):
    d = np.load(f"{L}/density_profile_{tag}.npz", allow_pickle=True)
    e = np.load(f"{L}/ebl_audit_{tag}.npz", allow_pickle=True)
    fr = {int(r["sid"]): r for r in d["frames"]}; pr = {int(r["k"]): r for r in d["pairs"]}
    eb = {int(r["sid"]): r for r in e["frames"]}
    out = {}
    for st, sids in (("charged", [s for s in fr if fr[s]["state"].startswith("charged")]), ("neutral", [s for s in fr if fr[s]["state"].startswith("neutral")])):
        out[f"F_{st}"] = rms([eb[s]["F_rms_err_meV"] for s in sids if s in eb])
        out[f"E_{st}"] = rms([1e3 * (eb[s]["E_model"] - eb[s]["E_dft"]) / eb[s]["nat"] for s in sids if s in eb])
        out[f"solvL1_{st}"] = mean([eb[s].get("rho_solv_L1_model_vs_dft", np.nan) for s in sids if s in eb])
        out[f"rmse3d_{st}"] = mean([fr[s]["rmse3d"] for s in sids])
        out[f"tailbot_{st}"] = mean([fr[s]["tail_bottom_L1_e"] for s in sids])
        out[f"tailtop_{st}"] = mean([fr[s]["tail_top_L1_e"] for s in sids])
        out[f"winL1_{st}"] = mean([fr[s]["L1_window_e"] for s in sids])
    out["dnL1"] = mean([pr[k]["L1_e"] for k in pr])
    return out


ST = {(arm, ep): struct_numbers(f"{run}_e{ep}") for arm, run in (("gate", "gate_vsolv"), ("prod", "prod500_w1000_ref")) for ep in (25, 30, 33)}
STRUCT_ROWS = [("力 rmse，带电", "F_charged"), ("力 rmse，中性", "F_neutral"), ("3D 密度 rmse，带电", "rmse3d_charged"), ("3D 密度 rmse，中性", "rmse3d_neutral"),
               ("真空侧密度尾部 L1，带电", "tailbot_charged"), ("真空侧密度尾部 L1，中性", "tailbot_neutral"), ("溶剂侧密度尾部 L1，带电", "tailtop_charged"),
               ("溶剂侧密度尾部 L1，中性", "tailtop_neutral"), ("面平均窗口 L1，带电", "winL1_charged"), ("面平均窗口 L1，中性", "winL1_neutral"),
               ("电荷响应 dn(z) L1（47 对）", "dnL1"), ("溶剂剖面 L1，带电", "solvL1_charged"), ("溶剂剖面 L1，中性", "solvL1_neutral"),
               ("E rmse，带电", "E_charged"), ("E rmse，中性", "E_neutral")]

# ---------------------------------------------------------------- data: charge response
CR = json.load(open(f"{L}/cr_all.json"))
cks = {c["label"]: c for c in CR["checkpoints"]}
prod_cks = sorted([c for c in CR["checkpoints"] if c["flag_in_training"] == "OFF"], key=lambda c: c["epoch"])
RG = CR["regions"]
regmean = {k: mean([r[k] for r in RG.values()]) for k in ("b1", "b2", "b3", "b4", "z_sheet_min", "z_sheet_max")}
regmean["z_adsorbate_C"] = [mean([r["z_adsorbate_C"][0] for r in RG.values() if r["z_adsorbate_C"]])]

# ---------------------------------------------------------------- data: Phi1D fix (pair 722)
pb = np.load(f"{B}/claude/2-1D_PB/exp_pair122_prod500/structure_pair_sid122_722.npz", allow_pickle=True)
pf = np.load(f"{B}/claude/2-1D_PB/exp_pair122_prod500/structure_pair_sid122_722_fix.npz", allow_pickle=True)
zphi = pb["z_phi_n"].astype(float)
res_before = (pb["phi_pred_cmp_n"] - pb["phi_ref_cmp_n"]).astype(float)
res_after = (pf["phi_pred_cmp_n"] - pf["phi_ref_cmp_n"]).astype(float)
res_before_c = (pb["phi_pred_cmp"] - pb["phi_ref_cmp"]).astype(float)
res_after_c = (pf["phi_pred_cmp"] - pf["phi_ref_cmp"]).astype(float)
mu_b = float(pb["mu_bound_n"]); A_CELL = 189.745476216; H_CELL = 45.0; K_C = 14.3996
step_pred = 4 * math.pi * K_C * mu_b / A_CELL; ramp_pred = -step_pred / H_CELL
def at(z, v, zz): return float(v[int(round(zz / (z[1] - z[0])))])
m2444 = (zphi >= 24) & (zphi <= 44.5)
obs_ramp_b = float(np.polyfit(zphi[m2444], res_before[m2444], 1)[0]); obs_ramp_a = float(np.polyfit(zphi[m2444], res_after[m2444], 1)[0])
obs_step_b = at(zphi, res_before, 18) - at(zphi, res_before, 2); obs_step_a = at(zphi, res_after, 18) - at(zphi, res_after, 2)

# ---------------------------------------------------------------- SVG helpers (reference palette classes s1..s8)
W, H = 960, 430
ML, MR, MT, MB = 72, 24, 58, 56
charts_js = {}


def nice_ticks(lo, hi, n=6):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** int(f"{raw:e}".split("e")[1])
    step = mag
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= n:
            break
    v = (int(lo / step) - 1) * step; t = []
    while v <= hi + 1e-12:
        if v >= lo - 1e-12:
            t.append(round(v, 10))
        v += step
    return t


def fmt(v):
    return f"{v:g}" if abs(v) < 1e4 else f"{v:.2e}"


def tw(text, px_ascii=7.0, px_cjk=13.0):
    return sum(px_cjk if ord(ch) > 0x2E7F else px_ascii for ch in text)


def line_chart(cid, series, xlabel, ylabel, regions=None, y0line=True, bands=None, xr=None, yr=None, unit="", hlines=None, xticks=None):
    """series: dict(x, y, label, cls, dash). bands: list of (x, ylo, yhi, cls) shaded ranges. hlines: [(y, label, cls)]."""
    xs = [v for s in series for v in s["x"]]; ys = [v for s in series for v in s["y"]]
    if bands:
        for bx, blo, bhi, _ in bands:
            ys += list(blo) + list(bhi)
    if hlines:
        ys += [h[0] for h in hlines]
    x0, x1 = (min(xs), max(xs)) if xr is None else xr
    if xr is None and regions is None:
        xp = 0.025 * (x1 - x0); x0 -= xp; x1 += xp
    ylo, yhi = (min(ys), max(ys)) if yr is None else yr
    if y0line:
        ylo, yhi = min(ylo, 0.0), max(yhi, 0.0)
    pad = 0.06 * (yhi - ylo if yhi > ylo else 1.0); ylo -= pad; yhi += pad
    pw, ph = W - ML - MR, H - MT - MB
    def X(v): return ML + (v - x0) / (x1 - x0) * pw
    def Y(v): return MT + (yhi - v) / (yhi - ylo) * ph
    o = [f'<svg id="{cid}" viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{html.escape(ylabel)}" class="fig" data-xlo="{x0}" data-xhi="{x1}">']
    if regions:
        for a, b, nm, lab in ((x0, regions["b1"], "vac", ""), (regions["b1"], regions["b2"], "el", "电极"), (regions["b2"], regions["b3"], "if", "界面"),
                              (regions["b3"], regions["b4"], "wa", "水层"), (regions["b4"], x1, "vac", "")):
            a, b = max(a, x0), min(b, x1)
            if b > a:
                o.append(f'<rect x="{X(a):.1f}" y="{MT}" width="{X(b)-X(a):.1f}" height="{ph}" class="band {nm}"/>')
                if lab:
                    o.append(f'<text x="{(X(a)+X(b))/2:.1f}" y="{MT-8}" class="bandlbl">{lab}</text>')
    if bands:
        for bx, blo, bhi, cls in bands:
            pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(bx, bhi)) + " " + " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(reversed(list(bx)), reversed(list(blo))))
            o.append(f'<polygon points="{pts}" class="rangeband {cls}"/>')
    for t in (xticks if xticks is not None else nice_ticks(x0, x1, 8)):
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{MT}" y2="{MT+ph}" class="grid"/>')
        o.append(f'<text x="{X(t):.1f}" y="{MT+ph+18}" class="tick">{fmt(t)}</text>')
    for t in nice_ticks(ylo, yhi, 6):
        o.append(f'<line x1="{ML}" x2="{ML+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>')
        o.append(f'<text x="{ML-8}" y="{Y(t)+4:.1f}" class="tick r">{fmt(t)}</text>')
    if y0line and ylo < 0 < yhi:
        o.append(f'<line x1="{ML}" x2="{ML+pw}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="zero"/>')
    if hlines:
        for yv, lab, cls in hlines:
            o.append(f'<line x1="{ML}" x2="{ML+pw}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" class="hline {cls}"/>')
            o.append(f'<text x="{ML+pw-4}" y="{Y(yv)-5:.1f}" class="hlbl r">{html.escape(lab)}</text>')
    o.append(f'<rect x="{ML}" y="{MT}" width="{pw}" height="{ph}" class="frame"/>')
    for s in series:
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(s["x"], s["y"]))
        dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
        o.append(f'<polyline points="{pts}" class="line {s["cls"]}"{dash}/>')
        if s.get("dots"):
            for a, b in zip(s["x"], s["y"]):
                o.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="4" class="dot {s["cls"]}"/>')
    lx = ML + pw
    for s in reversed(series):
        lx -= tw(s["label"]) + 34
        dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
        o.append(f'<line x1="{lx}" x2="{lx+22}" y1="{MT-34}" y2="{MT-34}" class="line {s["cls"]}"{dash}/>')
        o.append(f'<text x="{lx+27}" y="{MT-30}" class="lgd">{html.escape(s["label"])}</text>')
    o.append(f'<text x="{ML+pw/2:.1f}" y="{H-14}" class="axis">{html.escape(xlabel)}</text>')
    o.append(f'<text transform="translate(16,{MT+ph/2:.1f}) rotate(-90)" class="axis">{html.escape(ylabel)}</text>')
    o.append(f'<line class="xh" x1="0" x2="0" y1="{MT}" y2="{MT+ph}" style="opacity:0"/></svg>')
    xs0 = list(series[0]["x"])
    charts_js[cid] = {"x": [round(float(v), 4) for v in xs0], "unit": unit,
                      "series": [{"n": s["label"], "y": [round(float(v), 5) for v in np.interp(xs0, np.asarray(s["x"], float), np.asarray(s["y"], float))]} for s in series]}
    return "\n".join(o)


def bar_chart(cid, groups, ylabel, unit="", hline=None):
    """groups: list of (label, value, cls). Vertical bars, 4px rounded top, value labels."""
    n = len(groups); pw, ph = W - ML - MR, H - MT - MB
    vals = [g[1] for g in groups]; yhi = max(vals + ([hline[0]] if hline else [])) * 1.15; ylo = 0.0
    def Y(v): return MT + (yhi - v) / (yhi - ylo) * ph
    slot = pw / n; bw = min(64, slot * 0.6)
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{html.escape(ylabel)}" class="fig">']
    for t in nice_ticks(ylo, yhi, 6):
        o.append(f'<line x1="{ML}" x2="{ML+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>')
        o.append(f'<text x="{ML-8}" y="{Y(t)+4:.1f}" class="tick r">{fmt(t)}</text>')
    if hline:
        o.append(f'<line x1="{ML}" x2="{ML+pw}" y1="{Y(hline[0]):.1f}" y2="{Y(hline[0]):.1f}" class="hline s6"/>')
        o.append(f'<text x="{ML+pw-4}" y="{Y(hline[0])-5:.1f}" class="hlbl r">{html.escape(hline[1])}</text>')
    for i, (lab, v, cls) in enumerate(groups):
        cx = ML + slot * (i + 0.5); top = Y(v); base = Y(0)
        o.append(f'<rect x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{max(base-top,0):.1f}" rx="4" class="bar {cls}"><title>{html.escape(lab)}: {v:g} {unit}</title></rect>')
        o.append(f'<text x="{cx:.1f}" y="{top-6:.1f}" class="val">{v:g}</text>')
        for j, part in enumerate(lab.split("\n")):
            o.append(f'<text x="{cx:.1f}" y="{base+18+j*14:.1f}" class="tick">{html.escape(part)}</text>')
    o.append(f'<line x1="{ML}" x2="{ML+pw}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="zero"/>')
    o.append(f'<text transform="translate(16,{MT+ph/2:.1f}) rotate(-90)" class="axis">{html.escape(ylabel)}</text></svg>')
    return "\n".join(o)


def dot_rows(cid, rows, cols, xlabel, unit="%"):
    """rows: list of (label, {col: value}); cols: list of (name, cls). Horizontal dot plot, one row per quantity."""
    n = len(rows); rowh = 26; Hd = MT + n * rowh + MB
    pw = W - 300 - MR; ml = 300
    vals = [v for _, d in rows for v in d.values()]
    x0, x1 = min(min(vals), 0.0), max(max(vals), 0.0); pad = 0.08 * (x1 - x0); x0 -= pad; x1 += pad
    def X(v): return ml + (v - x0) / (x1 - x0) * pw
    o = [f'<svg viewBox="0 0 {W} {Hd}" width="100%" role="img" aria-label="{html.escape(xlabel)}" class="fig">']
    for t in nice_ticks(x0, x1, 8):
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{MT}" y2="{MT+n*rowh}" class="grid"/>')
        o.append(f'<text x="{X(t):.1f}" y="{MT+n*rowh+18}" class="tick">{fmt(t)}</text>')
    o.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{MT}" y2="{MT+n*rowh}" class="zero"/>')
    for i, (lab, d) in enumerate(rows):
        cy = MT + rowh * (i + 0.5)
        if i % 2 == 0:
            o.append(f'<rect x="{ml}" y="{cy-rowh/2:.1f}" width="{pw}" height="{rowh}" class="rowband"/>')
        o.append(f'<text x="{ml-10}" y="{cy+4:.1f}" class="rowlbl r">{html.escape(lab)}</text>')
        for name, cls in cols:
            if name in d and d[name] == d[name]:
                o.append(f'<circle cx="{X(d[name]):.1f}" cy="{cy:.1f}" r="5.5" class="dot {cls}"><title>{html.escape(lab)} · {html.escape(name)}: {d[name]:+.1f}{unit}</title></circle>')
    lx = ml
    for name, cls in cols:
        o.append(f'<circle cx="{lx+6}" cy="{MT-30}" r="5.5" class="dot {cls}"/>')
        o.append(f'<text x="{lx+16}" y="{MT-26}" class="lgd">{html.escape(name)}</text>'); lx += tw(name) + 40
    o.append(f'<text x="{ml+pw/2:.1f}" y="{Hd-12}" class="axis">{html.escape(xlabel)}</text></svg>')
    return "\n".join(o)


def pipeline_svg():
    """Hand-drawn flow: where v_new sits in the two-stage charge recursion."""
    bx = [("原子特征 + 电荷/密度头", 30, 40, 200, 56, "n"), ("第一阶段：1D PB 求解（不用头）\nφ*(z)，空腔 S_cav、S_diel、S_ion", 270, 40, 300, 56, "n"),
          ("反应电势 −e φ_solv\n（已有输入：补偿特征）", 610, 40, 320, 56, "old"),
          ("A_cav + A_diel + A_ion 在 φ = φ* 处\n对 n_e 求偏导 → v_new(r)", 270, 150, 300, 56, "new"),
          ("投影：接收高斯 σ_k 处的值与梯度\n×VSOLV_SIGN(−1)，不解泊松、不去均值", 610, 150, 320, 56, "new"),
          ("第二阶段：带头的新鲜 PB 求解\n能量、力、电势、费米、密度", 270, 260, 300, 56, "n"),
          ("训练损失（E/F/电势/费米/Φ1D/3D 密度/占据数）", 30, 370, 900, 40, "n")]
    o = ['<svg viewBox="0 0 960 430" width="100%" role="img" aria-label="流程图" class="fig diag">']
    o.append('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="arrhead"/></marker></defs>')
    for t, x, y, w, h, cls in bx:
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="box {cls}"/>')
        for i, line in enumerate(t.split("\n")):
            o.append(f'<text x="{x+w/2}" y="{y+h/2+(i-(len(t.split(chr(10)))-1)/2)*16+5}" class="boxtxt">{html.escape(line)}</text>')
    def arrow(x1, y1, x2, y2, cls="n"):
        o.append(f'<path d="M{x1},{y1} L{x2},{y2}" class="arrow {cls}" marker-end="url(#arr)"/>')
    arrow(230, 68, 268, 68); arrow(570, 68, 608, 68); arrow(420, 96, 420, 148, "new"); arrow(570, 178, 608, 178, "new")
    arrow(770, 96, 770, 148, "old"); arrow(770, 206, 770, 230, "old"); arrow(420, 206, 420, 258, "new")
    o.append('<path d="M770,230 L420,230" class="arrow old" marker-end="url(#arr)"/>')
    arrow(420, 316, 420, 368, "n"); arrow(130, 96, 130, 368, "n")
    o.append('<text x="440" y="122" class="lbl new">新增（可选开关 solvent_pb1d_vsolv_input）</text>')
    o.append('<text x="790" y="122" class="lbl old">原有的溶剂通道</text>')
    o.append('<text x="30" y="420" class="note">两条溶剂输入都用第一阶段的（滞后一步的）场，不加外层自洽循环。</text></svg>')
    return "\n".join(o)


# ---------------------------------------------------------------- figures
figs = []
def add(title, svg, note=""):
    figs.append((title, svg, note))

# F1 pipeline
add("图 1  v_new 在两阶段电荷递归中的位置", pipeline_svg())

# F2 eps sweep (Z3)
eps_lab = ["1e-30", "1e-12", "1e-8", "1e-6"]; jvp = [+0.01246, -0.01769, -0.01836, -0.02271]; fd = [-0.01319, -0.01327, -0.01793, -0.02216]
xs = [0, 1, 2, 3]
svg = line_chart("f2", [{"x": xs, "y": fd, "label": "有限差分 FD（h = 0.01 Å）", "cls": "s1", "dots": True},
                        {"x": xs, "y": jvp, "label": "自动微分二阶（JVP）", "cls": "s2", "dots": True, "dash": True}],
                 "面积正则化下限 eps（面积项 sqrt(|∇S|² + eps)）", "特征泛函沿位移的导数（任意单位）", y0line=True, unit="", xticks=xs)
svg = svg.replace('<text x="', '<text data-eps="1" x="', 0)
for i, lab in enumerate(eps_lab):
    svg = svg.replace(f'class="tick">{i}</text>', f'class="tick">{lab}</text>')
add("图 2  空腔项二阶导数对正则化下限的依赖（验收第 Z3 节，带电帧 sid 28）", svg,
    "一阶导数（v_cav 本身）在任何 eps 下都精确；外层依赖需要的二阶导数被下限的曲率 1/sqrt(eps) 支配：1e-12 时偏差 33%，1e-30 时符号错，1e-8 时 2.4%（A_cav 只变 0.15%）。最终：特征用 1e-8，能量项保持 1e-12。")

# F3/F4 cost
add("图 3  真实训练步时（三卡 DDP，每步 3 帧）", bar_chart("f3", [("生产 PB 阶段\n(20–22 轮, 关)", 1.43, "s1"), ("探针 关\n(预热 0)", 1.49, "s1"), ("探针 开\n(预热 0)", 1.64, "s2"), ("闸门 开\n(20–33 轮)", 1.62, "s2")], "步时均值 (s)", "s"),
    "探针 = 生产配置、预热 0、2 轮（3459536/3459537，同一节点 c301-003）；闸门 gate_vsolv 20–33 轮均值 1.59–1.66 s。输入使每步多 0.11–0.19 s（+7–13%）。")
add("图 4  显存峰值（rank 0）", bar_chart("f4", [("生产 (关)", 23.9, "s1"), ("探针 关", 23.9, "s1"), ("探针 开", 26.3, "s2"), ("闸门 开", 26.35, "s2")], "CUDA 峰值 (GiB)", "GiB", hline=(40.0, "A100 40 GB")),
    "+2.4 GiB（+10%）。单帧探针（全部参数冻结、非 DDP）给的是 +4.2/+6.5 GiB，与训练成本不同，用户指出后改测真实训练步。")

# F5-F7 ratios
def ratio_fig(cid, key, title):
    lo = [min(ratio_r[key])] * len(EP); hi = [max(ratio_r[key])] * len(EP)
    svg = line_chart(cid, [{"x": EP, "y": ratio_g[key], "label": "闸门(开) / 生产(关)", "cls": "s2", "dots": True},
                           {"x": EP, "y": ratio_r[key], "label": "同配置副本 w1000_ref / 生产", "cls": "s1", "dots": True, "dash": True}],
                     "训练轮次", f"验证 {KEY_CN[key]} 比值", y0line=False, bands=[(EP, lo, hi, "s1")], unit="", hlines=[(1.0, "1.0", "s6")], xticks=EP)
    add(title, svg, f"阴影 = 副本比值的范围（20–33 轮）= 同配置两次训练的随机底线。闸门比值均值 {mean(ratio_g[key]):.3f}，副本 {mean(ratio_r[key]):.3f}；闸门有 {outside[key]}/{len(EP)} 轮落在副本范围外。")
ratio_fig("f5", "F", "图 5  验证力误差比值：闸门 vs 生产，与随机底线对照")
ratio_fig("f6", "dens3d", "图 6  验证 3D 密度误差比值")
ratio_fig("f7", "occ", "图 7  验证占据数误差比值")
ratio_fig("f7b", "E", "图 8  验证能量误差比值（噪声内的例子）")

# F8 structural % changes
cols = [("e25", "s1"), ("e30", "s2"), ("e33", "s3")]
rows = []
for lab, key in STRUCT_ROWS:
    d = {}
    for ep, nm in ((25, "e25"), (30, "e30"), (33, "e33")):
        a, b = ST[("prod", ep)][key], ST[("gate", ep)][key]
        d[nm] = 100.0 * (b / a - 1.0) if a == a and a != 0 else float("nan")
    rows.append((lab, d))
add("图 9  结构量的相对变化：闸门相对生产（同轮次，47 对 94 帧；正 = 闸门更差）", dot_rows("f8", rows, cols, "闸门 / 生产 − 1（%）"),
    "三个轮次每个量都朝同一方向：力 +2–7%、3D 密度 rmse +1–2%、溶剂侧尾部 +2–13%（中性）变差；真空侧尾部 −6–9%、面平均窗口 L1 −1.5–4.4% 变好；电荷响应与溶剂剖面在 ±5% 内。E 的 ±5–18% 落在副本噪声带（0.81–1.04）内。")

# F9 charge response curves gate vs prod e33
ag, ap, ad = cks["gate_e33"]["aggregate_all"], cks["prod_e33"]["aggregate_all"], cks["prod_e33"]["aggregate_all"]
add("图 10  充电响应 R(z)/ΔN_e（47 对平均）：DFT、生产 e33、闸门 e33",
    line_chart("f9", [{"x": ad["z"], "y": ad["R_dft_per_e"], "label": "DFT", "cls": "s6"},
                      {"x": ap["z"], "y": ap["R_ml_per_e"], "label": "生产 e33（关）", "cls": "s1", "dash": True},
                      {"x": ag["z"], "y": ag["R_ml_per_e"], "label": "闸门 e33（开）", "cls": "s2"}], "z (Å)", "R / ΔN_e (1/Å)", regions=regmean, unit="1/Å"),
    "两臂几乎重合：新增势没有改变充电响应的形状；两者都把太多响应放进水层、电极片层不足。")

# F10 E/S vs epoch; F11 region shares
ep_ck = [c["epoch"] for c in prod_cks]
def es(c, sp): return mean([p["ratio"] for p in c["pairs"].values() if sp == "all" or p["split"] == sp])
add("图 11  充电响应误差比 E/S 随训练轮次（生产检查点，开关关）",
    line_chart("f10", [{"x": ep_ck, "y": [es(c, "train") for c in prod_cks], "label": "训练 27 对", "cls": "s1", "dots": True},
                       {"x": ep_ck, "y": [es(c, "val") for c in prod_cks], "label": "验证 20 对", "cls": "s2", "dots": True}], "训练轮次", "E / S", y0line=True, unit=""),
    "E = ∫|R_ML − R_DFT| dz，S = ∫|R_DFT| dz。0.98（33 轮）→ 0.55（200 轮）→ 0.50（499 轮）：200 轮后进入平台。电子数守恒每对每检查点 ≤ 0.007 e。")
def share(c, nm, side): return mean([p["regions"][nm][f"N_{side}"] / p[f"N_{side}"] for p in c["pairs"].values()])
add("图 12  额外电子的分区份额随训练轮次（虚线 = DFT）",
    line_chart("f11", [{"x": ep_ck, "y": [share(c, "electrode", "ml") for c in prod_cks], "label": "模型 电极", "cls": "s1", "dots": True},
                       {"x": ep_ck, "y": [share(c, "electrode", "dft") for c in prod_cks], "label": "DFT 电极", "cls": "s1", "dash": True},
                       {"x": ep_ck, "y": [share(c, "water", "ml") for c in prod_cks], "label": "模型 水层", "cls": "s2", "dots": True},
                       {"x": ep_ck, "y": [share(c, "water", "dft") for c in prod_cks], "label": "DFT 水层", "cls": "s2", "dash": True},
                       {"x": ep_ck, "y": [share(c, "interface", "ml") for c in prod_cks], "label": "模型 界面", "cls": "s3", "dots": True},
                       {"x": ep_ck, "y": [share(c, "interface", "dft") for c in prod_cks], "label": "DFT 界面", "cls": "s3", "dash": True}], "训练轮次", "份额 N_region / N", y0line=True, unit=""),
    "电极 = 片层原子 ±1.5 Å；成熟模型电极份额 0.37–0.40 对 DFT 0.47–0.48（亏 0.10 e），自 100 轮起不再变；水层已收敛到 DFT 附近；界面亏空 −0.08 e 几乎缺失。")

# F12 Phi1D fix
m = (zphi >= 0) & (zphi <= 44.95)
add("图 13  Φ1D 残差（模型 − DFT），中性 sid 722：修复前后",
    line_chart("f12", [{"x": zphi[m], "y": res_before[m], "label": "修复前", "cls": "s2"}, {"x": zphi[m], "y": res_after[m], "label": "修复后", "cls": "s1"}],
               "z (Å)", "Δφ (eV)", regions=regmean, unit="eV"),
    f"修复前 18–45 Å 一条斜线（斜率 {obs_ramp_b:+.5f} eV/Å）、与板下真空之间台阶 {obs_step_b:+.4f} eV；由 μ_bound = {mu_b:+.3f} e·Å 预测的台阶 4πKμ/A = {step_pred:+.4f} eV、斜坡 −台阶/H = {ramp_pred:+.5f} eV/Å。修复后斜率 {obs_ramp_a:+.5f}、台阶 {obs_step_a:+.4f} eV（剩余是模型误差）；rms 0.1009 → 0.0798 eV。带电 sid 122 不变（0.0640 → 0.0641）。")

# ---------------------------------------------------------------- tables
def table(head, rows, cls=""):
    return f'<div class="tw"><table class="{cls}"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + \
        "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>"

tbl_ratio = table(["指标", "闸门/生产 均值", "闸门 范围", "副本/生产 均值", "副本 范围", "闸门在副本范围外的轮数"],
                  [[KEY_CN[k], f"{mean(ratio_g[k]):.3f}", f"{min(ratio_g[k]):.3f}–{max(ratio_g[k]):.3f}", f"{mean(ratio_r[k]):.3f}", f"{min(ratio_r[k]):.3f}–{max(ratio_r[k]):.3f}", f"{outside[k]}/{len(EP)}"] for k, _ in KEYS])
tbl_struct = table(["量", "e25 生产", "e25 闸门", "e30 生产", "e30 闸门", "e33 生产", "e33 闸门"],
                   [[lab] + [f"{ST[(arm, ep)][key]:.4g}" for ep in (25, 30, 33) for arm in ("prod", "gate")] for lab, key in STRUCT_ROWS])
tbl_accept = table(["检查", "内容", "结果"], [
    ["C0", "−dλ_diel/dE 等于母模型的极化 p(E)（含局域场因子）", "相对偏差 7e-7 … 5e-11（E = 1e-4 … 1 eV/Å）"],
    ["C1", "固定 φ 的偏导有限差分，逐项（sid 28）", "cav 4e-6 / diel 5e-7 / ion 3e-6（壳层扰动）；随机方向 3e-5 / 1e-7 / 2e-5"],
    ["C2", "g(n, 活 φ) 与固定 φ 值逐位相等且带 d/dφ*", "通过（|∂/∂φ*| 最大 4e3）"],
    ["C3", "量级（eV/电子，sid 28/628）", "v_diel 网格 rms 0.24/0.28，峰 0.73/0.87 @ 16.35 Å；v_cav 峰 0.077；v_ion 峰 2.7e-3/3e-8；核处为零；反应电势 rms 2.5–3.3"],
    ["C4/C5", "单帧成本与显存", "偏导 0.13–0.41 s；带 create_graph 21–22 GiB → 一维场路径后 3.0–3.2 GiB"],
    ["1D vs 3D", "一维场路径与 torch_pb 三维路径", "A_diel 差 3.3e-5 eV（相对 2.6e-5），A_ion 相同，|E_loc| 最大差 3e-3 eV/Å"],
    ["S", "采样器 vs 静电通道对反应电势的投影（207 原子）", "值 7.5e-5，z 梯度 1.1e-4（谱求和后）"],
    ["M", "新增行的量级 / 通道行", "带电 3%，中性 61%（rms 0.119/0.135 对 3.77/0.236）"],
    ["F", "力：自动微分 vs 有限差分，六个分量", "开 0.210/0.211 meV/Å（关 0.209/0.211）"],
    ["G", "力损失对参数的梯度 AD vs FD", "开 9.3e-3 / 7.6e-6（关 6.3e-3 / 1.2e-7，FD 步长限制）"],
    ["X", "求解器容差 1e-3→1e-6、步长 h 扫描", "22 meV/Å 缺口不随容差与 h 变化 → 真正的导数不一致（随后定位）"],
    ["Y", "特征自身的导数 AD vs FD（线扫描拟合）", "修复后 −0.09148 vs −0.09145，斜率拟合 0.3%，拟合残差 7e-7"],
    ["Z/Z2/Z3", "路径归因（位置/φ/密度）与逐项 Hessian", "位置、φ 路径精确；密度路径缺口全在空腔项的正则化下限（图 2）"],
    ["T", "单帧成本（E+F / 力损失二阶反传）", "0.62→0.71 s，6.0→10.3 GiB / 1.03→1.16 s，16.9→23.4 GiB（检查点重算）"],
])
tbl_fix = table(["", "预测（μ_bound = +0.175 e·Å）", "观测（修复前）", "观测（修复后）"], [
    ["溶剂区与板下真空的台阶", f"4πKμ/A = {step_pred:+.4f} eV", f"{obs_step_b:+.4f} eV", f"{obs_step_a:+.4f} eV"],
    ["24–44.5 Å 匀强场", f"−台阶/H = {ramp_pred:+.5f} eV/Å", f"{obs_ramp_b:+.5f} eV/Å", f"{obs_ramp_a:+.5f} eV/Å"],
    ["中性 Φ1D 残差 rms", "", f"{rms(res_before):.4f} eV", f"{rms(res_after):.4f} eV"],
    ["带电 Φ1D 残差 rms", "", f"{rms(res_before_c):.4f} eV", f"{rms(res_after_c):.4f} eV"],
])

e33 = ST[("gate", 33)]; p33 = ST[("prod", 33)]
FIGS_HTML = "".join(f'<figure><figcaption>{html.escape(t)}</figcaption>{s}{f"<p class=note>{html.escape(n)}</p>" if n else ""}</figure>' for t, s, n in figs)

page = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Solvent Potential Input</title>
<style>
:root{{--bg:#f9f9f7;--surf:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mute:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--rule:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#4a3aa7;--s8:#e34948;
--band-el:#efe9dc;--band-if:#f3ecdc;--band-wa:#e2ebf3;--band-vac:#f3f2ee;--new:#fbe7dd;--old:#dfe9f6;--box:#fcfcfb}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0d0d0d;--surf:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--mute:#898781;--grid:#2c2c2a;--axis:#383835;--rule:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#0ca30c;--s7:#9085e9;--s8:#e66767;
--band-el:#2a2a24;--band-if:#2c2820;--band-wa:#20272e;--band-vac:#1e1e1c;--new:#3a2a22;--old:#22303f;--box:#1f1f1e}}}}
:root[data-theme="dark"]{{--bg:#0d0d0d;--surf:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--mute:#898781;--grid:#2c2c2a;--axis:#383835;--rule:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#0ca30c;--s7:#9085e9;--s8:#e66767;
--band-el:#2a2a24;--band-if:#2c2820;--band-wa:#20272e;--band-vac:#1e1e1c;--new:#3a2a22;--old:#22303f;--box:#1f1f1e}}
html,body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI","Noto Sans SC",sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:32px 16px 72px}}
h1{{font-size:1.75rem;margin:0 0 6px;text-wrap:balance}} h2{{font-size:1.2rem;margin:44px 0 12px;padding-top:14px;border-top:1px solid var(--rule)}} h3{{font-size:1.02rem;margin:22px 0 6px;color:var(--ink2)}}
p,li{{max-width:78ch}} .sub{{color:var(--ink2);margin:0 0 18px}} .note{{color:var(--ink2);font-size:.92rem;margin:6px 0 0}}
.tw{{overflow-x:auto;margin:10px 0 16px}} table{{border-collapse:collapse;width:100%;font-size:.88rem;font-variant-numeric:tabular-nums;background:var(--surf)}}
th,td{{padding:6px 9px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap;vertical-align:top}} th{{color:var(--ink2);font-weight:600}} td:first-child,th:first-child{{text-align:left;white-space:normal}}
table.wrap td{{white-space:normal;text-align:left}}
pre{{font:.86rem/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--surf);border:1px solid var(--grid);padding:12px 14px;overflow-x:auto;white-space:pre-wrap;max-width:none}}
figure{{margin:26px 0}} figcaption{{font-size:.95rem;color:var(--ink2);margin:0 0 6px}}
svg.fig{{display:block;background:var(--surf);border:1px solid var(--grid)}}
.grid{{stroke:var(--grid);stroke-width:1}} .frame{{fill:none;stroke:var(--axis)}} .zero{{stroke:var(--mute);stroke-width:1}} .hline{{stroke-width:1;stroke-dasharray:3 4}}
.tick{{font-size:11px;fill:var(--mute);text-anchor:middle}} .tick.r{{text-anchor:end}} .axis{{font-size:12px;fill:var(--ink2);text-anchor:middle}} .lgd,.rowlbl{{font-size:12px;fill:var(--ink)}} .rowlbl.r,.hlbl.r{{text-anchor:end}} .hlbl{{font-size:11px;fill:var(--ink2)}}
.val{{font-size:12px;fill:var(--ink);text-anchor:middle}} .bandlbl{{font-size:12px;fill:var(--mute);text-anchor:middle}}
.band.el{{fill:var(--band-el)}} .band.if{{fill:var(--band-if)}} .band.wa{{fill:var(--band-wa)}} .band.vac{{fill:var(--band-vac)}} .rowband{{fill:var(--band-vac)}}
.rangeband{{opacity:.18}} .rangeband.s1{{fill:var(--s1)}}
.line{{fill:none;stroke-width:2;stroke-linejoin:round}} .dot{{stroke:var(--surf);stroke-width:2}} .bar{{stroke:none}}
.s1{{stroke:var(--s1)}} .s2{{stroke:var(--s2)}} .s3{{stroke:var(--s3)}} .s4{{stroke:var(--s4)}} .s5{{stroke:var(--s5)}} .s6{{stroke:var(--s6)}} .s7{{stroke:var(--s7)}} .s8{{stroke:var(--s8)}}
.dot.s1,.bar.s1{{fill:var(--s1)}} .dot.s2,.bar.s2{{fill:var(--s2)}} .dot.s3,.bar.s3{{fill:var(--s3)}} .dot.s4,.bar.s4{{fill:var(--s4)}} .dot.s6,.bar.s6{{fill:var(--s6)}}
.xh{{stroke:var(--ink2);stroke-width:1}}
.diag .box{{fill:var(--box);stroke:var(--axis);stroke-width:1.2}} .diag .box.new{{fill:var(--new);stroke:var(--s2)}} .diag .box.old{{fill:var(--old);stroke:var(--s1)}}
.diag .boxtxt{{font-size:13px;fill:var(--ink);text-anchor:middle}} .diag .arrow{{fill:none;stroke:var(--ink2);stroke-width:1.6}} .diag .arrow.new{{stroke:var(--s2)}} .diag .arrow.old{{stroke:var(--s1)}}
.diag .arrhead{{fill:var(--ink2)}} .diag .lbl{{font-size:12px;fill:var(--ink2)}} .diag .lbl.new{{fill:var(--s2)}} .diag .lbl.old{{fill:var(--s1)}} .diag .note{{font-size:12px;fill:var(--ink2)}}
#tip{{position:fixed;pointer-events:none;background:var(--surf);border:1px solid var(--grid);border-radius:6px;padding:5px 9px;font-size:12.5px;opacity:0;z-index:9;font-variant-numeric:tabular-nums;box-shadow:0 2px 8px rgba(0,0,0,.12)}}
.kv{{display:grid;grid-template-columns:1fr;gap:8px;margin:12px 0}} .kv div{{background:var(--surf);border:1px solid var(--grid);border-radius:8px;padding:10px 14px}}
@media (min-width:720px){{.kv{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>溶剂有效势输入：理论、验证、闸门与最终改动</h1>
<p class="sub">v_new = δ(A_cav + A_diel + A_ion)/δn_e 在第一阶段场处取值，作为电荷递归的新输入。2026-09-21 至 09-22 的全部测试与结果；数据来自台账 RUNS.md 的对应条目与本页脚本直接读取的日志/文件。</p>

<h2>0 · 结论摘要</h2>
<div class="kv">
<div><b>物理输入本身通过全部导数验收。</b>偏导 FD 一致到 1e-6 量级；力 AD–FD 0.21 meV/Å（与原路径相同）；力损失参数梯度 9e-3/8e-6（FD 步长限制）；两处实现缺陷（总导数混入、面积正则化下限）在验收中被定位并修复。</div>
<div><b>训练闸门（34 轮，对照生产同轮次）没有指标在噪声外变好。</b>力 +5.6%（14/14 轮在副本范围外）、3D 密度 +1.4%（13/14）、占据数 +11%（13/14）；E、电势、费米、Φ1D、ρ_b 在同配置副本的随机底线内。</div>
<div><b>结构评估（e25/30/33）同一模式。</b>真空侧密度尾部好 6–9%，溶剂侧尾部差 2–13%，充电响应形状不变（两臂共同缺陷：电极片层亏 0.10 e）。</div>
<div><b>最终决定（用户）。</b>保留该输入（开关默认关，生产配置开）；修复 Φ1D 构造对带溶剂中性帧的缺陷（束缚电荷被清零、偶极修正仍含它 → 0.17 eV 台阶 + 匀强场）；以修复后的代码 + 输入开重训 500 轮（prod500_vsolv_fix，作业 3462229，gpu-a100 排队中）。</div>
</div>

<h2>1 · 背景：模型里溶剂电势是怎么进 SCF 的，缺了什么</h2>
<p>模型的“SCF”是 PolarMACE 的电荷递归（num_recursion_steps = 1）。溶剂通过补偿特征进入：第一阶段用密度头解一维 PB，得到横向均匀的溶剂电荷 ρ_ion + ρ_bound，其电势和场在接收高斯处的投影（G ≠ 0 项）加上板偶极 G0 项作为特征送入第二阶段。这就是 VASPsol++ 的 v_corr = eφ_solv + v_solv 中的第一项（反应电势）。第二项 v_solv = v_cav + v_diel + v_ion（论文 eqs. 59–67）是空腔、介电、离子自由能对电子密度的泛函导数，原来没有。</p>
<p>训练前先做了三个不训练的检查（生产 499 轮检查点、47 对）：tau·A 空腔能在生产里是开的（+3.915/+3.949 eV，我先前的“关着”说法有误）；配对充电残差的离散（0.20 eV）由 E_bl 项的带电−中性差解释（r 0.97/0.95），且它是输入场误差（溶剂剖面 L1 误差 42%/88%）而非重复计数；线性响应估计介质内能 U_int ≈ +0.25 eV/帧。用户随后的五点更正定下路线：先写出与一维闭合一致的自由能，再加第一阶段的 v_solv 输入，不加外层自洽。</p>
{figs[0][1]}
<p class="note">{html.escape(figs[0][0])}</p>

<h2>2 · 理论公式</h2>
<h3>2.1 母泛函（VASPsol++，电势形式，φ 为独立变量，取极大）</h3>
<pre>A[φ; n_e] = A_TXC + ∫ φ ρ_sol − (1/2K) ∫ |∇φ|² + A_cav[n_e] + n_mol ∫ S_diel[n_e] λ_diel(E) + n_max ∫ S_ion[n_e] λ_ion(φ)

λ_ion(φ)  = −(1/β) ln(1 − θ_b + θ_b cosh(z β φ)),   λ_ion(0) = 0,   θ_b = 0.436（格气离子）
λ_diel(E) = λ_rot + λ_pol + λ_sic
  λ_rot = −(1/β) ln( sinh(y)/y ),  y = p β E_loc            （转动熵）
  λ_pol = −½ α_pol E_loc² / K                               （电子极化）
  λ_sic = +½ (1/α_sic) [ (α0_rot g(y) + α_pol) E_loc ]² / K （自相互作用修正）
  E_loc = f_loc · E（局域场因子，pb1d_localfield），  K = 180.95 eV·Å
A_cav = τ ∫ |∇S_cav[n_e]| dV，离散形式 sqrt(|∇S|² + eps)（双 Stern 掩膜）</pre>
<h3>2.2 一维闭合 = λ_diel 在屏蔽真空场 E_scr[n_e] 处的面平均二次展开</h3>
<pre>a1(z)    = ⟨a(E_scr)⟩_plane                        （A_scr，响应 a(E) = f_loc n_mol s_diel (α0_rot g(y) + α_pol)/K）
p_off(z) = ⟨a(E_scr) E_scr⟩ − a1 ⟨E_scr⟩ + δp     （先验 + 头的残差）
ρ_b      = −d/dz [ w_b * (a1 E + p_off) ],  E = −d/dz (w_b * φ)
Λ_diel[φ; n_e] = Σ_z [ C_z[n_e] − ½ a1 E² − p_off E ] dz,   C_z = ⟨n_mol s_diel λ_diel(E_scr)⟩ + P_scr E_scr − ½ a1 E_scr²
A_1D[φ; n_e] = ∫ φ ρ_sol,eff − (1/2K) ∫ (dφ/dz)² + Λ_diel + n_max ∫ S_ion,z λ_ion(φ)
δA_1D/δφ = 0  ⇔  求解器残差 R(φ) = n_b(φ) + n_ion(φ) − L0(φ − φ_sol) + q_sol = 0</pre>
<p class="note">C_z 不进 PB 方程（不含 φ）但进 δA/δn_e，由母泛函在线性化点固定；δp 没有母自由能，按零自能的外加极化处理（声明，非推导）。</p>
<h3>2.3 包络定理与新电势</h3>
<pre>dA_solv*/dn_e(r) = ∂A_1D/∂n_e(r) |_{{φ = φ*}} = [ −e φ_solv(r) ] + v_new(r)
v_cav (r) = τ ∂A[S_cav3(n_e)]/∂n_e(r)                                   （三维，精确到离散形式）
v_ion (r) = n_max Σ_z λ_ion(φ*(z)) ∂S_ion,z[n_e]/∂n_e(r)
v_diel(r) = Σ_z [ ∂C_z/∂n_e(r) − ½ E*(z)² ∂a1_z/∂n_e(r) − E*(z) ∂p_off,z(先验)/∂n_e(r) ]</pre>
<p>实现上取 g(n, φ) = ∂_n A(n, φ)，φ 为独立输入，再在 φ = φ*(n) 处取值：对 n_e 的克隆节点求 autograd.grad（否则 φ* 由同一密度张量解出，得到的是总导数 ∂A/∂n + ∂A/∂φ·dφ*/dn，验收第 Y/Z 节抓到了这个 22 meV/Å 的差别）。φ* 是第一阶段的滞后场，与反应电势同一近似；create_graph 保留对 φ*、位置的依赖，力与损失梯度完整。</p>
<h3>2.4 投影与符号</h3>
<pre>v_new 是电势（每电子能量，电子势能约定）。在原子处取接收高斯 σ_k 平滑后的值与梯度（与静电通道对反应电势的投影相同），
不解泊松、不去均值。静电通道送的是物理符号电势 k ρ/G²；新增电子密度 δn_e 是物理电荷 −δn_e，因此特征 = VSOLV_SIGN · v_new，VSOLV_SIGN = −1。
第一阶段电势横向均匀 → E、f_loc、λ_diel、λ_ion 都是 z 的一维函数（在 PB 网格 z 轴上算再广播），只有空腔函数是三维。
正则化：能量项 eps = 1e-12（一阶导精确），特征项 eps = 1e-8（二阶导需要，A_cav 变 0.15%）。</pre>

<h2>3 · 实现</h2>
<ul>
<li><b>mace/modules/pb1d_vsolv.py</b>：lambda_diel、lambda_ion、field_terms_1d、free_energies、smoothed_value_and_gradient（截断谱求和 |G| &lt; 5/σ）、_vsolv_core（克隆节点偏导、enable_grad 内层）、vsolv_node_fields（torch.utils.checkpoint 重算，避免存 21 GiB 的二阶反传图）。</li>
<li><b>pb1d_backend.solve_graph</b>：vsolv_input / vsolv_sigmas 参数，返回 vsolv_node_fields；<b>extensions.py</b>：开关 solvent_pb1d_vsolv_input（默认 False），只在第一阶段（use_head = False）收集，_vsolv_projection_features 加到补偿特征的同一组行上；<b>arg_parser</b> 新增 --solvent_pb1d_vsolv_input。</li>
<li>环境变量 MACE_PB1D_VSOLV_AREA_EPS（默认 1e-8）、MACE_PB1D_VSOLV_NOCKPT、MACE_PB1D_VSOLV_DETACH（诊断切断 φ/n/位置路径）。</li>
<li>评估脚本对训练时开着开关的检查点用 KIT_VSOLV=1 打开同一路径（同一模型对象，参数集不变，只加特征行）。</li>
</ul>

<h2>4 · 训练前验收（单帧 sid 28 带电 / 628 中性，代码 9754b90 → 48051b5）</h2>
{tbl_accept}
<p>两处缺陷及修复：(1) 22 meV/Å 的力缺口 = 偏导取成了总导数（φ* 的历史经过同一密度张量），改为对 n_e.clone() 求导后路径贡献可加（φ −0.004、n −0.008、位置 −0.075），力 AD–FD 回到 0.10 meV/Å；(2) 剩余 5% 的特征导数缺口 = 空腔项二阶导数被正则化下限支配（图 2），特征改用 eps = 1e-8。</p>
{"".join(f'<figure><figcaption>{html.escape(t)}</figcaption>{s}{f"<p class=note>{html.escape(n)}</p>" if n else ""}</figure>' for t, s, n in figs[1:2])}

<h2>5 · 真实训练成本（三卡 DDP，全部参数，完整损失）</h2>
{"".join(f'<figure><figcaption>{html.escape(t)}</figcaption>{s}{f"<p class=note>{html.escape(n)}</p>" if n else ""}</figure>' for t, s, n in figs[2:4])}
<p class="note">按步拆分（ms）：关 数据 30–34 / 前向 541–548 / 损失 167–179 / 反传 727；开 39–48 / 593–612 / 183–208 / 782。预热 0 的探针里 Newton 上限命中与 15–16 次回退两臂都有，是探针条件不是输入。</p>

<h2>6 · 闸门：gate_vsolv（生产配置 + 开关开，34 轮）对生产 prod500_w1000_ref 同轮次</h2>
<p>参照一致性：配置只差 name、max_num_epochs 与开关；seed 123、预热 20、损失、1×3 卡相同。但同配置两次训练在预热阶段就不逐位一致（三卡 GPU 训练的运行间随机性；epoch 19 时 w1000_ref 对生产 loss +10.3%、F +11.7%），所以每个比值都要与同配置副本 w1000_ref/生产 的范围比。</p>
{tbl_ratio}
{"".join(f'<figure><figcaption>{html.escape(t)}</figcaption>{s}{f"<p class=note>{html.escape(n)}</p>" if n else ""}</figure>' for t, s, n in figs[4:8])}
<h3>结构评估（audits/ebl_audit.py + density_profile_eval.py，47 对 = 20 验证 + 27 训练）</h3>
{tbl_struct}
{"".join(f'<figure><figcaption>{html.escape(t)}</figcaption>{s}{f"<p class=note>{html.escape(n)}</p>" if n else ""}</figure>' for t, s, n in figs[8:10])}
<p>e33 终点：力 {p33['F_charged']:.2f}→{e33['F_charged']:.2f}（带电）/ {p33['F_neutral']:.2f}→{e33['F_neutral']:.2f}（中性）meV/Å；3D 密度 rmse {p33['rmse3d_charged']:.5f}→{e33['rmse3d_charged']:.5f}；电荷响应 L1 {p33['dnL1']:.3f}→{e33['dnL1']:.3f} e；溶剂剖面 L1 {p33['solvL1_charged']:.4f}→{e33['solvL1_charged']:.4f} / {p33['solvL1_neutral']:.4f}→{e33['solvL1_neutral']:.4f} e。配对充电只记录：e25 4.00 vs 4.04、e30 3.24 vs 3.56、e33 3.15 vs 3.01 eV（训练阶段的 +3–4 eV 偏置）。</p>

<h2>7 · 充电响应诊断（用户决定的依据之一：用已有检查点，不训练）</h2>
<p>定义先核对：47 对几何/晶胞/网格逐项相同，∫Δn dV = ΔN_e 到 0.007 e（模型）/ 0.0006 e（DFT）；E 是形状误差不是电子缺失。</p>
{"".join(f'<figure><figcaption>{html.escape(t)}</figcaption>{s}{f"<p class=note>{html.escape(n)}</p>" if n else ""}</figure>' for t, s, n in figs[10:12])}
<p>结论：早期“误差超过响应本身”只到 33 轮；成熟模型 E/S = 0.50，电极片层仍亏 0.10 e、界面亏空缺失、真空尾部多 0.07 e，200 轮后不再改善。开关开的检查点只有 0–33 轮（E/S 1.06 / 1.00，与同轮次关无差别）。用户据此决定保留输入、后续训练开、本轮不改训练。</p>

<h2>8 · 顺带发现并修复：Φ1D 构造对带溶剂中性帧的缺陷</h2>
<p>在 sid 122/722 配对页上，中性帧模型电势在 18–45 Å 是一条斜线。原因：一维电势 = 对 raw_total（POTCAR 中性原子 − GTO 电子密度 + POTCAR 离子 + 溶剂剖面）解周期泊松，再加 VASP 式锯齿偶极修正，修正的偶极取自 pred["dipole"]（多极头偶极 + 求解器溶剂偶极）。溶剂剖面进 raw_total 前乘系数 s_gauss/s_prof 对齐到高斯层净电荷；中性帧 s_gauss = 0，束缚电荷被清零，pred["dipole"] 里的 μ_bound 却还在。</p>
<pre>台阶 = 4π K μ_bound / A ,   匀强场 = −台阶 / H       （A = 189.7 Å²，H = 45 Å，K = 14.40 eV·Å/e²）</pre>
{tbl_fix}
{"".join(f'<figure><figcaption>{html.escape(t)}</figcaption>{s}{f"<p class=note>{html.escape(n)}</p>" if n else ""}</figure>' for t, s, n in figs[12:13])}
<p>修法（c16ab1b）：去掉系数，所有帧 raw_solvent = −raw_prof（明写符号约定；系数在带电帧恒为 −1.0000000000，无物理含义，还会掩盖离子电荷不配平）。影响范围：200 个带溶剂中性帧（sid 601–800）自 mix800 进训练起的 Φ1D 损失与验证指标；能量、力、求解器自身电势、电极电势与费米标量不受影响。闸门 3462092：中性 sid 722 rms 0.1009 → 0.0798 eV，斜坡 −0.00360 → −0.00012 eV/Å，台阶 +0.1685 → +0.0449 eV；带电 0.0640 → 0.0641；一轮训练试跑（预热 0，开关开）步时 1.67 s、26.24 GiB、无异常。</p>

<h2>9 · 最终改动与状态</h2>
<ul>
<li>代码：pb1d_vsolv.py 等（ad61501 → 48051b5，开关默认关）；loss.py 的 Φ1D 修复（c16ab1b/399b03e）；评估脚本（density_profile_eval.py、charge_response_report.py、struct_compare.py、gate_vs_prod_table.py 加副本底线列）。全部推送到 pb-s3d-energy（远端 7bb2759）。</li>
<li>生产配置：3-residual_3D/prod500_vsolv_fix = prod500_w1000_ref + solvent_pb1d_vsolv_input: True；seed 123、预热 20、E 1000 / F 100、三卡、500 轮；gpu-a100 48 h，预算 169200 s 到点写状态并自链续段（步时 +12% 使 500 轮装不进一个 48 h 作业）。作业 3462229，投递 09-22 06:58，至 09-23 02:30 仍在排队（分区 197 待运行、50 运行，优先级 1499 对队首 5645）。</li>
<li>对照：prod500_w1000_ref（开关关、旧构造）+ w1000_ref 底线；Φ1D 验证指标在 25% 的带溶剂中性帧上定义已变，跨修复前后不可直接比该项。</li>
<li>相关页面：充电响应诊断 https://claude.ai/artifact/NaL6dUc6YimsRfrNC5KhcS ；生产模型 sid 122/722 配对报告 https://claude.ai/artifact/NoEkrbdogoEctkv6wRTtM9 。</li>
</ul>
</main>
<div id="tip"></div>
<script>
const DATA = {json.dumps(charts_js, ensure_ascii=False)};
const tip = document.getElementById('tip');
for (const [cid, cfg] of Object.entries(DATA)) {{
  const svg = document.getElementById(cid); if (!svg) continue;
  const xlo = +svg.dataset.xlo, xhi = +svg.dataset.xhi; const ML = {ML}, MR = {MR}, W = {W};
  const xh = svg.querySelector('.xh');
  svg.addEventListener('mousemove', ev => {{
    const r = svg.getBoundingClientRect(); const fx = (ev.clientX - r.left) / r.width * W;
    if (fx < ML || fx > W - MR) {{ tip.style.opacity = 0; xh.style.opacity = 0; return; }}
    const xv = xlo + (fx - ML) / (W - MR - ML) * (xhi - xlo);
    let i = 0, best = 1e18; cfg.x.forEach((x, k) => {{ const d = Math.abs(x - xv); if (d < best) {{ best = d; i = k; }} }});
    const px = ML + (cfg.x[i] - xlo) / (xhi - xlo) * (W - MR - ML);
    xh.setAttribute('x1', px); xh.setAttribute('x2', px); xh.style.opacity = 0.5;
    tip.innerHTML = `x = ${{cfg.x[i]}}<br>` + cfg.series.map(s => `${{s.n}}: ${{s.y[i]}} ${{cfg.unit}}`).join('<br>');
    tip.style.left = (ev.clientX + 14) + 'px'; tip.style.top = (ev.clientY + 12) + 'px'; tip.style.opacity = 1;
  }});
  svg.addEventListener('mouseleave', () => {{ tip.style.opacity = 0; xh.style.opacity = 0; }});
}}
</script></body></html>"""
open(OUT, "w").write(page)
print(f"written {OUT}: {len(figs)} figures, {len(page)//1024} kB")
