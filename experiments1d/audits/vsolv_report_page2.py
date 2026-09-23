"""Report page (second version, user's review of 2026-09-23: fewer error metrics, rendered formulas,
training length per epoch, only the essential tests): the solvent effective-potential input
v_new = d(A_cav + A_diel + A_ion)/dn_e at the stage-1 field. Static HTML; formulas are rendered
offline with matplotlib mathtext into inline SVG (no CDN); the only script is the hover read-out.

  python vsolv_report_page2.py OUT.html
"""
import html
import io
import json
import math
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["mathtext.fontset"] = "dejavusans"
B = "/scratch/08384/tg876840/tmp/c-MACEsol"
L = f"{B}/claude/2-1D_PB/exp_vsolv/logs"
OUT = sys.argv[1]


# ---------------------------------------------------------------- formulas -> inline SVG
def tex(formula, size=13):
    fig = plt.figure(figsize=(0.1, 0.1)); fig.text(0, 0, formula, fontsize=size)
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.06, transparent=True)
    except Exception as exc:
        raise SystemExit(f"mathtext failed on: {formula[:80]} -> {str(exc)[:120]}")
    finally:
        plt.close(fig)
    s = buf.getvalue().decode()
    s = s[s.index("<svg"):]
    w = float(re.search(r'width="([0-9.]+)pt"', s).group(1)); h = float(re.search(r'height="([0-9.]+)pt"', s).group(1))
    vb = re.search(r'viewBox="([^"]+)"', s).group(1)
    head = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="' + vb + '" style="width:min(100%,' + f"{w*1.35:.0f}" + 'px);height:auto;display:block" role="img" class="math">'
    s = re.sub(r'<svg[^>]*>', lambda m: head, s, count=1)
    s = re.sub(r'<metadata>.*?</metadata>', '', s, flags=re.S)
    s = s.replace('fill="#000000"', 'fill="currentColor"').replace('fill:#000000', 'fill:currentColor').replace('stroke="#000000"', 'stroke="currentColor"').replace('stroke:#000000', 'stroke:currentColor')
    return f'<div class="eq">{s}</div>'


EQ = {
    "A": tex(r"$A[\phi;\,n_e]=A_{TXC}+\int\phi\,\rho_{sol}\,dV-\frac{1}{2K}\int|\nabla\phi|^{2}\,dV+A_{cav}[n_e]+n_{mol}\int S_{diel}[n_e]\,\lambda_{diel}(E)\,dV+n_{max}\int S_{ion}[n_e]\,\lambda_{ion}(\phi)\,dV$"),
    "lion": tex(r"$\lambda_{ion}(\phi)=-\frac{1}{\beta}\ln\left(1-\theta_b+\theta_b\cosh(z\beta\phi)\right),\qquad \lambda_{ion}(0)=0,\qquad \theta_b=0.436$"),
    "ldiel": tex(r"$\lambda_{diel}=\lambda_{rot}+\lambda_{pol}+\lambda_{sic},\quad \lambda_{rot}=-\frac{1}{\beta}\ln\frac{\sinh y}{y},\ \ y=p\beta E_{loc},\quad \lambda_{pol}=-\frac{\alpha_{pol}E_{loc}^{2}}{2K},\quad \lambda_{sic}=\frac{[(\alpha_{rot}\,g(y)+\alpha_{pol})E_{loc}]^{2}}{2\alpha_{sic}K},\quad E_{loc}=f_{loc}E$"),
    "cav": tex(r"$A_{cav}=\tau\int|\nabla S_{cav}[n_e]|\,dV\;\approx\;\tau\int\sqrt{|\nabla S|^{2}+\varepsilon}\,dV,\qquad \varepsilon_{energy}=10^{-12},\ \ \varepsilon_{feature}=10^{-8}$"),
    "closure": tex(r"$\rho_b=-\frac{d}{dz}\left[w_b*(a_1E+p_{off})\right],\quad E=-\frac{d}{dz}(w_b*\phi),\quad a_1(z)=\langle a(E_{scr})\rangle,\quad p_{off}(z)=\langle a(E_{scr})E_{scr}\rangle-a_1\langle E_{scr}\rangle+\delta p$"),
    "stat": tex(r"$\frac{\delta A_{1D}}{\delta\phi}=0\ \Leftrightarrow\ R(\phi)=n_b(\phi)+n_{ion}(\phi)-L_0(\phi-\phi_{sol})+q_{sol}=0$"),
    "env": tex(r"$\frac{\delta A^{*}_{solv}}{\delta n_e(\mathbf{r})}=\left.\frac{\partial A_{1D}}{\partial n_e(\mathbf{r})}\right|_{\phi=\phi^{*}}=-e\,\phi_{solv}(\mathbf{r})+v_{new}(\mathbf{r})$"),
    "vnew": tex(r"$v_{new}(\mathbf{r})=\left.\frac{\partial\,(A_{cav}+A_{diel}+A_{ion})}{\partial n_e(\mathbf{r})}\right|_{\phi=\phi^{*}}=v_{cav}+v_{diel}+v_{ion}$"),
    "vterms": tex(r"$v_{cav}=\tau\frac{\partial A_{cav}}{\partial n_e(\mathbf{r})},\quad v_{ion}=n_{max}\sum_z\lambda_{ion}(\phi^{*}_z)\frac{\partial S_{ion,z}}{\partial n_e(\mathbf{r})},\quad v_{diel}=\sum_z\left[\frac{\partial C_z}{\partial n_e(\mathbf{r})}-\frac{E^{*2}_z}{2}\frac{\partial a_{1,z}}{\partial n_e(\mathbf{r})}-E^{*}_z\frac{\partial p_{off,z}}{\partial n_e(\mathbf{r})}\right]$"),
    "partial": tex(r"$g(n,\phi)=\partial_n A(n,\phi),\qquad v_{new}=g(n_e,\ \phi^{*}[n_e])$"),
    "proj": tex(r"$F_k(\mathbf{R}_i)=-\int v_{new}(\mathbf{r})\,G_{\sigma_k}(\mathbf{r}-\mathbf{R}_i)\,d\mathbf{r},\qquad \mathbf{F}^{(1)}_k(\mathbf{R}_i)=-\nabla_{\mathbf{R}_i}\int v_{new}(\mathbf{r})\,G_{\sigma_k}(\mathbf{r}-\mathbf{R}_i)\,d\mathbf{r}$"),
    "phi1d": tex(r"$\phi_{1D}(z)=\mathcal{P}^{-1}\left[\rho_{neutral}-\rho_e^{GTO}+\rho_{ion}^{POTCAR}+\rho_{solv}\right](z)+\phi_{saw}(z;\,\mu),\qquad \mu=\mu_{pred}$"),
    "step": tex(r"$\Delta\phi_{step}=\frac{4\pi K\,\mu_b}{A},\qquad E_{ramp}=-\frac{\Delta\phi_{step}}{H},\qquad K=14.40\ \mathrm{eV\cdot Å},\ A=189.7\ \mathrm{Å^2},\ H=45\ \mathrm{Å}$"),
}

# ---------------------------------------------------------------- data
KEYS = [("F", r"RMSE_F=\s*([0-9.]+)"), ("dens3d", r"RMSE_density_3d=([0-9.]+)"), ("E", r"RMSE_E_per_atom=\s*([0-9.]+)"),
        ("Phi1D", r"RMSE_potential_1d_profile=([0-9.]+)"), ("occ", r"RMSE_occ_aug=([0-9.]+)")]
KEY_CN = {"F": "力 F", "dens3d": "3D 密度", "E": "能量 E", "Phi1D": "一维电势 Φ1D", "occ": "占据数"}


def parse_log(path):
    out, ts = {}, {}
    for line in open(path, errors="ignore"):
        m = re.search(r"^(\S+ \S+) INFO: Epoch (\d+): ", line)
        if not m:
            continue
        row = {}
        for k, pat in KEYS:
            mm = re.search(pat, line); row[k] = float(mm.group(1)) if mm else float("nan")
        out[int(m.group(2))] = row
    return out


gate = parse_log(f"{B}/3-residual_3D/gate_vsolv/run.log"); prod = parse_log(f"{B}/3-residual_3D/prod500_w1000_ref/run.log"); ref = parse_log(f"{B}/3-residual_3D/w1000_ref/run.log")
EP = [e for e in range(20, 34) if e in gate and e in prod and e in ref]
rg = {k: [gate[e][k] / prod[e][k] for e in EP] for k, _ in KEYS}; rr = {k: [ref[e][k] / prod[e][k] for e in EP] for k, _ in KEYS}
outside = {k: sum(1 for v in rg[k] if v > max(rr[k]) or v < min(rr[k])) for k, _ in KEYS}


def mean(x):
    x = [v for v in x if v == v]; return sum(x) / len(x) if x else float("nan")


def rms(x):
    x = [v for v in x if v == v]; return math.sqrt(sum(v * v for v in x) / len(x)) if x else float("nan")


def struct_numbers(tag):
    d = np.load(f"{L}/density_profile_{tag}.npz", allow_pickle=True); e = np.load(f"{L}/ebl_audit_{tag}.npz", allow_pickle=True)
    fr = {int(r["sid"]): r for r in d["frames"]}; pr = {int(r["k"]): r for r in d["pairs"]}; eb = {int(r["sid"]): r for r in e["frames"]}
    out = {}
    for st in ("charged", "neutral"):
        sids = [s for s in fr if fr[s]["state"].startswith(st)]
        out[f"F_{st}"] = rms([eb[s]["F_rms_err_meV"] for s in sids if s in eb]); out[f"rmse3d_{st}"] = mean([fr[s]["rmse3d"] for s in sids])
        out[f"tailbot_{st}"] = mean([fr[s]["tail_bottom_L1_e"] for s in sids]); out[f"tailtop_{st}"] = mean([fr[s]["tail_top_L1_e"] for s in sids])
        out[f"solvL1_{st}"] = mean([eb[s].get("rho_solv_L1_model_vs_dft", np.nan) for s in sids if s in eb])
    out["dnL1"] = mean([pr[k]["L1_e"] for k in pr]); return out


ST = {(arm, ep): struct_numbers(f"{run}_e{ep}") for arm, run in (("gate", "gate_vsolv"), ("prod", "prod500_w1000_ref")) for ep in (25, 30, 33)}
STRUCT_ROWS = [("力误差，带电帧", "F_charged"), ("力误差，中性帧", "F_neutral"), ("3D 密度误差，带电帧", "rmse3d_charged"), ("3D 密度误差，中性帧", "rmse3d_neutral"),
               ("真空侧电子尾部误差", "tailbot_charged"), ("溶剂侧电子尾部误差，中性帧", "tailtop_neutral"), ("充电响应 Δn(z) 误差", "dnL1"), ("溶剂剖面误差，带电帧", "solvL1_charged")]
CR = json.load(open(f"{L}/cr_all.json")); cks = {c["label"]: c for c in CR["checkpoints"]}
prod_cks = sorted([c for c in CR["checkpoints"] if c["flag_in_training"] == "OFF"], key=lambda c: c["epoch"])
RG = CR["regions"]; regmean = {k: mean([r[k] for r in RG.values()]) for k in ("b1", "b2", "b3", "b4")}
pb = np.load(f"{B}/claude/2-1D_PB/exp_pair122_prod500/structure_pair_sid122_722.npz", allow_pickle=True)
pf = np.load(f"{B}/claude/2-1D_PB/exp_pair122_prod500/structure_pair_sid122_722_fix.npz", allow_pickle=True)
zphi = pb["z_phi_n"].astype(float); res_b = (pb["phi_pred_cmp_n"] - pb["phi_ref_cmp_n"]).astype(float); res_a = (pf["phi_pred_cmp_n"] - pf["phi_ref_cmp_n"]).astype(float)
mu_b = float(pb["mu_bound_n"]); A_CELL, H_CELL, K_C = 189.745476216, 45.0, 14.3996
step_pred = 4 * math.pi * K_C * mu_b / A_CELL; ramp_pred = -step_pred / H_CELL
def at(v, zz): return float(v[int(round(zz / (zphi[1] - zphi[0])))])
m2444 = (zphi >= 24) & (zphi <= 44.5)
ramp_b = float(np.polyfit(zphi[m2444], res_b[m2444], 1)[0]); ramp_a = float(np.polyfit(zphi[m2444], res_a[m2444], 1)[0])
step_b = at(res_b, 18) - at(res_b, 2); step_a = at(res_a, 18) - at(res_a, 2)
# epoch wall times from the log timestamps
def epoch_seconds(path, lo, hi):
    t = {}
    import datetime
    for line in open(path, errors="ignore"):
        m = re.search(r"^(\S+ \S+) INFO: Epoch (\d+):", line)
        if m: t[int(m.group(2))] = datetime.datetime.strptime(m.group(1)[:23], "%Y-%m-%d %H:%M:%S.%f")
    d = [(t[e] - t[e - 1]).total_seconds() for e in range(lo, hi + 1) if e in t and e - 1 in t]
    d = [x for x in d if x < 3000]
    return mean(d), t
ep_prod, tprod = epoch_seconds(f"{B}/3-residual_3D/prod500_w1000_ref/run.log", 21, 33)
ep_gate, _ = epoch_seconds(f"{B}/3-residual_3D/gate_vsolv/run.log", 21, 33)
ep_prod_late, _ = epoch_seconds(f"{B}/3-residual_3D/prod500_w1000_ref/run.log", 100, 499)
ep_warm, _ = epoch_seconds(f"{B}/3-residual_3D/prod500_w1000_ref/run.log", 1, 19)
total_prod_h = (tprod[499] - tprod[0]).total_seconds() / 3600
est_new_h = (20 * ep_warm * 1.12 + 480 * ep_gate) / 3600

# ---------------------------------------------------------------- SVG helpers
W, H = 960, 430; ML, MR, MT, MB = 72, 24, 58, 56; charts_js = {}


def tw(text): return sum(13.0 if ord(ch) > 0x2E7F else 7.0 for ch in text)


def nice_ticks(lo, hi, n=6):
    span = hi - lo
    if span <= 0: return [lo]
    mag = 10 ** int(f"{span / n:e}".split("e")[1]); step = mag
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= n: break
    v = (int(lo / step) - 1) * step; t = []
    while v <= hi + 1e-12:
        if v >= lo - 1e-12: t.append(round(v, 10))
        v += step
    return t


def fmt(v): return f"{v:g}" if abs(v) < 1e4 else f"{v:.2e}"


def line_chart(cid, series, xlabel, ylabel, regions=None, y0line=True, bands=None, unit="", hlines=None, xticks=None):
    xs = [v for s in series for v in s["x"]]; ys = [v for s in series for v in s["y"]]
    if bands:
        for _, blo, bhi, _ in bands: ys += list(blo) + list(bhi)
    if hlines: ys += [h[0] for h in hlines]
    x0, x1 = min(xs), max(xs)
    if regions is None:
        xp = 0.025 * (x1 - x0); x0 -= xp; x1 += xp
    ylo, yhi = min(ys), max(ys)
    if y0line: ylo, yhi = min(ylo, 0.0), max(yhi, 0.0)
    pad = 0.06 * (yhi - ylo if yhi > ylo else 1.0); ylo -= pad; yhi += pad
    pw, ph = W - ML - MR, H - MT - MB
    def X(v): return ML + (v - x0) / (x1 - x0) * pw
    def Y(v): return MT + (yhi - v) / (yhi - ylo) * ph
    o = [f'<svg id="{cid}" viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{html.escape(ylabel)}" class="fig" data-xlo="{x0}" data-xhi="{x1}">']
    if regions:
        for a, b, nm, lab in ((x0, regions["b1"], "vac", ""), (regions["b1"], regions["b2"], "el", "电极片层"), (regions["b2"], regions["b3"], "if", "界面"), (regions["b3"], regions["b4"], "wa", "水层"), (regions["b4"], x1, "vac", "")):
            a, b = max(a, x0), min(b, x1)
            if b > a:
                o.append(f'<rect x="{X(a):.1f}" y="{MT}" width="{X(b)-X(a):.1f}" height="{ph}" class="band {nm}"/>')
                if lab: o.append(f'<text x="{(X(a)+X(b))/2:.1f}" y="{MT-8}" class="bandlbl">{lab}</text>')
    if bands:
        for bx, blo, bhi, cls in bands:
            pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(bx, bhi)) + " " + " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(reversed(list(bx)), reversed(list(blo))))
            o.append(f'<polygon points="{pts}" class="rangeband {cls}"/>')
    for t in (xticks if xticks is not None else nice_ticks(x0, x1, 8)):
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{MT}" y2="{MT+ph}" class="grid"/>'); o.append(f'<text x="{X(t):.1f}" y="{MT+ph+18}" class="tick">{fmt(t)}</text>')
    for t in nice_ticks(ylo, yhi, 6):
        o.append(f'<line x1="{ML}" x2="{ML+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>'); o.append(f'<text x="{ML-8}" y="{Y(t)+4:.1f}" class="tick r">{fmt(t)}</text>')
    if y0line and ylo < 0 < yhi: o.append(f'<line x1="{ML}" x2="{ML+pw}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="zero"/>')
    if hlines:
        for yv, lab, cls in hlines:
            o.append(f'<line x1="{ML}" x2="{ML+pw}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" class="hline {cls}"/>'); o.append(f'<text x="{ML+pw-4}" y="{Y(yv)-5:.1f}" class="hlbl r">{html.escape(lab)}</text>')
    o.append(f'<rect x="{ML}" y="{MT}" width="{pw}" height="{ph}" class="frame"/>')
    for s in series:
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(s["x"], s["y"])); dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
        o.append(f'<polyline points="{pts}" class="line {s["cls"]}"{dash}/>')
        if s.get("dots"):
            for a, b in zip(s["x"], s["y"]): o.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="4" class="dot {s["cls"]}"/>')
    lx = ML + pw
    for s in reversed(series):
        lx -= tw(s["label"]) + 34; dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
        o.append(f'<line x1="{lx}" x2="{lx+22}" y1="{MT-34}" y2="{MT-34}" class="line {s["cls"]}"{dash}/>'); o.append(f'<text x="{lx+27}" y="{MT-30}" class="lgd">{html.escape(s["label"])}</text>')
    o.append(f'<text x="{ML+pw/2:.1f}" y="{H-14}" class="axis">{html.escape(xlabel)}</text>'); o.append(f'<text transform="translate(16,{MT+ph/2:.1f}) rotate(-90)" class="axis">{html.escape(ylabel)}</text>')
    o.append(f'<line class="xh" x1="0" x2="0" y1="{MT}" y2="{MT+ph}" style="opacity:0"/></svg>')
    xs0 = list(series[0]["x"])
    charts_js[cid] = {"x": [round(float(v), 4) for v in xs0], "unit": unit, "series": [{"n": s["label"], "y": [round(float(v), 4) for v in np.interp(xs0, np.asarray(s["x"], float), np.asarray(s["y"], float))]} for s in series]}
    return "\n".join(o)


def bar_chart(groups, ylabel, unit=""):
    n = len(groups); pw, ph = W - ML - MR, H - MT - MB; yhi = max(g[1] for g in groups) * 1.15
    def Y(v): return MT + (yhi - v) / yhi * ph
    slot = pw / n; bw = min(90, slot * 0.5)
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{html.escape(ylabel)}" class="fig">']
    for t in nice_ticks(0, yhi, 6):
        o.append(f'<line x1="{ML}" x2="{ML+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>'); o.append(f'<text x="{ML-8}" y="{Y(t)+4:.1f}" class="tick r">{fmt(t)}</text>')
    for i, (lab, v, cls) in enumerate(groups):
        cx = ML + slot * (i + 0.5); top = Y(v); base = Y(0)
        o.append(f'<rect x="{cx-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{base-top:.1f}" rx="4" class="bar {cls}"><title>{html.escape(lab)}: {v:g} {unit}</title></rect>')
        o.append(f'<text x="{cx:.1f}" y="{top-6:.1f}" class="val">{v:g} {unit}</text>')
        for j, part in enumerate(lab.split("\n")): o.append(f'<text x="{cx:.1f}" y="{base+18+j*15:.1f}" class="tick">{html.escape(part)}</text>')
    o.append(f'<line x1="{ML}" x2="{ML+pw}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="zero"/>')
    o.append(f'<text transform="translate(16,{MT+ph/2:.1f}) rotate(-90)" class="axis">{html.escape(ylabel)}</text></svg>')
    return "\n".join(o)


def dot_rows(rows, cols, xlabel):
    n = len(rows); rowh = 30; Hd = MT + n * rowh + MB; ml = 300; pw = W - ml - MR
    vals = [v for _, d in rows for v in d.values()]; x0, x1 = min(min(vals), 0.0), max(max(vals), 0.0); pad = 0.08 * (x1 - x0); x0 -= pad; x1 += pad
    def X(v): return ml + (v - x0) / (x1 - x0) * pw
    o = [f'<svg viewBox="0 0 {W} {Hd}" width="100%" role="img" aria-label="{html.escape(xlabel)}" class="fig">']
    for t in nice_ticks(x0, x1, 8):
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{MT}" y2="{MT+n*rowh}" class="grid"/>'); o.append(f'<text x="{X(t):.1f}" y="{MT+n*rowh+18}" class="tick">{fmt(t)}</text>')
    o.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{MT}" y2="{MT+n*rowh}" class="zero"/>')
    for i, (lab, d) in enumerate(rows):
        cy = MT + rowh * (i + 0.5)
        if i % 2 == 0: o.append(f'<rect x="{ml}" y="{cy-rowh/2:.1f}" width="{pw}" height="{rowh}" class="rowband"/>')
        o.append(f'<text x="{ml-10}" y="{cy+4:.1f}" class="rowlbl r">{html.escape(lab)}</text>')
        for name, cls in cols:
            if name in d and d[name] == d[name]:
                o.append(f'<circle cx="{X(d[name]):.1f}" cy="{cy:.1f}" r="6" class="dot {cls}"><title>{html.escape(lab)} · {html.escape(name)}: {d[name]:+.1f}%</title></circle>')
    lx = ml
    for name, cls in cols:
        o.append(f'<circle cx="{lx+6}" cy="{MT-30}" r="6" class="dot {cls}"/>'); o.append(f'<text x="{lx+16}" y="{MT-26}" class="lgd">{html.escape(name)}</text>'); lx += tw(name) + 40
    o.append(f'<text x="{ml+pw/2:.1f}" y="{Hd-12}" class="axis">{html.escape(xlabel)}</text></svg>')
    return "\n".join(o)


def pipeline_svg():
    bx = [("原子特征 + 电荷 / 密度头", 30, 40, 200, 56, "n"), ("第一阶段：一维 PB 求解（不用头）\nφ*(z)，空腔 S_cav、S_diel、S_ion", 270, 40, 300, 56, "n"),
          ("反应电势 −e φ_solv\n（已有输入：补偿特征）", 610, 40, 320, 56, "old"), ("A_cav + A_diel + A_ion 在 φ = φ* 处\n对 n_e 求偏导 → v_new(r)", 270, 150, 300, 56, "new"),
          ("投影：接收高斯 σ_k 处的值与梯度\n×(−1)，不解泊松、不去均值", 610, 150, 320, 56, "new"), ("第二阶段：带头的 PB 求解\n能量、力、电势、费米、密度", 270, 260, 300, 56, "n"),
          ("训练损失（E / F / 电势 / 费米 / Φ1D / 3D 密度 / 占据数）", 30, 370, 900, 40, "n")]
    o = ['<svg viewBox="0 0 960 430" width="100%" role="img" aria-label="流程图" class="fig diag">',
         '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="arrhead"/></marker></defs>']
    for t, x, y, w, h, cls in bx:
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="box {cls}"/>')
        lines = t.split("\n")
        for i, line in enumerate(lines): o.append(f'<text x="{x+w/2}" y="{y+h/2+(i-(len(lines)-1)/2)*16+5}" class="boxtxt">{html.escape(line)}</text>')
    def arrow(x1, y1, x2, y2, cls="n"): o.append(f'<path d="M{x1},{y1} L{x2},{y2}" class="arrow {cls}" marker-end="url(#arr)"/>')
    arrow(230, 68, 268, 68); arrow(570, 68, 608, 68); arrow(420, 96, 420, 148, "new"); arrow(570, 178, 608, 178, "new"); arrow(770, 96, 770, 148, "old"); arrow(770, 206, 770, 230, "old"); arrow(420, 206, 420, 258, "new")
    o.append('<path d="M770,230 L420,230" class="arrow old" marker-end="url(#arr)"/>'); arrow(420, 316, 420, 368); arrow(130, 96, 130, 368)
    o.append('<text x="440" y="122" class="lbl new">新增（开关 solvent_pb1d_vsolv_input）</text><text x="790" y="122" class="lbl old">原有的溶剂通道</text>')
    o.append('<text x="30" y="420" class="note">两条溶剂输入都用第一阶段（滞后一步）的场，不加外层自洽循环。</text></svg>')
    return "\n".join(o)


# ---------------------------------------------------------------- figures
fig_pipe = pipeline_svg()
lo = [min(rr["F"])] * len(EP); hi = [max(rr["F"])] * len(EP)
fig_F = line_chart("fF", [{"x": EP, "y": rg["F"], "label": "开 / 关（闸门 / 生产）", "cls": "s2", "dots": True}, {"x": EP, "y": rr["F"], "label": "同配置重复训练 / 生产（随机底线）", "cls": "s1", "dots": True, "dash": True}],
                   "训练轮次 (epoch)", "验证集力误差比值", y0line=False, bands=[(EP, lo, hi, "s1")], hlines=[(1.0, "1.0", "s6")], xticks=EP)
rows = []
for lab, key in STRUCT_ROWS:
    d = {}
    for ep, nm in ((25, "第 25 轮"), (30, "第 30 轮"), (33, "第 33 轮")):
        a, b = ST[("prod", ep)][key], ST[("gate", ep)][key]; d[nm] = 100.0 * (b / a - 1.0)
    rows.append((lab, d))
fig_struct = dot_rows(rows, [("第 25 轮", "s1"), ("第 30 轮", "s2"), ("第 33 轮", "s3")], "开 / 关 − 1（%）；正值 = 开着更差")
ag, ap = cks["gate_e33"]["aggregate_all"], cks["prod_e33"]["aggregate_all"]
fig_resp = line_chart("fR", [{"x": ap["z"], "y": ap["R_dft_per_e"], "label": "DFT", "cls": "s6"}, {"x": ap["z"], "y": ap["R_ml_per_e"], "label": "关（生产，第 33 轮）", "cls": "s1", "dash": True}, {"x": ag["z"], "y": ag["R_ml_per_e"], "label": "开（闸门，第 33 轮）", "cls": "s2"}],
                      "z (Å)", "充电响应 R / ΔN_e (1/Å)", regions=regmean, unit="1/Å")
ep_ck = [c["epoch"] for c in prod_cks]
def es(c, sp): return mean([p["ratio"] for p in c["pairs"].values() if p["split"] == sp])
fig_es = line_chart("fES", [{"x": ep_ck, "y": [es(c, "train") for c in prod_cks], "label": "训练集 27 对", "cls": "s1", "dots": True}, {"x": ep_ck, "y": [es(c, "val") for c in prod_cks], "label": "验证集 20 对", "cls": "s2", "dots": True}],
                    "训练轮次 (epoch)", "响应误差 / 响应大小  E/S", y0line=True)
fig_fix = line_chart("fFix", [{"x": zphi, "y": res_b, "label": "修复前", "cls": "s2"}, {"x": zphi, "y": res_a, "label": "修复后", "cls": "s1"}], "z (Å)", "一维电势残差 模型 − DFT (eV)", regions=regmean, unit="eV")
fig_time = bar_chart([("关\n生产 prod500_w1000_ref", round(ep_prod), "s1"), ("开\n闸门 gate_vsolv", round(ep_gate), "s2")], "每个 epoch 的墙钟时间 (s)，PB 阶段第 21–33 轮均值", "s")


def table(head, rows, cls=""):
    return f'<div class="tw"><table class="{cls}"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>"


tbl_ratio = table(["验证指标", "开 / 关 均值（20–33 轮）", "随机底线范围", "开在底线外的轮数"],
                  [[KEY_CN[k], f"{mean(rg[k]):.3f}", f"{min(rr[k]):.3f} – {max(rr[k]):.3f}", f"{outside[k]} / {len(EP)}"] for k, _ in KEYS])
tbl_acc = table(["检查", "结果"], [
    ["偏导 vs 有限差分（固定 φ，逐项）", "相对偏差 1e-6 量级（空腔 4e-6、介电 5e-7、离子 3e-6）"],
    ["自由能表达式 = 母模型", "−dλ_diel/dE 与母模型极化 p(E) 相对偏差 ≤ 7e-7"],
    ["力：自动微分 vs 有限差分（六分量）", "开 0.21 meV/Å，关 0.21 meV/Å"],
    ["力损失对参数的梯度 AD vs FD", "开 9e-3 / 8e-6（关 6e-3 / 1e-7，受 FD 步长限制）"],
    ["特征自身的导数 AD vs FD", "修复两处缺陷后 0.3%（−0.0915 vs −0.0915）"],
], "wrap")
tbl_fix = table(["", "预测（μ_b = %+.3f e·Å）" % mu_b, "修复前", "修复后"], [
    ["溶剂区与板下真空的台阶", f"{step_pred:+.3f} eV", f"{step_b:+.3f} eV", f"{step_a:+.3f} eV"],
    ["24–44.5 Å 的匀强场", f"{ramp_pred:+.4f} eV/Å", f"{ramp_b:+.4f} eV/Å", f"{ramp_a:+.4f} eV/Å"],
    ["中性帧 Φ1D 残差 rms", "", f"{rms(res_b):.3f} eV", f"{rms(res_a):.3f} eV"],
])
e33, p33 = ST[("gate", 33)], ST[("prod", 33)]


def fig(title, svg, note=""):
    return f'<figure><figcaption>{html.escape(title)}</figcaption>{svg}{("<p class=note>" + html.escape(note) + "</p>") if note else ""}</figure>'


page = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Solvent Potential Input</title>
<style>
:root{{--bg:#f9f9f7;--surf:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mute:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--rule:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s6:#008300;--band-el:#efe9dc;--band-if:#f3ecdc;--band-wa:#e2ebf3;--band-vac:#f3f2ee;--new:#fbe7dd;--old:#dfe9f6;--box:#fcfcfb}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0d0d0d;--surf:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--mute:#898781;--grid:#2c2c2a;--axis:#383835;--rule:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s6:#0ca30c;--band-el:#2a2a24;--band-if:#2c2820;--band-wa:#20272e;--band-vac:#1e1e1c;--new:#3a2a22;--old:#22303f;--box:#1f1f1e}}}}
:root[data-theme="dark"]{{--bg:#0d0d0d;--surf:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--mute:#898781;--grid:#2c2c2a;--axis:#383835;--rule:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s6:#0ca30c;--band-el:#2a2a24;--band-if:#2c2820;--band-wa:#20272e;--band-vac:#1e1e1c;--new:#3a2a22;--old:#22303f;--box:#1f1f1e}}
html,body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 system-ui,-apple-system,"Segoe UI","Noto Sans SC",sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:32px 16px 72px}}
h1{{font-size:1.75rem;margin:0 0 6px;text-wrap:balance}} h2{{font-size:1.2rem;margin:44px 0 12px;padding-top:14px;border-top:1px solid var(--rule)}} h3{{font-size:1rem;margin:22px 0 6px;color:var(--ink2)}}
p,li{{max-width:78ch}} .sub{{color:var(--ink2);margin:0 0 18px}} .note{{color:var(--ink2);font-size:.92rem;margin:6px 0 0}}
.eq{{margin:10px 0 12px;overflow-x:auto;padding:2px 0}} .eq svg{{color:var(--ink)}}
.tw{{overflow-x:auto;margin:10px 0 16px}} table{{border-collapse:collapse;width:100%;font-size:.9rem;font-variant-numeric:tabular-nums;background:var(--surf)}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap;vertical-align:top}} th{{color:var(--ink2);font-weight:600}} td:first-child,th:first-child{{text-align:left;white-space:normal}}
table.wrap td{{white-space:normal;text-align:left}}
figure{{margin:26px 0}} figcaption{{font-size:.95rem;color:var(--ink2);margin:0 0 6px}}
svg.fig{{display:block;background:var(--surf);border:1px solid var(--grid)}}
.grid{{stroke:var(--grid);stroke-width:1}} .frame{{fill:none;stroke:var(--axis)}} .zero{{stroke:var(--mute);stroke-width:1}} .hline{{stroke-width:1;stroke-dasharray:3 4}}
.tick{{font-size:11px;fill:var(--mute);text-anchor:middle}} .tick.r{{text-anchor:end}} .axis{{font-size:12px;fill:var(--ink2);text-anchor:middle}} .lgd,.rowlbl{{font-size:12px;fill:var(--ink)}} .rowlbl.r,.hlbl.r{{text-anchor:end}} .hlbl{{font-size:11px;fill:var(--ink2)}}
.val{{font-size:13px;fill:var(--ink);text-anchor:middle}} .bandlbl{{font-size:12px;fill:var(--mute);text-anchor:middle}}
.band.el{{fill:var(--band-el)}} .band.if{{fill:var(--band-if)}} .band.wa{{fill:var(--band-wa)}} .band.vac{{fill:var(--band-vac)}} .rowband{{fill:var(--band-vac)}}
.rangeband{{opacity:.18}} .rangeband.s1{{fill:var(--s1)}}
.line{{fill:none;stroke-width:2;stroke-linejoin:round}} .dot{{stroke:var(--surf);stroke-width:2}}
.s1{{stroke:var(--s1)}} .s2{{stroke:var(--s2)}} .s3{{stroke:var(--s3)}} .s6{{stroke:var(--s6)}}
.dot.s1,.bar.s1{{fill:var(--s1)}} .dot.s2,.bar.s2{{fill:var(--s2)}} .dot.s3,.bar.s3{{fill:var(--s3)}} .dot.s6{{fill:var(--s6)}}
.xh{{stroke:var(--ink2);stroke-width:1}}
.diag .box{{fill:var(--box);stroke:var(--axis);stroke-width:1.2}} .diag .box.new{{fill:var(--new);stroke:var(--s2)}} .diag .box.old{{fill:var(--old);stroke:var(--s1)}}
.diag .boxtxt{{font-size:13px;fill:var(--ink);text-anchor:middle}} .diag .arrow{{fill:none;stroke:var(--ink2);stroke-width:1.6}} .diag .arrow.new{{stroke:var(--s2)}} .diag .arrow.old{{stroke:var(--s1)}}
.diag .arrhead{{fill:var(--ink2)}} .diag .lbl{{font-size:12px;fill:var(--ink2)}} .diag .lbl.new{{fill:var(--s2)}} .diag .lbl.old{{fill:var(--s1)}} .diag .note{{font-size:12px;fill:var(--ink2)}}
#tip{{position:fixed;pointer-events:none;background:var(--surf);border:1px solid var(--grid);border-radius:6px;padding:5px 9px;font-size:12.5px;opacity:0;z-index:9;font-variant-numeric:tabular-nums;box-shadow:0 2px 8px rgba(0,0,0,.12)}}
.kv{{display:grid;grid-template-columns:1fr;gap:8px;margin:12px 0}} .kv div{{background:var(--surf);border:1px solid var(--grid);border-radius:8px;padding:10px 14px}}
@media (min-width:720px){{.kv{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>溶剂有效势输入：理论、验证、训练对照与最终改动</h1>
<p class="sub">把 VASPsol++ 的溶剂有效势 v_solv = δ(A_cav + A_diel + A_ion)/δn_e 在第一阶段场处算出来，作为电荷递归的一个新输入。本页只放对结论有影响的内容；全部数字来自台账 RUNS.md 或本页脚本直接读取的日志与数据。</p>

<h2>0 · 结论</h2>
<div class="kv">
<div><b>输入本身是对的。</b>偏导与有限差分一致到 1e-6，力的自动微分与有限差分差 0.21 meV/Å（与原路径相同），力损失的参数梯度一致到 FD 步长极限。</div>
<div><b>训练对照没有变好。</b>34 轮闸门对生产同轮次：力误差 +5.6%，14/14 轮落在同配置重复训练的随机底线之外；3D 密度 +1.4%（13/14 轮）；能量、电势、Φ1D 都在底线内。充电响应曲线两臂重合。</div>
<div><b>训练变慢 12%。</b>每个 epoch 从 {ep_prod:.0f} s 到 {ep_gate:.0f} s；显存峰值 23.9 → 26.3 GiB。500 轮按此速度约 {est_new_h:.0f} h，超过一个 48 h 作业。</div>
<div><b>最终决定（用户）。</b>保留该输入；修复顺带发现的 Φ1D 构造缺陷（中性带溶剂帧的束缚电荷被清零 → 0.17 eV 假台阶）；以修复后的代码、输入开着重训 500 轮（prod500_vsolv_fix，作业 3462229，gpu-a100 排队中）。</div>
</div>

<h2>1 · 它在模型里的位置</h2>
<p>模型的自洽是 PolarMACE 的一步电荷递归。溶剂原来只通过一条通道进入：第一阶段用密度头解一维 Poisson–Boltzmann，得到横向均匀的溶剂电荷（离子 + 束缚），把它的电势和场投影到接收高斯上作为补偿特征。这对应 VASPsol++ 修正势 v_corr = eφ_solv + v_solv 里的第一项。第二项 v_solv 是空腔、介电、离子三项自由能对电子密度的导数，原来没有；这次把它加成第二条通道，走同样的投影。</p>
{fig("图 1  新输入在两阶段电荷递归中的位置", fig_pipe)}

<h2>2 · 理论</h2>
<h3>2.1 母泛函（VASPsol++，电势形式，φ 为独立变量）</h3>
{EQ["A"]}
{EQ["lion"]}
{EQ["ldiel"]}
{EQ["cav"]}
<p class="note">K = 180.95 eV·Å（库仑常数），f_loc 是局域场因子，g(y) 是转动响应函数；三个 λ 都是母模型的原式，实现时逐项与母模型极化 p(E) = −dλ_diel/dE 核对到 7e-7。</p>
<h3>2.2 一维闭合 = 介电项在屏蔽真空场处的面平均二次展开</h3>
{EQ["closure"]}
{EQ["stat"]}
<p class="note">a₁ 是面平均响应，p_off 是先验极化偏置加头的残差；对 φ 取驻点正好给出求解器的残差方程（含电荷约束与偶极反馈项），这一点作为检查而不是假设。</p>
<h3>2.3 包络定理：哪一项是新的</h3>
{EQ["env"]}
{EQ["vnew"]}
{EQ["vterms"]}
<p class="note">E*、φ* 来自第一阶段的求解（滞后一步，与反应电势同一近似）。a₁、p_off、C_z 通过 S_diel、S_ion、局域场因子和 E_scr 依赖 n_e，这整条显式路径都参与求导。</p>
<h3>2.4 偏导的实现与投影</h3>
{EQ["partial"]}
{EQ["proj"]}
<p class="note">偏导对 n_e 的克隆节点求，否则 φ* 由同一密度张量解出，得到的是总导数（多出 ∂A/∂φ·dφ*/dn，验收时表现为 22 meV/Å 的力缺口）。v_new 是每电子的能量，新增电子是物理电荷 −δn_e，所以特征取 −v_new；F_k 是接收高斯 σ_k 处的平滑值，F_k⁽¹⁾ 是其梯度，与静电通道对反应电势的投影完全相同。面积项的正则化下限：能量用 1e-12，特征用 1e-8（二阶导数被下限曲率支配，1e-12 时偏 33%）。</p>

<h2>3 · 训练前的验证（单帧 sid 28 带电 / 628 中性）</h2>
{tbl_acc}
<p>过程中定位并修复了两处实现缺陷：总导数混入（改为对克隆节点求偏导）和面积正则化下限使二阶导数失真（特征改用 1e-8）。</p>

<h2>4 · 训练成本</h2>
{fig("图 2  每个 epoch 的墙钟时间（三卡，213 步/epoch，含验证）", fig_time, f"预热阶段每 epoch {ep_warm:.0f} s（20 轮），PB 阶段生产 {ep_prod:.0f} s、闸门 {ep_gate:.0f} s（+{100*(ep_gate/ep_prod-1):.0f}%）。生产 500 轮实测 {total_prod_h:.1f} h；开着输入按闸门速度 20 × {ep_warm*1.12:.0f} s + 480 × {ep_gate:.0f} s ≈ {est_new_h:.0f} h，故新生产分两段自链。显存峰值 23.9 → 26.3 GiB（40 GB 卡）。")}

<h2>5 · 训练对照：开 vs 关，同轮次</h2>
<p>闸门 gate_vsolv 用生产配置加开关训练 34 轮，与生产 prod500_w1000_ref 的 0–33 轮比。同配置、同 seed 的两次训练本身就不逐位一致（三卡 GPU 的运行间随机性），所以每个比值都和同配置重复训练 w1000_ref 对生产的比值范围对照：只有整段都落在范围外的差别才算信号。</p>
{fig("图 3  验证集力误差：开/关 比值与随机底线", fig_F, f"闸门 {outside['F']}/{len(EP)} 轮在底线外，均值 {mean(rg['F']):.3f}；差距随训练从 1.17 收窄到 1.03 但没有进入底线。")}
{tbl_ratio}
{fig("图 4  结构量的相对变化（47 对 94 帧，第 25/30/33 轮）", fig_struct, "三个轮次每个量方向一致：力与 3D 密度变差，真空侧电子尾部变好，溶剂侧尾部（中性帧）变差，充电响应与溶剂剖面在 ±5% 内。")}
{fig("图 5  充电响应 R(z)/ΔN_e（47 对平均，第 33 轮）", fig_resp, "两臂几乎重合：新输入没有改变响应的形状。两臂共同的缺陷是电极片层上的响应不足、水层过多。")}

<h2>6 · 充电响应随训练的变化（保留输入的决定依据）</h2>
{fig("图 6  响应误差比 E/S 随训练轮次（生产检查点）", fig_es, "E = ∫|R_ML − R_DFT| dz，S = ∫|R_DFT| dz。33 轮 0.98 → 200 轮 0.55 → 499 轮 0.50，之后平台；成熟模型电极片层仍少 0.10 e。电子总数每对每检查点守恒到 0.007 e。")}

<h2>7 · 顺带发现并修复：一维电势构造对中性带溶剂帧的缺陷</h2>
<p>一维电势的构造：对总电荷曲线解周期泊松，再加 VASP 式锯齿偶极修正，修正用的偶极取自模型的偶极输出（含求解器的溶剂偶极）。溶剂剖面进曲线前曾按“净电荷对齐到高斯层”乘一个系数；中性帧该系数为 0，束缚电荷被整个抹掉，而偶极修正里的 μ_b 还在，于是留下一个台阶和一个整胞匀强场：</p>
{EQ["phi1d"]}
{EQ["step"]}
{tbl_fix}
{fig("图 7  中性 sid 722 的一维电势残差，修复前后", fig_fix, "修法：去掉系数，所有帧 ρ_solv 按明确符号约定直接进入（提交 c16ab1b）。影响 200 个带溶剂中性帧（sid 601–800）自进训练起的 Φ1D 损失；带电帧数值不变（0.0640 → 0.0641 eV）。")}

<h2>8 · 最终改动与状态</h2>
<ul>
<li>代码：新模块 pb1d_vsolv.py 与接线（开关 solvent_pb1d_vsolv_input，默认关）；loss.py 的一维电势修复；均已推送（远端 7d1fdb4）。</li>
<li>新生产 3-residual_3D/prod500_vsolv_fix：生产配置 + 开关开，seed 123，预热 20 轮，500 轮，三卡 gpu-a100；作业 3462229 于 09-22 06:58 投递，至 09-23 03:00 仍排队（分区 197 个待运行、50 个运行，优先级 1499 对队首 5645）。预算到点写状态并自动续段。</li>
<li>对照方式不变：同轮次对 prod500_w1000_ref，附 w1000_ref 底线；Φ1D 验证指标在 25% 的带溶剂中性帧上定义已变，跨修复前后不可直接比该项。</li>
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
print(f"written {OUT}: {len(page)//1024} kB; epoch s prod {ep_prod:.0f} gate {ep_gate:.0f} warm {ep_warm:.0f}; prod total {total_prod_h:.1f} h; est new {est_new_h:.1f} h")
