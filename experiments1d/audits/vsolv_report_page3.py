"""Report page (third version, English; user 2026-09-23: fewer error metrics, rendered formulas, training length
per epoch, essential tests only, predicted potential curves added to the Phi1D-fix section): the solvent effective-potential input
v_new = d(A_cav + A_diel + A_ion)/dn_e at the stage-1 field. Static HTML; formulas are rendered
offline with matplotlib mathtext into inline SVG (no CDN); the only script is the hover read-out.

  python vsolv_report_page3.py OUT.html
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
KEY_CN = {"F": "forces F", "dens3d": "3-D density", "E": "energy E", "Phi1D": "1-D potential Φ1D", "occ": "occupancies"}


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
STRUCT_ROWS = [("force error, charged frames", "F_charged"), ("force error, neutral frames", "F_neutral"), ("3-D density error, charged", "rmse3d_charged"), ("3-D density error, neutral", "rmse3d_neutral"),
               ("vacuum-side electron tail error", "tailbot_charged"), ("solvent-side electron tail error, neutral", "tailtop_neutral"), ("charging response Δn(z) error", "dnL1"), ("solvent profile error, charged", "solvL1_charged")]
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
        for a, b, nm, lab in ((x0, regions["b1"], "vac", ""), (regions["b1"], regions["b2"], "el", "electrode"), (regions["b2"], regions["b3"], "if", "interface"), (regions["b3"], regions["b4"], "wa", "water"), (regions["b4"], x1, "vac", "")):
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
    bx = [("atom features + charge / density heads", 30, 40, 200, 56, "n"), ("stage 1: 1-D PB solve (no head)\nφ*(z); cavities S_cav, S_diel, S_ion", 270, 40, 300, 56, "n"),
          ("reaction potential −e φ_solv\n(existing input: compensation features)", 610, 40, 320, 56, "old"), ("∂(A_cav + A_diel + A_ion)/∂n_e at φ = φ*\n→ v_new(r)", 270, 150, 300, 56, "new"),
          ("projection: value + gradient at the receiver\nGaussians σ_k, × (−1); no Poisson solve, no de-meaning", 610, 150, 320, 56, "new"), ("stage 2: PB solve with the head\nenergy, forces, potential, Fermi level, density", 270, 260, 300, 56, "n"),
          ("training loss (E / F / potential / Fermi / Φ1D / 3-D density / occupancies)", 30, 370, 900, 40, "n")]
    o = ['<svg viewBox="0 0 960 430" width="100%" role="img" aria-label="pipeline" class="fig diag">',
         '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="arrhead"/></marker></defs>']
    for t, x, y, w, h, cls in bx:
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="box {cls}"/>')
        lines = t.split("\n")
        for i, line in enumerate(lines): o.append(f'<text x="{x+w/2}" y="{y+h/2+(i-(len(lines)-1)/2)*16+5}" class="boxtxt">{html.escape(line)}</text>')
    def arrow(x1, y1, x2, y2, cls="n"): o.append(f'<path d="M{x1},{y1} L{x2},{y2}" class="arrow {cls}" marker-end="url(#arr)"/>')
    arrow(230, 68, 268, 68); arrow(570, 68, 608, 68); arrow(420, 96, 420, 148, "new"); arrow(570, 178, 608, 178, "new"); arrow(770, 96, 770, 148, "old"); arrow(770, 206, 770, 230, "old"); arrow(420, 206, 420, 258, "new")
    o.append('<path d="M770,230 L420,230" class="arrow old" marker-end="url(#arr)"/>'); arrow(420, 316, 420, 368); arrow(130, 96, 130, 368)
    o.append('<text x="440" y="122" class="lbl new">new (flag solvent_pb1d_vsolv_input)</text><text x="790" y="122" class="lbl old">existing solvent channel</text>')
    o.append('<text x="30" y="420" class="note">Both solvent inputs use the stage-1 (lagged) field; no outer self-consistency loop is added.</text></svg>')
    return "\n".join(o)


# ---------------------------------------------------------------- figures
fig_pipe = pipeline_svg()
lo = [min(rr["F"])] * len(EP); hi = [max(rr["F"])] * len(EP)
fig_F = line_chart("fF", [{"x": EP, "y": rg["F"], "label": "ON / OFF (gate / production)", "cls": "s2", "dots": True}, {"x": EP, "y": rr["F"], "label": "same-config replicate / production (noise floor)", "cls": "s1", "dots": True, "dash": True}],
                   "epoch", "validation force-error ratio", y0line=False, bands=[(EP, lo, hi, "s1")], hlines=[(1.0, "1.0", "s6")], xticks=EP)
rows = []
for lab, key in STRUCT_ROWS:
    d = {}
    for ep, nm in ((25, "epoch 25"), (30, "epoch 30"), (33, "epoch 33")):
        a, b = ST[("prod", ep)][key], ST[("gate", ep)][key]; d[nm] = 100.0 * (b / a - 1.0)
    rows.append((lab, d))
fig_struct = dot_rows(rows, [("epoch 25", "s1"), ("epoch 30", "s2"), ("epoch 33", "s3")], "ON / OFF − 1 (%); positive = worse with the input")
ag, ap = cks["gate_e33"]["aggregate_all"], cks["prod_e33"]["aggregate_all"]
fig_resp = line_chart("fR", [{"x": ap["z"], "y": ap["R_dft_per_e"], "label": "DFT", "cls": "s6"}, {"x": ap["z"], "y": ap["R_ml_per_e"], "label": "OFF (production, epoch 33)", "cls": "s1", "dash": True}, {"x": ag["z"], "y": ag["R_ml_per_e"], "label": "ON (gate, epoch 33)", "cls": "s2"}],
                      "z (Å)", "charging response R / ΔN_e (1/Å)", regions=regmean, unit="1/Å")
ep_ck = [c["epoch"] for c in prod_cks]
def es(c, sp): return mean([p["ratio"] for p in c["pairs"].values() if p["split"] == sp])
fig_es = line_chart("fES", [{"x": ep_ck, "y": [es(c, "train") for c in prod_cks], "label": "train, 27 pairs", "cls": "s1", "dots": True}, {"x": ep_ck, "y": [es(c, "val") for c in prod_cks], "label": "validation, 20 pairs", "cls": "s2", "dots": True}],
                    "epoch", "response error / response size  E/S", y0line=True)
fig_fix = line_chart("fFix", [{"x": zphi, "y": res_b, "label": "before the fix", "cls": "s2"}, {"x": zphi, "y": res_a, "label": "after the fix", "cls": "s1"}], "z (Å)", "1-D potential residual, model − DFT (eV)", regions=regmean, unit="eV")
phi_ref = pb["phi_ref_cmp_n"].astype(float); phi_b = pb["phi_pred_cmp_n"].astype(float); phi_a = pf["phi_pred_cmp_n"].astype(float)
fig_phi = line_chart("fPhi", [{"x": zphi, "y": phi_ref, "label": "DFT", "cls": "s6"}, {"x": zphi, "y": phi_b, "label": "model, before the fix", "cls": "s2", "dash": True}, {"x": zphi, "y": phi_a, "label": "model, after the fix", "cls": "s1"}],
                     "z (Å)", "plane-averaged potential φ(z) − φ(upper vacuum) (eV)", regions=regmean, unit="eV")
mz = (zphi >= 18.0) & (zphi <= 44.5)
fig_phiz = line_chart("fPhiZ", [{"x": zphi[mz], "y": phi_ref[mz], "label": "DFT", "cls": "s6"}, {"x": zphi[mz], "y": phi_b[mz], "label": "model, before the fix", "cls": "s2", "dash": True}, {"x": zphi[mz], "y": phi_a[mz], "label": "model, after the fix", "cls": "s1"}],
                      "z (Å)", "φ(z) − φ(upper vacuum) (eV), z = 18–44.5 Å", unit="eV")
fig_time = bar_chart([("OFF\nproduction prod500_w1000_ref", round(ep_prod), "s1"), ("ON\ngate gate_vsolv", round(ep_gate), "s2")], "wall time per epoch (s), PB phase, mean of epochs 21–33", "s")


def table(head, rows, cls=""):
    return f'<div class="tw"><table class="{cls}"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>"


tbl_ratio = table(["validation metric", "ON / OFF, mean over epochs 20–33", "noise-floor range", "epochs with ON outside the floor"],
                  [[KEY_CN[k], f"{mean(rg[k]):.3f}", f"{min(rr[k]):.3f} – {max(rr[k]):.3f}", f"{outside[k]} / {len(EP)}"] for k, _ in KEYS])
tbl_acc = table(["check", "result"], [
    ["partial derivative vs finite differences (φ fixed, term by term)", "relative deviation at the 1e-6 level (cavity 4e-6, dielectric 5e-7, ionic 3e-6)"],
    ["free-energy expressions = parent model", "−dλ_diel/dE against the parent's polarisation p(E): relative deviation ≤ 7e-7"],
    ["forces: autodiff vs finite differences (six components)", "ON 0.21 meV/Å, OFF 0.21 meV/Å"],
    ["force-loss parameter gradient, AD vs FD", "ON 9e-3 / 8e-6 (OFF 6e-3 / 1e-7; limited by the FD step)"],
    ["derivative of the feature itself, AD vs FD", "0.3% after the two fixes (−0.0915 vs −0.0915)"],
], "wrap")
tbl_fix = table(["", "predicted (μ_b = %+.3f e·Å)" % mu_b, "before the fix", "after the fix"], [
    ["step between the solvent region and the vacuum below the slab", f"{step_pred:+.3f} eV", f"{step_b:+.3f} eV", f"{step_a:+.3f} eV"],
    ["uniform field over 24–44.5 Å", f"{ramp_pred:+.4f} eV/Å", f"{ramp_b:+.4f} eV/Å", f"{ramp_a:+.4f} eV/Å"],
    ["Φ1D residual rms, neutral frame", "", f"{rms(res_b):.3f} eV", f"{rms(res_a):.3f} eV"],
])
e33, p33 = ST[("gate", 33)], ST[("prod", 33)]


def fig(title, svg, note=""):
    return f'<figure><figcaption>{html.escape(title)}</figcaption>{svg}{("<p class=note>" + html.escape(note) + "</p>") if note else ""}</figure>'


page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Solvent Potential Input</title>
<style>
:root{{--bg:#f9f9f7;--surf:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mute:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--rule:rgba(11,11,11,.10);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s6:#008300;--band-el:#efe9dc;--band-if:#f3ecdc;--band-wa:#e2ebf3;--band-vac:#f3f2ee;--new:#fbe7dd;--old:#dfe9f6;--box:#fcfcfb}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0d0d0d;--surf:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--mute:#898781;--grid:#2c2c2a;--axis:#383835;--rule:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s6:#0ca30c;--band-el:#2a2a24;--band-if:#2c2820;--band-wa:#20272e;--band-vac:#1e1e1c;--new:#3a2a22;--old:#22303f;--box:#1f1f1e}}}}
:root[data-theme="dark"]{{--bg:#0d0d0d;--surf:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--mute:#898781;--grid:#2c2c2a;--axis:#383835;--rule:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s6:#0ca30c;--band-el:#2a2a24;--band-if:#2c2820;--band-wa:#20272e;--band-vac:#1e1e1c;--new:#3a2a22;--old:#22303f;--box:#1f1f1e}}
html,body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}}
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
<h1>The solvent effective-potential input: theory, verification, training comparison, final changes</h1>
<p class="sub">The VASPsol++ solvent effective potential v_solv = δ(A_cav + A_diel + A_ion)/δn_e, evaluated at the stage-1 field and fed to the charge recursion as a new input. Only what bears on the conclusions is shown; every number comes from the ledger RUNS.md or is read directly from the logs and data files by the page script.</p>

<h2>0 · Conclusions</h2>
<div class="kv">
<div><b>The input itself is correct.</b> Partial derivatives agree with finite differences to 1e-6; forces from autodiff and finite differences differ by 0.21 meV/Å (same as the existing path); the force-loss parameter gradient agrees to the finite-difference step limit.</div>
<div><b>Training did not improve.</b> A 34-epoch gate against production at matched epochs: force error +5.6%, outside the same-configuration replicate's noise floor in 14/14 epochs; 3-D density +1.4% (13/14 epochs); energy, potential and Φ1D stay inside the floor. The charging-response curves of the two arms coincide.</div>
<div><b>Training is 12% slower.</b> One epoch goes from {ep_prod:.0f} s to {ep_gate:.0f} s; peak GPU memory 23.9 → 26.3 GiB. At that rate 500 epochs take about {est_new_h:.0f} h, more than one 48 h job.</div>
<div><b>Final decision (user).</b> Keep the input; fix the Φ1D construction defect found on the way (the bound charge of neutral solvated frames was zeroed → a spurious 0.17 eV step); retrain 500 epochs with the fixed code and the input on (prod500_vsolv_fix, job 3462229, queued on gpu-a100).</div>
</div>

<h2>1 · Where it sits in the model</h2>
<p>The model's self-consistency is PolarMACE's one-step charge recursion. The solvent used to enter through one channel only: stage 1 solves the 1-D Poisson–Boltzmann equation with the density head, obtains the laterally uniform solvent charge (ionic + bound) and projects its potential and field onto the receiver Gaussians as compensation features. That is the first term of the VASPsol++ correction potential v_corr = eφ_solv + v_solv. The second term, the derivative of the cavity, dielectric and ionic free energies with respect to the electron density, was missing; it is now added as a second channel through the same projection.</p>
{fig("Figure 1  Where the new input enters the two-stage charge recursion", fig_pipe)}

<h2>2 · Theory</h2>
<h3>2.1 Parent functional (VASPsol++, potential form, φ the independent variable)</h3>
{EQ["A"]}
{EQ["lion"]}
{EQ["ldiel"]}
{EQ["cav"]}
<p class="note">K = 180.95 eV·Å (Coulomb constant), f_loc the local-field factor, g(y) the rotational response function; the three λ are the parent's own expressions, checked term by term against the parent's polarisation p(E) = −dλ_diel/dE to 7e-7.</p>
<h3>2.2 The 1-D closure = the plane-averaged quadratic expansion of the dielectric term about the screened-vacuum field</h3>
{EQ["closure"]}
{EQ["stat"]}
<p class="note">a₁ is the plane-mean response, p_off the prior polarisation offset plus the head's residual; stationarity in φ gives exactly the solver's residual equation (charge constraint and dipole feedback included), which was verified rather than assumed.</p>
<h3>2.3 Envelope theorem: which term is new</h3>
{EQ["env"]}
{EQ["vnew"]}
{EQ["vterms"]}
<p class="note">E* and φ* come from the stage-1 solve (lagged by one step, the same approximation as the reaction potential). a₁, p_off and C_z depend on n_e through S_diel, S_ion, the local-field factor and E_scr; that whole explicit path is differentiated.</p>
<h3>2.4 Implementation of the partial derivative and the projection</h3>
{EQ["partial"]}
{EQ["proj"]}
<p class="note">The partial is taken with respect to a clone of n_e; otherwise φ*, solved from the same density tensor, turns it into a total derivative (an extra ∂A/∂φ·dφ*/dn, which showed up as a 22 meV/Å force gap during acceptance). v_new is an energy per electron and an added electron is the physical charge −δn_e, so the feature is −v_new; F_k is the value smoothed with the receiver Gaussian σ_k and F_k⁽¹⁾ its gradient, exactly the projection the electrostatic channel applies to the reaction potential. Regularisation floor of the area term: 1e-12 for the energy, 1e-8 for the feature (the second derivative is dominated by the floor's curvature; 33% off at 1e-12).</p>

<h2>3 · Verification before training (single frames sid 28 charged / 628 neutral)</h2>
{tbl_acc}
<p>Two implementation defects were located and fixed on the way: the total derivative leaking in (fixed by differentiating with respect to the cloned node) and the area floor distorting the second derivative (the feature now uses 1e-8).</p>

<h2>4 · Training cost</h2>
{fig("Figure 2  Wall time per epoch (3 GPUs, 213 steps per epoch, validation included)", fig_time, f"Warm-up epochs {ep_warm:.0f} s each (20 epochs); PB phase production {ep_prod:.0f} s, gate {ep_gate:.0f} s (+{100*(ep_gate/ep_prod-1):.0f}%). Production measured {total_prod_h:.1f} h for 500 epochs; with the input on, 20 × {ep_warm*1.12:.0f} s + 480 × {ep_gate:.0f} s ≈ {est_new_h:.0f} h, hence the new production run chains two segments. Peak GPU memory 23.9 → 26.3 GiB on a 40 GB card.")}

<h2>5 · Training comparison: ON vs OFF at matched epochs</h2>
<p>The gate gate_vsolv trains 34 epochs with the production configuration plus the flag and is compared with epochs 0–33 of production prod500_w1000_ref. Two trainings with the same configuration and seed are not bit-identical (run-to-run randomness of 3-GPU training), so every ratio is read against the range of the same-configuration replicate w1000_ref over production: only a difference outside that range over the whole stretch counts as a signal.</p>
{fig("Figure 3  Validation force error: ON/OFF ratio against the noise floor", fig_F, f"The gate lies outside the floor in {outside['F']}/{len(EP)} epochs, mean ratio {mean(rg['F']):.3f}; the gap narrows from 1.17 to 1.03 with training but never enters the floor.")}
{tbl_ratio}
{fig("Figure 4  Relative change of structural quantities (47 pairs, 94 frames, epochs 25/30/33)", fig_struct, "All three epochs move the same way: forces and 3-D density worse, vacuum-side electron tail better, solvent-side tail (neutral frames) worse, charging response and solvent profile within ±5%.")}
{fig("Figure 5  Charging response R(z)/ΔN_e, mean over 47 pairs, epoch 33", fig_resp, "The two arms nearly coincide: the new input does not change the shape of the response. The defect shared by both arms is too little response on the electrode sheet and too much in the water.")}

<h2>6 · Charging response along training (basis of the decision to keep the input)</h2>
{fig("Figure 6  Response-error ratio E/S versus epoch (production checkpoints)", fig_es, "E = ∫|R_ML − R_DFT| dz, S = ∫|R_DFT| dz. 0.98 at epoch 33 → 0.55 at 200 → 0.50 at 499, then a plateau; the mature model still lacks 0.10 e on the electrode sheet. The total added electron count is conserved to 0.007 e on every pair and checkpoint.")}

<h2>7 · Found and fixed on the way: the 1-D potential construction for neutral solvated frames</h2>
<p>The 1-D potential is built by solving the periodic Poisson equation for the total charge profile and adding a VASP-style sawtooth dipole correction whose dipole is the model's dipole output (which includes the solver's solvent dipole). Before entering the profile, the solvent charge used to be multiplied by a coefficient that matched its net charge to the Gaussian layer; for a neutral frame that coefficient is 0, so the whole bound charge was wiped out while the μ_b in the dipole correction stayed, leaving a step and a uniform field across the cell:</p>
{EQ["phi1d"]}
{EQ["step"]}
{tbl_fix}
{fig("Figure 7  Neutral frame sid 722: plane-averaged 1-D potential, DFT and the model before and after the fix", fig_phi, "Aligned to the upper vacuum, whole cell. The slab sits at 6.6–9.4 Å, explicit water up to 16.7 Å, the implicit electrolyte beyond.")}
{fig("Figure 8  The same curves for z = 18–44.5 Å (implicit electrolyte and upper vacuum)", fig_phiz, "Before the fix the model's potential is a straight line from +0.10 eV at 18 Å to −0.02 eV at 44 Å where DFT is flat at 0; after the fix it is flat too.")}
{fig("Figure 9  1-D potential residual (model − DFT), neutral sid 722, before and after the fix", fig_fix, "Fix: the coefficient is removed and ρ_solv enters with the explicit sign convention for every frame (commit c16ab1b). It affects the Φ1D loss of the 200 solvated neutral frames (sid 601–800) since they entered training; charged frames are unchanged (0.0640 → 0.0641 eV).")}

<h2>8 · Final changes and status</h2>
<ul>
<li>Code: the new module pb1d_vsolv.py and its wiring (flag solvent_pb1d_vsolv_input, default off); the 1-D potential fix in loss.py; all pushed (remote 34dc818).</li>
<li>New production run 3-residual_3D/prod500_vsolv_fix: production configuration + flag on, seed 123, 20 warm-up epochs, 500 epochs, 3 GPUs on gpu-a100; job 3462229 submitted 09-22 06:58, still queued at 04:00 on 09-23 (partition: 195 pending, 50 running; priority 1499 against 5645 at the head). It writes its state at the time budget and re-submits its own continuation.</li>
<li>Comparison protocol unchanged: matched epochs against prod500_w1000_ref with the w1000_ref noise floor; the Φ1D validation metric changed its definition on the 25% solvated-neutral frames, so that one metric is not comparable across the fix.</li>
<li>Related pages: charging-response diagnosis https://claude.ai/artifact/NaL6dUc6YimsRfrNC5KhcS ; pair report sid 122/722 of the production model https://claude.ai/artifact/NoEkrbdogoEctkv6wRTtM9 .</li>
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
