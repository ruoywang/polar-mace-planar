"""Static HTML page (tables + hand-drawn SVG figures, no JavaScript) from charge_response_report.py's JSON.

  python charge_response_page.py REPORT.json CONCLUSIONS.txt OUT.html [EARLY_LABEL MATURE_LABEL]
CONCLUSIONS.txt: plain paragraphs (blank-line separated) shown in the last section; lines starting with
"- " become list items. EARLY/MATURE pick the two checkpoints drawn in the curve figures (defaults:
first and last checkpoint of the JSON).
"""
import html
import json
import sys

R = json.load(open(sys.argv[1]))
CONCL = open(sys.argv[2]).read() if sys.argv[2] != "-" else ""
OUT = sys.argv[3]
CKS = R["checkpoints"]
LABELS = [c["label"] for c in CKS]
EARLY = sys.argv[4] if len(sys.argv) > 4 else LABELS[0]
MATURE = sys.argv[5] if len(sys.argv) > 5 else LABELS[-1]
byl = {c["label"]: c for c in CKS}
REG = ["vacuum_below", "electrode", "interface", "water", "vacuum_above"]
REG_CN = {"vacuum_below": "下真空", "electrode": "电极", "interface": "界面", "water": "水层", "vacuum_above": "上真空"}
REP = [28, 60, 1]


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


# ---------- SVG helpers ----------
W, H = 960, 420
ML, MR, MT, MB = 70, 24, 56, 54   # margins; legend + region labels live in the top margin, never on data


def nice_ticks(lo, hi, n=6):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** int(f"{raw:e}".split("e")[1])
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= n:
            break
    start = (int(lo / step) - 1) * step
    t = []
    v = start
    while v <= hi + 1e-12:
        if v >= lo - 1e-12:
            t.append(round(v, 10))
        v += step
    return t


def fmt(v):
    return f"{v:g}" if abs(v) < 1e4 else f"{v:.2e}"


def svg_plot(series, xlabel, ylabel, regions=None, y0line=True, title=None, xr=None, yr=None):
    """series: list of dict(x, y, label, cls, dash). regions: dict b1..b4 (A) for shading."""
    xs = [v for s in series for v in s["x"]]; ys = [v for s in series for v in s["y"]]
    x0, x1 = (min(xs), max(xs)) if xr is None else xr
    ylo, yhi = (min(ys), max(ys)) if yr is None else yr
    if y0line:
        ylo, yhi = min(ylo, 0.0), max(yhi, 0.0)
    pad = 0.06 * (yhi - ylo if yhi > ylo else 1.0); ylo -= pad; yhi += pad
    pw, ph = W - ML - MR, H - MT - MB

    def X(v): return ML + (v - x0) / (x1 - x0) * pw
    def Y(v): return MT + (yhi - v) / (yhi - ylo) * ph
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{html.escape(title or ylabel)}" class="fig">']
    if regions:
        bands = [(x0, regions["b1"], "vacuum_below"), (regions["b1"], regions["b2"], "electrode"),
                 (regions["b2"], regions["b3"], "interface"), (regions["b3"], regions["b4"], "water"), (regions["b4"], x1, "vacuum_above")]
        for a, b, nm in bands:
            a, b = max(a, x0), min(b, x1)
            if b > a:
                out.append(f'<rect x="{X(a):.1f}" y="{MT}" width="{X(b)-X(a):.1f}" height="{ph}" class="band {nm}"/>')
                if nm in ("electrode", "interface", "water"):
                    out.append(f'<text x="{(X(a)+X(b))/2:.1f}" y="{MT-8}" class="bandlbl">{REG_CN[nm]}</text>')
        if "z_sheet_min" in regions:
            for zz, lbl in ((regions["z_sheet_min"], ""), (regions["z_sheet_max"], "C/Ni/N 片层")):
                out.append(f'<line x1="{X(zz):.1f}" x2="{X(zz):.1f}" y1="{MT}" y2="{MT+ph}" class="atomline"/>')
            if regions.get("z_adsorbate_C"):
                zc = regions["z_adsorbate_C"][0]
                out.append(f'<line x1="{X(zc):.1f}" x2="{X(zc):.1f}" y1="{MT}" y2="{MT+ph}" class="atomline ads"/>')
    # axes + grid
    for t in nice_ticks(x0, x1, 8):
        out.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{MT}" y2="{MT+ph}" class="grid"/>')
        out.append(f'<text x="{X(t):.1f}" y="{MT+ph+18}" class="tick">{fmt(t)}</text>')
    for t in nice_ticks(ylo, yhi, 6):
        out.append(f'<line x1="{ML}" x2="{ML+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>')
        out.append(f'<text x="{ML-8}" y="{Y(t)+4:.1f}" class="tick r">{fmt(t)}</text>')
    if y0line and ylo < 0 < yhi:
        out.append(f'<line x1="{ML}" x2="{ML+pw}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="zero"/>')
    out.append(f'<rect x="{ML}" y="{MT}" width="{pw}" height="{ph}" class="frame"/>')
    for s in series:
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(s["x"], s["y"]))
        dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
        out.append(f'<polyline points="{pts}" class="line {s["cls"]}"{dash}/>')
    # legend in the top-right margin row
    lx = ML + pw
    for s in reversed(series):
        tw = 7 * len(s["label"]) + 34
        lx -= tw
        dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
        out.append(f'<line x1="{lx}" x2="{lx+22}" y1="{MT-32}" y2="{MT-32}" class="line {s["cls"]}"{dash}/>')
        out.append(f'<text x="{lx+27}" y="{MT-28}" class="lgd">{html.escape(s["label"])}</text>')
    out.append(f'<text x="{ML+pw/2:.1f}" y="{H-14}" class="axis">{html.escape(xlabel)}</text>')
    out.append(f'<text transform="translate(16,{MT+ph/2:.1f}) rotate(-90)" class="axis">{html.escape(ylabel)}</text>')
    if title:
        out.append(f'<text x="{ML}" y="{MT-32}" class="ttl">{html.escape(title)}</text>')
    out.append("</svg>")
    return "\n".join(out)


# ---------- tables ----------
def th(cells): return "<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>"
def td(cells, cls=None): return f'<tr{" class=%s" % cls if cls else ""}>' + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


rows = []
for c in CKS:
    rows.append(td([c["label"], c["epoch"], "EMA 0.99", c["flag_in_training"], c["tag"]]))
tbl_ck = "<table><thead>" + th(["标签", "训练轮次", "权重", "训练时开关", "npz 标签"]) + "</thead><tbody>" + "".join(rows) + "</tbody></table>"

V = R["verification"]
ver_rows = [
    ("配对几何", f"47 对位置逐原子一致，max |Δr| = {max(v['pos_max_diff'] for v in V.values()):.1e} Å"),
    ("晶胞", f"带电/中性晶胞相同：{all(v['cell_equal'] for v in V.values())}"),
    ("密度网格", f"格点数 nx/ny/nz 相同：{all(v['grid_equal'] for v in V.values())}；网格晶格 = 晶胞：{all(v['lattice_equal'] and v['lattice_is_cell'] for v in V.values())}；有效平面 valid_iz 相同：{all(v['valid_iz_equal'] for v in V.values())}"),
    ("符号与单位", "DFT 网格为净电荷密度（电子为负，e/Å^3）；模型用同一约定的 GTO 密度在同一格点求值；本页所有量乘 −1 后以“新增电子数为正”表示"),
    ("电子数守恒", "见表 A 的 max|N − ΔN_e| 列：窗口积分与 −q(带电) 的偏差"),
    ("ΔN_e", f"q(带电) 范围 {min(v['q_charged'] for v in V.values()):+.3f} … {max(v['q_charged'] for v in V.values()):+.3f} e，中性帧 q = 0：{all(abs(v['q_neutral']) < 1e-9 for v in V.values())}"),
]
tbl_ver = "<table><tbody>" + "".join(td([a, b]) for a, b in ver_rows) + "</tbody></table>"

RG = R["regions"]
reg_rows = [
    ("电极", f"[z_sheet_min − 1.5, z_sheet_max + 1.5]，均值 [{mean([r['b1'] for r in RG.values()]):.2f}, {mean([r['b2'] for r in RG.values()]):.2f}] Å；片层 = 与 C 中位高度相差 < 1 Å 的 C 原子 + Ni + N（{int(mean([r['n_sheet'] for r in RG.values()]))} 个原子，z {mean([r['z_sheet_min'] for r in RG.values()]):.2f}–{mean([r['z_sheet_max'] for r in RG.values()]):.2f} Å）"),
    ("界面", f"(z_sheet_max + 1.5, z_sheet_max + 4.0]，均值到 {mean([r['b3'] for r in RG.values()]):.2f} Å；含吸附 C（z ≈ {mean([r['z_adsorbate_C'][0] for r in RG.values() if r['z_adsorbate_C']]):.2f} Å）与第一层水"),
    ("水层", f"(z_sheet_max + 4.0, z_water_max + 1.0]，均值到 {mean([r['b4'] for r in RG.values()]):.2f} Å"),
    ("真空", "窗口内其余部分（片层下方、水层上方）"),
    ("份额分母", "各侧各自的窗口积分 N（模型用 N_ML，DFT 用 N_DFT）；这是空间分区指标，不等于 Ni 原子电荷"),
]
tbl_reg = "<table><tbody>" + "".join(td([a, b]) for a, b in reg_rows) + "</tbody></table>"

# table A
rowsA = []
for c in CKS:
    for sp in ("train", "val"):
        P = [p for p in c["pairs"].values() if p["split"] == sp]
        rowsA.append(td([c["label"], c["epoch"], c["flag_in_training"], sp, len(P),
                         f"{mean([p['E'] for p in P]):.3f}", f"{mean([p['S'] for p in P]):.3f}",
                         f"{mean([p['ratio'] for p in P]):.3f}", f"{median([p['ratio'] for p in P]):.3f}",
                         f"{max(abs(p['N_ml'] - p['dNe']) for p in P):.4f}", f"{max(abs(p['N_dft'] - p['dNe']) for p in P):.4f}",
                         f"{mean([abs(p['zc_ml'] - p['zc_dft']) for p in P]):.3f}"], cls="t" if sp == "train" else "v"))
tblA = "<table><thead>" + th(["检查点", "轮次", "开关", "集", "对数", "E = ∫|R_ML − R_DFT| dz (e)", "S = ∫|R_DFT| dz (e)", "E/S 均值", "E/S 中位", "max|N_ML − ΔN_e|", "max|N_DFT − ΔN_e|", "|质心误差| (Å)"]) + "</thead><tbody>" + "".join(rowsA) + "</tbody></table>"

# table B: regions
rowsB = []
for c in CKS:
    for sp in ("train", "val"):
        P = [p for p in c["pairs"].values() if p["split"] == sp]
        cells = [c["label"], c["flag_in_training"], sp]
        for nm in ("electrode", "interface", "water"):
            nml = mean([p["regions"][nm]["N_ml"] for p in P]); nd = mean([p["regions"][nm]["N_dft"] for p in P])
            sml = mean([p["regions"][nm]["N_ml"] / p["N_ml"] for p in P]); sd = mean([p["regions"][nm]["N_dft"] / p["N_dft"] for p in P])
            cells.append(f"{nml:+.3f} / {nd:+.3f}"); cells.append(f"{sml:.2f} / {sd:.2f}")
        cells.append(" · ".join(f"{REG_CN[nm]} {mean([p['regions'][nm]['E'] / p['E'] for p in P]):.2f}" for nm in ("electrode", "interface", "water")))
        rowsB.append(td(cells, cls="t" if sp == "train" else "v"))
tblB = "<table><thead>" + th(["检查点", "开关", "集", "电极 N_ML / N_DFT (e)", "电极份额", "界面 N_ML / N_DFT (e)", "界面份额", "水层 N_ML / N_DFT (e)", "水层份额", "误差 E 的区域分布"]) + "</thead><tbody>" + "".join(rowsB) + "</tbody></table>"

# ---------- figures ----------
figs = []
ce, cm = byl[EARLY], byl[MATURE]
regmean = {k: mean([r[k] for r in RG.values()]) for k in ("b1", "b2", "b3", "b4", "z_sheet_min", "z_sheet_max")}
regmean["z_adsorbate_C"] = [mean([r["z_adsorbate_C"][0] for r in RG.values() if r["z_adsorbate_C"]])]
agg_e, agg_m = ce["aggregate_all"], cm["aggregate_all"]
figs.append(("图 1  全部 47 对的平均响应 R(z)/ΔN_e（每对按各自 DFT 电子数归一）：DFT 与两个检查点",
             svg_plot([{"x": agg_m["z"], "y": agg_m["R_dft_per_e"], "label": "DFT", "cls": "dft"},
                       {"x": agg_e["z"], "y": agg_e["R_ml_per_e"], "label": f"模型 {EARLY}", "cls": "early", "dash": True},
                       {"x": agg_m["z"], "y": agg_m["R_ml_per_e"], "label": f"模型 {MATURE}", "cls": "mature"}],
                      "z (Å)", "R / ΔN_e  (1/Å)", regions=regmean)))
err_series = []
cls_cycle = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"]
for i, c in enumerate(CKS):
    a = c["aggregate_all"]
    err_series.append({"x": a["z"], "y": [u - v for u, v in zip(a["R_ml_per_e"], a["R_dft_per_e"])], "label": c["label"], "cls": cls_cycle[i % 8]})
figs.append(("图 2  平均响应误差 (R_ML − R_DFT)/ΔN_e：按检查点", svg_plot(err_series, "z (Å)", "(R_ML − R_DFT) / ΔN_e  (1/Å)", regions=regmean)))
figs.append(("图 3  沿 z 累计的新增电子数 C(z)/ΔN_e（全部 47 对平均）",
             svg_plot([{"x": agg_m["z"], "y": agg_m["C_dft_per_e"], "label": "DFT", "cls": "dft"},
                       {"x": agg_e["z"], "y": agg_e["C_ml_per_e"], "label": f"模型 {EARLY}", "cls": "early", "dash": True},
                       {"x": agg_m["z"], "y": agg_m["C_ml_per_e"], "label": f"模型 {MATURE}", "cls": "mature"}],
                      "z (Å)", "C / ΔN_e", regions=regmean)))
n = 4
for k in REP:
    pe, pm = ce["pairs"].get(str(k)) or ce["pairs"].get(k), cm["pairs"].get(str(k)) or cm["pairs"].get(k)
    if not (pe and pm and "curve" in pm):
        continue
    reg = RG[str(k)] if str(k) in RG else RG[k]
    ttl = f"图 {n}  配对 {k}（{pm['split']}，q = {pm['q']:+.3f} e，ΔN_e = {pm['dNe']:.3f}）：R(z)"
    figs.append((ttl, svg_plot([{"x": pm["curve"]["z"], "y": pm["curve"]["R_dft"], "label": "DFT", "cls": "dft"},
                                {"x": pe["curve"]["z"], "y": pe["curve"]["R_ml"], "label": f"模型 {EARLY}", "cls": "early", "dash": True},
                                {"x": pm["curve"]["z"], "y": pm["curve"]["R_ml"], "label": f"模型 {MATURE}", "cls": "mature"}],
                               "z (Å)", "R(z)  (e/Å)", regions=reg)))
    n += 1
    figs.append((f"图 {n}  配对 {k}：累计新增电子数 C(z)",
                 svg_plot([{"x": pm["curve"]["z"], "y": pm["curve"]["C_dft"], "label": "DFT", "cls": "dft"},
                           {"x": pe["curve"]["z"], "y": pe["curve"]["C_ml"], "label": f"模型 {EARLY}", "cls": "early", "dash": True},
                           {"x": pm["curve"]["z"], "y": pm["curve"]["C_ml"], "label": f"模型 {MATURE}", "cls": "mature"}],
                          "z (Å)", "C(z)  (e)", regions=reg)))
    n += 1
# region shares vs epoch (production OFF checkpoints only, sorted by epoch)
prod = sorted([c for c in CKS if c["flag_in_training"] == "OFF"], key=lambda c: c["epoch"])
if len(prod) >= 2:
    ep = [c["epoch"] for c in prod]
    ser = []
    for nm, cls in (("electrode", "c0"), ("water", "c2"), ("interface", "c4")):
        ser.append({"x": ep, "y": [mean([p["regions"][nm]["N_ml"] / p["N_ml"] for p in c["pairs"].values()]) for c in prod], "label": f"模型 {REG_CN[nm]}", "cls": cls})
        ser.append({"x": ep, "y": [mean([p["regions"][nm]["N_dft"] / p["N_dft"] for p in c["pairs"].values()]) for c in prod], "label": f"DFT {REG_CN[nm]}", "cls": cls, "dash": True})
    figs.append((f"图 {n}  区域份额随训练轮次（生产检查点，开关 OFF；虚线 = DFT）", svg_plot(ser, "训练轮次", "份额 N_region / N", y0line=True)))
    n += 1
    ser = []
    for sp, cls in (("train", "c0"), ("val", "c2")):
        ser.append({"x": ep, "y": [mean([p["ratio"] for p in c["pairs"].values() if p["split"] == sp]) for c in prod], "label": f"E/S {sp}", "cls": cls})
    figs.append((f"图 {n}  响应误差比 E/S 随训练轮次（生产检查点）", svg_plot(ser, "训练轮次", "E / S", y0line=True)))

# ---------- conclusions ----------
def render_text(t):
    out = []
    for para in [p for p in t.strip().split("\n\n") if p.strip()]:
        lines = para.strip().split("\n")
        if all(l.startswith("- ") for l in lines):
            out.append("<ul>" + "".join(f"<li>{html.escape(l[2:])}</li>" for l in lines) + "</ul>")
        else:
            out.append(f"<p>{html.escape(' '.join(lines))}</p>")
    return "\n".join(out)


defs = html.escape(R["definitions"].split("Definitions")[1].split("Writes")[0]) if "Definitions" in R["definitions"] else ""

page = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Charging Response Audit</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#f6f4ee;--fg:#1d2126;--mute:#5d646d;--rule:#d8d3c7;--card:#fffdf8;--dft:#1d2126;--early:#b0762a;--mature:#1f6fb2;
--band-el:#e6dfcf;--band-if:#efe6d3;--band-w:#dce7ef;--band-v:#f2f0ea;--grid:#e8e3d8;--c0:#1f6fb2;--c1:#3d8fd1;--c2:#c2542b;--c3:#e08a5a;--c4:#4c8c3f;--c5:#7ab86a;--c6:#7a4fa3;--c7:#a98ad0;--t:#f4efe3;--v:#eaf0f5}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#161a1f;--fg:#e8e6df;--mute:#a4a9b0;--rule:#343b44;--card:#1d222a;--dft:#e8e6df;--early:#e0a25a;--mature:#6db1f2;
--band-el:#2a2f2a;--band-if:#2f2c26;--band-w:#232c34;--band-v:#1b1f24;--grid:#262c33;--c0:#6db1f2;--c1:#98c9f5;--c2:#f08a63;--c3:#f5b494;--c4:#8fcf7f;--c5:#b7e2ab;--c6:#b494de;--c7:#d1bdf0;--t:#242821;--v:#1f272e}}}}
:root[data-theme="dark"]{{--bg:#161a1f;--fg:#e8e6df;--mute:#a4a9b0;--rule:#343b44;--card:#1d222a;--dft:#e8e6df;--early:#e0a25a;--mature:#6db1f2;
--band-el:#2a2f2a;--band-if:#2f2c26;--band-w:#232c34;--band-v:#1b1f24;--grid:#262c33;--c0:#6db1f2;--c1:#98c9f5;--c2:#f08a63;--c3:#f5b494;--c4:#8fcf7f;--c5:#b7e2ab;--c6:#b494de;--c7:#d1bdf0;--t:#242821;--v:#1f272e}}
html,body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 "IBM Plex Sans",system-ui,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:32px 16px 64px}}
h1{{font-size:1.7rem;margin:0 0 6px;text-wrap:balance}} h2{{font-size:1.15rem;margin:40px 0 12px;padding-top:12px;border-top:1px solid var(--rule)}}
p{{max-width:72ch}} .sub{{color:var(--mute);margin:0 0 20px}}
table{{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums;background:var(--card)}} .tw{{overflow-x:auto;margin:10px 0 18px}}
th,td{{padding:6px 8px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}} th{{font-weight:600;color:var(--mute);text-align:right}} td:first-child,th:first-child,td:nth-child(2){{text-align:left}}
tr.t td{{background:var(--t)}} tr.v td{{background:var(--v)}}
pre{{font:.84rem/1.5 "IBM Plex Mono",monospace;background:var(--card);padding:12px 14px;border:1px solid var(--rule);overflow-x:auto;white-space:pre-wrap}}
figure{{margin:26px 0}} figcaption{{font-size:.92rem;color:var(--mute);margin:0 0 6px}}
svg.fig{{display:block;background:var(--card);border:1px solid var(--rule)}}
.band.electrode{{fill:var(--band-el)}} .band.interface{{fill:var(--band-if)}} .band.water{{fill:var(--band-w)}} .band.vacuum_below,.band.vacuum_above{{fill:var(--band-v)}}
.bandlbl{{font:12px "IBM Plex Sans",sans-serif;fill:var(--mute);text-anchor:middle}}
.atomline{{stroke:var(--mute);stroke-width:1;stroke-dasharray:2 3}} .atomline.ads{{stroke:var(--c2)}}
.grid{{stroke:var(--grid);stroke-width:1}} .frame{{fill:none;stroke:var(--rule)}} .zero{{stroke:var(--mute);stroke-width:1}}
.tick{{font:11px "IBM Plex Mono",monospace;fill:var(--mute);text-anchor:middle}} .tick.r{{text-anchor:end}}
.axis{{font:12px "IBM Plex Sans",sans-serif;fill:var(--mute);text-anchor:middle}} .lgd{{font:12px "IBM Plex Sans",sans-serif;fill:var(--fg)}} .ttl{{font:12px "IBM Plex Sans",sans-serif;fill:var(--mute)}}
.line{{fill:none;stroke-width:2;stroke-linejoin:round}} .line.dft{{stroke:var(--dft);stroke-width:2.4}} .line.early{{stroke:var(--early)}} .line.mature{{stroke:var(--mature)}}
.line.c0{{stroke:var(--c0)}} .line.c1{{stroke:var(--c1)}} .line.c2{{stroke:var(--c2)}} .line.c3{{stroke:var(--c3)}} .line.c4{{stroke:var(--c4)}} .line.c5{{stroke:var(--c5)}} .line.c6{{stroke:var(--c6)}} .line.c7{{stroke:var(--c7)}}
ul{{max-width:80ch}}
</style></head><body><main>
<h1>充电密度响应诊断</h1>
<p class="sub">Δn(r) = n(带电) − n(中性)，模型 vs DFT，47 个真实配对（训练 27 / 验证 20），NiN-C 片层 + CO2 吸附物 + 44 水。诊断只用已有检查点，不改训练。</p>

<h2>1. 检查点</h2>
<div class="tw">{tbl_ck}</div>
<p>生产（prod）在代码 a39cf46 训练，当时不存在该开关（OFF）；闸门（gate）在 59815d1 / 88c9a7c 训练，开关 ON，只有 0–33 轮。所有检查点为 mace 的 EMA 权重（衰减 0.99）。评估：早期六个检查点用 b396deb，成熟检查点见各作业日志首行。</p>

<h2>2. 指标定义与核对</h2>
<pre>{defs}</pre>
<div class="tw">{tbl_ver}</div>
<div class="tw">{tbl_reg}</div>

<h2>3. 表 A  响应误差与电子数守恒</h2>
<div class="tw">{tblA}</div>
<p>E 是空间分布的绝对值积分误差，不是电子数缺失：max|N − ΔN_e| 列表明两侧窗口积分都给出正确的新增电子数。</p>

<h2>4. 表 B  分区有符号电子数与份额</h2>
<div class="tw">{tblB}</div>

<h2>5. 响应曲线</h2>
{"".join(f'<figure><figcaption>{html.escape(t)}</figcaption>{s}</figure>' for t, s in figs)}

<h2>6. 结论</h2>
{render_text(CONCL)}
</main></body></html>"""
open(OUT, "w").write(page)
print(f"written {OUT}: {len(figs)} figures, {len(CKS)} checkpoints")
