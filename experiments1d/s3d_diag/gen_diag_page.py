"""Round-1 offline diagnosis page (tables only) from diag_F1, diag_H1, diag_H2, diag_ridge, diag_conv results.json.
Usage: python gen_diag_page.py <out.html>"""
import json
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).parent
F = json.load(open(H / "diag_F1/results.json")); H1 = json.load(open(H / "diag_H1/results.json"))
H2 = json.load(open(H / "diag_H2/results.json")); RG = json.load(open(H / "diag_ridge/results.json"))
CV = json.load(open(H / "diag_conv/results.json"))
VAL = [28, 30, 43, 60, 61, 62, 69, 79, 83, 94, 128, 134, 148, 153, 159, 177, 180, 185, 186, 189]
HP = [3, 40, 81, 111, 164, 196]; FP = [1, 52, 152]


def fm(R, sids, tag, ch, key):
    return float(np.mean([R["frames"][str(s)][tag][ch][key] for s in sids]))


def pm(R, ks, tag, ch, key):
    return float(np.mean([R["pairs"][str(k)][tag][ch][key] for k in ks]))


def tr(cells, cls=""):
    return f'<tr class="{cls}">' + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def table(head, rows):
    return ('<div class="tw"><table><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


f3 = lambda v: f"{v:.3f}"
f4 = lambda v: f"{v:.4f}"

# ---------------- table 1: bound channel
SETS = [("训练 3 对（1、52、152）", F, FP, [("现有系数", "V0"), ("每帧自由系数，正则 1e-2", "V1@0.01"), ("每帧自由系数，正则 3e-3", "V1@0.003"),
                                           ("配对共用系数，正则 1e-2", "V2@0.01"), ("配对共用系数，正则 3e-3", "V2@0.003")]),
        ("训练 6 对（拟合读取头）", H2, HP, [("现有读取头", "V0"), ("重拟合读取头，正则 1e-2", "V3@0.01"), ("重拟合读取头，正则 3e-3", "V3@0.003")]),
        ("验证 20 对", H2, VAL, [("现有读取头", "V0"), ("重拟合读取头，正则 1e-2", "V3@0.01"), ("重拟合读取头，正则 3e-3", "V3@0.003")]),
        ("测试配对 122（只展示）", H2, [122], [("现有读取头", "V0"), ("重拟合读取头，正则 1e-2", "V3@0.01"), ("重拟合读取头，正则 3e-3", "V3@0.003")])]
rows1 = []
for sname, R, ks, vs in SETS:
    first = True
    for vname, tag in vs:
        C, N = ks, [k + 600 for k in ks]
        rows1.append(tr([sname if first else "", vname, f3(fm(R, C, tag, "b", "eps_pa")), f3(fm(R, N, tag, "b", "eps_pa")),
                         f3(fm(R, C, tag, "b", "eps_lat")), f3(fm(R, N, tag, "b", "eps_lat")), f3(pm(R, ks, tag, "b", "eps_lat")),
                         f3(fm(R, C, tag, "b", "lat_sq")), f3(fm(R, N, tag, "b", "lat_sq")), f3(pm(R, ks, tag, "b", "lat_sq"))],
                        "grp" if first else ""))
        first = False
T1 = table(["帧组", "变体", "一维 带电", "一维 中性", "横向 L1 带电", "横向 L1 中性", "横向 L1 充电响应", "横向平方 带电", "横向平方 中性",
            "横向平方 充电响应"], rows1)

# ---------------- table 2: ionic channel (charged frames)
rows2 = [tr(["训练 3 对", "现有系数", f3(fm(F, FP, "V0", "i", "eps_lat")), f3(fm(F, FP, "V0", "i", "lat_sq")), "—"], "grp"),
         tr(["", "每帧自由系数，正则 3e-3（带门控）", f3(fm(F, FP, "V1@0.01", "i", "eps_lat")), f3(fm(F, FP, "V1@0.01", "i", "lat_sq")), "—"]),
         tr(["", "候选背景，现有系数", f3(fm(H1, FP, "V4c0", "i", "eps_lat")), f3(fm(H1, FP, "V4c0", "i", "lat_sq")), "—"]),
         tr(["", "候选背景，重新拟合，正则 3e-3", f3(fm(H1, FP, "V4@0.003", "i", "eps_lat")), f3(fm(H1, FP, "V4@0.003", "i", "lat_sq")), "—"]),
         tr(["训练 6 对", "现有读取头", f3(fm(H2, HP, "V0", "i", "eps_lat")), f3(fm(H2, HP, "V0", "i", "lat_sq")), "—"], "grp"),
         tr(["", "重拟合读取头，正则 3e-3", f3(fm(H2, HP, "V3@0.003", "i", "eps_lat")), f3(fm(H2, HP, "V3@0.003", "i", "lat_sq")), "—"]),
         tr(["验证 20 对", "现有读取头", f3(fm(H2, VAL, "V0", "i", "eps_lat")), f3(fm(H2, VAL, "V0", "i", "lat_sq")), "—"], "grp"),
         tr(["", "重拟合读取头，正则 1e-2", f3(fm(H2, VAL, "V3@0.01", "i", "eps_lat")), f3(fm(H2, VAL, "V3@0.01", "i", "lat_sq")), "—"]),
         tr(["", "重拟合读取头，正则 3e-3", f3(fm(H2, VAL, "V3@0.003", "i", "eps_lat")), f3(fm(H2, VAL, "V3@0.003", "i", "lat_sq")), "—"]),
         tr(["训练 3 对，中性帧", "去掉门控的自由系数，正则 3e-3（不是现有模型的能力）",
             f3(fm(F, [k + 600 for k in FP], "V1u@0.003", "i", "eps_lat")), f3(fm(F, [k + 600 for k in FP], "V1u@0.003", "i", "lat_sq")), "现有模型此处恒为零，误差比 1.000"], "grp")]
T2 = table(["帧组", "变体", "横向 L1", "横向平方", "说明"], rows2)

# ---------------- table 3: potential and self-energy of the total solvent charge
rows3 = []
for sname, R, ks, vs in SETS[:3]:
    first = True
    for vname, tag in vs:
        C, N = ks, [k + 600 for k in ks]
        rows3.append(tr([sname if first else "", vname, f4(fm(R, C, tag, "t", "phi_rms_err")), f4(fm(R, N, tag, "t", "phi_rms_err")),
                         f4(pm(R, ks, tag, "t", "phi_rms_err")),
                         f"{fm(R, C, tag, 't', 'eself_ml'):.3f}（{fm(R, C, tag, 't', 'eself_dft'):.3f}）",
                         f"{fm(R, N, tag, 't', 'eself_ml'):.3f}（{fm(R, N, tag, 't', 'eself_dft'):.3f}）",
                         f"{pm(R, ks, tag, 't', 'eself_ml'):.3f}（{pm(R, ks, tag, 't', 'eself_dft'):.3f}）"], "grp" if first else ""))
        first = False
T3 = table(["帧组", "变体", "电势 rms 误差 带电 (eV)", "中性 (eV)", "充电响应 (eV)", "静电自能 带电（DFT）(eV)", "中性（DFT）(eV)", "充电响应（DFT）(eV)"], rows3)

# ---------------- table 4: convergence / ridge path, sid 1
d = RG["frames"]["1"]; dc = CV["frames"]["1"]
fits = {f["label"]: f for f in RG["fits"]}; fitc = {f["label"]: f for f in CV["fits"]}
rows4 = [tr(["束缚", "现有系数", "—", "—", f"{fits['RIDGE b sid 1 ridge_rel=0.03']['x0_norm']:.0f}", f3(d["V0"]["b"]["J_hold"]), f3(d["V0"]["b"]["eps_lat"]),
             f4(d["V0"]["t"]["phi_rms_err"]), f"{d['V0']['t']['eself_ml']:.3f}"], "grp")]
for rr in ("0.03", "0.01", "0.003", "0.001", "0.0003", "0.0001", "3e-05"):
    lab = f"RIDGE b sid 1 ridge_rel={float(rr)}"
    o = d[lab]; f = fits[lab]
    rows4.append(tr(["", f"正则 {float(rr):g}", "是" if f["istop"] in (1, 2) else f"否（{f['itn']} 次上限）", f["itn"],
                     f"{f['xnorm']:.0f}", f3(o["b"]["J_hold"]), f3(o["b"]["eps_lat"]), f4(o["t"]["phi_rms_err"]), f"{o['t']['eself_ml']:.3f}"]))
lab = "CONV b sid 1 precond=jacobi damp_rel=None iter=8000"; o = dc[lab]; f = fitc[lab]
rows4.append(tr(["", "不加正则", f"否（{f['itn']} 次上限）", f["itn"], f"{f['xnorm']:.2e}", f3(o["b"]["J_hold"]), f3(o["b"]["eps_lat"]),
                 f4(o["t"]["phi_rms_err"]), f"{o['t']['eself_ml']:.3f}"]))
rows4.append(tr(["离子", "现有系数", "—", "—", f"{fits['RIDGE i sid 1 ridge_rel=0.01']['x0_norm']:.0f}", f3(d["V0"]["i"]["J_hold"]), f3(d["V0"]["i"]["eps_lat"]), "—", "—"], "grp"))
for rr in ("0.01", "0.001", "0.0001"):
    lab = f"RIDGE i sid 1 ridge_rel={float(rr)}"; o = d[lab]; f = fits[lab]
    rows4.append(tr(["", f"正则 {float(rr):g}", "是" if f["istop"] in (1, 2) else f"否（{f['itn']} 次上限）", f["itn"], f"{f['xnorm']:.0f}",
                     f3(o["i"]["J_hold"]), f3(o["i"]["eps_lat"]), "—", "—"]))
lab = "CONV i sid 1 precond=jacobi damp_rel=None iter=8000"; o = dc[lab]; f = fitc[lab]
rows4.append(tr(["", "不加正则", f"否（{f['itn']} 次上限）", f["itn"], f"{f['xnorm']:.2e}", f3(o["i"]["J_hold"]), f3(o["i"]["eps_lat"]), "—", "—"]))
T4 = table(["通道", "拟合", "按判据收敛", "迭代次数", "系数的模", "留出点横向平方误差", "横向 L1", "合计电势 rms 误差 (eV)", "合计静电自能 (eV)"], rows4)
eself_dft_1 = d["V0"]["t"]["eself_dft"]

# ---------------- table 5: round trip and envelope shares
rt = [(int(s), o) for X in (F, H1) for s, o in X["roundtrip"].items()]
sh = [(int(s), x["repro"]["target_share_where_env_below"]) for X in (F, H2) for s, x in X["frames"].items()
      if "repro" in x and "b_L1" in x["repro"].get("target_share_where_env_below", {})]
rng = lambda v: f"{min(v):.3f} 到 {max(v):.3f}"
rows5 = [tr(["网格往返，束缚电荷横向 L1", rng([o["b"]["eps_lat"] for s, o in rt if s <= 200]), rng([o["b"]["eps_lat"] for s, o in rt if s > 600])]),
         tr(["网格往返，离子电荷横向 L1", rng([o["i"]["eps_lat"] for s, o in rt if s <= 200]), "中性帧离子横向电荷极小，不列"]),
         tr(["网格往返，束缚电荷三维峰值下降", f"{100 * np.mean([1 - o['b']['max3d_rt'] / o['b']['max3d_dft'] for s, o in rt if s <= 200]):.1f}%",
             f"{100 * np.mean([1 - o['b']['max3d_rt'] / o['b']['max3d_dft'] for s, o in rt if s > 600]):.1f}%"]),
         tr(["束缚横向目标落在包络 &lt; 1e-3 处，平方口径", rng([x["b"]["0.001"] for s, x in sh if s <= 200]), rng([x["b"]["0.001"] for s, x in sh if s > 600])]),
         tr(["束缚横向目标落在包络 &lt; 1e-3 处，绝对值口径", rng([x["b_L1"]["0.001"] for s, x in sh if s <= 200]), rng([x["b_L1"]["0.001"] for s, x in sh if s > 600])]),
         tr(["束缚横向目标落在包络 &lt; 1e-2 处，绝对值口径", rng([x["b_L1"]["0.01"] for s, x in sh if s <= 200]), rng([x["b_L1"]["0.01"] for s, x in sh if s > 600])])]
T5 = table(["量", f"带电帧（{len([1 for s, o in rt if s <= 200])} 帧）", f"中性帧（{len([1 for s, o in rt if s > 600])} 帧）"], rows5)

nfits = len(F["fits"]) + len(H2["fits"]) + sum(1 for f in H1["fits"] if f["label"].startswith("V4"))
nconv = sum(1 for X in (F, H2) for f in X["fits"] if f["istop"] in (1, 2)) + sum(1 for f in H1["fits"] if f["label"].startswith("V4") and f["istop"] in (1, 2))

page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>三维溶剂电荷离线诊断</title>
<style>
:root{{--bg:#f9f9f7;--surf:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e1e0d9;--rule:rgba(11,11,11,.10);--hl:#eef3fb}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0d0d0d;--surf:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--grid:#2c2c2a;--rule:rgba(255,255,255,.10);--hl:#18202c}}}}
:root[data-theme="dark"]{{--bg:#0d0d0d;--surf:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--grid:#2c2c2a;--rule:rgba(255,255,255,.10);--hl:#18202c}}
html,body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.75 system-ui,-apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans SC",sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:32px 16px 72px}} h1{{font-size:1.6rem;margin:0 0 8px}}
h2{{font-size:1.15rem;margin:40px 0 10px;padding-top:12px;border-top:1px solid var(--rule)}}
p,li{{max-width:48em}} .sub,.note{{color:var(--ink2)}} .note{{font-size:.9rem}}
.tw{{overflow-x:auto;margin:10px 0 14px}} table{{border-collapse:collapse;font-size:.86rem;font-variant-numeric:tabular-nums;background:var(--surf);width:100%}}
th,td{{padding:6px 9px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}} th{{color:var(--ink2);font-weight:600}}
td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){{text-align:left;white-space:normal}} tr.grp td{{border-top:1px solid var(--ink2)}}
</style></head><body><main>
<h1>三维溶剂电荷离线诊断：第一轮</h1>
<p class="sub">模型 prod500_vsolv_fix，500 轮最终权重。不训练，不改生产代码。所有拟合都用模型自己的函数重建三维场：
每帧一次前向，截取原子坐标、节点特征、读取头系数、最终求解的空腔场和一维剖面；用截取量重算的修正场与模型逐位一致（54 帧最大相对差 0）。</p>

<h2>做法</h2>
<ul>
<li>模型的三维溶剂电荷 = 一维剖面铺满每个 z 平面 + 横向修正。横向修正 = 包络 × 原子高斯基（宽度 0.5、1、2 Å，角动量至 2）的组合，再减去每层平均 × 包络。
束缚电荷的包络是介电函数梯度的模除以其最大值，离子电荷的包络是离子可达函数。包络与投影固定后，系数到横向修正是线性映射。</li>
<li>拟合只针对横向部分：目标是 DFT 场减去它自己的每层平均。最小化均匀 DFT 网格点上的平方误差（训练集 3 对用 200 万点，读取头用 100 万点），
在不相交的留出点上检验，再在 500 × 168 × 168 全网格上算误差。一维（平面平均）误差单独列出，任何横向变体都改变不了它。</li>
<li>求解：scipy LSQR，正算子就是模型的计算，转置由自动求导给出，不存矩阵。按列范数预处理（随机探测估计）。
正则加在原始系数的改动上，强度 = 相对值 × 算子范数。离子通道保留现有的总电荷门控，中性帧去掉门控的结果单列，不算现有模型的能力。</li>
<li>误差定义：横向 L1 = ∫|ML⊥ − DFT⊥| / ∫|DFT⊥|，横向平方 = ∫(ML⊥ − DFT⊥)² / ∫DFT⊥²，⊥ 表示各自减去每层平均；一维 = 平面平均之差的 L1 比。
充电响应 = 同一几何的带电帧减中性带溶剂帧。最小二乘优化的是平方误差，L1 是结果，不是优化目标。</li>
<li>表 1 到表 3 的 {nfits} 次拟合中 {nconv} 次按判据收敛（残差满足最小二乘判据），最优性残差都在 1e-8 量级。没加正则的拟合不收敛，只在表 4 里作参照。</li>
</ul>

<h2>1 · 束缚电荷</h2>
{T1}
<p class="note">一维误差各变体相同，因为横向修正每层平均为零。读取头 6 对和验证 20 对的一维中性帧误差（0.73、0.92）高于训练 3 对（0.58），是帧组不同。</p>

<h2>2 · 离子电荷（带电帧）</h2>
{T2}
<p class="note">候选背景 = 一维离子剖面 × 离子可达函数 / 其每层平均（每层平均不变）。sid 152 有 60 个平面的离子可达函数平均恰为 0 而一维剖面不为 0（≤ 1.5e-9 e/Å³，为其最大值的百万分之一），
这些平面退回平铺背景并已记录。离子横向部分本身很小，网格往返就丢掉 33% 到 50%（表 5）。</p>

<h2>3 · 合计溶剂电荷的电势与静电自能</h2>
{T3}
<p class="note">电势由周期泊松方程在 DFT 网格上求得，G = 0 分量去掉；rms 取整个晶胞。括号内为 DFT。</p>

<h2>4 · 收敛与正则路径（sid 1，自由系数）</h2>
{T4}
<p class="note">sid 1 合计溶剂电荷的 DFT 静电自能 {eself_dft_1:.3f} eV。不加正则时最小值存在，但对应极大的系数，几千次迭代内收敛不到。</p>

<h2>5 · 网格往返与包络比例</h2>
{T5}
<p class="note">网格往返 = DFT 场在模型网格（100 × 100 × 300）点上三线性取值，再用模型自己的插值回到 DFT 网格。它衡量这套重采样的失真，不是任何表示的下限。</p>
</main></body></html>
"""
Path(sys.argv[1]).write_text(page)
print("wrote", sys.argv[1], len(page))
