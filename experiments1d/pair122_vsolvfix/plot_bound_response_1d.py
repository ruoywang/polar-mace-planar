"""Plane-integrated bound-charge charging response R_b(z) = A * <rho_b(charged) - rho_b(neutral)>_xy, DFT vs ML,
pair sid 122 / 722 (user question 2026-09-30). DFT: data/solvent3d_grid full grids; ML: the 1-D bound profiles of the
final stage-2 solve from structure_pair_sid122_722.npz (the lateral residual has zero plane mean, so the ML plane
average is exactly this profile). Writes bound_response_1d.html (Chinese) next to this script."""
import base64, io
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

HERE = Path(__file__).parent
G = Path("/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_vsolv_fix/data/solvent3d_grid")
fm.fontManager.ttflist.append(fm.FontEntry(fname="/usr/share/fonts/google-droid/DroidSansFallback.ttf", name="CJKFallback",
                                           style="normal", variant="normal", weight=400, stretch="normal", size="scalable"))
plt.rcParams.update({"font.family": ["DejaVu Sans", "CJKFallback"], "font.size": 13, "axes.labelsize": 13, "legend.fontsize": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})

p = np.load(HERE / "structure_pair_sid122_722.npz", allow_pickle=True)
lat = np.load(G / "sid_122_meta.npz")["lattice"]; A = abs(np.cross(lat[0], lat[1])[2]); Lz = lat[2, 2]
db = np.load(G / "sid_122_b.npy").astype(np.float64) - np.load(G / "sid_722_b.npy").astype(np.float64)
nz = db.shape[0]; zd = np.arange(nz) * Lz / nz; dzd = Lz / nz
Rd = A * db.mean(axis=(1, 2))
zm = p["z_solve"].astype(float); dzm = zm[1] - zm[0]
Rm = A * (p["rho_bound"] - p["rho_bound_n"]).astype(float)
Rm_on_d = np.interp(zd, zm, Rm, period=Lz)
zat = p["z_atoms"].astype(float)


def parts(z, R, d):
    pos, neg = np.clip(R, 0, None), np.clip(R, None, 0)
    return dict(tot=R.sum() * d, pos=pos.sum() * d, zpos=(z * pos).sum() / pos.sum(), neg=neg.sum() * d, zneg=(z * neg).sum() / neg.sum(),
                pk=R.max(), zpk=z[np.argmax(R)], mn=R.min(), zmn=z[np.argmin(R)],
                edge=R[(z >= 14) & (z < 19)].sum() * d, far=R[(z >= 19)].sum() * d)


PD, PM = parts(zd, Rd, dzd), parts(zm, Rm, dzm)
diff = Rm_on_d - Rd
L1 = np.abs(diff).sum() * dzd; S = np.abs(Rd).sum() * dzd


def nice(v):
    for s in (0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5):
        if v / s <= 7:
            return s
    return 1.0


def png(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


Z0, Z1 = 5.0, 35.0
fig, ax = plt.subplots(figsize=(12, 5.4))
ax.axhline(0, color="#898781", lw=1)
ax.plot(zd, Rd, color="#3a3a3a", lw=2.2, label="DFT")
ax.plot(zm, Rm, color="#2a78d6", lw=2.2, label="ML")
top = max(Rd.max(), Rm.max()); bot = min(Rd.min(), Rm.min()); st = nice(top - bot)
ax.set_ylim(np.floor(bot / st) * st, np.ceil(top / st) * st); ax.set_yticks(np.arange(np.floor(bot / st) * st, np.ceil(top / st) * st + st / 2, st))
ax.set_xlim(Z0, Z1); ax.set_xticks(np.arange(Z0, Z1 + 0.1, 5))
ax.set_xlabel("z (Å)"); ax.set_ylabel("A × ⟨Δρ_b⟩_xy (e/Å)")
ax.grid(color="#e1e0d9", lw=0.8); ax.set_axisbelow(True); ax.legend(loc="upper right", frameon=False)
F1 = png(fig)

fig, ax = plt.subplots(figsize=(12, 4.2))
ax.axhline(0, color="#898781", lw=1)
ax.plot(zd, diff, color="#eb6834", lw=2.0, label="ML − DFT")
v = np.abs(diff[(zd >= Z0) & (zd <= Z1)]).max(); st = nice(2 * v); lim = np.ceil(v / st) * st
ax.set_ylim(-lim, lim); ax.set_yticks(np.arange(-lim, lim + st / 2, st))
ax.set_xlim(Z0, Z1); ax.set_xticks(np.arange(Z0, Z1 + 0.1, 5))
ax.set_xlabel("z (Å)"); ax.set_ylabel("ML − DFT (e/Å)")
ax.grid(color="#e1e0d9", lw=0.8); ax.set_axisbelow(True); ax.legend(loc="upper right", frameon=False)
F2 = png(fig)

rows = [("正的部分 积分 (e)", f"{PD['pos']:+.4f}", f"{PM['pos']:+.4f}"), ("正的部分 重心 z (Å)", f"{PD['zpos']:.2f}", f"{PM['zpos']:.2f}"),
        ("负的部分 积分 (e)", f"{PD['neg']:+.4f}", f"{PM['neg']:+.4f}"), ("负的部分 重心 z (Å)", f"{PD['zneg']:.2f}", f"{PM['zneg']:.2f}"),
        ("峰值 (e/Å)", f"{PD['pk']:+.4f}", f"{PM['pk']:+.4f}"), ("峰值位置 z (Å)", f"{PD['zpk']:.2f}", f"{PM['zpk']:.2f}"),
        ("谷值 (e/Å)", f"{PD['mn']:+.4f}", f"{PM['mn']:+.4f}"), ("谷值位置 z (Å)", f"{PD['zmn']:.2f}", f"{PM['zmn']:.2f}"),
        ("14–19 Å 积分 (e)", f"{PD['edge']:+.4f}", f"{PM['edge']:+.4f}"), ("19 Å 以外 积分 (e)", f"{PD['far']:+.4f}", f"{PM['far']:+.4f}"),
        ("总积分 (e)", f"{PD['tot']:+.4f}", f"{PM['tot']:+.4f}")]
tbl = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>束缚电荷充电响应</title>
<style>
:root{{--bg:#f9f9f7;--surf:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e1e0d9;--rule:rgba(11,11,11,.10)}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0d0d0d;--surf:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--grid:#2c2c2a;--rule:rgba(255,255,255,.10)}}}}
:root[data-theme="dark"]{{--bg:#0d0d0d;--surf:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--grid:#2c2c2a;--rule:rgba(255,255,255,.10)}}
html,body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.75 system-ui,-apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans SC",sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:32px 16px 72px}} h1{{font-size:1.6rem;margin:0 0 8px}}
p{{max-width:46em}} .sub{{color:var(--ink2)}} .note{{color:var(--ink2);font-size:.9rem}}
.tw{{overflow-x:auto;margin:14px 0 20px}} table{{border-collapse:collapse;font-size:.92rem;font-variant-numeric:tabular-nums;background:var(--surf)}}
th,td{{padding:7px 14px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}} th{{color:var(--ink2)}} td:first-child,th:first-child{{text-align:left}}
figure{{margin:26px 0}} figcaption{{color:var(--ink2);font-size:.95rem;margin:0 0 6px}}
figure img{{display:block;width:100%;height:auto;border:1px solid var(--grid);border-radius:4px;background:#fcfcfb}}
</style></head><body><main>
<h1>束缚电荷充电响应沿 z 的分布：sid 122 − 722</h1>
<p class="sub">同一几何下带电帧 sid 122（q = {float(p['q_ion']) * -1:+.4f} e）减中性带溶剂帧 sid 722 的束缚电荷，在每个 z 平面上做平面积分：
A × ⟨Δρ_b⟩_xy，单位 e/Å，对 z 积分即为电荷。DFT 来自 VASPsol 的 RHOB 全网格；ML 是模型 prod500_vsolv_fix 最后一次溶剂求解的一维束缚电荷剖面，
横向残差每层平均为零，所以 ML 的平面平均就是这条剖面。</p>
<div class="tw"><table><thead><tr><th>量</th><th>DFT</th><th>ML</th></tr></thead><tbody>{tbl}</tbody></table></div>
<p class="note">两条曲线之差的积分 ∫|ML − DFT| dz = {L1:.4f} e，DFT 自身 ∫|DFT| dz = {S:.4f} e，比值 {L1 / S:.3f}。显式原子最高处 z = {zat.max():.2f} Å。</p>
<figure><figcaption>束缚电荷充电响应的平面积分，DFT 与 ML</figcaption><img src="data:image/png;base64,{F1}" alt="bound charge response along z, DFT and ML"></figure>
<figure><figcaption>ML − DFT</figcaption><img src="data:image/png;base64,{F2}" alt="ML minus DFT"></figure>
</main></body></html>
"""
(HERE / "bound_response_1d.html").write_text(html)
for k in ("pos", "zpos", "neg", "zneg", "pk", "zpk", "mn", "zmn", "edge", "far", "tot"):
    print(f"{k:5s} DFT {PD[k]:+.4f}  ML {PM[k]:+.4f}")
print(f"L1 {L1:.4f}  S {S:.4f}  ratio {L1 / S:.3f}")
