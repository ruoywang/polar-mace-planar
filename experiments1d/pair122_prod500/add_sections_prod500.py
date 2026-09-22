"""Insert sections 7b/7c (electron density with the ion baseline restored) and 10 (bands from the
fully-ML CHGCAR) into the prod500 pair page, the way the published w200 page got them.
Usage: python add_sections_prod500.py <page.html> <out.html> [bands.png] [bands.json]
Inputs in this directory: structure_pair_sid122_722.npz, electron_profiles_prod500.npz,
ml_sol_prod500/build_summary.npz; the band figure/json come from bands_prod500.py (optional: the
section is written only when both files exist).
"""
from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path("/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/exp_pair122_prod500")
GEN = Path("/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/tools1d/gen_prod500_pair_docs_page.py")
PAGE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
PNG = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "bands_122_sol_prod500.png"
BJS = Path(sys.argv[4]) if len(sys.argv) > 4 else HERE / "bands_122_sol_prod500.json"

# reuse the page's chart helpers verbatim (from "W, H_MAIN" up to the end of line_chart)
src = GEN.read_text().splitlines()
i0 = next(i for i, l in enumerate(src) if l.startswith("W, H_MAIN, H_SUB"))
i1 = next(i for i, l in enumerate(src) if l.startswith("# ==== section 0"))
ns: dict = {"np": np}
exec("\n".join(src[i0:i1]), ns)
line_chart, charts_js = ns["line_chart"], ns["charts_js"]
ML, MT, H_SUB = ns["ML"], ns["MT"], ns["H_SUB"]

e = np.load(HERE / "electron_profiles_prod500.npz")
pair = np.load(HERE / "structure_pair_sid122_722.npz", allow_pickle=True)
sid, sid_n = int(pair["sid"]), int(pair["sid_n"])
z_dft_ref = pair["z_dft"].astype(float)
WLO, WHI = float(z_dft_ref.min()), float(z_dft_ref.max())

stats, charts = {}, {}
for tag, cid in (("charged", "ne"), ("neutral", "nen")):
    z = e[f"z_{tag}_dft"].astype(float); dft = e[f"ne_{tag}_dft"].astype(float); ml = e[f"ne_{tag}_ml"].astype(float)
    m = (z >= WLO) & (z <= WHI)
    zw, dftw, diff = z[m], dft[m], (ml - dft)[m]
    charts[cid] = line_chart(cid, [("DFT reference", zw, dftw, "s6", False), ("model (ML CHGCAR)", zw, ml[m], "s1", True)],
                             "z (Å)", "electron density (e/Å³)", legend_xy=(ML + 16, MT + 12), unit="e/Å³")
    charts[cid + "res"] = line_chart(cid + "res", [("difference model−DFT", zw, diff, "s1", False)],
                                     "z (Å)", "difference (e/Å³)", h=H_SUB, unit="e/Å³")
    stats[tag] = dict(peak=float(dftw.max()), zpeak=float(zw[np.argmax(dftw)]), rms=float(np.sqrt((diff ** 2).mean())),
                      amax=float(np.abs(diff).max()), azm=float(zw[np.argmax(np.abs(diff))]),
                      rel=float(np.sqrt((diff ** 2).mean()) / dftw.max() * 100), nelect=float(e[f"nelect_{tag}_dft"]),
                      nelect_ml=float(e[f"nelect_{tag}_ml"]))

# the section-5 net-density error on the same window, for the cross-check sentence
z_grid = pair["z_grid"].astype(float); nbar_c = pair["nbar_model"].astype(float); nbar_dft_c = pair["nbar_dft"].astype(float)
m_on_ref = np.interp(z_dft_ref, z_grid, nbar_c)
ls_c = float(np.dot(m_on_ref, nbar_dft_c) / np.dot(m_on_ref, m_on_ref))
mzc = z_grid[(z_grid >= WLO) & (z_grid <= WHI)]
rms_net = float(np.sqrt((np.interp(mzc, z_grid, nbar_c * ls_c) - np.interp(mzc, z_dft_ref, nbar_dft_c)) ** 2).mean())
bs = {}
for st, dd in (("charged", "ml_sol_prod500"), ("neutral", "ml_neutral_prod500")):
    p = HERE / dd / "build_summary.npz"
    if p.exists():
        b = np.load(p); bs[st] = dict(clamped=int(b["n_clamped"]), scale=float(b["scale"]), occ=float(b["occ_rmse"]), res=float(b["res_rmse_win"]))

sc, sn = stats["charged"], stats["neutral"]
def bnote(st):
    b = bs.get(st)
    return (f" Build: {b['clamped']} grid points clamped at zero, renormalisation factor {b['scale']:.5f}, "
            f"augmentation-occupancy rmse {b['occ']:.2e}, net-density rmse on the grid window {b['res']:.2e} e/Å³.") if b else ""

section = f"""<h2>7b · Charged: electron density with the ion baseline restored, model vs DFT</h2>
<div class="card">{charts['ne']}</div>
<div class="card">{charts['neres']}</div>
<p class="note">Sections 5–7 show the <i>net</i> density n̄ = ion − electron, which is the quantity the model
regresses. This is the same charged frame with the ion baseline put back, i.e. the valence electron density
itself, on the same z window. Construction, as in the band pipeline: baseline = n<sub>e</sub>(DFT) + n<sub>net</sub>(DFT)
(the ion term, a full-grid identity), then n<sub>e</sub>(ML) = baseline − n<sub>net</sub>(ML), hard-clamped at zero and
renormalised to the DFT electron count ({sc['nelect']:.4f} e). Peak DFT density {sc['peak']:.3f} e/Å³ at z = {sc['zpeak']:.2f} Å
(the carbon sheet); rms difference {sc['rms']:.2e} e/Å³, {sc['rel']:.2f} % of that peak; largest deviation {sc['amax']:.2e} e/Å³
at z = {sc['azm']:.2f} Å. The baseline is common to both curves, so the gap is the section-5 net-density error carried onto
the absolute scale (rms {rms_net:.2e} e/Å³ there against {sc['rms']:.2e} e/Å³ here; the difference is the clamp and the
renormalisation). This is the density from which the section-10 bands were computed.{bnote('charged')}</p>

<h2>7c · Neutral: electron density with the ion baseline restored, model vs DFT</h2>
<div class="card">{charts['nen']}</div>
<div class="card">{charts['nenres']}</div>
<p class="note">Same construction for the neutral twin (sid {sid_n}, solvated, {sn['nelect']:.4f} e). Peak DFT density
{sn['peak']:.3f} e/Å³; rms difference {sn['rms']:.2e} e/Å³, {sn['rel']:.2f} % of the peak; largest deviation {sn['amax']:.2e} e/Å³
at z = {sn['azm']:.2f} Å.{bnote('neutral')}</p>"""

page = PAGE.read_text()
anchor = "<h2>8 · Solvent:"
assert page.count(anchor) == 1, f"anchor found {page.count(anchor)} times"
page = page.replace(anchor, section + "\n\n" + anchor)
marker = "const DATA = {"
assert page.count(marker) == 1
page = page.replace(marker, marker + json.dumps(charts_js, ensure_ascii=False)[1:-1] + ", ")

if PNG.exists() and BJS.exists():
    bj = json.load(open(BJS))
    b64 = base64.b64encode(PNG.read_bytes()).decode()
    sec10 = f"""<h2>10 · Band structure from the fully-ML CHGCAR (solvated)</h2>
<p>Non-SCF bands (ICHARG = 11) from the fully-ML CHGCAR of section 7b (density + augmentation occupancies, hard-clamped)
against the DFT reference, both with the VASPsol solvent of the source calculation and the same INCAR/KPOINTS
(hexagonal path Γ–M–K–Γ, {bj['nk']} k-points, {bj['nbands']} bands). Each run is plotted against its own Fermi level; {bj['nwin']} bands around
the band edge are shown. Fermi-window RMSE {bj['rmse_window']:.3f} eV; all-band RMSE {bj['rmse_all']:.3f} eV; Fermi-level difference
ML − DFT {bj['fermi_diff']:+.3f} eV (DFT E_F {bj['ef_dft']:.3f} eV, ML {bj['ef_ml']:.3f} eV). Reference bands: the earlier DFT non-SCF run
of this structure (exp_band/sid122_bands/dft_sol), unchanged.</p>
<div class="card"><img src="data:image/png;base64,{b64}" alt="band structure sid 122, DFT vs fully-ML CHGCAR of prod500_w1000_ref" style="width:100%;display:block"></div>
"""
    tip_anchor = '<div id="tip"></div>'
    assert page.count(tip_anchor) == 1
    # keep the section inside the .pg container: the container closes right before the tip div
    page = page.replace("</div>\n" + tip_anchor, sec10 + "</div>\n" + tip_anchor)
    assert sec10 in page
    print(f"section 10 inserted (window rmse {bj['rmse_window']:.3f} eV)")
else:
    print("section 10 skipped: band figure / json not present")

OUT.write_text(page)
for tag, s in stats.items():
    print(f"{tag:8s} peak {s['peak']:.4f} @ {s['zpeak']:.2f}  rms {s['rms']:.4e}  max {s['amax']:.4e} @ {s['azm']:.2f}  ({s['rel']:.3f} % of peak)  NELECT dft {s['nelect']:.4f} ml {s['nelect_ml']:.4f}")
print(f"section-5 net-density rms, same window: {rms_net:.4e}")
print("wrote", OUT, OUT.stat().st_size, "bytes")
