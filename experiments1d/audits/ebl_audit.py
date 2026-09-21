"""Physics audit of the baseline-coupling term E_bl (user 2026-09-21, three questions,
no training):

  Q1  double counting? -- put E_bl, the 1-D compensation (cross + self), the slab dipole
      correction and the medium's internal free energy under ONE set of definitions on
      the solver's own 1-D grid: recompute E_bl from the exported fields, verify the
      reciprocity identity E_bl = int rho_base * phi_solv (same periodic G!=0 operator),
      split the 1-D compensation into cross_net + self, and estimate the medium internal
      free energy that the interaction form omits (linear response:
      U_int = -1/2 int rho_solv * phi_tot = -1/2 int rho_solv * (phi_sol + phi_solv)).
  Q2  input-field error? -- the same E_bl integral with the DFT's own 1-D solvent charge
      (rb_z_vasp + ion_z_vasp from data/dft_solvent1d_ref.npz) against the same baseline
      potential; compare per frame and per pair with the model-field value.
  Q3  deletion stakes without retraining -- per-frame E_bl (absolute-energy shift per
      state group) and the force contribution -dE_bl/dR (rms, max) against the model's
      force error.

Conventions: the exported baseline potential pbz (bl_export["phi"]) is the plane-averaged
VASP potential in the electron-PE convention (energy of an electron, eV), zero-mean; the
physics-sign potential of a positive unit charge is -pbz. The 1-D periodic Poisson
operator used here is phi(G) = K rho(G) / G^2, G != 0, K = 4 pi e^2/(4 pi eps0) = 180.95 eV A.
The solver's phi_z sign convention is DETERMINED numerically (both hypotheses printed).

Env: KIT_RUN, KIT_MODEL_OBJ, KIT_CKPT, KIT_TAG, KIT_OUT, KIT_PAIRS / KIT_STRIDE as in
pair_terms_prod.py; KIT_FORCES=0 skips the force pass.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ.get(
    "KIT_MACE_REPO",
    "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.pb1d_backend as PB

RUN = os.environ["KIT_RUN"]
TAG = os.environ.get("KIT_TAG", "ebl_audit")
OUT = os.path.abspath(os.environ.get("KIT_OUT", os.path.join(RUN, "logs")))
STRIDE = int(os.environ.get("KIT_STRIDE", "6"))
DO_FORCES = os.environ.get("KIT_FORCES", "1") != "0"
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")
os.chdir(RUN)
K_COUL = 180.9512816816869  # eV*A per e^2: 4*pi*27.211386/1.8897261

PAIRS_VAL = [28, 30, 43, 60, 61, 62, 69, 79, 83, 94,
             128, 134, 148, 153, 159, 177, 180, 185, 186, 189]

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
MO, CK = os.environ["KIT_MODEL_OBJ"], os.environ["KIT_CKPT"]
model = torch.load(f=MO, map_location=device).to(device)
model.load_state_dict(torch.load(CK, map_location=device)["model"], strict=True)
EPOCH = int(CK.rsplit("epoch-", 1)[1].split(".")[0])
model.eval(); model._pb1d_epoch = EPOCH
if os.environ.get("KIT_VSOLV", "") not in ("", "0"):   # checkpoint trained with the stage-1 effective-potential input
    model.solvent_pb1d_vsolv_input = True
print(f"[{TAG}] solvent_pb1d_vsolv_input = {getattr(model, 'solvent_pb1d_vsolv_input', False)}")
for p in model.parameters():
    p.requires_grad_(False)
print(f"[{TAG}] {os.path.basename(MO)} @ epoch {EPOCH}; solvent_baseline_coupling={model.solvent_baseline_coupling} "
      f"cavity={model.solvent_cavity_energy} s3d={model.solvent3d_energy}; LIVE_POS={os.environ.get('MACE_PB1D_LIVE_POS','unset')} "
      f"GRAD_PASSES={os.environ.get('MACE_PB1D_GRAD_PASSES','unset')}; forces pass {'on' if DO_FORCES else 'off'}")

# capture the solver's result dict of the LAST solve of a forward (stage 2 runs after stage 1)
_cap = {}
_bk = PB.PB1DBackend.solve_graph
def _wrap_bk(self, *a, **k):
    _cap["b"] = self
    r = _bk(self, *a, **k)
    _cap["r"] = r
    return r
PB.PB1DBackend.solve_graph = _wrap_bk


def _evict():
    b = _cap.get("b")
    if b is not None:
        for nm in ("_bl_ram", "_grids", "_solvers", "_c_units"):
            d = getattr(b, nm, None)
            if isinstance(d, dict):
                d.clear()
    torch.cuda.empty_cache()


z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin",
               "sample_id": "sample_id", "fermi_level": "Fermi", "potential": "potential_diff",
               "solvated": "solvated"},
    arrays_keys={"forces": "forces"})
atoms_by_sid, split_by_sid, train_order = {}, {}, []
for split in ("train", "val"):
    p = os.path.join(RUN, "data", f"{split}.xyz")
    if os.path.exists(p):
        for a in read(p, ":"):
            s = int(a.info["sample_id"]); atoms_by_sid[s] = a; split_by_sid[s] = split
            if split == "train":
                train_order.append(s)
if os.environ.get("KIT_PAIRS"):
    pairs = [int(x) for x in os.environ["KIT_PAIRS"].split(",") if x.strip()]
else:
    tp = sorted({(s if s <= 200 else s - 600) for s in train_order[::STRIDE]})
    pairs = list(PAIRS_VAL) + [k for k in tp if k not in PAIRS_VAL]
pairs = [k for k in pairs if k in atoms_by_sid and (k + 600) in atoms_by_sid]

# DFT 1-D solvent reference
ref = np.load(os.path.join(RUN, "data", "dft_solvent1d_ref.npz"), allow_pickle=True)
ref_row = {int(s): i for i, s in enumerate(ref["sample_ids"])}
ref_z, ref_lz = np.asarray(ref["z_A"], float), float(ref["lz_A"])
ref_rb, ref_ion = np.asarray(ref["rb_z_vasp"], float), np.asarray(ref["ion_z_vasp"], float)


def poisson_1d(rho, dz):
    """Periodic G!=0 potential (physics sign, eV per unit charge) of a 1-D charge density
    rho (e/A^3, laterally uniform): phi(G) = K rho(G)/G^2."""
    n = rho.shape[0]
    g = 2.0 * np.pi * np.fft.fftfreq(n, d=dz)
    rk = np.fft.fft(rho)
    out = np.zeros_like(rk)
    m = np.abs(g) > 1e-14
    out[m] = K_COUL * rk[m] / (g[m] ** 2)
    return np.fft.ifft(out).real


def second_deriv(f, dz):
    n = f.shape[0]
    g = 2.0 * np.pi * np.fft.fftfreq(n, d=dz)
    if n % 2 == 0:
        g[n // 2] = 0.0
    return np.fft.ifft(-(g ** 2) * np.fft.fft(f)).real


def resample(f, n_new):
    """Band-limited periodic resampling."""
    n = f.shape[0]
    F = np.fft.rfft(f) / n
    G = np.zeros(n_new // 2 + 1, dtype=complex)
    m = min(F.shape[0], G.shape[0])
    G[:m] = F[:m]
    return np.fft.irfft(G * n_new, n=n_new)


def run(sid, want_forces):
    a = atoms_by_sid[sid]
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    rec = {"sid": sid, "nat": len(a), "q": float(a.info.get("total_charge", 0.0)),
           "state": ("charged" if abs(float(a.info.get("total_charge", 0.0))) > 1e-6 else "neutral")
                    + (" NiN44" if len(a) == 207 else " NiN88")}
    if want_forces:
        b["positions"].requires_grad_(True)
        with torch.enable_grad():
            pred = model(b.to_dict(), compute_force=False, training=False)
            E = pred["energy"].sum()
            ebl = pred["baseline_coupling_energy_g"].sum()
            gE = torch.autograd.grad(E, b["positions"], retain_graph=True)[0]
            gB = torch.autograd.grad(ebl, b["positions"], allow_unused=True)[0]
        F = -gE.detach().cpu().numpy()
        Fbl = -gB.detach().cpu().numpy() if gB is not None else np.zeros_like(F)
        fd = np.asarray(a.get_array("forces"))
        rec["F_rms_err_meV"] = 1e3 * float(np.sqrt(((F - fd) ** 2).mean()))
        rec["F_rms_err_noEbl_meV"] = 1e3 * float(np.sqrt(((F - Fbl - fd) ** 2).mean()))
        rec["Fbl_rms_meV"] = 1e3 * float(np.sqrt((Fbl ** 2).mean()))
        rec["Fbl_max_meV"] = 1e3 * float(np.abs(Fbl).max())
        rec["Fbl_rms_water_meV"] = 1e3 * float(np.sqrt((Fbl[np.isin(a.get_chemical_symbols(), ["O", "H"])] ** 2).mean()))
    else:
        with torch.no_grad():
            pred = model(b.to_dict(), compute_force=False, training=False)
    rec["E_model"] = float(pred["energy"].sum()); rec["E_dft"] = float(a.info["energy"])
    for k in ("baseline_coupling_energy_g", "compensation_periodic_1d_energy",
              "compensation_slab_correction_energy", "cavity_energy_g", "solvent3d_energy_g",
              "electrostatic_energy"):
        rec[k] = float(pred[k].sum())
    r = _cap["r"]
    bl = r.get("bl_export")
    z = r["z"].detach().cpu().numpy().astype(float)
    rho_i = r["rho_ion_z"].detach().cpu().numpy().astype(float)
    rho_b = r["rho_bound_z"].detach().cpu().numpy().astype(float)
    phi_z = r["phi_z"].detach().cpu().numpy().astype(float)
    H = float(r["height"]); n = z.shape[0]; dz = H / n
    pbz = bl["phi"].detach().cpu().numpy().astype(float); area = float(bl["area"])
    assert abs(float(bl["dz"]) - dz) < 1e-9 and pbz.shape[0] == n, (bl["dz"], dz, pbz.shape, n)
    rho_s = rho_i + rho_b
    rec.update(H=H, n=n, dz=dz, area=area)
    # Q1: recomputed E_bl, reciprocity, cross/self split, medium internal energy estimate
    rec["Ebl_recomp"] = float((rho_s * (-pbz)).sum() * dz * area)
    phi_solv = poisson_1d(rho_s, dz)                   # physics sign
    rho_base = second_deriv(pbz, dz) / K_COUL           # rho_base = -(1/K) d2(-pbz)/dz2
    rec["Ebl_recip"] = float((rho_base * phi_solv).sum() * dz * area)
    rec["q_solv"] = float(rho_s.sum() * dz * area); rec["q_ion"] = float(rho_i.sum() * dz * area)
    rec["q_base_1d"] = float(rho_base.sum() * dz * area)
    rec["self_solv"] = float(0.5 * (rho_s * phi_solv).sum() * dz * area)
    rec["cross_net_from_E1D"] = rec["compensation_periodic_1d_energy"] - rec["self_solv"]
    # solver phi_z convention: hypothesis A physics (phi_tot = -pbz + phi_solv), B electron-PE
    hA = (-pbz + phi_solv); hB = (pbz - phi_solv)
    for nm, h in (("A", hA), ("B", hB)):
        d = phi_z - h; d = d - d.mean()
        rec[f"phi_conv_{nm}_rms"] = float(np.sqrt((d ** 2).mean()))
    rec["phi_z_rms"] = float(np.sqrt(((phi_z - phi_z.mean()) ** 2).mean()))
    phi_tot_phys = phi_z if rec["phi_conv_A_rms"] <= rec["phi_conv_B_rms"] else -phi_z
    phi_tot_phys = phi_tot_phys - phi_tot_phys.mean()
    rec["Uint_lin_solver"] = float(-0.5 * (rho_s * phi_tot_phys).sum() * dz * area)
    rec["Uint_lin_recon"] = float(-0.5 * (rho_s * (-pbz + phi_solv + 0.0)).sum() * dz * area)  # without the net-charge potential (not exported)
    rec["half_cross_total_proxy"] = float(0.5 * (rec["cross_net_from_E1D"] + rec["Ebl_recomp"]))
    # Q2: E_bl with the DFT solvent charge on the reference grid (500 pts, dz 0.09)
    if sid in ref_row:
        i = ref_row[sid]
        rb, ion = ref_rb[i], ref_ion[i]
        dz_r = ref_lz / rb.shape[0]
        pbz_r = resample(pbz, rb.shape[0])
        # unit convention of the reference: pick the reading that integrates to the ionic charge
        Q_dens = float(ion.sum() * dz_r * area)            # if e/A^3
        Q_vals = float(ion.sum() / ion.shape[0])            # if VASP "values" (sum/N = charge)
        rec["dft_Qion_as_density"], rec["dft_Qion_as_values"] = Q_dens, Q_vals
        conv = "density" if abs(abs(Q_dens) - abs(rec["q_ion"])) <= abs(abs(Q_vals) - abs(rec["q_ion"])) else "values"
        scale = 1.0 if conv == "density" else 1.0 / (dz_r * area)
        rho_s_dft = (rb + ion) * scale
        rec["dft_conv"] = conv
        rec["Ebl_dft_field"] = float((rho_s_dft * (-pbz_r)).sum() * dz_r * area)
        rec["q_solv_dft"] = float(rho_s_dft.sum() * dz_r * area)
        rec["q_ion_dft"] = float(ion.sum() * scale * dz_r * area)
        rho_s_m_r = resample(rho_s, rb.shape[0])
        rec["rho_solv_L1_model_vs_dft"] = float(np.abs(rho_s_m_r - rho_s_dft).sum() * dz_r * area)
        rec["rho_solv_L1_dft"] = float(np.abs(rho_s_dft).sum() * dz_r * area)
        zr = ref_z if ref_z.shape[0] == rb.shape[0] else np.arange(rb.shape[0]) * dz_r
        rec["mu_solv_model"] = float((rho_s_m_r * zr).sum() * dz_r * area)
        rec["mu_solv_dft"] = float((rho_s_dft * zr).sum() * dz_r * area)
        rec["_rho_s_dft"] = rho_s_dft; rec["_rho_s_model_r"] = rho_s_m_r; rec["_pbz_r"] = pbz_r
    rec["_rho_i"], rec["_rho_b"], rec["_pbz"], rec["_phi_z"], rec["_z"] = rho_i, rho_b, pbz, phi_z, z
    del pred, b
    _evict()
    return rec


t0 = time.time(); F = {}
for i, k in enumerate(pairs):
    for sid in (k, k + 600):
        F[sid] = run(sid, DO_FORCES)
    if i == 0:
        c, nn = F[k], F[k + 600]
        print(f"\n=== conventions, first pair (sid {k} / {k+600}) ===")
        for s, r in ((k, c), (k + 600, nn)):
            print(f"   sid {s}: grid n={r['n']} dz={r['dz']:.4f} A, area {r['area']:.2f} A^2, q_ion(model) {r['q_ion']:+.4f} e, "
                  f"q_solv {r['q_solv']:+.4f}, q_base_1d {r['q_base_1d']:+.4f}; E_bl exported {r['baseline_coupling_energy_g']:+.5f} "
                  f"recomputed {r['Ebl_recomp']:+.5f} reciprocity {r['Ebl_recip']:+.5f}")
            print(f"           solver phi_z convention: rms(phi_z - [physics -pbz+phi_solv]) {r['phi_conv_A_rms']:.3f}, "
                  f"rms(phi_z - [electron-PE]) {r['phi_conv_B_rms']:.3f}, phi_z rms {r['phi_z_rms']:.3f} eV")
            if "dft_conv" in r:
                print(f"           DFT ref ion charge: as e/A^3 {r['dft_Qion_as_density']:+.4f}, as VASP values {r['dft_Qion_as_values']:+.4f} "
                      f"-> read as {r['dft_conv']}; q_ion_dft {r['q_ion_dft']:+.4f}, q_solv_dft {r['q_solv_dft']:+.4f}")
    if (i + 1) % 10 == 0:
        print(f"   {i+1}/{len(pairs)} pairs, {time.time()-t0:.0f} s", flush=True)
print(f"   all {len(pairs)} pairs ({2*len(pairs)} frames) in {time.time()-t0:.0f} s")

np.savez(os.path.join(OUT, f"ebl_audit_{TAG}.npz"),
         frames=np.array([{kk: v for kk, v in r.items()} for r in F.values()], dtype=object), epoch=EPOCH)


def rms(x): x = np.asarray(x, float); x = x[np.isfinite(x)]; return float(np.sqrt((x ** 2).mean())) if x.size else float("nan")
def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 2 else float("nan")


allf = list(F.values())
print(f"\n=== Q1: identities on all {len(allf)} frames ===")
print(f"   E_bl exported vs recomputed from the exported fields: max |diff| {max(abs(r['baseline_coupling_energy_g']-r['Ebl_recomp']) for r in allf):.2e} eV")
print(f"   reciprocity int rho_base*phi_solv vs E_bl: max |diff| {max(abs(r['Ebl_recip']-r['Ebl_recomp']) for r in allf):.2e} eV, "
      f"E_bl rms {rms([r['Ebl_recomp'] for r in allf]):.4f}")
print(f"   1-D baseline charge integrates to {np.mean([r['q_base_1d'] for r in allf]):+.2e} e (zero-mean potential -> zero net charge by construction)")
print(f"   solver phi_z convention votes: physics {sum(r['phi_conv_A_rms'] < r['phi_conv_B_rms'] for r in allf)}, electron-PE "
      f"{sum(r['phi_conv_A_rms'] >= r['phi_conv_B_rms'] for r in allf)}; median residual rms of the winner "
      f"{np.median([min(r['phi_conv_A_rms'], r['phi_conv_B_rms']) for r in allf]):.3f} eV (net-charge potential + dipole correction not in the reconstruction)")
for st in sorted({r["state"] for r in allf}):
    R = [r for r in allf if r["state"] == st]
    print(f"   {st:14s} n={len(R):2d}: E_bl {np.mean([r['Ebl_recomp'] for r in R]):+.3f}  1D cross_net {np.mean([r['cross_net_from_E1D'] for r in R]):+.3f}  "
          f"1D self {np.mean([r['self_solv'] for r in R]):+.3f}  slab dip {np.mean([r['compensation_slab_correction_energy'] for r in R]):+.3f}  "
          f"U_int(lin, solver phi) {np.mean([r['Uint_lin_solver'] for r in R]):+.3f}  half-cross proxy {np.mean([r['half_cross_total_proxy'] for r in R]):+.3f} eV")

print(f"\n=== Q2 + pairs: model-field vs DFT-field E_bl, charged minus neutral ===")
for split in ("val", "train"):
    P = [k for k in pairs if split_by_sid[k] == split and split_by_sid[k + 600] == split]
    if not P:
        continue
    res = np.array([(F[k]["E_model"] - F[k + 600]["E_model"]) - (F[k]["E_dft"] - F[k + 600]["E_dft"]) for k in P])
    dm = np.array([F[k]["Ebl_recomp"] - F[k + 600]["Ebl_recomp"] for k in P])
    dd = np.array([F[k].get("Ebl_dft_field", np.nan) - F[k + 600].get("Ebl_dft_field", np.nan) for k in P])
    du = np.array([F[k]["Uint_lin_solver"] - F[k + 600]["Uint_lin_solver"] for k in P])
    dq = np.array([F[k]["q_ion"] - F[k + 600]["q_ion"] for k in P]); dqd = np.array([F[k].get("q_ion_dft", np.nan) - F[k + 600].get("q_ion_dft", np.nan) for k in P])
    print(f"   {split} ({len(P)} pairs): residual rmse {rms(res):.4f} bias {res.mean():+.4f} std {res.std():.4f}")
    print(f"      dE_bl model-field: mean {dm.mean():+.4f} rms {rms(dm):.4f} | DFT-field: mean {np.nanmean(dd):+.4f} rms {rms(dd):.4f} | "
          f"corr(model, DFT) {corr(dm, dd):+.3f} | rms(model - DFT) {rms(dm - dd):.4f}")
    print(f"      corr(resid, dE_bl model) {corr(res, dm):+.3f}; corr(resid, dE_bl DFT-field) {corr(res, dd):+.3f}; "
          f"resid - dE_bl(model): std {np.std(res - dm):.4f}; resid - dE_bl(model) + dE_bl(DFT): rmse {rms(res - dm + dd):.4f} bias {np.nanmean(res - dm + dd):+.4f} std {np.nanstd(res - dm + dd):.4f}")
    print(f"      d q_ion model {dq.mean():+.4f} vs DFT {np.nanmean(dqd):+.4f} e; d U_int(lin): mean {du.mean():+.4f} rms {rms(du):.4f} corr(resid) {corr(res, du):+.3f}; "
          f"resid - dU_int: bias {np.mean(res - du):+.4f} std {np.std(res - du):.4f}")
    print(f"      {'k':>4} {'resid':>7} {'dEbl_m':>8} {'dEbl_dft':>8} {'dUint':>7} {'dq_ion_m':>8} {'dq_ion_d':>8} {'L1(m-d) c':>9} {'L1(m-d) n':>9}")
    for k, r_, a_, b_, u_ in zip(P, res, dm, dd, du):
        print(f"      {k:>4} {r_:+7.3f} {a_:+8.4f} {b_:+8.4f} {u_:+7.3f} {F[k]['q_ion']-F[k+600]['q_ion']:+8.4f} "
              f"{F[k].get('q_ion_dft', np.nan)-F[k+600].get('q_ion_dft', np.nan):+8.4f} {F[k].get('rho_solv_L1_model_vs_dft', np.nan):9.4f} {F[k+600].get('rho_solv_L1_model_vs_dft', np.nan):9.4f}")
    # per-frame E_bl model vs DFT field
    em = np.array([F[s]["Ebl_recomp"] for k in P for s in (k, k + 600)]); ed = np.array([F[s].get("Ebl_dft_field", np.nan) for k in P for s in (k, k + 600)])
    print(f"      per-frame E_bl: model-field mean {em.mean():+.4f}, DFT-field mean {np.nanmean(ed):+.4f}, model - DFT mean {np.nanmean(em-ed):+.4f} rms {rms(em-ed):.4f}, corr {corr(em, ed):+.3f}")

if DO_FORCES:
    print(f"\n=== Q3: deletion stakes without retraining ===")
    for st in sorted({r["state"] for r in allf}):
        R = [r for r in allf if r["state"] == st]
        print(f"   {st:14s} n={len(R):2d}: E_bl mean {np.mean([r['Ebl_recomp'] for r in R]):+.3f} eV = {1e3*np.mean([r['Ebl_recomp']/r['nat'] for r in R]):+.2f} meV/atom "
              f"(deleting shifts the absolute energy by minus this); force error rms {np.mean([r['F_rms_err_meV'] for r in R]):.2f} -> without E_bl force "
              f"{np.mean([r['F_rms_err_noEbl_meV'] for r in R]):.2f} meV/A; -dE_bl/dR rms {np.mean([r['Fbl_rms_meV'] for r in R]):.2f} (water atoms {np.mean([r['Fbl_rms_water_meV'] for r in R]):.2f}), max {np.mean([r['Fbl_max_meV'] for r in R]):.1f} meV/A")
print("\nDONE")
