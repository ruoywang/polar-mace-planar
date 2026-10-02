"""Where does the I2 degradation come from? (follow-up to onedim_cross.py, 2026-10-02; no training.)
For each frame: the production closure inputs are captured (electron density n_e^ML, solute potential cvhar3, the final
solver call's arguments, delta_p); the DFT electron density n_e^DFT is built as in onedim_cross.py. Then
  (1) cavity geometry from the two densities (tp.create_cavity_torch): plane means of s_diel / s_ion, per-column
      position of the dielectric edge (s_diel = 0.5 crossing from the solute side), fraction of points with |ds| > 0.5;
      density difference in the edge region (0.005 < n_e^ML < 0.05 e/A^3): rmse and mean of n_e^DFT - n_e^ML;
  (2) closure outputs with the ML source and either density: L1 change of A_scr (a1), S_ion_z, prior, w_env;
  (3) mixed 1-D solves from the captured solver arguments, replacing ONE closure quantity at a time by its DFT-density
      version (a1 | p_off = prior + delta_p | s_ion), all three (= I2), and all three with delta_p off (= I4 cavity part):
      rho_b / rho_ion against the DFT plane averages (L1, L2, 1-D potential rms), per frame and charged - neutral response.
Usage: python cavity_check.py <out.json>   (cwd with ./data, ./cal1_train.json); env KIT_PAIRS, KIT_DEVICE
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")

import numpy as np
import torch
from ase.io import read

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils
import mace.modules.pb1d_backend as PB
from mace.modules.solvent3d import _interp3_periodic

OUT = sys.argv[1] if len(sys.argv) > 1 else "cavity_check.json"
MODEL = os.environ.get("KIT_MODEL", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_vsolv_fix/models/prod500_vsolv_fix.model")
PAIRS = [int(s) for s in os.environ.get("KIT_PAIRS", "1 52 152").replace("+", " ").split()]
devname = os.environ.get("KIT_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
dev = torch.device(devname)
if dev.type == "cpu":
    _jl = torch.jit.load
    torch.jit.load = lambda f, map_location=None, **kw: _jl(f, map_location="cpu", **kw)
model = torch.load(MODEL, map_location=dev).to(dev); model.eval()
for p in model.parameters():
    p.requires_grad_(False)
backend = model._get_pb1d_backend(); tp = backend._tp; params = backend.params
K_EVA = 14.39964546866782
man_s = json.load(open("data/solvent3d_grid_manifest.json")); ENT_S = {int(k): v for k, v in man_s["entries"].items()}
man_n = json.load(open("data/density3d_net_grid_manifest_npy.json")); ENT_N = {int(k): v for k, v in man_n["entries"].items()}
ATOMS = {}
for sp in ("train", "val", "test"):
    for a in read(f"data/{sp}.xyz", ":"):
        a.info["_split"] = sp; ATOMS[int(a.info["sample_id"])] = a
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin", "sample_id": "sample_id",
                                    "solvated": "solvated", "fermi_level": "Fermi", "potential": "potential_diff"}, arrays_keys={"forces": "forces"})
T0 = time.time()

CAP = {}
_orig_clo = PB.closure_from_fields
_orig_solve = PB.Solver1D.solve


def clo_cap(n_e, cv, grid, params_, tp_):
    CAP["ne"] = n_e.detach().clone(); CAP["cv"] = cv.detach().clone(); CAP["grid"] = grid
    return _orig_clo(n_e, cv, grid, params_, tp_)


def solve_cap(self, **kw):
    CAP["solver"] = self; CAP["kw"] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kw.items()}
    return _orig_solve(self, **kw)


PB.closure_from_fields = clo_cap
PB.Solver1D.solve = solve_cap


def dft_planes(sid):
    e = ENT_S[sid]
    rb = np.asarray(np.load("data/" + e["path_b"], mmap_mode="r"), dtype=np.float64).mean(axis=(1, 2))
    ri = np.asarray(np.load("data/" + e["path_i"], mmap_mode="r"), dtype=np.float64).mean(axis=(1, 2))
    with np.load("data/" + e["meta_path"]) as m:
        lat = np.asarray(m["lattice"], dtype=np.float64)
    return rb, ri, lat


def dft_net_on_model_grid(sid, mshape, volume):
    e = ENT_N[sid]
    rho = torch.as_tensor(np.asarray(np.load(e["path"], mmap_mode="r"), dtype=np.float64), device=dev).permute(2, 1, 0).contiguous()
    nx, ny, nz = mshape
    ii, jj, kk = torch.meshgrid(torch.arange(nx, device=dev), torch.arange(ny, device=dev), torch.arange(nz, device=dev), indexing="ij")
    fm = torch.stack([ii.reshape(-1) / nx, jj.reshape(-1) / ny, kk.reshape(-1) / nz], dim=1).double()
    return _interp3_periodic(rho, fm).reshape(nx, ny, nz) * volume


def forward(sid):
    atoms = ATOMS[sid]
    cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(dev)
    hold = {}; orig = backend.solve_graph

    def spy(*a, **kw):
        res = orig(*a, **kw); hold["res"] = res; return res
    backend.solve_graph = spy
    try:
        with torch.no_grad():
            model(batch.to_dict(), training=False, compute_force=False)
    finally:
        backend.solve_graph = orig
    return hold["res"]


def poisson1d(rho, lz):
    n = rho.size; rg = np.fft.rfft(rho); g = 2 * np.pi * np.fft.rfftfreq(n, d=lz / n)
    pg = np.zeros_like(rg); pg[1:] = 4 * np.pi * K_EVA * rg[1:] / g[1:] ** 2
    return np.fft.irfft(pg, n=n)


def on_dft_z(prof, z_solve, lz, nz_dft):
    zd = np.arange(nz_dft) * lz / nz_dft
    return np.interp(zd, np.concatenate([z_solve, [lz]]), np.concatenate([prof, [prof[0]]]))


def m1d(m, d, lz):
    e = m - d
    return dict(L1=float(np.abs(e).sum() / np.abs(d).sum()), L2=float((e ** 2).sum() / (d ** 2).sum()),
                phi_rms_err=float(np.sqrt(((poisson1d(m, lz) - poisson1d(d, lz)) ** 2).mean())))


def edge_z(s, lz):
    """per column: first z (from z = 0 upward, i.e. from the solute side of the window) where s >= 0.5; the solute sits
    low in the cell and the solvent above it, so this is the dielectric edge facing the solvent."""
    nx, ny, nz = s.shape
    above = (s >= 0.5)
    idx = torch.where(above, torch.arange(nz, device=dev)[None, None, :].expand_as(s), torch.full_like(above, nz, dtype=torch.long)).min(dim=2).values
    return idx.double() * lz / nz


RES = []
for k in PAIRS:
    rec = dict(pair=k, split=ATOMS[k].info["_split"], frames={}, response={})
    profs = {}
    for sid in (k, k + 600):
        rb_d, ri_d, lat = dft_planes(sid); lz = lat[2, 2]; nzd = rb_d.size
        res = forward(sid); grid = CAP["grid"]; ne_ml = CAP["ne"]; cv = CAP["cv"]; solver = CAP["solver"]; kw = dict(CAP["kw"])
        V = float(grid.volume); mshape = tuple(ne_ml.shape)
        net_dft = dft_net_on_model_grid(sid, mshape, V)
        bl_row = backend._bl_index.get(sid)
        neutral_v = torch.from_numpy(np.ascontiguousarray(backend._bl_arr[bl_row][backend._bl_take])).to(dev).double()[0]
        ne_dft = torch.clamp((neutral_v - net_dft) / V, min=0.0)
        fr = {}
        with torch.no_grad():
            # (1) cavities
            si_ml, sd_ml, sc_ml = tp.create_cavity_torch(ne_ml, grid, params); si_df, sd_df, sc_df = tp.create_cavity_torch(ne_dft, grid, params)
            ez_ml, ez_df = edge_z(sd_ml, lz), edge_z(sd_df, lz)
            region = (ne_ml > 0.005) & (ne_ml < 0.05)
            fr["cavity"] = dict(sdiel_plane_L1=float((sd_df.mean((0, 1)) - sd_ml.mean((0, 1))).abs().sum() / sd_ml.mean((0, 1)).abs().sum()),
                                sion_plane_L1=float((si_df.mean((0, 1)) - si_ml.mean((0, 1))).abs().sum() / si_ml.mean((0, 1)).abs().sum()),
                                edge_dz_median=float((ez_df - ez_ml).median()), edge_dz_q25=float((ez_df - ez_ml).quantile(0.25)), edge_dz_q75=float((ez_df - ez_ml).quantile(0.75)),
                                edge_dz_absmedian=float((ez_df - ez_ml).abs().median()),
                                frac_sdiel_flip=float(((sd_df - sd_ml).abs() > 0.5).double().mean()),
                                edge_region_points=int(region.sum()), ne_diff_rmse=float(((ne_dft - ne_ml)[region] ** 2).mean().sqrt()),
                                ne_diff_mean=float((ne_dft - ne_ml)[region].mean()), ne_ml_mean_region=float(ne_ml[region].mean()),
                                ne_diff_rmse_all=float(((ne_dft - ne_ml) ** 2).mean().sqrt()))
            # (2) closure outputs
            clo_ml = _orig_clo(ne_ml, cv, grid, params, tp); clo_df = _orig_clo(ne_dft, cv, grid, params, tp)
            fr["closure"] = {q: dict(L1=float((clo_df[q] - clo_ml[q]).abs().sum() / clo_ml[q].abs().sum().clamp(min=1e-30)),
                                    ml_absmean=float(clo_ml[q].abs().mean()), dft_absmean=float(clo_df[q].abs().mean()))
                             for q in ("A_scr", "S_ion_z", "prior", "w_env")}
            # (3) mixed solves
            f = backend.solve_upsample
            up = lambda x: PB.fourier_upsample(x, f)
            dp = res["delta_p"].detach().double()
            a_df = torch.clamp(up(clo_df["A_scr"]), min=0.0); s_df = torch.clamp(up(clo_df["S_ion_z"]), min=0.0); p_df = up(clo_df["prior"])
            a_ml = torch.clamp(up(clo_ml["A_scr"]), min=0.0); s_ml = torch.clamp(up(clo_ml["S_ion_z"]), min=0.0); p_ml = up(clo_ml["prior"])
            chk = dict(a1_repro=float((a_ml - kw["a1"]).abs().max()), sion_repro=float((s_ml - kw["s_ion"]).abs().max()), poff_repro=float((p_ml + dp - kw["p_off"]).abs().max()))
            mixes = {"M0 production": dict(), "Ma a1<-DFT": dict(a1=a_df), "Mp prior<-DFT": dict(p_off=p_df + dp), "Ms s_ion<-DFT": dict(s_ion=s_df),
                     "Mall (=I2 cavity)": dict(a1=a_df, p_off=p_df + dp, s_ion=s_df), "Mall dp off": dict(a1=a_df, p_off=p_df, s_ion=s_df)}
            fr["mix"] = {}; profs[sid] = {}
            for nm, sub in mixes.items():
                kk = dict(kw); kk.update(sub); kk["phi_init"] = None
                out = _orig_solve(solver, **kk)
                rb = -(out["n_b"] / V).cpu().numpy(); ri = -(out["n_ion"] / V).cpu().numpy(); zs = solver.z.cpu().numpy()
                rbd, rid = on_dft_z(rb, zs, lz, nzd), on_dft_z(ri, zs, lz, nzd)
                profs[sid][nm] = (rbd, rid)
                fr["mix"][nm] = dict(b=m1d(rbd, rb_d, lz), i=m1d(rid, ri_d, lz), t_phi=m1d(rbd + rid, rb_d + ri_d, lz)["phi_rms_err"],
                                     newton_exit=str(out.get("solver_exit", "")).split("'newton_exit': ")[-1][:8])
        # ---- polarization-divergence consistency (lateral_polarization.py check failed): rebuild the solver's own rho_b
        # from the closure ingredients, term by term, and along the 3-D route; everything on the 300-plane model grid
        with torch.no_grad():
            import math
            from mace.modules.pb1d_closure import EDEPS, response_a3
            sigma_b = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
            w_b3 = tp._normalized_gaussian_kernel_g(grid, sigma_b)
            a3z = response_a3(sd_ml.new_zeros(1), sd_ml.new_ones(1), params, tp)[0] * sd_ml; eps3 = 1.0 + EDEPS * a3z
            ex, ey, ez, emag = grid.grad_from_recip(-torch.conj(w_b3) * grid.fft(cv))
            a3 = response_a3(emag / eps3, sd_ml, params, tp); A1 = a3.mean((0, 1)); ez_s = ez / eps3
            prior300 = (a3 * ez_s).mean((0, 1)) - A1 * ez_s.mean((0, 1))
            nzm = cv.shape[2]
            kz = 2.0 * math.pi * torch.fft.fftfreq(nzm, d=lz / nzm, device=dev, dtype=torch.float64); wb1 = torch.exp(-0.5 * (sigma_b * kz) ** 2)
            dzs = lambda x: torch.fft.ifft(1j * kz * wb1 * torch.fft.fft(x)).real
            phi600 = res["phi_z"].detach().double(); E600 = -(solver._core @ phi600)        # the solver's E_z = -(D WB phi)
            ez_1d = torch.fft.ifft(-1j * kz * wb1 * torch.fft.fft(phi600[::f])).real
            rho_sol = res["rho_bound_z"].detach().double()
            B_mat = solver.bound_matrix(kw["a1"]); nb_phi = B_mat @ phi600; nb_off = solver.bound_offset(kw["p_off"])
            rho_phi_part = -(nb_phi / V); rho_off_part = -(nb_off / V)
            dp600 = res["delta_p"].detach().double(); prior600 = kw["p_off"] - dp600
            r_noDp = dzs(A1 * ez_1d + prior300); r_Dp = dzs(A1 * ez_1d + prior300 + dp600[::f])
            Ez3 = ez_s - ez_s.mean((0, 1))[None, None, :] + ez_1d[None, None, :]
            g3 = grid.ifft_real(torch.conj(w_b3) * grid.div_real_vector(a3 * ex / eps3, a3 * ey / eps3, a3 * Ez3)); gm = g3.mean((0, 1))
            cor = lambda a, b: float((a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum()))
            L1r = lambda a, b: float((a - b).abs().sum() / b.abs().sum())
            rs = rho_sol[::f]
            fr["pol_check"] = dict(prior_repro_maxdiff=float((prior300 - clo_ml["prior"]).abs().max()), a1_repro=float((A1 - clo_ml["A_scr"]).abs().max()),
                                   E_L1_vs_solver=L1r(ez_1d, E600[::f]), rho_split_repro=float((rho_phi_part + rho_off_part - rho_sol).abs().max()),
                                   rms_rho_phi_part=float(rho_phi_part.pow(2).mean().sqrt()), rms_rho_off_part=float(rho_off_part.pow(2).mean().sqrt()), rms_rho=float(rho_sol.pow(2).mean().sqrt()),
                                   rms_dz_prior=float(dzs(prior300).pow(2).mean().sqrt()), rms_dz_dp=float(dzs(dp600[::f]).pow(2).mean().sqrt()), rms_dz_a1E=float(dzs(A1 * ez_1d).pow(2).mean().sqrt()),
                                   r_noDp=dict(corr=cor(r_noDp, rs), L1=L1r(r_noDp, rs)), r_Dp=dict(corr=cor(r_Dp, rs), L1=L1r(r_Dp, rs)), g3_mean=dict(corr=cor(gm, rs), L1=L1r(gm, rs)),
                                   g3_vs_r_noDp=dict(corr=cor(gm, r_noDp), L1=L1r(gm, r_noDp)))
            pc = fr["pol_check"]
            print(f"[{time.time() - T0:5.0f}s] sid {sid} pol-check: prior repro {pc['prior_repro_maxdiff']:.1e} a1 repro {pc['a1_repro']:.1e} | E_z 1-D vs solver L1 {pc['E_L1_vs_solver']:.3f} | rho split repro {pc['rho_split_repro']:.1e}; "
                  f"rms rho {pc['rms_rho']:.2e} = phi-part {pc['rms_rho_phi_part']:.2e} + offset-part {pc['rms_rho_off_part']:.2e}; dz-terms a1E {pc['rms_dz_a1E']:.2e} prior {pc['rms_dz_prior']:.2e} dp {pc['rms_dz_dp']:.2e} | "
                  f"1-D rebuild noDp corr {pc['r_noDp']['corr']:+.3f} L1 {pc['r_noDp']['L1']:.3f}; with Dp corr {pc['r_Dp']['corr']:+.3f} L1 {pc['r_Dp']['L1']:.3f}; 3-D route <g> corr {pc['g3_mean']['corr']:+.3f} L1 {pc['g3_mean']['L1']:.3f}; <g> vs 1-D noDp corr {pc['g3_vs_r_noDp']['corr']:+.3f} L1 {pc['g3_vs_r_noDp']['L1']:.3f}", flush=True)
        fr["check"] = chk
        rec["frames"][str(sid)] = fr
        c = fr["cavity"]
        print(f"[{time.time() - T0:5.0f}s] sid {sid}: edge dz DFT-ML median {c['edge_dz_median']:+.3f} A (|dz| median {c['edge_dz_absmedian']:.3f}, q25/q75 {c['edge_dz_q25']:+.3f}/{c['edge_dz_q75']:+.3f}); "
              f"s_diel plane L1 {c['sdiel_plane_L1']:.3f}, s_ion {c['sion_plane_L1']:.3f}, flip frac {c['frac_sdiel_flip']:.4f}; edge-region ne diff rmse {c['ne_diff_rmse']:.4f} (mean {c['ne_diff_mean']:+.4f}, ne {c['ne_ml_mean_region']:.4f}) | "
              f"closure L1: A_scr {fr['closure']['A_scr']['L1']:.3f} S_ion {fr['closure']['S_ion_z']['L1']:.3f} prior {fr['closure']['prior']['L1']:.3f} w_env {fr['closure']['w_env']['L1']:.3f} | repro {chk} | "
              + " ; ".join(f"{nm}: b L1 {v['b']['L1']:.3f} phi {v['b']['phi_rms_err']:.3f}" for nm, v in fr["mix"].items()), flush=True)
    rb_c, ri_c, lat = dft_planes(k); rb_n, ri_n, _ = dft_planes(k + 600); lz = lat[2, 2]
    for nm in profs[k]:
        mb = profs[k][nm][0] - profs[k + 600][nm][0]; mi = profs[k][nm][1] - profs[k + 600][nm][1]
        rec["response"][nm] = dict(b=m1d(mb, rb_c - rb_n, lz), i=m1d(mi, ri_c - ri_n, lz), t_phi=m1d(mb + mi, (rb_c + ri_c) - (rb_n + ri_n), lz)["phi_rms_err"])
    print(f"[{time.time() - T0:5.0f}s] pair {k} response: " + " ; ".join(f"{nm}: b L1 {v['b']['L1']:.3f} phi {v['b']['phi_rms_err']:.3f} t phi {v['t_phi']:.3f}" for nm, v in rec["response"].items()), flush=True)
    RES.append(rec); json.dump(RES, open(OUT, "w"), indent=1)
print(f"[{time.time() - T0:5.0f}s] done", flush=True)
