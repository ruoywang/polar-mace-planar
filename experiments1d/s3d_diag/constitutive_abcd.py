"""Constitutive comparison A / B / C / D (reviewer 2026-10-02; no training, no production-code change).
Six training frames. All fields on the model grid (100 x 100 x 300), production operators (TorchGrid spectral grad / div,
w_b Gaussian with sigma_b = R_B or A_K, response_a3 with the local-field factor and Langevin saturation), bound charge
rho_b = w_b div( a(|E|) E ) exactly as pb1d_solver.bound_matrix writes it in 1-D; compared against the DFT bound charge
(solvent3d_grid RHOB) on the DFT grid: total L1, lateral L1 / squared, plane-average L1, net charge, 3-D bound potential
rms; charged - neutral response per pair.
    A   DFT total electrostatic field (production Poisson operator on the DFT solute net charge + DFT RHOB + RHOION, plane
        mean of E_z replaced by the DFT 1-D potential label potential1d_potcar_cache phi_eV, so that the laterally uniform
        dipole-correction convention is the label's own), DFT cavity, response recomputed at |E_A|
    B   current approximate 3-D field (screened vacuum field E_vac / eps3 from the DFT solute source and the DFT cavity,
        plane mean of E_z from the same DFT 1-D potential), response recomputed at |E_B|
    C   field B, response frozen at the screened vacuum field |E_vac / eps3| (the production choice)
    D   field A, MODEL cavity (s_diel from the model electron density), response recomputed at |E_A|
    P   production construction for reference (model cavity, model source, plane mean from the solver's own phi), with
        and without the learned delta_p (the delta_p part enters only the plane mean)
Also: native-solve-grid (600 planes) three-term decomposition of the solver's rho_b (a1 E, prior, delta_p) -- exact --
and the discretisation error of the 300-plane rebuild reported separately; convention checks: solver phi vs the DFT phi_eV
label (production frames), reconstructed <phi_tot^DFT>_xy vs phi_eV (raw / mean-removed / line-removed).
Usage: python constitutive_abcd.py <out.json>   (cwd with ./data, ./cal1_train.json); env KIT_PAIRS, KIT_DEVICE
"""
from __future__ import annotations

import json
import math
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
from mace.modules.pb1d_closure import EDEPS, response_a3
from mace.modules.solvent3d import _interp1_periodic, _interp3_periodic

OUT = sys.argv[1] if len(sys.argv) > 1 else "constitutive_abcd.json"
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
POT = np.load("data/potential1d_potcar_cache.npz"); POT_IDX = {int(s): i for i, s in enumerate(POT["sample_ids"])}
REF1D = np.load("data/dft_solvent1d_ref.npz"); REF_IDX = {int(s): i for i, s in enumerate(REF1D["sample_ids"])}
ATOMS = {}
for sp in ("train", "val", "test"):
    for a in read(f"data/{sp}.xyz", ":"):
        a.info["_split"] = sp; ATOMS[int(a.info["sample_id"])] = a
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin", "sample_id": "sample_id",
                                    "solvated": "solvated", "fermi_level": "Fermi", "potential": "potential_diff"}, arrays_keys={"forces": "forces"})
T0 = time.time()
print(f"model {os.path.basename(MODEL)}; device {dev}; pairs {PAIRS}; sigma_b {params['R_B'] if params['R_B'] > 0 else params['A_K']}; "
      f"potential label grid {POT['z_A'].shape}, align_mask mean {float(POT['align_mask'].mean()):.3f}", flush=True)

CAP = {}
_orig_clo = PB.closure_from_fields; _orig_solve = PB.Solver1D.solve


def clo_cap(n_e, cv, grid, params_, tp_):
    CAP["ne"] = n_e.detach().clone(); CAP["cv"] = cv.detach().clone(); CAP["grid"] = grid
    return _orig_clo(n_e, cv, grid, params_, tp_)


def solve_cap(self, **kw):
    CAP["solver"] = self; CAP["kw"] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kw.items()}
    return _orig_solve(self, **kw)


PB.closure_from_fields = clo_cap; PB.Solver1D.solve = solve_cap


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


def label3d(sid):
    e = ENT_S[sid]
    rb = np.asarray(np.load("data/" + e["path_b"], mmap_mode="r"), dtype=np.float64)
    ri = np.asarray(np.load("data/" + e["path_i"], mmap_mode="r"), dtype=np.float64)
    with np.load("data/" + e["meta_path"]) as m:
        lat = np.asarray(m["lattice"], dtype=np.float64)
    return rb, ri, lat


def dft_to_model(arr_zyx, mshape):
    """DFT grid (nz,ny,nx) -> model grid [nx,ny,nz], trilinear periodic."""
    rho = torch.as_tensor(arr_zyx, device=dev).permute(2, 1, 0).contiguous()
    nx, ny, nz = mshape
    ii, jj, kk = torch.meshgrid(torch.arange(nx, device=dev), torch.arange(ny, device=dev), torch.arange(nz, device=dev), indexing="ij")
    fm = torch.stack([ii.reshape(-1) / nx, jj.reshape(-1) / ny, kk.reshape(-1) / nz], dim=1).double()
    return _interp3_periodic(rho, fm).reshape(nx, ny, nz)


def to_dft(F3, shape):
    nz, ny, nx = shape
    iy, ix = torch.meshgrid(torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing="ij")
    fxy = torch.stack([ix.reshape(-1).double() / nx, iy.reshape(-1).double() / ny], dim=1)
    M = np.empty(shape)
    for z0 in range(0, nz, 25):
        z1 = min(z0 + 25, nz)
        fz = torch.arange(z0, z1, device=dev, dtype=torch.float64) / nz
        frac = torch.cat([fxy.repeat(z1 - z0, 1), fz.repeat_interleave(ny * nx)[:, None]], dim=1)
        M[z0:z1] = _interp3_periodic(F3, frac).reshape(z1 - z0, ny, nx).cpu().numpy()
    return M


def poisson(rho, lat):
    nz, ny, nx = rho.shape
    rg = np.fft.rfftn(rho); Bm = 2.0 * np.pi * np.linalg.inv(lat).T
    hx = np.fft.rfftfreq(nx) * nx; hy = np.fft.fftfreq(ny) * ny; hz = np.fft.fftfreq(nz) * nz
    G = (hz[:, None, None, None] * Bm[2] + hy[None, :, None, None] * Bm[1] + hx[None, None, :, None] * Bm[0])
    G2 = (G * G).sum(-1); G2[0, 0, 0] = 1.0
    pg = 4.0 * np.pi * K_EVA * rg / G2; pg[0, 0, 0] = 0.0
    return np.fft.irfftn(pg, s=rho.shape)


def metrics(M, D, lat, dV, pD=None):
    d = M - D; pm, pd = M.mean(axis=(1, 2)), D.mean(axis=(1, 2)); Ml, Dl = M - pm[:, None, None], D - pd[:, None, None]
    pM = poisson(M, lat); pD = poisson(D, lat) if pD is None else pD
    return dict(L1=float(np.abs(d).sum() / np.abs(D).sum()), L2=float((d ** 2).sum() / (D ** 2).sum()),
                eps_lat=float(np.abs(Ml - Dl).sum() / np.abs(Dl).sum()), lat_sq=float(((Ml - Dl) ** 2).sum() / (Dl ** 2).sum()),
                lat_corr=float((Ml * Dl).sum() / np.sqrt((Ml * Ml).sum() * (Dl * Dl).sum())),
                eps_pa=float(np.abs(pm - pd).sum() / np.abs(pd).sum()), pa_corr=float((pm * pd).sum() / np.sqrt((pm * pm).sum() * (pd * pd).sum())),
                net_err_e=float(d.sum()) * dV, net_ml_e=float(M.sum()) * dV, net_dft_e=float(D.sum()) * dV,
                phi_rms_err=float(np.sqrt(((pM - pD) ** 2).mean())), phi_rms_dft=float(np.sqrt((pD ** 2).mean())),
                norm_ratio=float(np.sqrt((M * M).sum() / (D * D).sum())), lat_norm_ratio=float(np.sqrt((Ml * Ml).sum() / (Dl * Dl).sum()))), pD


def interp_z(prof, z_src, lz, nz_dst):
    zd = np.arange(nz_dst) * lz / nz_dst
    zs = np.asarray(z_src) % lz; o = np.argsort(zs); zs, pr = zs[o], np.asarray(prof)[o]
    return np.interp(zd, np.concatenate([zs, [zs[0] + lz]]), np.concatenate([pr, [pr[0]]]))


def line_removed_rms(d, lz):
    z = np.arange(d.size) * lz / d.size; A = np.stack([z, np.ones_like(z)], 1)
    c, *_ = np.linalg.lstsq(A, d, rcond=None); r = d - A @ c
    return float(np.sqrt((r ** 2).mean())), float(c[0])


RES = []
for k in PAIRS:
    rec = dict(pair=k, split=ATOMS[k].info["_split"], frames={}, response={})
    fields_all = {}
    for sid in (k, k + 600):
        Db, Di, lat = label3d(sid); shape = Db.shape; nzd = shape[0]; lz = lat[2, 2]; dV = abs(np.linalg.det(lat)) / Db.size
        res = forward(sid); grid = CAP["grid"]; ne_ml = CAP["ne"]; cv_ml = CAP["cv"]; solver = CAP["solver"]; kw = CAP["kw"]
        V = float(grid.volume); mshape = tuple(ne_ml.shape); nzm = mshape[2]
        bl_row = backend._bl_index.get(sid)
        fields_bl = torch.from_numpy(np.ascontiguousarray(backend._bl_arr[bl_row][backend._bl_take])).to(dev).double()
        neutral_v, phi_base = fields_bl[0], fields_bl[1]
        with torch.no_grad():
            # ---- DFT inputs on the model grid (e/A^3 for densities)
            net_dft = dft_to_model(np.asarray(np.load(ENT_N[sid]["path"], mmap_mode="r"), dtype=np.float64), mshape)
            ne_dft = torch.clamp((neutral_v - net_dft * V) / V, min=0.0)
            rb_dft3 = dft_to_model(Db, mshape); ri_dft3 = dft_to_model(Di, mshape)
            cv_dft = phi_base - grid.ifft_real(grid.l0_inv_op(grid.fft(net_dft * V)))              # solute source, production assembly
            phi_solv = grid.ifft_real(grid.l0_inv_op(grid.fft(-V * (rb_dft3 + ri_dft3))))        # solvent part in the same convention
            phi_tot3 = cv_dft + phi_solv
            # DFT 1-D total potential label on the model planes
            ip = POT_IDX[sid]; phi_lab = interp_z(POT["phi_eV"][ip], POT["z_A"][ip], lz, nzm); phi_lab_t = torch.as_tensor(phi_lab, device=dev)
            amask = interp_z(POT["align_mask"][ip], POT["z_A"][ip], lz, nzm) > 0.5
            # ---- convention checks
            phi_solver = res["phi_z"].detach().double()[::2]                                      # 600 -> 300 planes
            dsol = (phi_solver - phi_lab_t).cpu().numpy(); dsol_al = dsol - dsol[amask].mean() if amask.any() else dsol - dsol.mean()
            pm3 = phi_tot3.mean(dim=(0, 1)).cpu().numpy(); d3 = pm3 - phi_lab; d3_m = d3 - (d3[amask].mean() if amask.any() else d3.mean())
            lr, slope = line_removed_rms(d3, lz)
            chk = dict(solver_vs_label_rms_aligned=float(np.sqrt((dsol_al ** 2).mean())), recon_vs_label_rms_meanremoved=float(np.sqrt((d3_m ** 2).mean())),
                       recon_vs_label_rms_lineremoved=lr, recon_vs_label_slope_eV_per_A=slope, label_rms=float(np.sqrt(((phi_lab - phi_lab.mean()) ** 2).mean())),
                       ne_plane_L1=float((ne_ml.mean((0, 1)) - ne_dft.mean((0, 1))).abs().sum() / ne_dft.mean((0, 1)).abs().sum()),
                       rb3_resample_L1=float(np.abs(rb_dft3.mean((0, 1)).cpu().numpy() - interp_z(Db.mean(axis=(1, 2)), np.arange(nzd) * lz / nzd, lz, nzm)).sum()
                                             / np.abs(interp_z(Db.mean(axis=(1, 2)), np.arange(nzd) * lz / nzd, lz, nzm)).sum()))
            # ---- cavities, kernel, screened fields
            si_d, sd_d, _ = tp.create_cavity_torch(ne_dft, grid, params); si_m, sd_m, _ = tp.create_cavity_torch(ne_ml, grid, params)
            sigma_b = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
            w_b = tp._normalized_gaussian_kernel_g(grid, sigma_b)
            kz = 2.0 * math.pi * torch.fft.fftfreq(nzm, d=lz / nzm, device=dev, dtype=torch.float64); wb1 = torch.exp(-0.5 * (sigma_b * kz) ** 2)
            ez1 = lambda phi1: torch.fft.ifft(-1j * kz * wb1 * torch.fft.fft(phi1)).real          # 1-D field -(w_b d phi/dz)
            resp0 = response_a3(sd_d.new_zeros(1), sd_d.new_ones(1), params, tp)[0]
            eps_d = 1.0 + EDEPS * resp0 * sd_d; eps_m = 1.0 + EDEPS * resp0 * sd_m
            gradw = lambda phi3: grid.grad_from_recip(-torch.conj(w_b) * grid.fft(phi3))        # (ex, ey, ez, |E|) = -(w_b * grad phi)
            exv, eyv, ezv, _ = gradw(cv_dft)                                                     # DFT-source vacuum field
            exs, eys, ezs = exv / eps_d, eyv / eps_d, ezv / eps_d                                # screened (production heuristic), DFT cavity
            exT, eyT, ezT, _ = gradw(phi_tot3)                                                   # DFT total field (periodic Poisson part)
            Ez_lab = ez1(phi_lab_t)
            EA = (exT, eyT, ezT - ezT.mean((0, 1))[None, None, :] + Ez_lab[None, None, :])
            EB = (exs, eys, ezs - ezs.mean((0, 1))[None, None, :] + Ez_lab[None, None, :])
            mag = lambda E: torch.sqrt(E[0] ** 2 + E[1] ** 2 + E[2] ** 2)
            div_b = lambda a, E: grid.ifft_real(torch.conj(w_b) * grid.div_real_vector(a * E[0], a * E[1], a * E[2]))
            cands = {}
            cands["A"] = div_b(response_a3(mag(EA), sd_d, params, tp), EA)
            cands["B"] = div_b(response_a3(mag(EB), sd_d, params, tp), EB)
            cands["C"] = div_b(response_a3(mag((exs, eys, ezs)), sd_d, params, tp), EB)
            cands["D"] = div_b(response_a3(mag(EA), sd_m, params, tp), EA)
            # production construction (reference): model cavity, model source, plane mean from the solver's phi
            exm, eym, ezm, _ = gradw(cv_ml); exsm, eysm, ezsm = exm / eps_m, eym / eps_m, ezm / eps_m
            Ez_sol = ez1(res["phi_z"].detach().double()[::2])
            EP = (exsm, eysm, ezsm - ezsm.mean((0, 1))[None, None, :] + Ez_sol[None, None, :])
            cands["P_noDp"] = div_b(response_a3(mag((exsm, eysm, ezsm)), sd_m, params, tp), EP)
            dp300 = res["delta_p"].detach().double()[::2]
            cands["P"] = cands["P_noDp"] + torch.fft.ifft(1j * kz * wb1 * torch.fft.fft(dp300)).real[None, None, :]
            # A-field with the frozen response (how much does freezing cost when the field is right?)
            cands["A_frozen"] = div_b(response_a3(mag((exs, eys, ezs)), sd_d, params, tp), EA)
            # ---- native-grid (600) three-term decomposition of the solver's rho_b, and the 300-grid rebuild error
            phi600 = res["phi_z"].detach().double(); dp600 = res["delta_p"].detach().double(); prior600 = kw["p_off"] - dp600
            t_phi = -(solver.bound_matrix(kw["a1"]) @ phi600) / V; t_pr = -(solver.bound_offset(prior600)) / V; t_dp = -(solver.bound_offset(dp600)) / V
            rho600 = res["rho_bound_z"].detach().double()
            dzs1 = lambda x: torch.fft.ifft(1j * kz * wb1 * torch.fft.fft(x)).real
            clo = _orig_clo(ne_ml, cv_ml, grid, params, tp)
            r300 = dzs1(clo["A_scr"] * Ez_sol + clo["prior"] + dp300)
            dec = dict(rms_rho=float(rho600.pow(2).mean().sqrt()), rms_a1E=float(t_phi.pow(2).mean().sqrt()), rms_prior=float(t_pr.pow(2).mean().sqrt()),
                       rms_dp=float(t_dp.pow(2).mean().sqrt()), rms_a1E_plus_prior=float((t_phi + t_pr).pow(2).mean().sqrt()),
                       sum_repro_600=float((t_phi + t_pr + t_dp - rho600).abs().max()),
                       rebuild300_vs_600_L1=float((r300 - rho600[::2]).abs().sum() / rho600[::2].abs().sum()),
                       rebuild300_vs_600_corr=float((r300 * rho600[::2]).sum() / torch.sqrt((r300 ** 2).sum() * (rho600[::2] ** 2).sum())),
                       physical_mean_P_rms=float((clo["A_scr"] * Ez_sol + clo["prior"]).pow(2).mean().sqrt()), dp_rms=float(dp300.pow(2).mean().sqrt()),
                       aEscr_mean_rms=float(((response_a3(mag((exsm, eysm, ezsm)), sd_m, params, tp) * ezsm).mean((0, 1))).pow(2).mean().sqrt()))
        # ---- metrics against the DFT bound charge on the DFT grid
        pD = poisson(Db, lat); fr = dict(check=chk, decomposition=dec, cand={})
        fields_all[sid] = {}
        for nm, F3 in cands.items():
            M = to_dft(F3, shape); m, _ = metrics(M, Db, lat, dV, pD); fr["cand"][nm] = m; fields_all[sid][nm] = M
        # the DFT bound charge resampled through the model grid (what the model grid can represent of RHOB)
        Mrt = to_dft(rb_dft3, shape); fr["cand"]["DFT_roundtrip"], _ = metrics(Mrt, Db, lat, dV, pD)
        fr["cand"]["production_1D_only"] = dict(eps_pa=float(np.abs(interp_z(res["rho_bound_z"].detach().cpu().numpy(), np.arange(600) * lz / 600, lz, nzd) - Db.mean(axis=(1, 2))).sum() / np.abs(Db.mean(axis=(1, 2))).sum()))
        rec["frames"][str(sid)] = fr
        c = fr["cand"]
        print(f"[{time.time() - T0:5.0f}s] sid {sid} (q {float(ATOMS[sid].info['total_charge']):+.3f}) checks: solver-vs-label phi rms {chk['solver_vs_label_rms_aligned']:.3f} eV (label rms {chk['label_rms']:.2f}); "
              f"recon <phi>-label rms mean-removed {chk['recon_vs_label_rms_meanremoved']:.3f} line-removed {chk['recon_vs_label_rms_lineremoved']:.3f} (slope {chk['recon_vs_label_slope_eV_per_A']:+.4f} eV/A); ne plane L1 {chk['ne_plane_L1']:.3f} | "
              f"600-grid terms rms: rho {dec['rms_rho']:.2e} a1E {dec['rms_a1E']:.2e} prior {dec['rms_prior']:.2e} a1E+prior {dec['rms_a1E_plus_prior']:.2e} dp {dec['rms_dp']:.2e} (sum repro {dec['sum_repro_600']:.1e}); 300-rebuild L1 {dec['rebuild300_vs_600_L1']:.3f}", flush=True)
        print(f"[{time.time() - T0:5.0f}s]   sid {sid} vs DFT rho_b:  " + " | ".join(f"{nm}: L1 {c[nm]['L1']:.3f} lat {c[nm]['eps_lat']:.3f} pa {c[nm]['eps_pa']:.3f} norm {c[nm]['norm_ratio']:.2f} phi {c[nm]['phi_rms_err']:.3f}"
                                                                        for nm in ("A", "B", "C", "D", "A_frozen", "P_noDp", "P", "DFT_roundtrip")) + f" | production 1-D pa {c['production_1D_only']['eps_pa']:.3f}", flush=True)
    # charged - neutral response
    Dbc, _, lat = label3d(k); Dbn, _, _ = label3d(k + 600); dV = abs(np.linalg.det(lat)) / Dbc.size
    for nm in fields_all[k]:
        m, _ = metrics(fields_all[k][nm] - fields_all[k + 600][nm], Dbc - Dbn, lat, dV); rec["response"][nm] = m
    print(f"[{time.time() - T0:5.0f}s] pair {k} response vs DFT: " + " | ".join(f"{nm}: L1 {rec['response'][nm]['L1']:.3f} lat {rec['response'][nm]['eps_lat']:.3f} pa {rec['response'][nm]['eps_pa']:.3f} phi {rec['response'][nm]['phi_rms_err']:.3f}" for nm in ("A", "B", "C", "D", "A_frozen", "P_noDp", "P")), flush=True)
    RES.append(rec); json.dump(RES, open(OUT, "w"), indent=1)
    del fields_all
print(f"[{time.time() - T0:5.0f}s] done", flush=True)
