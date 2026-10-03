"""Field-response-consistent bound charge on the DFT native grid (reviewer 2026-10-02, step 2 of the single route; offline, no
training). The same total potential gives the 3-D field, the non-linear response is evaluated at that field, and the bound
charge is the divergence of that one polarization field; the 1-D production solution is only an INITIAL GUESS; no prior, no
delta_p, no screened-vacuum-field approximation anywhere.
Fixed point (backend conventions, native grid 500 x 168 x 168):
    phi_tot = phi_sol + L(rho_b + rho_i),   L(rho) = ifft(l0_inv(fft(-V rho)))  (the production Poisson operator)
    E       = -(w_b grad phi_tot)             (w_b = normalized Gaussian, sigma_b = R_B or A_K, as in the closure)
    rho_b   = w_b div( a(|E|, s_diel) E )     (response_a3, the production response function)
Solved by Anderson-accelerated fixed-point iteration (history m, damping beta); the ion charge is held at DFT RHOION in every
run (bound channel only).  Runs (per frame):
    S0   DFT solute potential (PHI - L(RHOB + RHOION)), DFT cavity, start from rho_b = 0
    S1   same inputs, start from the production 1-D solution B(z) (laterally uniform)
    S2   DFT solute potential, MODEL cavity (production electron density, trilinearly upsampled), start from B(z)
    S3   MODEL solute potential (production cv, Fourier-upsampled) + model cavity, start from B(z)
    S3b  S3 with the plane mean of the model solute potential replaced by the DFT one's (lateral model error only)
Also: plain Picard steps F(B), F(F(B)), F(F(F(B))) with DFT inputs (what a few explicit field updates would give), the
error of the initial guess itself, and the model-vs-DFT solute-potential difference.  Per iteration: relative residual
||F(x)-x||/||F(x)||, L1 / lateral L1 / plane-average L1 / norm ratio vs native RHOB, wall time of the F evaluation
(CPU, OMP threads as set; a measurement of this script, not of production).  Final: full metrics incl. bound-potential rms;
charged-minus-neutral response per pair; plane averages and a central xz slice saved to sc_slices/.
Usage: python selfconsistent_native.py <out.json>;  env KIT_PAIRS, KIT_RUNS (e.g. S0+S1+S2+S3+S3b), KIT_MAXIT, KIT_M, KIT_BETA,
KIT_TOL, KIT_PICARD (1/0), KIT_DEVICE, KIT_DFT, KIT_MODEL
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
from mace.modules.pb1d_closure import EDEPS, response_a3
from mace.modules.solvent3d import _interp3_periodic

OUT = sys.argv[1] if len(sys.argv) > 1 else "selfconsistent_native.json"
MODEL = os.environ.get("KIT_MODEL", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_vsolv_fix/models/prod500_vsolv_fix.model")
PAIRS = [int(s) for s in os.environ.get("KIT_PAIRS", "1 52 152").replace("+", " ").split()]
RUNS = os.environ.get("KIT_RUNS", "S0+S1+S2+S3+S3b").replace("+", " ").split()
MAXIT = int(os.environ.get("KIT_MAXIT", "60")); M_HIST = int(os.environ.get("KIT_M", "12")); BETA = float(os.environ.get("KIT_BETA", "0.05"))
TOL = float(os.environ.get("KIT_TOL", "1e-3")); PICARD = os.environ.get("KIT_PICARD", "1") == "1"
PRECOND = os.environ.get("KIT_PRECOND", "none")   # none | eps : residual scaled by 1/(1 + EDEPS a0 s_diel) (local dielectric Jacobi preconditioner)
DFT = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single")
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
SIGMA_B = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
ATOMS = {}
for sp in ("train", "val", "test"):
    for a in read(f"data/{sp}.xyz", ":"):
        a.info["_split"] = sp; ATOMS[int(a.info["sample_id"])] = a
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin", "sample_id": "sample_id",
                                    "solvated": "solvated", "fermi_level": "Fermi", "potential": "potential_diff"}, arrays_keys={"forces": "forces"})
T0 = time.time()
SLICES = os.environ.get("KIT_SLICES", "sc_slices"); os.makedirs(SLICES, exist_ok=True)
print(f"model {os.path.basename(MODEL)}; device {dev} {torch.cuda.get_device_name(dev) if dev.type == 'cuda' else ''}; threads {torch.get_num_threads()}; pairs {PAIRS}; runs {RUNS}; maxit {MAXIT} m {M_HIST} beta {BETA} tol {TOL}; precond {PRECOND}; sigma_b {SIGMA_B}", flush=True)
RESP0 = float(response_a3(torch.zeros(1, device=dev), torch.ones(1, device=dev), params, tp)[0])
CAP = {}
_orig_clo = PB.closure_from_fields


def clo_cap(n_e, cv, grid, params_, tp_):
    CAP["ne"] = n_e.detach().clone(); CAP["cv"] = cv.detach().clone(); CAP["grid"] = grid
    return _orig_clo(n_e, cv, grid, params_, tp_)


PB.closure_from_fields = clo_cap


def dft_dir(sid):
    return f"{DFT}/1-44_GCE/cal_{sid}" if sid <= 200 else f"{DFT}/5-44_neutral_withsolv/cal_{sid - 600}"


def read_grid_fast(path):
    with open(path) as f:
        f.readline(); s = float(f.readline())
        lat = np.array([[float(x) for x in f.readline().split()] for _ in range(3)]) * s
        line = f.readline().split()
        try:
            counts = [int(x) for x in line]
        except ValueError:
            counts = [int(x) for x in f.readline().split()]
        nat = sum(counts); f.readline()
        for _ in range(nat):
            f.readline()
        f.readline()
        nx, ny, nz = [int(x) for x in f.readline().split()]
        need = nx * ny * nz
        rest = f.read()
    arr = np.fromstring(rest, sep=" ", dtype=np.float64)[:need]; del rest
    assert arr.size == need, (path, arr.size, need)
    return lat, torch.as_tensor(np.ascontiguousarray(arr.reshape(nz, ny, nx).transpose(2, 1, 0)), device=dev)


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
    return hold["res"], CAP["ne"], CAP["cv"]


def resample_tri(F3, shape):
    nx, ny, nz = shape
    ii, jj, kk = torch.meshgrid(torch.arange(nx, device=dev), torch.arange(ny, device=dev), torch.arange(nz, device=dev), indexing="ij")
    out = torch.empty(shape, device=dev)
    step = max(1, int(2_000_000 // (ny * nz)))
    for x0 in range(0, nx, step):
        x1 = min(x0 + step, nx)
        fm = torch.stack([ii[x0:x1].reshape(-1) / nx, jj[x0:x1].reshape(-1) / ny, kk[x0:x1].reshape(-1) / nz], dim=1).double()
        out[x0:x1] = _interp3_periodic(F3, fm).reshape(x1 - x0, ny, nz)
    return out


def resample_fourier(F3, shape):
    G = torch.fft.fftn(F3)
    for ax, (n, N) in enumerate(zip(F3.shape, shape)):
        m = min(n, N); half = (m - 1) // 2
        ks = torch.arange(-half, half + 1, device=dev)
        Gn = torch.zeros([N if i == ax else s for i, s in enumerate(G.shape)], dtype=G.dtype, device=dev)
        Gn.index_copy_(ax, ks % N, G.index_select(ax, ks % n)); G = Gn
    return torch.fft.ifftn(G).real * (float(np.prod(shape)) / float(np.prod(F3.shape)))


def interp_z(prof, z_src, lz, nz_dst):
    zd = np.arange(nz_dst) * lz / nz_dst
    zs = np.asarray(z_src) % lz; o = np.argsort(zs); zs, pr = zs[o], np.asarray(prof)[o]
    return np.interp(zd, np.concatenate([zs, [zs[0] + lz]]), np.concatenate([pr, [pr[0]]]))


def line_removed(d, lz):
    z = np.arange(d.size) * lz / d.size; A = np.stack([z, np.ones_like(z)], 1)
    c, *_ = np.linalg.lstsq(A, d, rcond=None); r = d - A @ c
    return float(np.sqrt((r ** 2).mean())), float(c[0])


def poisson_np(rho_zyx, lat):
    nz, ny, nx = rho_zyx.shape
    rg = np.fft.rfftn(rho_zyx); Bm = 2.0 * np.pi * np.linalg.inv(lat).T
    hx = np.fft.rfftfreq(nx) * nx; hy = np.fft.fftfreq(ny) * ny; hz = np.fft.fftfreq(nz) * nz
    G = (hz[:, None, None, None] * Bm[2] + hy[None, :, None, None] * Bm[1] + hx[None, None, :, None] * Bm[0])
    G2 = (G * G).sum(-1); G2[0, 0, 0] = 1.0
    pg = 4.0 * np.pi * K_EVA * rg / G2; pg[0, 0, 0] = 0.0
    return np.fft.irfftn(pg, s=rho_zyx.shape)


def metrics(M, D, lat, dV, pD=None):
    d = M - D; pm, pd = M.mean(axis=(1, 2)), D.mean(axis=(1, 2)); Ml, Dl = M - pm[:, None, None], D - pd[:, None, None]
    pM = poisson_np(M, lat); pD = poisson_np(D, lat) if pD is None else pD
    return dict(L1=float(np.abs(d).sum() / np.abs(D).sum()), L2=float((d ** 2).sum() / (D ** 2).sum()),
                eps_lat=float(np.abs(Ml - Dl).sum() / np.abs(Dl).sum()), lat_sq=float(((Ml - Dl) ** 2).sum() / (Dl ** 2).sum()),
                lat_corr=float((Ml * Dl).sum() / np.sqrt((Ml * Ml).sum() * (Dl * Dl).sum())),
                eps_pa=float(np.abs(pm - pd).sum() / np.abs(pd).sum()), pa_corr=float(((pm - pm.mean()) * (pd - pd.mean())).sum() / np.sqrt(((pm - pm.mean()) ** 2).sum() * ((pd - pd.mean()) ** 2).sum())),
                net_err_e=float(d.sum()) * dV, net_ml_e=float(M.sum()) * dV, net_dft_e=float(D.sum()) * dV,
                phi_rms_err=float(np.sqrt(((pM - pD) ** 2).mean())), phi_rms_dft=float(np.sqrt((pD ** 2).mean())),
                norm_ratio=float(np.sqrt((M * M).sum() / (D * D).sum())), lat_norm_ratio=float(np.sqrt((Ml * Ml).sum() / (Dl * Dl).sum()))), pD


def quick(Mt, Dt):
    """torch metrics on [nx,ny,nz] tensors: L1, lateral L1, plane-average L1, norm ratio, lateral corr."""
    d = Mt - Dt; pm = Mt.mean((0, 1)); pd = Dt.mean((0, 1)); Ml = Mt - pm[None, None, :]; Dl = Dt - pd[None, None, :]
    return dict(L1=float(d.abs().sum() / Dt.abs().sum()), eps_lat=float((Ml - Dl).abs().sum() / Dl.abs().sum()),
                eps_pa=float((pm - pd).abs().sum() / pd.abs().sum()), norm_ratio=float(Mt.norm() / Dt.norm()),
                lat_corr=float((Ml * Dl).sum() / (Ml.norm() * Dl.norm())))


def make_F(gd, phi_sol, rho_i, s_d, w_b, V):
    def F(rho_b):
        phi = phi_sol + gd.ifft_real(gd.l0_inv_op(gd.fft(-V * (rho_b + rho_i))))
        ex, ey, ez, em = gd.grad_from_recip(-torch.conj(w_b) * gd.fft(phi)); del phi
        a = response_a3(em, s_d, params, tp); del em
        return gd.ifft_real(w_b * gd.div_real_vector(a * ex, a * ey, a * ez))
    return F


def anderson(F, x0, Dt, m=12, beta=0.05, maxit=60, tol=1e-3, label="", P=None):
    """Anderson-accelerated fixed-point iteration x = F(x); with P the mixing / history use the preconditioned residual P (F(x) - x)
    while the stopping residual stays ||F(x) - x|| / ||F(x)||. Returns (last iterate g, history, status)."""
    x = x0.clone(); hist = []; status = "maxit"; g = None
    DX = []; DF = []; Gm = torch.zeros((0, 0), device=dev); x_prev = f_prev = None     # difference history + incremental Gram matrix
    for k in range(maxit):
        t = time.time(); g = F(x)
        if dev.type == "cuda": torch.cuda.synchronize()
        tF = time.time() - t
        f = g - x; res = float(f.norm() / g.norm())
        if P is not None: f = P * f
        q = quick(g, Dt); q.update(k=k, res_rel=res, t_F=tF); hist.append(q)
        if k % 5 == 0 or res < tol or k == maxit - 1:
            print(f"      [{label}] k {k:2d} res {res:.2e} L1 {q['L1']:.3f} lat {q['eps_lat']:.3f} pa {q['eps_pa']:.3f} norm {q['norm_ratio']:.3f} corr {q['lat_corr']:+.3f} ({tF:.1f} s/F)", flush=True)
        if not np.isfinite(res) or res > 1e4:
            status = "diverged"; break
        if res < tol:
            status = "converged"; break
        if x_prev is not None:
            dx = (x - x_prev).reshape(-1); df = (f - f_prev).reshape(-1)
            row = torch.stack([torch.dot(df, d_) for d_ in DF] + [torch.dot(df, df)])
            Gn = torch.empty((len(DF) + 1, len(DF) + 1), device=dev); Gn[:-1, :-1] = Gm; Gn[-1, :] = row; Gn[:, -1] = row; Gm = Gn
            DX.append(dx); DF.append(df)
            if len(DX) > m:
                DX.pop(0); DF.pop(0); Gm = Gm[1:, 1:].clone()
        x_prev = x; f_prev = f
        if DX:
            fv = f.reshape(-1); bvec = torch.stack([torch.dot(d_, fv) for d_ in DF])
            A = Gm + 1e-12 * torch.trace(Gm) * torch.eye(Gm.shape[0], device=dev)
            gam = torch.linalg.solve(A, bvec)
            xn = (x + beta * f).reshape(-1).clone()
            for i in range(len(DX)):
                xn.add_(DX[i] + beta * DF[i], alpha=-float(gam[i]))
            x = xn.reshape(x.shape)
        else:
            x = x + beta * f
    return g, hist, status


zyx = lambda t: t.permute(2, 1, 0).contiguous().cpu().numpy()
RES = []
for kpair in PAIRS:
    rec = dict(pair=kpair, split=ATOMS[kpair].info["_split"], frames={}, response={}, settings=dict(maxit=MAXIT, m=M_HIST, beta=BETA, tol=TOL, runs=RUNS))
    keep = {}
    for sid in (kpair, kpair + 600):
        d = dft_dir(sid); t1 = time.time()
        lat, chg = read_grid_fast(f"{d}/CHGCAR"); _, phi3 = read_grid_fast(f"{d}/PHI"); _, rb_raw = read_grid_fast(f"{d}/RHOB"); _, ri_raw = read_grid_fast(f"{d}/RHOION")
        shape = tuple(chg.shape); V = float(abs(np.linalg.det(lat))); dV = V / chg.numel(); lz = float(lat[2, 2]); nz = shape[2]
        gd = tp.TorchGrid(lat, shape, device=str(dev), dtype=torch.float64, rspec=True)
        w_b = tp._normalized_gaussian_kernel_g(gd, SIGMA_B)
        res, ne_ml, cv_ml = forward(sid)
        fr = dict(check=dict(read_s=time.time() - t1, native=list(shape)), runs={}, picard={}, init={}, solute_potential={})
        keep[sid] = {}
        with torch.no_grad():
            ne_d = torch.clamp(chg / V, min=0.0); del chg
            rho_b_ref = -(rb_raw / V); rho_i_ref = -(ri_raw / V); del rb_raw, ri_raw
            Dz = zyx(rho_b_ref); pD = poisson_np(Dz, lat); keep[sid]["ref"] = Dz
            _, s_d, _ = tp.create_cavity_torch(ne_d, gd, params); s_d = torch.clamp(s_d, 0.0, 1.0)
            phi_sol = phi3 - gd.ifft_real(gd.l0_inv_op(gd.fft(-V * (rho_b_ref + rho_i_ref)))); del phi3
            # model inputs on the native grid
            _, s_m, _ = tp.create_cavity_torch(resample_tri(ne_ml, shape), gd, params); s_m = torch.clamp(s_m, 0.0, 1.0)
            cv_m = resample_fourier(cv_ml, shape)
            cv_m_pm = cv_m - cv_m.mean((0, 1))[None, None, :] + phi_sol.mean((0, 1))[None, None, :]
            dcv = cv_m - phi_sol; dpm = dcv.mean((0, 1)).cpu().numpy(); dl = dcv - dcv.mean((0, 1))[None, None, :]
            lr, slope = line_removed(dpm, lz)
            fr["solute_potential"] = dict(model_minus_dft_lateral_rms=float(dl.pow(2).mean().sqrt()), dft_lateral_rms=float((phi_sol - phi_sol.mean((0, 1))[None, None, :]).pow(2).mean().sqrt()),
                                          model_minus_dft_pa_rms_lineremoved=lr, model_minus_dft_pa_slope_eV_per_A=slope); del dcv, dl
            fr["check"]["cavity_model_vs_dft_L1"] = float((s_m - s_d).abs().sum() / s_d.sum())
            # initial guess: production 1-D bound charge (600 planes) -> native planes, laterally uniform
            b600 = res["rho_bound_z"].detach().double().cpu().numpy()
            b_nat = torch.as_tensor(interp_z(b600, np.arange(b600.size) * lz / b600.size, lz, nz), device=dev)
            x_1d = b_nat[None, None, :].expand(shape).contiguous()
            fr["init"]["B1D"] = quick(x_1d, rho_b_ref); fr["init"]["B1D"]["net_e"] = float(x_1d.sum() * dV); fr["init"]["zero"] = quick(torch.zeros_like(x_1d), rho_b_ref)
            print(f"[{time.time() - T0:5.0f}s] sid {sid} (q {float(ATOMS[sid].info['total_charge']):+.3f}) read {fr['check']['read_s']:.0f} s; 1-D init vs RHOB: L1 {fr['init']['B1D']['L1']:.3f} pa {fr['init']['B1D']['eps_pa']:.3f} norm {fr['init']['B1D']['norm_ratio']:.3f}; "
                  f"model cv - DFT solute: lateral rms {fr['solute_potential']['model_minus_dft_lateral_rms']:.4f} eV (DFT lateral {fr['solute_potential']['dft_lateral_rms']:.3f}), pa line-removed {lr:.4f}, slope {slope:+.5f}; cavity L1 diff {fr['check']['cavity_model_vs_dft_L1']:.4f}", flush=True)
            F_dft = make_F(gd, phi_sol, rho_i_ref, s_d, w_b, V)
            if PICARD:
                p = x_1d
                for j in (1, 2, 3):
                    t = time.time(); p = F_dft(p); q = quick(p, rho_b_ref); q["t_F"] = time.time() - t; fr["picard"][f"F^{j}(B1D)"] = q
                print(f"         Picard from B1D (DFT inputs): " + " | ".join(f"{kk}: L1 {v['L1']:.3f} lat {v['eps_lat']:.3f} pa {v['eps_pa']:.3f} norm {v['norm_ratio']:.3f}" for kk, v in fr["picard"].items()) + f"  ({fr['picard']['F^1(B1D)']['t_F']:.1f} s/F)", flush=True)
                del p
            specs = {"S0": (F_dft, torch.zeros_like(x_1d)), "S1": (F_dft, x_1d),
                     "S2": (make_F(gd, phi_sol, rho_i_ref, s_m, w_b, V), x_1d),
                     "S3": (make_F(gd, cv_m, rho_i_ref, s_m, w_b, V), x_1d),
                     "S3b": (make_F(gd, cv_m_pm, rho_i_ref, s_m, w_b, V), x_1d)}
            cav_of = {"S0": s_d, "S1": s_d, "S2": s_m, "S3": s_m, "S3b": s_m}
            for name in RUNS:
                if name not in specs:
                    continue
                Fn, x0 = specs[name]; t = time.time()
                P = None if PRECOND == "none" else 1.0 / (1.0 + EDEPS * RESP0 * cav_of[name])
                if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize()
                g, hist, status = anderson(Fn, x0, rho_b_ref, m=M_HIST, beta=BETA, maxit=MAXIT, tol=TOL, label=f"{sid} {name}", P=P)
                if dev.type == "cuda": torch.cuda.synchronize()
                peak_gib = float(torch.cuda.max_memory_allocated(dev)) / 2 ** 30 if dev.type == "cuda" else None
                del P
                Mz = zyx(g); full, _ = metrics(Mz, Dz, lat, dV, pD)
                fr["runs"][name] = dict(status=status, iterations=len(hist), wall_s=time.time() - t, t_F_mean=float(np.mean([h["t_F"] for h in hist])), final=full, history=hist,
                                        precond=PRECOND, m=M_HIST, beta=BETA, device=str(dev), peak_gpu_gib=peak_gib)
                keep[sid][name] = Mz
                np.savez_compressed(f"{SLICES}/sid{sid}_{name}.npz", pa_cand=Mz.mean(axis=(1, 2)), pa_ref=Dz.mean(axis=(1, 2)), xz_cand=Mz[:, shape[1] // 2, :], xz_ref=Dz[:, shape[1] // 2, :], lz=lz)
                print(f"[{time.time() - T0:5.0f}s] sid {sid} {name}: {status} after {len(hist)} it ({time.time() - t:.0f} s; {'%.2f GiB peak' % peak_gib if peak_gib else 'cpu'}; precond {PRECOND}) | final L1 {full['L1']:.3f} lat {full['eps_lat']:.3f} ({full['lat_corr']:+.3f}) pa {full['eps_pa']:.3f} norm {full['norm_ratio']:.3f} phi {full['phi_rms_err']:.3f} eV", flush=True)
                del g
            del F_dft, specs, x_1d, phi_sol, cv_m, cv_m_pm, s_d, s_m, rho_b_ref, rho_i_ref, ne_d
        rec["frames"][str(sid)] = fr
    dV = abs(np.linalg.det(lat)) / keep[kpair]["ref"].size
    Dr = keep[kpair]["ref"] - keep[kpair + 600]["ref"]
    for nm in [n for n in keep[kpair] if n != "ref" and n in keep[kpair + 600]]:
        m, _ = metrics(keep[kpair][nm] - keep[kpair + 600][nm], Dr, lat, dV); rec["response"][nm] = m
    if rec["response"]:
        print(f"[{time.time() - T0:5.0f}s] pair {kpair} response vs DFT: " + " | ".join(f"{nm}: L1 {v['L1']:.3f} lat {v['eps_lat']:.3f} ({v['lat_corr']:+.2f}) pa {v['eps_pa']:.3f} norm {v['norm_ratio']:.2f} phi {v['phi_rms_err']:.3f}" for nm, v in rec["response"].items()), flush=True)
    RES.append(rec); json.dump(RES, open(OUT, "w"), indent=1); del keep
print(f"[{time.time() - T0:5.0f}s] done", flush=True)
