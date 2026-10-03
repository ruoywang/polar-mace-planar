"""Full self-consistent solvent closure on the DFT native grid (reviewer 2026-10-02, step 2 item 4; offline, no training):
bound AND ion charge from the same total potential, with the production's ion model, neutrality through the potential
reference, and the production's dipole correction inside the fixed point. No DFT ion charge and no DFT mean potential enter
the model-input run.
    phi(r)   = cv(r) + L(rho_b + rho_i)(r) + saw_z(c_unit d),   d = p_sol + mean((rho_b + rho_i) V z) - mean((rho_b + rho_i) V) center_z
    c        : constant with  int rho_i[phi + c] dV = -Q_total   (neutrality; the G = 0 mode the periodic Poisson operator drops)
    rho_i(r) = -ion_density_values(phi + c, s_ion, params, V) / V        (the 1-D solver's ion model and sign, pointwise in 3-D)
    rho_b(r) = w_b div( a(|E|, s_diel) E ),   E = -(w_b grad phi)
L = the production Poisson operator (ifft l0_inv fft(-V rho)); saw_z = cdipol_potential_1d (the 1-D solver's sawtooth) with
c_unit, center_z, indmin exactly as pb1d_backend builds them (here for the native nz).
Runs:  FD  DFT inputs through the production assembly: cv = phi_base - L(net_label V), cavities from CHGCAR, p_sol from the net
           label  (tests the boundary / dipole / ion treatment itself against RHOB, RHOION and PHI)
       FM  model inputs: production cv (Fourier-upsampled), cavities from the production electron density (trilinear), p_sol =
           the solver's val_ion_dipole_z
Both start from the production 1-D solutions B(z), I(z) laterally uniform. Anderson on the stacked (rho_b, rho_i) with the
dielectric Jacobi preconditioner on the bound block (KIT_PRECOND=eps) or none.
Metrics (native grid): rho_b vs RHOB, rho_i vs RHOION, sum vs RHOB + RHOION; <phi> vs <PHI> (mean-removed rms, slope);
charged-minus-neutral response per pair; peak GPU memory, s per evaluation, iterations.
Usage: python fullclosure_native.py <out.json>; env KIT_PAIRS, KIT_RUNS (FD+FM), KIT_MAXIT, KIT_M, KIT_BETA, KIT_TOL, KIT_PRECOND, KIT_DEVICE, KIT_DFT, KIT_MODEL
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
from mace.modules.pb1d_solver import cdipol_potential_1d, ion_density_derivative, ion_density_values
from mace.modules.solvent3d import _interp3_periodic

OUT = sys.argv[1] if len(sys.argv) > 1 else "fullclosure_native.json"
MODEL = os.environ.get("KIT_MODEL", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_vsolv_fix/models/prod500_vsolv_fix.model")
PAIRS = [int(s) for s in os.environ.get("KIT_PAIRS", "1 52 152").replace("+", " ").split()]
RUNS = os.environ.get("KIT_RUNS", "FD+FM").replace("+", " ").split()
MAXIT = int(os.environ.get("KIT_MAXIT", "70")); M_HIST = int(os.environ.get("KIT_M", "20")); BETA = float(os.environ.get("KIT_BETA", "0.5"))
TOL = float(os.environ.get("KIT_TOL", "1e-3")); PRECOND = os.environ.get("KIT_PRECOND", "eps")
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
man_n = json.load(open("data/density3d_net_grid_manifest_npy.json")); ENT_N = {int(k): v for k, v in man_n["entries"].items()}
ATOMS = {}
for sp in ("train", "val", "test"):
    for a in read(f"data/{sp}.xyz", ":"):
        a.info["_split"] = sp; ATOMS[int(a.info["sample_id"])] = a
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin", "sample_id": "sample_id",
                                    "solvated": "solvated", "fermi_level": "Fermi", "potential": "potential_diff"}, arrays_keys={"forces": "forces"})
T0 = time.time()
SLICES = os.environ.get("KIT_SLICES", "fc_slices"); os.makedirs(SLICES, exist_ok=True)
print(f"model {os.path.basename(MODEL)}; device {dev} {torch.cuda.get_device_name(dev) if dev.type == 'cuda' else ''}; threads {torch.get_num_threads()}; pairs {PAIRS}; runs {RUNS}; maxit {MAXIT} m {M_HIST} beta {BETA} tol {TOL}; precond {PRECOND}; sigma_b {SIGMA_B}", flush=True)
print("ion params: " + ", ".join(f"{k}={params[k]}" for k in ("LION", "LNLION", "ZBETA", "theta_b", "n_max", "invBETA", "LVAC", "SOL_Z0", "SOL_Z1") if k in params), flush=True)
RESP0 = float(response_a3(torch.zeros(1, device=dev), torch.ones(1, device=dev), params, tp)[0])
CAP = {}
_orig_clo = PB.closure_from_fields; _orig_solve = PB.Solver1D.solve


def clo_cap(n_e, cv, grid, params_, tp_):
    CAP["ne"] = n_e.detach().clone(); CAP["cv"] = cv.detach().clone(); CAP["grid"] = grid
    return _orig_clo(n_e, cv, grid, params_, tp_)


def solve_cap(self, **kw):
    CAP["kw"] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kw.items()}
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
    return hold["res"], CAP["ne"], CAP["cv"], CAP["kw"]


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


def line_fit(d, z):
    A = np.stack([z, np.ones_like(z)], 1); c, *_ = np.linalg.lstsq(A, d, rcond=None); r = d - A @ c
    return float(c[0]), float(np.sqrt((r ** 2).mean()))


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
    d = Mt - Dt; pm = Mt.mean((0, 1)); pd = Dt.mean((0, 1)); Ml = Mt - pm[None, None, :]; Dl = Dt - pd[None, None, :]
    return dict(L1=float(d.abs().sum() / Dt.abs().sum()), eps_lat=float((Ml - Dl).abs().sum() / Dl.abs().sum()),
                eps_pa=float((pm - pd).abs().sum() / pd.abs().sum()), norm_ratio=float(Mt.norm() / Dt.norm()), lat_corr=float((Ml * Dl).sum() / (Ml.norm() * Dl.norm())))


class FullClosure:
    """G(rho_b, rho_i) -> (rho_b', rho_i', aux) with neutrality shift and dipole sawtooth."""

    def __init__(self, gd, cv, s_diel, s_ion, p_sol, q_total, V, lz, c_unit, center_z, indmin, w_b):
        self.gd, self.cv, self.s_d, self.s_i, self.p_sol, self.Q, self.V, self.lz = gd, cv, s_diel, s_ion, p_sol, q_total, V, lz
        self.c_unit, self.center_z, self.indmin, self.w_b = c_unit, center_z, indmin, w_b
        self.nz = int(cv.shape[2]); self.z = torch.arange(self.nz, device=dev, dtype=torch.float64) * (lz / self.nz)
        self.dV = V / float(cv.numel()); self.c_last = 0.0

    def rho_ion(self, phi_c):
        return -ion_density_values(phi_c, self.s_i, params, self.V) / self.V

    def shift(self, phi):
        """c with int rho_i[phi + c] dV = -Q (Newton with the analytic derivative, bracket-safe)."""
        c = torch.tensor(self.c_last, device=dev); target = -self.Q
        for _ in range(30):
            g = float(self.rho_ion(phi + c).sum() * self.dV) - target
            dg = float((-ion_density_derivative(phi + c, self.s_i, params, self.V) / self.V).sum() * self.dV)
            if abs(g) < 1e-9 * max(1.0, abs(target)) or dg == 0.0:
                break
            step = -g / dg
            c = c + torch.clamp(torch.tensor(step, device=dev), -2.0, 2.0)
        self.c_last = float(c)
        return c

    def potential(self, rho_b, rho_i):
        rho = rho_b + rho_i
        d = self.p_sol + float((rho * self.V * self.z[None, None, :]).mean() - (rho * self.V).mean() * self.center_z)
        saw = cdipol_potential_1d(self.nz, self.lz, torch.tensor(self.c_unit * max(-20.0, min(20.0, d)), device=dev), self.indmin, dev)
        phi = self.cv + self.gd.ifft_real(self.gd.l0_inv_op(self.gd.fft(-self.V * rho))) + saw[None, None, :]
        return phi, d

    def __call__(self, rho_b, rho_i):
        phi, d = self.potential(rho_b, rho_i)
        c = self.shift(phi)
        rho_i_new = self.rho_ion(phi + c)
        ex, ey, ez, em = self.gd.grad_from_recip(-torch.conj(self.w_b) * self.gd.fft(phi))
        a = response_a3(em, self.s_d, params, tp); del em
        rho_b_new = self.gd.ifft_real(self.w_b * self.gd.div_real_vector(a * ex, a * ey, a * ez))
        return rho_b_new, rho_i_new, dict(dipole=d, shift=float(c), q_ion=float(rho_i_new.sum() * self.dV))


def anderson2(G, xb0, xi0, Db, Di, m, beta, maxit, tol, Pb=None, label=""):
    nb = xb0.numel()
    x = torch.cat([xb0.reshape(-1), xi0.reshape(-1)]); shape = xb0.shape
    hist = []; status = "maxit"; gb = gi = None; aux = {}
    DX = []; DF = []; Gm = torch.zeros((0, 0), device=dev); x_prev = f_prev = None     # difference history + incremental Gram matrix
    for k in range(maxit):
        t = time.time(); gb, gi, aux = G(x[:nb].reshape(shape), x[nb:].reshape(shape))
        if dev.type == "cuda": torch.cuda.synchronize()
        tG = time.time() - t
        g = torch.cat([gb.reshape(-1), gi.reshape(-1)]); f = g - x; res = float(f.norm() / g.norm())
        qb = quick(gb, Db); qi = quick(gi, Di)
        hist.append(dict(k=k, res_rel=res, t_G=tG, b_L1=qb["L1"], b_lat=qb["eps_lat"], b_pa=qb["eps_pa"], b_norm=qb["norm_ratio"], i_L1=qi["L1"], i_pa=qi["eps_pa"], i_norm=qi["norm_ratio"], **aux))
        if k % 5 == 0 or res < tol or k == maxit - 1:
            print(f"      [{label}] k {k:2d} res {res:.2e} | bound L1 {qb['L1']:.3f} lat {qb['eps_lat']:.3f} pa {qb['eps_pa']:.3f} norm {qb['norm_ratio']:.3f} | ion L1 {qi['L1']:.3f} pa {qi['eps_pa']:.3f} norm {qi['norm_ratio']:.3f} | dip {aux['dipole']:+.3f} shift {aux['shift']:+.4f} q_ion {aux['q_ion']:+.4f} ({tG:.1f} s/G)", flush=True)
        if not np.isfinite(res) or res > 1e4:
            status = "diverged"; break
        if res < tol:
            status = "converged"; break
        if Pb is not None:
            f = torch.cat([(Pb.reshape(-1) * f[:nb]), f[nb:]])
        if x_prev is not None:
            dx = x - x_prev; df = f - f_prev
            row = torch.stack([torch.dot(df, d_) for d_ in DF] + [torch.dot(df, df)])
            Gn = torch.empty((len(DF) + 1, len(DF) + 1), device=dev); Gn[:-1, :-1] = Gm; Gn[-1, :] = row; Gn[:, -1] = row; Gm = Gn
            DX.append(dx); DF.append(df)
            if len(DX) > m:
                DX.pop(0); DF.pop(0); Gm = Gm[1:, 1:].clone()
        x_prev = x; f_prev = f
        if DX:
            bvec = torch.stack([torch.dot(d_, f) for d_ in DF])
            A = Gm + 1e-12 * torch.trace(Gm) * torch.eye(Gm.shape[0], device=dev)
            gam = torch.linalg.solve(A, bvec)
            xn = (x + beta * f).clone()
            for i in range(len(DX)):
                xn.add_(DX[i] + beta * DF[i], alpha=-float(gam[i]))
            x = xn
        else:
            x = x + beta * f
    return gb, gi, hist, status, aux


zyx = lambda t: t.permute(2, 1, 0).contiguous().cpu().numpy()
RES = []
for kpair in PAIRS:
    rec = dict(pair=kpair, split=ATOMS[kpair].info["_split"], frames={}, response={}, settings=dict(maxit=MAXIT, m=M_HIST, beta=BETA, tol=TOL, precond=PRECOND, runs=RUNS, device=str(dev)))
    keep = {}
    for sid in (kpair, kpair + 600):
        atoms = ATOMS[sid]; q_total = float(atoms.info["total_charge"])
        d = dft_dir(sid); t1 = time.time()
        lat, chg = read_grid_fast(f"{d}/CHGCAR"); _, phi3 = read_grid_fast(f"{d}/PHI"); _, rb_raw = read_grid_fast(f"{d}/RHOB"); _, ri_raw = read_grid_fast(f"{d}/RHOION")
        shape = tuple(chg.shape); V = float(abs(np.linalg.det(lat))); dV = V / chg.numel(); lz = float(lat[2, 2]); nz = shape[2]
        gd = tp.TorchGrid(lat, shape, device=str(dev), dtype=torch.float64, rspec=True)
        w_b = tp._normalized_gaussian_kernel_g(gd, SIGMA_B)
        res, ne_ml, cv_ml, kw = forward(sid)
        bl_row = backend._bl_index.get(sid)
        fields_bl = torch.from_numpy(np.ascontiguousarray(backend._bl_arr[bl_row][backend._bl_take])).to(dev).double()
        neutral_v, phi_base = fields_bl[0], fields_bl[1]; del fields_bl
        cell_np = np.asarray(atoms.get_cell(), dtype=np.float64); c_unit = backend._c_unit(cell_np)
        center_z = 0.5 * float(cell_np[0, 2] + cell_np[1, 2] + cell_np[2, 2]); nouth = nz // 2; indmin = int((nouth + int(0.5 * nz) + 10 * nz) % nz + 1)
        fr = dict(check=dict(read_s=time.time() - t1, native=list(shape), q_total=q_total, c_unit=c_unit, center_z=center_z, indmin=indmin), runs={}, init={}, inputs={})
        keep[sid] = {}
        with torch.no_grad():
            ne_d = torch.clamp(chg / V, min=0.0); del chg
            rho_b_ref = -(rb_raw / V); rho_i_ref = -(ri_raw / V); del rb_raw, ri_raw
            Db = zyx(rho_b_ref); Di = zyx(rho_i_ref); pDb = poisson_np(Db, lat); pDi = poisson_np(Di, lat); keep[sid]["ref_b"] = Db; keep[sid]["ref_i"] = Di
            fr["check"]["q_ion_dft"] = float(rho_i_ref.sum() * dV); fr["check"]["q_bound_dft"] = float(rho_b_ref.sum() * dV)
            pm_PHI = phi3.mean((0, 1)).cpu().numpy(); del phi3
            zt = torch.arange(nz, device=dev, dtype=torch.float64) * (lz / nz)
            # DFT inputs through the production assembly
            net_nat = torch.as_tensor(np.asarray(np.load(ENT_N[sid]["path"], mmap_mode="r"), dtype=np.float64), device=dev).permute(2, 1, 0).contiguous()
            cv_D = resample_fourier(phi_base, shape) - gd.ifft_real(gd.l0_inv_op(gd.fft(net_nat * V)))
            p_sol_D = float((net_nat * V * zt[None, None, :]).mean() - (net_nat * V).mean() * center_z); del net_nat
            s_i_D, s_d_D, _ = tp.create_cavity_torch(ne_d, gd, params); s_d_D = torch.clamp(s_d_D, 0.0, 1.0); s_i_D = torch.clamp(s_i_D, 0.0, 1.0)
            # model inputs
            cv_M = resample_fourier(cv_ml, shape)
            s_i_M, s_d_M, _ = tp.create_cavity_torch(resample_tri(ne_ml, shape), gd, params); s_d_M = torch.clamp(s_d_M, 0.0, 1.0); s_i_M = torch.clamp(s_i_M, 0.0, 1.0)
            p_sol_M = float(kw["val_ion_dipole_z"])
            fr["inputs"] = dict(p_sol_dft_label=p_sol_D, p_sol_model_solver=p_sol_M, solver_c_unit=float(kw["c_unit"]), solver_center_z=float(kw["center_z"]),
                                cv_model_minus_dft_lateral_rms=float(((cv_M - cv_M.mean((0, 1))[None, None, :]) - (cv_D - cv_D.mean((0, 1))[None, None, :])).pow(2).mean().sqrt()),
                                s_diel_L1_model_vs_dft=float((s_d_M - s_d_D).abs().sum() / s_d_D.sum()), s_ion_L1_model_vs_dft=float((s_i_M - s_i_D).abs().sum() / s_i_D.sum()))
            # initial guess: production 1-D solutions
            b600 = res["rho_bound_z"].detach().double().cpu().numpy(); i600 = res["rho_ion_z"].detach().double().cpu().numpy()
            x_b = torch.as_tensor(interp_z(b600, np.arange(b600.size) * lz / b600.size, lz, nz), device=dev)[None, None, :].expand(shape).contiguous()
            x_i = torch.as_tensor(interp_z(i600, np.arange(i600.size) * lz / i600.size, lz, nz), device=dev)[None, None, :].expand(shape).contiguous()
            fr["init"] = dict(bound=quick(x_b, rho_b_ref), ion=quick(x_i, rho_i_ref), q_ion_1d=float(x_i.sum() * dV))
            print(f"[{time.time() - T0:5.0f}s] sid {sid} (q {q_total:+.3f}) read {fr['check']['read_s']:.0f} s | DFT q_ion {fr['check']['q_ion_dft']:+.4f} q_bound {fr['check']['q_bound_dft']:+.4f}; 1-D init q_ion {fr['init']['q_ion_1d']:+.4f}, bound L1 {fr['init']['bound']['L1']:.3f} pa {fr['init']['bound']['eps_pa']:.3f}, ion L1 {fr['init']['ion']['L1']:.3f} pa {fr['init']['ion']['eps_pa']:.3f} | "
                  f"p_sol DFT label {p_sol_D:+.3f} model(solver) {p_sol_M:+.3f} e*A; c_unit {c_unit:.5f} (solver {fr['inputs']['solver_c_unit']:.5f}); cv lateral diff {fr['inputs']['cv_model_minus_dft_lateral_rms']:.4f} eV; s_diel L1 diff {fr['inputs']['s_diel_L1_model_vs_dft']:.4f} s_ion {fr['inputs']['s_ion_L1_model_vs_dft']:.4f}", flush=True)
            specs = {"FD": (cv_D, s_d_D, s_i_D, p_sol_D), "FM": (cv_M, s_d_M, s_i_M, p_sol_M)}
            for name in RUNS:
                if name not in specs:
                    continue
                cv, s_d, s_i, p_sol = specs[name]
                G = FullClosure(gd, cv, s_d, s_i, p_sol, q_total, V, lz, c_unit, center_z, indmin, w_b)
                Pb = None if PRECOND == "none" else 1.0 / (1.0 + EDEPS * RESP0 * s_d)
                if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize()
                t = time.time()
                gb, gi, hist, status, aux = anderson2(G, x_b, x_i, rho_b_ref, rho_i_ref, M_HIST, BETA, MAXIT, TOL, Pb=Pb, label=f"{sid} {name}")
                if dev.type == "cuda": torch.cuda.synchronize()
                peak = float(torch.cuda.max_memory_allocated(dev)) / 2 ** 30 if dev.type == "cuda" else None
                phi_fin, d_fin = G.potential(gb, gi); c_fin = G.shift(phi_fin); pm_phi = (phi_fin + c_fin).mean((0, 1)).cpu().numpy(); del phi_fin
                dpm = pm_phi - pm_PHI; sl, lr = line_fit(dpm, zt.cpu().numpy())
                Mb = zyx(gb); Mi = zyx(gi); fb, _ = metrics(Mb, Db, lat, dV, pDb); fi, _ = metrics(Mi, Di, lat, dV, pDi); ft, _ = metrics(Mb + Mi, Db + Di, lat, dV)
                fr["runs"][name] = dict(status=status, iterations=len(hist), wall_s=time.time() - t, t_G_mean=float(np.mean([h["t_G"] for h in hist])), peak_gpu_gib=peak,
                                        final_bound=fb, final_ion=fi, final_total=ft, dipole=d_fin, shift=float(c_fin), q_ion=aux["q_ion"],
                                        phi_pa_vs_PHI=dict(rms_meanremoved=float(np.sqrt(((dpm - dpm.mean()) ** 2).mean())), slope=sl, rms_lineremoved=lr, PHI_pa_rms=float(np.sqrt(((pm_PHI - pm_PHI.mean()) ** 2).mean()))),
                                        history=hist)
                keep[sid][name + "_b"] = Mb; keep[sid][name + "_i"] = Mi
                np.savez_compressed(f"{SLICES}/sid{sid}_{name}.npz", pa_b=Mb.mean(axis=(1, 2)), pa_b_ref=Db.mean(axis=(1, 2)), pa_i=Mi.mean(axis=(1, 2)), pa_i_ref=Di.mean(axis=(1, 2)),
                                    xz_b=Mb[:, shape[1] // 2, :], xz_b_ref=Db[:, shape[1] // 2, :], xz_i=Mi[:, shape[1] // 2, :], xz_i_ref=Di[:, shape[1] // 2, :], pm_phi=pm_phi, pm_PHI=pm_PHI, lz=lz)
                print(f"[{time.time() - T0:5.0f}s] sid {sid} {name}: {status} after {len(hist)} it ({time.time() - t:.0f} s; {('%.2f GiB' % peak) if peak else 'cpu'}) | bound L1 {fb['L1']:.3f} lat {fb['eps_lat']:.3f} ({fb['lat_corr']:+.3f}) pa {fb['eps_pa']:.3f} norm {fb['norm_ratio']:.3f} phi {fb['phi_rms_err']:.3f} | "
                      f"ion L1 {fi['L1']:.3f} pa {fi['eps_pa']:.3f} norm {fi['norm_ratio']:.3f} q {aux['q_ion']:+.4f} | total L1 {ft['L1']:.3f} phi {ft['phi_rms_err']:.3f} | <phi> vs <PHI>: rms {fr['runs'][name]['phi_pa_vs_PHI']['rms_meanremoved']:.4f} eV slope {sl:+.5f} line-removed {lr:.4f} | dipole {d_fin:+.3f} shift {float(c_fin):+.4f}", flush=True)
                del G, Pb, gb, gi
            del cv_D, cv_M, s_d_D, s_d_M, s_i_D, s_i_M, x_b, x_i, rho_b_ref, rho_i_ref, ne_d
        rec["frames"][str(sid)] = fr
    dV = abs(np.linalg.det(lat)) / keep[kpair]["ref_b"].size
    for name in RUNS:
        if name + "_b" in keep[kpair] and name + "_b" in keep[kpair + 600]:
            rb, _ = metrics(keep[kpair][name + "_b"] - keep[kpair + 600][name + "_b"], keep[kpair]["ref_b"] - keep[kpair + 600]["ref_b"], lat, dV)
            ri, _ = metrics(keep[kpair][name + "_i"] - keep[kpair + 600][name + "_i"], keep[kpair]["ref_i"] - keep[kpair + 600]["ref_i"], lat, dV)
            rec["response"][name] = dict(bound=rb, ion=ri)
            print(f"[{time.time() - T0:5.0f}s] pair {kpair} {name} response vs DFT: bound L1 {rb['L1']:.3f} lat {rb['eps_lat']:.3f} ({rb['lat_corr']:+.2f}) pa {rb['eps_pa']:.3f} phi {rb['phi_rms_err']:.3f} | ion L1 {ri['L1']:.3f} pa {ri['eps_pa']:.3f} phi {ri['phi_rms_err']:.3f}", flush=True)
    RES.append(rec); json.dump(RES, open(OUT, "w"), indent=1); del keep
print(f"[{time.time() - T0:5.0f}s] done", flush=True)
