"""Polarization-divergence candidate for the lateral bound charge (reviewer plan 2026-10-02; no training).

With the model's OWN inputs and the production constitutive relation, smoothing and units (pb1d_solver.bound_matrix:
E = -(w_b * grad phi), P = a * E, rho_b = w_b * div P, sigma_b = R_B or A_K), an approximate 3-D polarization field is
built on the model grid:
    a3(r)      = response_a3(|E_scr|, s_diel3)            (the closure's saturated response at the screened vacuum field)
    phi_tot(r) = cvhar3(r) + [phi_1D(z) - <cvhar3>_xy(z)]  (3-D solute potential + laterally uniform solvent potential)
    E(r)       = -(w_b * grad phi_tot),  P = a3 E,  g = w_b * div P,  g_perp = g - <g>_xy
Consistency check printed per frame: <g>_xy against the solver's own rho_b(z) (correlation, L1) -- a wrong sign shows
up there. Candidate field on the DFT grid: rho(alpha) = B_ML(z) + (1 - alpha) delta_current + alpha g_perp; alpha = 0 is
the production prediction, alpha = 1 the pure polarization-divergence lateral distribution. One alpha shared across the
training frames (closed-form least squares, clipped to [0, 1]) is fixed and then evaluated on the validation pairs.
Reported: lateral L1 / squared error (single frames, charging response), bound / ionic / total potential rms, plane
average, net charge, rearrangement (cancellation, layer addition), per pair. Usage: python lateral_polarization.py <out.json>
Env: KIT_FIT_PAIRS (train, alpha fit), KIT_VAL_PAIRS, KIT_TEST_PAIRS, KIT_DEVICE.
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
from mace.modules.solvent3d import _interp1_periodic, _interp3_periodic, normalized_gradient_envelope

OUT = sys.argv[1] if len(sys.argv) > 1 else "lateral_polarization.json"
MODEL = os.environ.get("KIT_MODEL", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_vsolv_fix/models/prod500_vsolv_fix.model")
ints = lambda k, d: [int(s) for s in os.environ.get(k, d).replace("+", " ").split()]
FIT_PAIRS = ints("KIT_FIT_PAIRS", "1 52 152"); VAL_PAIRS = ints("KIT_VAL_PAIRS", "28 30 43 60 61 62 69 79 83 94 128 134 148 153 159 177 180 185 186 189")
TEST_PAIRS = ints("KIT_TEST_PAIRS", "122")
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
man = json.load(open("data/solvent3d_grid_manifest.json")); ENT = {int(k): v for k, v in man["entries"].items()}
ATOMS = {}
for sp in ("train", "val", "test"):
    for a in read(f"data/{sp}.xyz", ":"):
        a.info["_split"] = sp; ATOMS[int(a.info["sample_id"])] = a
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin", "sample_id": "sample_id",
                                    "solvated": "solvated", "fermi_level": "Fermi", "potential": "potential_diff"}, arrays_keys={"forces": "forces"})
T0 = time.time()
print(f"model {os.path.basename(MODEL)}; device {dev}; fit pairs {FIT_PAIRS}; val {len(VAL_PAIRS)} pairs; test {TEST_PAIRS}; "
      f"R_B {params['R_B']} A_K {params['A_K']} LNLDIEL {params['LNLDIEL']}", flush=True)

CAP = {}
_orig_clo = PB.closure_from_fields


def clo_cap(n_e, cv, grid, params_, tp_):
    CAP["ne"] = n_e.detach().clone(); CAP["cv"] = cv.detach().clone(); CAP["grid"] = grid
    return _orig_clo(n_e, cv, grid, params_, tp_)


PB.closure_from_fields = clo_cap


def label(sid):
    e = ENT[sid]
    rb = np.asarray(np.load("data/" + e["path_b"], mmap_mode="r"), dtype=np.float64)
    ri = np.asarray(np.load("data/" + e["path_i"], mmap_mode="r"), dtype=np.float64)
    with np.load("data/" + e["meta_path"]) as m:
        lat = np.asarray(m["lattice"], dtype=np.float64)
    return rb, ri, lat


def to_dft(F3, shape):
    nz, ny, nx = shape
    iy, ix = torch.meshgrid(torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing="ij")
    fxy = torch.stack([ix.reshape(-1).double() / nx, iy.reshape(-1).double() / ny], dim=1)
    M = np.empty(shape)
    with torch.no_grad():
        for z0 in range(0, nz, 25):
            z1 = min(z0 + 25, nz)
            fz = torch.arange(z0, z1, device=dev, dtype=torch.float64) / nz
            frac = torch.cat([fxy.repeat(z1 - z0, 1), fz.repeat_interleave(ny * nx)[:, None]], dim=1)
            M[z0:z1] = _interp3_periodic(F3, frac).reshape(z1 - z0, ny, nx).cpu().numpy()
    return M


def capture(sid, shape):
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
    res = hold["res"]; grid = CAP["grid"]; ne = CAP["ne"]; cv = CAP["cv"]
    nz = shape[0]
    with torch.no_grad():
        # ---- the candidate, on the model grid, with the production relation
        s_ion3, s_diel3, _ = tp.create_cavity_torch(ne, grid, params)
        sigma_b = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
        w_b = tp._normalized_gaussian_kernel_g(grid, sigma_b)
        zero1 = s_diel3.new_zeros(1)
        a3_zero = response_a3(zero1, s_diel3.new_ones(1), params, tp)[0] * s_diel3
        eps3 = 1.0 + EDEPS * a3_zero
        ex, ey, ez, emag = grid.grad_from_recip(-torch.conj(w_b) * grid.fft(cv))
        a3 = response_a3(emag / eps3, s_diel3, params, tp)
        # the closure's own screened vacuum field (3-D) carries the lateral structure; its plane mean along z is replaced
        # by the 1-D solver's total field E_z = -(w_b * d phi_1D / dz), so that <g>_xy = w_b d/dz (a1 E_z + prior), the
        # solver's own bound charge without delta_p (consistency check below)
        ex_s, ey_s, ez_s = ex / eps3, ey / eps3, ez / eps3
        phi_s = res["phi_z"].detach().double(); f = phi_s.numel() // cv.shape[2]
        phi_1d = phi_s[::f] if f > 1 else phi_s                           # solve grid -> model grid planes
        nzm = cv.shape[2]; lz_ = float(atoms.cell[2, 2])
        kz = 2.0 * math.pi * torch.fft.fftfreq(nzm, d=lz_ / nzm, device=dev, dtype=torch.float64)
        wb1 = torch.exp(-0.5 * (sigma_b * kz) ** 2)
        ez_1d = torch.fft.ifft(-1j * kz * wb1 * torch.fft.fft(phi_1d)).real
        Ez3 = ez_s - ez_s.mean(dim=(0, 1))[None, None, :] + ez_1d[None, None, :]
        g = grid.ifft_real(torch.conj(w_b) * grid.div_real_vector(a3 * ex_s, a3 * ey_s, a3 * Ez3))
        g_mean = g.mean(dim=(0, 1)); g_perp = g - g_mean[None, None, :]
        Bz = res["rho_bound_z"].detach().double(); Bi = res["rho_ion_z"].detach().double()
        B_model_planes = _interp1_periodic(Bz, torch.arange(cv.shape[2], device=dev, dtype=torch.float64) / cv.shape[2])
        chk = dict(corr_gmean_B=float((g_mean * B_model_planes).sum() / torch.sqrt((g_mean ** 2).sum() * (B_model_planes ** 2).sum())),
                   L1_gmean_B=float((g_mean - B_model_planes).abs().sum() / B_model_planes.abs().sum()),
                   gperp_rms=float(g_perp.pow(2).mean().sqrt()), dsup_rms=float(res["s3d_obs"]["d_sup_b"].pow(2).mean().sqrt()))
        env_b = normalized_gradient_envelope(s_diel3, torch.as_tensor(np.array(atoms.cell[:], dtype=np.float64), device=dev))
        out = dict(q=float(atoms.info["total_charge"]),
                   B=_interp1_periodic(Bz, torch.arange(nz, device=dev, dtype=torch.float64) / nz).cpu().numpy(),
                   Bi=_interp1_periodic(Bi, torch.arange(nz, device=dev, dtype=torch.float64) / nz).cpu().numpy(),
                   delta=to_dft(res["s3d_obs"]["d_sup_b"].detach().double(), shape), delta_i=to_dft(res["s3d_obs"]["d_sup_i"].detach().double(), shape),
                   g_perp=to_dft(g_perp, shape), env=to_dft(env_b, shape), chk=chk)
    return out


def poisson(rho, lat):
    nz, ny, nx = rho.shape
    rg = np.fft.rfftn(rho); Bm = 2.0 * np.pi * np.linalg.inv(lat).T
    hx = np.fft.rfftfreq(nx) * nx; hy = np.fft.fftfreq(ny) * ny; hz = np.fft.fftfreq(nz) * nz
    G = (hz[:, None, None, None] * Bm[2] + hy[None, :, None, None] * Bm[1] + hx[None, None, :, None] * Bm[0])
    G2 = (G * G).sum(-1); G2[0, 0, 0] = 1.0
    pg = 4.0 * np.pi * K_EVA * rg / G2; pg[0, 0, 0] = 0.0
    return np.fft.irfftn(pg, s=rho.shape)


def metrics(M, D, lat, dV, phi_D=None):
    d = M - D; pm, pd = M.mean(axis=(1, 2)), D.mean(axis=(1, 2)); Ml, Dl = M - pm[:, None, None], D - pd[:, None, None]
    pM = poisson(M, lat); pD = poisson(D, lat) if phi_D is None else phi_D
    return dict(L1=float(np.abs(d).sum() / np.abs(D).sum()), L2=float((d ** 2).sum() / (D ** 2).sum()),
                corr=float((M * D).sum() / np.sqrt((M * M).sum() * (D * D).sum())),
                eps_lat=float(np.abs(Ml - Dl).sum() / np.abs(Dl).sum()), lat_sq=float(((Ml - Dl) ** 2).sum() / (Dl ** 2).sum()),
                lat_corr=float((Ml * Dl).sum() / np.sqrt((Ml * Ml).sum() * (Dl * Dl).sum())),
                eps_pa=float(np.abs(pm - pd).sum() / np.abs(pd).sum()), net_err_e=float(d.sum()) * dV,
                phi_rms_err=float(np.sqrt(((pM - pD) ** 2).mean())), phi_rms_dft=float(np.sqrt((pD ** 2).mean()))), pM


def rearr(M, D, Bd, E, dV):
    pz = np.abs(D).mean(axis=(1, 2)); win = pz > 0.01 * pz.max(); W = win[:, None, None]
    have = M - Bd; nol = (np.abs(D) < 0.01 * np.abs(D).max()) & (Bd > 0.1 * Bd.max()) & W; lay = (D > 0.1 * D.max()) & W
    need = -Bd
    r = lambda m: float(have[m].sum() / need[m].sum()) if m.any() else None
    return dict(cancel=r(nol), cancel_env_lt_1e3=r(nol & (E < 1e-3)), cancel_env_ge_1e2=r(nol & (E >= 1e-2)),
                layer=float(have[lay].sum() / (D - Bd)[lay].sum()) if lay.any() else None,
                outside_ml=float(np.abs(M[~win]).sum() / np.abs(D).sum()), outside_dft=float(np.abs(D[~win]).sum() / np.abs(D).sum()))


FR = {}
def prep(sid):
    Db, Di, lat = label(sid); shape = Db.shape
    C = capture(sid, shape)
    FR[sid] = dict(C=C, Db=Db, Di=Di, lat=lat, dV=abs(np.linalg.det(lat)) / Db.size)
    r = C["chk"]
    print(f"[{time.time() - T0:5.0f}s] sid {sid} (q {C['q']:+.3f}): <g>_xy vs solver rho_b: corr {r['corr_gmean_B']:+.3f} L1 {r['L1_gmean_B']:.3f}; "
          f"g_perp rms {r['gperp_rms']:.2e} vs current residual rms {r['dsup_rms']:.2e}", flush=True)


def field(sid, alpha):
    F = FR[sid]; C = F["C"]
    return C["B"][:, None, None] + (1 - alpha) * C["delta"] + alpha * C["g_perp"]


# ---------------- alpha from the training frames (closed form, shared)
fit_sids = [s for k in FIT_PAIRS for s in (k, k + 600)]
for s in fit_sids:
    prep(s)
num = den = 0.0; per = {}
for s in fit_sids:
    F = FR[s]; C = F["C"]; r = F["Db"] - C["B"][:, None, None] - C["delta"]; sdir = C["g_perp"] - C["delta"]
    n_, d_ = float((sdir * r).sum()), float((sdir * sdir).sum()); num += n_; den += d_; per[s] = n_ / d_
ALPHA = min(1.0, max(0.0, num / den))
print(f"[{time.time() - T0:5.0f}s] shared alpha = {ALPHA:.4f} (unclipped {num / den:.4f}); per-frame unclipped: " + ", ".join(f"{s}: {v:+.3f}" for s, v in per.items()), flush=True)
ALPHAS = [("a0", 0.0), ("afit", ALPHA), ("a1", 1.0)]
RES = dict(alpha=ALPHA, alpha_unclipped=num / den, per_frame_alpha={str(k): v for k, v in per.items()}, pairs=[])


def eval_pair(k):
    for s in (k, k + 600):
        if s not in FR:
            prep(s)
    rec = dict(pair=k, split=ATOMS[k].info["_split"], frames={}, response={}, chk={str(s): FR[s]["C"]["chk"] for s in (k, k + 600)})
    fields = {}
    for s in (k, k + 600):
        F = FR[s]; C = F["C"]; D, Di, lat, dV = F["Db"], F["Di"], F["lat"], F["dV"]
        ION = C["Bi"][:, None, None] + C["delta_i"]
        phiD_b = poisson(D, lat); phiD_t = poisson(D + Di, lat)
        fr = {}
        for nm, a in ALPHAS:
            M = field(s, a); mb, pMb = metrics(M, D, lat, dV, phiD_b); mt, _ = metrics(M + ION, D + Di, lat, dV, phiD_t)
            fr[nm] = dict(b=mb, t=dict(phi_rms_err=mt["phi_rms_err"], L1=mt["L1"]), rearr=rearr(M, D, np.broadcast_to(C["B"][:, None, None], M.shape), C["env"], dV))
            fields[(s, nm)] = M
        rec["frames"][str(s)] = fr
    for nm, a in ALPHAS:
        Dr = FR[k]["Db"] - FR[k + 600]["Db"]; Mr = fields[(k, nm)] - fields[(k + 600, nm)]
        Dri = Dr + (FR[k]["Di"] - FR[k + 600]["Di"]); Mri = Mr + (FR[k]["C"]["Bi"][:, None, None] + FR[k]["C"]["delta_i"]) - (FR[k + 600]["C"]["Bi"][:, None, None] + FR[k + 600]["C"]["delta_i"])
        mb, _ = metrics(Mr, Dr, FR[k]["lat"], FR[k]["dV"]); mt, _ = metrics(Mri, Dri, FR[k]["lat"], FR[k]["dV"])
        rec["response"][nm] = dict(b=mb, t=dict(phi_rms_err=mt["phi_rms_err"], L1=mt["L1"]))
    f0, fa, f1 = (rec["frames"][str(k)][n]["b"] for n in ("a0", "afit", "a1")); r0, ra, r1 = (rec["response"][n]["b"] for n in ("a0", "afit", "a1"))
    print(f"[{time.time() - T0:5.0f}s] pair {k} ({rec['split']}): charged lat L1 {f0['eps_lat']:.3f} -> afit {fa['eps_lat']:.3f} / a1 {f1['eps_lat']:.3f} (lat sq {f0['lat_sq']:.3f} -> {fa['lat_sq']:.3f} / {f1['lat_sq']:.3f}; "
          f"phi_b {f0['phi_rms_err']:.4f} -> {fa['phi_rms_err']:.4f} / {f1['phi_rms_err']:.4f}) | response lat L1 {r0['eps_lat']:.3f} -> {ra['eps_lat']:.3f} / {r1['eps_lat']:.3f} (lat sq {r0['lat_sq']:.3f} -> {ra['lat_sq']:.3f} / {r1['lat_sq']:.3f}; "
          f"phi_t {rec['response']['a0']['t']['phi_rms_err']:.4f} -> {rec['response']['afit']['t']['phi_rms_err']:.4f} / {rec['response']['a1']['t']['phi_rms_err']:.4f})", flush=True)
    for s in (k, k + 600):
        FR.pop(s, None)
    return rec


for k in FIT_PAIRS + VAL_PAIRS + TEST_PAIRS:
    RES["pairs"].append(eval_pair(k)); json.dump(RES, open(OUT, "w"), indent=1)
print(f"[{time.time() - T0:5.0f}s] done", flush=True)
