"""Grid attribution for the constitutive relation rho_b = w_b div(a(|E|) E)  (reviewer 2026-10-02, step 1; no training).
Same six frames as abcd_native.py; reference = native RHOB (500 x 168 x 168, 0.09 A) read from the raw files.

NATIVE stage (true field -(w_b grad PHI) on the native grid; only the cavity density changes):
    A    native CHGCAR density (abcd_native A, recomputed as the in-run reference)
    R1   native density -> model grid by trilinear point sampling (the production resampling operator) -> back to the
         native grid (trilinear, as D used for the model density) -> cavity            [cavity-path resampling alone]
    R2   package route: n_e = clamp((neutral_v - net_label V)/V) on the model grid (density3d_net_grid label trilinearly
         sampled, neutral baseline from the baseline cache) -> native -> cavity           [the density source of the
         model-grid A in constitutive_abcd.py; production uses the same assembly with its own net density]
MODEL-GRID stage (production grid, 100 x 100 x 300). Metrics 'mg' = against RHOB resampled to the model grid with the same
operator; 'nat' = candidate trilinearly upsampled to the native grid, against native RHOB:
    M1   PHI and CHGCAR trilinearly sampled to the model grid; relation evaluated there      [coarse-grid computation alone]
    M1f  as M1, but PHI / CHGCAR / RHOB resampled by Fourier truncation (low-pass) instead of point sampling
    M2   cavity of M1; field from the production-assembly rebuilt potential cv_dft + Poisson(RHOB_m + RHOION_m),
         cv_dft = phi_base - Poisson(net_label_m V)                                       [field reconstruction route]
    M3   M2 with the plane mean of E_z replaced by the 1-D potential label's (the field of the model-grid A)
    M4   field of M1, cavity of the package route (R2 density on the model grid)          [cavity route on the model grid]
    M5   M3 field + package cavity = the model-grid A of constitutive_abcd.py (sanity: L1 1.16-1.53 expected)
Resolution series for the direct route (trilinear mapping): 50x50x150, 100x100x300 (= M1), 134x134x400, native (= A).
Charged-minus-neutral response per pair for every candidate, on the native grid.
Usage: python grid_attribution.py <out.json>   (cwd with ./data, ./cal1_train.json); env KIT_PAIRS, KIT_DEVICE, KIT_DFT, KIT_MODEL
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
from mace.modules.pb1d_closure import response_a3
from mace.modules.solvent3d import _interp3_periodic

OUT = sys.argv[1] if len(sys.argv) > 1 else "grid_attribution.json"
MODEL = os.environ.get("KIT_MODEL", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_vsolv_fix/models/prod500_vsolv_fix.model")
PAIRS = [int(s) for s in os.environ.get("KIT_PAIRS", "1 52 152").replace("+", " ").split()]
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
POT = np.load("data/potential1d_potcar_cache.npz"); POT_IDX = {int(s): i for i, s in enumerate(POT["sample_ids"])}
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


def clo_cap(n_e, cv, grid, params_, tp_):
    CAP["ne"] = n_e.detach().clone(); CAP["grid"] = grid
    return _orig_clo(n_e, cv, grid, params_, tp_)


PB.closure_from_fields = clo_cap


def dft_dir(sid):
    return f"{DFT}/1-44_GCE/cal_{sid}" if sid <= 200 else f"{DFT}/5-44_neutral_withsolv/cal_{sid - 600}"


def read_grid_fast(path):
    """VASP CHGCAR-style grid file -> (lattice, tensor [nx, ny, nz])."""
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
    with torch.no_grad():
        model(batch.to_dict(), training=False, compute_force=False)
    return CAP["ne"], CAP["grid"]


def resample_tri(F3, shape):
    """Trilinear periodic point sampling of F3 [nx,ny,nz] at the points of a grid of `shape` (down- or up-sampling)."""
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
    """Fourier truncation / zero padding (Nyquist of an even axis dropped); values preserved, not sums."""
    G = torch.fft.fftn(F3)
    for ax, (n, N) in enumerate(zip(F3.shape, shape)):
        m = min(n, N); half = (m - 1) // 2
        ks = torch.arange(-half, half + 1, device=dev)
        Gn = torch.zeros([N if i == ax else s for i, s in enumerate(G.shape)], dtype=G.dtype, device=dev)
        Gn.index_copy_(ax, ks % N, G.index_select(ax, ks % n)); G = Gn
    return torch.fft.ifftn(G).real * (float(np.prod(shape)) / float(np.prod(F3.shape)))


def make_grid(lat, shape):
    return tp.TorchGrid(lat, tuple(shape), device=str(dev), dtype=torch.float64, rspec=True)


def rho_from(gd, phi3, ne, Ez_pa=None):
    """rho_b = w_b div( a(|E|) E ), E = -(w_b grad phi3), cavity from ne; optional plane mean of E_z replaced by Ez_pa[nz]."""
    s_i, s_d, _ = tp.create_cavity_torch(ne, gd, params); s_d = torch.clamp(s_d, 0.0, 1.0); del s_i
    w_b = tp._normalized_gaussian_kernel_g(gd, SIGMA_B)
    ex, ey, ez, em = gd.grad_from_recip(-torch.conj(w_b) * gd.fft(phi3))
    if Ez_pa is not None:
        ez = ez - ez.mean((0, 1))[None, None, :] + Ez_pa[None, None, :]; em = torch.sqrt(ex ** 2 + ey ** 2 + ez ** 2)
    a = response_a3(em, s_d, params, tp)
    return gd.ifft_real(w_b * gd.div_real_vector(a * ex, a * ey, a * ez)), s_d


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


def interp_z(prof, z_src, lz, nz_dst):
    zd = np.arange(nz_dst) * lz / nz_dst
    zs = np.asarray(z_src) % lz; o = np.argsort(zs); zs, pr = zs[o], np.asarray(prof)[o]
    return np.interp(zd, np.concatenate([zs, [zs[0] + lz]]), np.concatenate([pr, [pr[0]]]))


def line_removed(d, lz):
    z = np.arange(d.size) * lz / d.size; A = np.stack([z, np.ones_like(z)], 1)
    c, *_ = np.linalg.lstsq(A, d, rcond=None); r = d - A @ c
    return float(np.sqrt((r ** 2).mean())), float(c[0])


def cav_diff(s_ref, s_new, dV):
    d = (s_new - s_ref)
    return dict(L1=float(d.abs().sum() / s_ref.sum()), frac_gt_0p1=float((d.abs() > 0.1).double().mean()), vol_diff_A3=float(d.sum() * dV))


zyx = lambda t: t.permute(2, 1, 0).contiguous().cpu().numpy()
EXTRA_GRIDS = [(50, 50, 150), (134, 134, 400)]
RES = []
for k in PAIRS:
    rec = dict(pair=k, split=ATOMS[k].info["_split"], frames={}, response={})
    keep = {}
    for sid in (k, k + 600):
        d = dft_dir(sid); t1 = time.time()
        lat, chg = read_grid_fast(f"{d}/CHGCAR"); _, phi3 = read_grid_fast(f"{d}/PHI"); _, rb_raw = read_grid_fast(f"{d}/RHOB"); _, ri_raw = read_grid_fast(f"{d}/RHOION")
        shape = tuple(chg.shape); V = float(abs(np.linalg.det(lat))); dV = V / chg.numel(); lz = float(lat[2, 2])
        ne_ml, gm = forward(sid); mshape = tuple(ne_ml.shape); nzm = mshape[2]; dVm = V / float(np.prod(mshape))
        bl_row = backend._bl_index.get(sid)
        fields_bl = torch.from_numpy(np.ascontiguousarray(backend._bl_arr[bl_row][backend._bl_take])).to(dev).double()
        neutral_v, phi_base = fields_bl[0], fields_bl[1]; del fields_bl
        gd = make_grid(lat, shape)
        fr = dict(check=dict(read_s=time.time() - t1, native=list(shape), model=list(mshape)), cand={}, cand_mg={}, cavity={}, potential={}, series={})
        keep[sid] = {}
        with torch.no_grad():
            ne_d = torch.clamp(chg / V, min=0.0); del chg
            rho_b_ref = -(rb_raw / V); rho_i_ref = -(ri_raw / V); del rb_raw, ri_raw
            Dz = zyx(rho_b_ref); pD = poisson_np(Dz, lat); keep[sid]["ref"] = Dz
            fr["check"]["ne_e"] = float(ne_d.sum() * dV)

            def record(name, F_native_xyz, extra=None):
                M = zyx(F_native_xyz); m, _ = metrics(M, Dz, lat, dV, pD)
                if extra: m.update(extra)
                fr["cand"][name] = m; keep[sid][name] = M

            # ---------------- native stage: cavity density variants, true field
            rA, sA = rho_from(gd, phi3, ne_d); record("A", rA); del rA
            ne_rt = resample_tri(resample_tri(ne_d, mshape), shape)
            r1, s1 = rho_from(gd, phi3, ne_rt); fr["cavity"]["R1"] = cav_diff(sA, s1, dV); record("R1", r1); del r1, s1, ne_rt
            net_dft = resample_tri(torch.as_tensor(np.asarray(np.load(ENT_N[sid]["path"], mmap_mode="r"), dtype=np.float64), device=dev).permute(2, 1, 0).contiguous(), mshape)
            ne_pkg_m = torch.clamp((neutral_v - net_dft * V) / V, min=0.0)
            fr["check"]["ne_pkg_e"] = float(ne_pkg_m.sum() * dVm); fr["check"]["ne_ml_e"] = float(ne_ml.sum() * dVm)
            ne_pkg = resample_tri(ne_pkg_m, shape)
            r2, s2 = rho_from(gd, phi3, ne_pkg); fr["cavity"]["R2"] = cav_diff(sA, s2, dV); record("R2", r2); del r2, s2, ne_pkg
            # cavity of the model density upsampled (D's cavity), for the interpolation-vs-network bound
            _, sD = tp.create_cavity_torch(resample_tri(ne_ml, shape), gd, params)[:2]
            fr["cavity"]["D_model_ne"] = cav_diff(sA, torch.clamp(sD, 0.0, 1.0), dV); del sD
            # ---------------- model-grid stage
            rb_m = resample_tri(rho_b_ref, mshape); ri_m = resample_tri(rho_i_ref, mshape)
            Dm = zyx(rb_m); pDm = poisson_np(Dm, lat)
            phi_m = resample_tri(phi3, mshape); ne_m = torch.clamp(resample_tri(ne_d, mshape), min=0.0)

            def record_mg(name, F_m_xyz, Dref_m, pDref_m):
                m, _ = metrics(zyx(F_m_xyz), Dref_m, lat, dVm, pDref_m); fr["cand_mg"][name] = m
                record(name, resample_tri(F_m_xyz, shape))

            rM1, sM1 = rho_from(gm, phi_m, ne_m); record_mg("M1", rM1, Dm, pDm); del rM1
            # Fourier-resampled inputs and reference
            phi_f = resample_fourier(phi3, mshape); ne_f = torch.clamp(resample_fourier(ne_d, mshape), min=0.0); rb_f = resample_fourier(rho_b_ref, mshape)
            rM1f, _ = rho_from(gm, phi_f, ne_f); Dmf = zyx(rb_f); record_mg("M1f", rM1f, Dmf, poisson_np(Dmf, lat)); del rM1f, phi_f, ne_f, rb_f, Dmf
            # rebuilt potential (production assembly)
            cv_dft = phi_base - gm.ifft_real(gm.l0_inv_op(gm.fft(net_dft * V)))
            phi_solv_m = gm.ifft_real(gm.l0_inv_op(gm.fft(-V * (rb_m + ri_m))))
            phi_reb = cv_dft + phi_solv_m
            dphi = (phi_reb - phi_m); dpm = dphi.mean((0, 1)).cpu().numpy(); dlat = dphi - dphi.mean((0, 1))[None, None, :]
            lr, slope = line_removed(dpm, lz)
            # solute part check: cv_dft vs (PHI_m - Poisson(solvent_m))
            dcv = cv_dft - (phi_m - phi_solv_m); dcv_lat = dcv - dcv.mean((0, 1))[None, None, :]
            fr["potential"] = dict(rebuilt_minus_mapped_lateral_rms=float(dlat.pow(2).mean().sqrt()), rebuilt_minus_mapped_pa_rms_lineremoved=lr,
                                   rebuilt_minus_mapped_pa_slope_eV_per_A=slope, mapped_lateral_rms=float((phi_m - phi_m.mean((0, 1))[None, None, :]).pow(2).mean().sqrt()),
                                   cv_minus_true_solute_lateral_rms=float(dcv_lat.pow(2).mean().sqrt()), cv_minus_true_solute_pa_rms_lineremoved=line_removed(dcv.mean((0, 1)).cpu().numpy(), lz)[0])
            del dphi, dlat, dcv, dcv_lat
            rM2, _ = rho_from(gm, phi_reb, ne_m); record_mg("M2", rM2, Dm, pDm); del rM2
            ip = POT_IDX[sid]; phi_lab_t = torch.as_tensor(interp_z(POT["phi_eV"][ip], POT["z_A"][ip], lz, nzm), device=dev)
            kz = 2.0 * math.pi * torch.fft.fftfreq(nzm, d=lz / nzm, device=dev, dtype=torch.float64); wb1 = torch.exp(-0.5 * (SIGMA_B * kz) ** 2)
            Ez_lab = torch.fft.ifft(-1j * kz * wb1 * torch.fft.fft(phi_lab_t)).real
            rM3, _ = rho_from(gm, phi_reb, ne_m, Ez_pa=Ez_lab); record_mg("M3", rM3, Dm, pDm); del rM3
            rM4, sM4 = rho_from(gm, phi_m, ne_pkg_m); record_mg("M4", rM4, Dm, pDm); del rM4
            fr["cavity"]["M4_vs_M1_modelgrid"] = cav_diff(sM1, sM4, dVm); del sM1, sM4
            rM5, _ = rho_from(gm, phi_reb, ne_pkg_m, Ez_pa=Ez_lab); record_mg("M5", rM5, Dm, pDm); del rM5, phi_reb, phi_solv_m, cv_dft
            # ---------------- resolution series (direct route, trilinear mapping)
            for gs in EXTRA_GRIDS:
                gx = make_grid(lat, gs); dVx = V / float(np.prod(gs))
                rX, _ = rho_from(gx, resample_tri(phi3, gs), torch.clamp(resample_tri(ne_d, gs), min=0.0))
                Dx = zyx(resample_tri(rho_b_ref, gs)); mg, _ = metrics(zyx(rX), Dx, lat, dVx)
                nat, _ = metrics(zyx(resample_tri(rX, shape)), Dz, lat, dV, pD)
                fr["series"]["x".join(map(str, gs))] = dict(mg=mg, nat=nat); del rX, Dx, gx
            del phi3, ne_d, rho_b_ref, rho_i_ref, rb_m, ri_m, phi_m, ne_m, net_dft, ne_pkg_m
        rec["frames"][str(sid)] = fr; c = fr["cand"]; cm = fr["cand_mg"]
        print(f"[{time.time() - T0:5.0f}s] sid {sid} (q {float(ATOMS[sid].info['total_charge']):+.3f}) native {shape} model {mshape}; electrons native {fr['check']['ne_e']:.1f} pkg {fr['check']['ne_pkg_e']:.1f} ml {fr['check']['ne_ml_e']:.1f} | "
              + " | ".join(f"{nm}: L1 {c[nm]['L1']:.3f} lat {c[nm]['eps_lat']:.3f} ({c[nm]['lat_corr']:+.2f}) pa {c[nm]['eps_pa']:.3f} norm {c[nm]['norm_ratio']:.2f}" for nm in ("A", "R1", "R2")), flush=True)
        print(f"         cavity vs A: R1 L1 {fr['cavity']['R1']['L1']:.4f} R2 {fr['cavity']['R2']['L1']:.4f} D {fr['cavity']['D_model_ne']['L1']:.4f} | model grid (mg / nat): "
              + " | ".join(f"{nm}: {cm[nm]['L1']:.3f}/{c[nm]['L1']:.3f} lat {cm[nm]['eps_lat']:.3f} ({cm[nm]['lat_corr']:+.2f}) norm {cm[nm]['norm_ratio']:.2f}" for nm in ("M1", "M1f", "M2", "M3", "M4", "M5")), flush=True)
        print(f"         potential: rebuilt-mapped lateral rms {fr['potential']['rebuilt_minus_mapped_lateral_rms']:.4f} eV (mapped lateral rms {fr['potential']['mapped_lateral_rms']:.4f}), pa line-removed {fr['potential']['rebuilt_minus_mapped_pa_rms_lineremoved']:.4f}, slope {fr['potential']['rebuilt_minus_mapped_pa_slope_eV_per_A']:+.5f} eV/A; "
              f"cv vs true solute lateral rms {fr['potential']['cv_minus_true_solute_lateral_rms']:.4f} | series (mg/nat L1): "
              + " ".join(f"{g}: {v['mg']['L1']:.3f}/{v['nat']['L1']:.3f}" for g, v in fr["series"].items()), flush=True)
    dV = abs(np.linalg.det(lat)) / keep[k]["ref"].size
    Dr = keep[k]["ref"] - keep[k + 600]["ref"]
    names = [n for n in keep[k] if n != "ref"]
    for nm in names:
        m, _ = metrics(keep[k][nm] - keep[k + 600][nm], Dr, lat, dV); rec["response"][nm] = m
    print(f"[{time.time() - T0:5.0f}s] pair {k} response vs DFT (native): " + " | ".join(f"{nm}: L1 {rec['response'][nm]['L1']:.3f} lat {rec['response'][nm]['eps_lat']:.3f} ({rec['response'][nm]['lat_corr']:+.2f}) pa {rec['response'][nm]['eps_pa']:.3f}" for nm in names), flush=True)
    RES.append(rec); json.dump(RES, open(OUT, "w"), indent=1); del keep
print(f"[{time.time() - T0:5.0f}s] done", flush=True)
