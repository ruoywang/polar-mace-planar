"""A / B / C / D on the DFT NATIVE grid (500 x 168 x 168, 0.09 A), from the raw VASPsol files PHI (total potential,
electron-energy convention = the solver's phi), CHGCAR (electron density), RHOB, RHOION -- the construction that
reproduced DFT RHOB to 0.18 % on 2026-09-10 (bound_coeff_decompose.py step 1), extended to B / C / D and to the full 3-D
comparison. No resampling of any DFT quantity. Grid operators: torch_pb.TorchGrid on the native grid; w_b Gaussian with
sigma_b = R_B or A_K; rho_b = w_b div( a(|E|) E ) with response_a3 (local-field factor folded into a, as in the closure).
    A        E = -(w_b grad PHI), DFT cavity (CHGCAR), response recomputed at |E|
    A_frozen E as A, response frozen at the screened vacuum field
    B        E_vac = -(w_b grad phi_sol), phi_sol = PHI - Poisson(RHOB + RHOION) in the same convention, screened by eps3 of the
             DFT cavity, plane mean of E_z replaced by the DFT total field's; response recomputed at |E_B|
    C        field B, response frozen at |E_vac / eps3| (the production choice)
    D        E as A, MODEL cavity (model electron density of the production forward, trilinearly upsampled from the
             model grid to the native grid), response recomputed
Metrics on the native grid against -RHOB/V: total L1 / L2, lateral L1 / squared / corr, plane-average L1 / corr, net
charge, norm ratios, 3-D bound potential rms; charged - neutral response per pair. Also: native RHOB plane mean vs the
training-package label (must be identical), the Sept-style plane-average L1 of A.
Usage: python abcd_native.py <out.json>   (cwd with ./data, ./cal1_train.json); env KIT_PAIRS, KIT_DEVICE, KIT_DFT
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

OUT = sys.argv[1] if len(sys.argv) > 1 else "abcd_native.json"
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
man_s = json.load(open("data/solvent3d_grid_manifest.json")); ENT_S = {int(k): v for k, v in man_s["entries"].items()}
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
    """VASP CHGCAR-style grid file -> (lattice, tensor [nx, ny, nz]); fast numpy parsing."""
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
        rest = f.read()                                   # grid numbers first; parsing stops at the augmentation text
    arr = np.fromstring(rest, sep=" ", dtype=np.float64)[:need]; del rest
    assert arr.size == need, (path, arr.size, need)
    return lat, torch.as_tensor(np.ascontiguousarray(arr.reshape(nz, ny, nx).transpose(2, 1, 0)), device=dev)


def forward_ne(sid):
    atoms = ATOMS[sid]
    cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(dev)
    with torch.no_grad():
        model(batch.to_dict(), training=False, compute_force=False)
    return CAP["ne"]


def upsample_to(F3, shape):
    nx, ny, nz = shape
    ii, jj, kk = torch.meshgrid(torch.arange(nx, device=dev), torch.arange(ny, device=dev), torch.arange(nz, device=dev), indexing="ij")
    out = torch.empty(shape, device=dev)
    for x0 in range(0, nx, 12):
        x1 = min(x0 + 12, nx)
        fm = torch.stack([ii[x0:x1].reshape(-1) / nx, jj[x0:x1].reshape(-1) / ny, kk[x0:x1].reshape(-1) / nz], dim=1).double()
        out[x0:x1] = _interp3_periodic(F3, fm).reshape(x1 - x0, ny, nz)
    return out


def poisson_np(rho_zyx, lat):
    nz, ny, nx = rho_zyx.shape
    rg = np.fft.rfftn(rho_zyx); Bm = 2.0 * np.pi * np.linalg.inv(lat).T
    hx = np.fft.rfftfreq(nx) * nx; hy = np.fft.fftfreq(ny) * ny; hz = np.fft.fftfreq(nz) * nz
    G = (hz[:, None, None, None] * Bm[2] + hy[None, :, None, None] * Bm[1] + hx[None, None, :, None] * Bm[0])
    G2 = (G * G).sum(-1); G2[0, 0, 0] = 1.0
    pg = 4.0 * np.pi * K_EVA * rg / G2; pg[0, 0, 0] = 0.0
    return np.fft.irfftn(pg, s=rho_zyx.shape)


def metrics(M, D, lat, dV, pD=None):
    """M, D numpy (nz, ny, nx)."""
    d = M - D; pm, pd = M.mean(axis=(1, 2)), D.mean(axis=(1, 2)); Ml, Dl = M - pm[:, None, None], D - pd[:, None, None]
    pM = poisson_np(M, lat); pD = poisson_np(D, lat) if pD is None else pD
    return dict(L1=float(np.abs(d).sum() / np.abs(D).sum()), L2=float((d ** 2).sum() / (D ** 2).sum()),
                eps_lat=float(np.abs(Ml - Dl).sum() / np.abs(Dl).sum()), lat_sq=float(((Ml - Dl) ** 2).sum() / (Dl ** 2).sum()),
                lat_corr=float((Ml * Dl).sum() / np.sqrt((Ml * Ml).sum() * (Dl * Dl).sum())),
                eps_pa=float(np.abs(pm - pd).sum() / np.abs(pd).sum()), pa_corr=float(((pm - pm.mean()) * (pd - pd.mean())).sum() / np.sqrt(((pm - pm.mean()) ** 2).sum() * ((pd - pd.mean()) ** 2).sum())),
                net_err_e=float(d.sum()) * dV, net_ml_e=float(M.sum()) * dV, net_dft_e=float(D.sum()) * dV,
                phi_rms_err=float(np.sqrt(((pM - pD) ** 2).mean())), phi_rms_dft=float(np.sqrt((pD ** 2).mean())),
                norm_ratio=float(np.sqrt((M * M).sum() / (D * D).sum())), lat_norm_ratio=float(np.sqrt((Ml * Ml).sum() / (Dl * Dl).sum()))), pD


zyx = lambda t: t.permute(2, 1, 0).contiguous().cpu().numpy()
RES = []
for k in PAIRS:
    rec = dict(pair=k, split=ATOMS[k].info["_split"], frames={}, response={})
    keep = {}
    for sid in (k, k + 600):
        d = dft_dir(sid); t1 = time.time()
        lat, chg = read_grid_fast(f"{d}/CHGCAR"); _, phi3 = read_grid_fast(f"{d}/PHI"); _, rb_raw = read_grid_fast(f"{d}/RHOB"); _, ri_raw = read_grid_fast(f"{d}/RHOION")
        shape = tuple(chg.shape); V = float(abs(np.linalg.det(lat))); dV = V / chg.numel(); lz = float(lat[2, 2])
        gd = tp.TorchGrid(lat, shape, device=str(dev), dtype=torch.float64, rspec=True)
        with torch.no_grad():
            ne_d = torch.clamp(chg / V, min=0.0); del chg
            rho_b_ref = -(rb_raw / V); rho_i_ref = -(ri_raw / V); del rb_raw, ri_raw
            # label consistency: native RHOB plane mean vs the training-package label
            e = ENT_S[sid]; lab = np.asarray(np.load("data/" + e["path_b"], mmap_mode="r"), dtype=np.float64).mean(axis=(1, 2))
            pm_native = rho_b_ref.mean(dim=(0, 1)).cpu().numpy()
            chk = dict(label_vs_native_pa_L1=float(np.abs(pm_native - lab).sum() / np.abs(lab).sum()), read_s=time.time() - t1,
                       ne_e=float(ne_d.sum() * dV), int_abs_rhob_e=float(rho_b_ref.abs().sum() * dV))
            s_ion_d, s_diel_d, _ = tp.create_cavity_torch(ne_d, gd, params); s_diel_d = torch.clamp(s_diel_d, 0.0, 1.0); del s_ion_d
            sigma_b = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
            w_b = tp._normalized_gaussian_kernel_g(gd, sigma_b)
            gradw = lambda f3: gd.grad_from_recip(-torch.conj(w_b) * gd.fft(f3))
            div_b = lambda a, E: gd.ifft_real(w_b * gd.div_real_vector(a * E[0], a * E[1], a * E[2]))
            mag = lambda E: torch.sqrt(E[0] ** 2 + E[1] ** 2 + E[2] ** 2)
            exT, eyT, ezT, emagT = gradw(phi3)
            ET = (exT, eyT, ezT)
            # solute potential (same convention): PHI minus the solvent part
            phi_solv = gd.ifft_real(gd.l0_inv_op(gd.fft(-V * (rho_b_ref + rho_i_ref))))
            phi_sol = phi3 - phi_solv; del phi3, phi_solv
            exv, eyv, ezv, _ = gradw(phi_sol); del phi_sol
            resp0 = response_a3(s_diel_d.new_zeros(1), s_diel_d.new_ones(1), params, tp)[0]
            eps_d = 1.0 + EDEPS * resp0 * s_diel_d
            exs, eys, ezs = exv / eps_d, eyv / eps_d, ezv / eps_d; del exv, eyv, ezv
            EB = (exs, eys, ezs - ezs.mean((0, 1))[None, None, :] + ezT.mean((0, 1))[None, None, :])
            a_frozen = response_a3(mag((exs, eys, ezs)), s_diel_d, params, tp)
            cands = {}
            cands["A"] = div_b(response_a3(emagT, s_diel_d, params, tp), ET)
            cands["A_frozen"] = div_b(a_frozen, ET)
            cands["B"] = div_b(response_a3(mag(EB), s_diel_d, params, tp), EB)
            cands["C"] = div_b(a_frozen, EB)
            del a_frozen, EB, exs, eys, ezs, eps_d, s_diel_d
            # D: model cavity (production electron density, upsampled to the native grid)
            ne_m = upsample_to(forward_ne(sid), shape)
            s_ion_m, s_diel_m, _ = tp.create_cavity_torch(ne_m, gd, params); s_diel_m = torch.clamp(s_diel_m, 0.0, 1.0); del s_ion_m, ne_m
            cands["D"] = div_b(response_a3(emagT, s_diel_m, params, tp), ET)
            del s_diel_m, ET, exT, eyT, ezT, emagT
            Dz = zyx(rho_b_ref); pD = poisson_np(Dz, lat)
            fr = dict(check=chk, cand={})
            keep[sid] = {}
            for nm, F in cands.items():
                M = zyx(F); m, _ = metrics(M, Dz, lat, dV, pD); fr["cand"][nm] = m; keep[sid][nm] = M
            keep[sid]["ref"] = Dz
            del cands
        rec["frames"][str(sid)] = fr; c = fr["cand"]
        print(f"[{time.time() - T0:5.0f}s] sid {sid} (q {float(ATOMS[sid].info['total_charge']):+.3f}) native {shape}; label-vs-native RHOB pa L1 {chk['label_vs_native_pa_L1']:.2e}; electrons {chk['ne_e']:.1f}; read {chk['read_s']:.0f}s | "
              + " | ".join(f"{nm}: L1 {c[nm]['L1']:.3f} lat {c[nm]['eps_lat']:.3f} (corr {c[nm]['lat_corr']:+.2f}) pa {c[nm]['eps_pa']:.3f} (corr {c[nm]['pa_corr']:+.2f}) norm {c[nm]['norm_ratio']:.2f} phi {c[nm]['phi_rms_err']:.3f}" for nm in ("A", "A_frozen", "B", "C", "D")), flush=True)
    # response: twin frames share the cell, so the last frame's lattice applies
    dV = abs(np.linalg.det(lat)) / keep[k]["ref"].size
    Dr = keep[k]["ref"] - keep[k + 600]["ref"]
    for nm in ("A", "A_frozen", "B", "C", "D"):
        m, _ = metrics(keep[k][nm] - keep[k + 600][nm], Dr, lat, dV); rec["response"][nm] = m
    print(f"[{time.time() - T0:5.0f}s] pair {k} response vs DFT: " + " | ".join(f"{nm}: L1 {rec['response'][nm]['L1']:.3f} lat {rec['response'][nm]['eps_lat']:.3f} pa {rec['response'][nm]['eps_pa']:.3f} phi {rec['response'][nm]['phi_rms_err']:.3f}" for nm in ("A", "A_frozen", "B", "C", "D")), flush=True)
    RES.append(rec); json.dump(RES, open(OUT, "w"), indent=1); del keep
print(f"[{time.time() - T0:5.0f}s] done", flush=True)
