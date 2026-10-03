"""Ion analogue of candidate A: the production ion model evaluated at the DFT potential itself (native grid).
    rho_i^A(r) = -ion_density_values(PHI + c, s_ion^DFT, params, V) / V,  c fixed by neutrality int rho_i dV = -Q_total
Variants: potential PHI as read; PHI smoothed with w_b (sigma_b); s_ion from CHGCAR (DFT) and from the model density (D-type).
Compared with -RHOION/V: L1, lateral L1 / corr, plane-average L1 / corr, norm ratio, net charge, plus the plane profiles saved.
Also reports the shift c, and the fraction of the solvent-box volume where the ion response is saturated (|ZBETA (PHI+c)| > 100).
Usage: python ion_A_check.py <out.json>; env KIT_PAIRS, KIT_DEVICE, KIT_DFT, KIT_MODEL
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
from mace.modules.pb1d_solver import ion_density_values
from mace.modules.solvent3d import _interp3_periodic

OUT = sys.argv[1] if len(sys.argv) > 1 else "ion_A_check.json"
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
SIGMA_B = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
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
    CAP["ne"] = n_e.detach().clone()
    return _orig_clo(n_e, cv, grid, params_, tp_)


PB.closure_from_fields = clo_cap


def forward_ne(sid):
    atoms = ATOMS[sid]
    cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(dev)
    with torch.no_grad():
        model(batch.to_dict(), training=False, compute_force=False)
    return CAP["ne"]


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


def quick(Mt, Dt, dV):
    d = Mt - Dt; pm = Mt.mean((0, 1)); pd = Dt.mean((0, 1)); Ml = Mt - pm[None, None, :]; Dl = Dt - pd[None, None, :]
    return dict(L1=float(d.abs().sum() / Dt.abs().sum()), eps_lat=float((Ml - Dl).abs().sum() / Dl.abs().sum()), lat_corr=float((Ml * Dl).sum() / (Ml.norm() * Dl.norm())),
                eps_pa=float((pm - pd).abs().sum() / pd.abs().sum()), pa_corr=float(((pm - pm.mean()) * (pd - pd.mean())).sum() / ((pm - pm.mean()).norm() * (pd - pd.mean()).norm())),
                norm_ratio=float(Mt.norm() / Dt.norm()), net_e=float(Mt.sum() * dV), net_dft_e=float(Dt.sum() * dV))


def rho_ion_shifted(phi, s_i, V, dV, Q):
    """ion density at phi + c with c from neutrality (bracket + bisection)."""
    target = -Q
    g = lambda c: float((-ion_density_values(phi + c, s_i, params, V) / V).sum() * dV) - target
    lo, hi = -0.1, 0.1; glo, ghi = g(lo), g(hi); step = 0.2
    while glo * ghi > 0.0 and step < 1e3:
        lo -= step; hi += step; glo, ghi = g(lo), g(hi); step *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi); gm = g(mid)
        if glo * gm <= 0.0:
            hi, ghi = mid, gm
        else:
            lo, glo = mid, gm
        if hi - lo < 1e-5:
            break
    c = 0.5 * (lo + hi)
    return -ion_density_values(phi + c, s_i, params, V) / V, c


RES = []
for kpair in PAIRS:
    for sid in (kpair, kpair + 600):
        atoms = ATOMS[sid]; Q = float(atoms.info["total_charge"])
        d = dft_dir(sid)
        lat, chg = read_grid_fast(f"{d}/CHGCAR"); _, phi3 = read_grid_fast(f"{d}/PHI"); _, ri_raw = read_grid_fast(f"{d}/RHOION")
        shape = tuple(chg.shape); V = float(abs(np.linalg.det(lat))); dV = V / chg.numel(); lz = float(lat[2, 2])
        gd = tp.TorchGrid(lat, shape, device=str(dev), dtype=torch.float64, rspec=True)
        with torch.no_grad():
            ne_d = torch.clamp(chg / V, min=0.0); del chg
            rho_i_ref = -(ri_raw / V); del ri_raw
            s_i_d, s_d_d, _ = tp.create_cavity_torch(ne_d, gd, params); s_i_d = torch.clamp(s_i_d, 0.0, 1.0)
            s_i_m, _, _ = tp.create_cavity_torch(resample_tri(forward_ne(sid), shape), gd, params); s_i_m = torch.clamp(s_i_m, 0.0, 1.0)
            w_b = tp._normalized_gaussian_kernel_g(gd, SIGMA_B)
            phi_s = gd.ifft_real(w_b * gd.fft(phi3))
            out = dict(sid=sid, pair=kpair, q_total=Q, q_ion_dft=float(rho_i_ref.sum() * dV), s_ion_vol_dft_A3=float(s_i_d.sum() * dV), s_ion_vol_model_A3=float(s_i_m.sum() * dV), cands={})
            profs = {"z": np.arange(shape[2]) * lz / shape[2], "ref": rho_i_ref.mean((0, 1)).cpu().numpy()}
            for name, phi, s_i in (("PHI_dftcavity", phi3, s_i_d), ("PHIsmoothed_dftcavity", phi_s, s_i_d), ("PHI_modelcavity", phi3, s_i_m)):
                r, c = rho_ion_shifted(phi, s_i, V, dV, Q)
                m = quick(r, rho_i_ref, dV); m["shift_eV"] = c
                x = params["ZBETA"] * (phi + c); m["saturated_frac_in_ion_region"] = float(((x.abs() > 100.0) & (s_i > 0.5)).double().sum() / (s_i > 0.5).double().sum())
                m["ion_region_pot_mean_eV"] = float((phi + c)[s_i > 0.5].mean()); m["ion_region_pot_std_eV"] = float((phi + c)[s_i > 0.5].std())
                out["cands"][name] = m; profs[name] = r.mean((0, 1)).cpu().numpy()
            np.savez_compressed(f"ionA_profiles_sid{sid}.npz", **profs)
        RES.append(out); json.dump(RES, open(OUT, "w"), indent=1)
        print(f"[{time.time() - T0:5.0f}s] sid {sid} (q {Q:+.3f}) DFT q_ion {out['q_ion_dft']:+.4f}; s_ion volume DFT {out['s_ion_vol_dft_A3']:.0f} model {out['s_ion_vol_model_A3']:.0f} A^3 | "
              + " | ".join(f"{k}: L1 {v['L1']:.3f} lat {v['eps_lat']:.3f} ({v['lat_corr']:+.2f}) pa {v['eps_pa']:.3f} ({v['pa_corr']:+.2f}) norm {v['norm_ratio']:.3f} shift {v['shift_eV']:+.3f} sat {v['saturated_frac_in_ion_region']:.3f} pot {v['ion_region_pot_mean_eV']:+.3f}+-{v['ion_region_pot_std_eV']:.3f}" for k, v in out["cands"].items()), flush=True)
print(f"[{time.time() - T0:5.0f}s] done", flush=True)
