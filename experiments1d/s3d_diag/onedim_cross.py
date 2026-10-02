"""1-D cross-substitution I0..I4 (reviewer plan 2026-10-02; no training, no production-code change).

The 1-D bound / ionic profiles of prod500_vsolv_fix are recomputed with the solute electrostatic source and the
cavity-generating electron density swapped between ML and DFT, and the learned polarization correction delta_p
kept / frozen at its baseline value / switched off. For every combination the dielectric closure (cavity, screened
field, response, prior) is recomputed by the production code itself:
    config   solute source   cavity density   delta_p
    I0       ML              ML               as predicted               (= the production model)
    I1       DFT             ML               frozen at I0
    I2       ML              DFT              frozen at I0
    I3       DFT             DFT              frozen at I0
    I4       DFT             DFT              off (0)
"Solute source" = the net SOLUTE charge (valence electrons + smeared ion cores, the density3d_net_grid label; no solvent
charge, no reaction potential) pushed through the model's own assembly: n_e = clamp((neutral_v - net V)/V, 0),
cvhar3 = phi_base - Poisson(net), cvhar_z = <cvhar3>_xy, solute dipole from the same electron profile. Mechanics:
backend._gto_net_density_g is intercepted for the solute-density call only (recognised by the atomic sigmas) and
returns the FFT of the DFT net charge resampled to the model grid; pb1d_backend.closure_from_fields is intercepted to
swap the cavity density; model.pb1d_head.delta_p is intercepted to replay / zero the correction. Nothing else changes.
Reported per frame and config: rho_b(z), rho_ion(z) vs the DFT plane averages of RHOB / RHOION (L1, L2), their 1-D
periodic potentials (rms error, bound / ion / total), q_ion, mu_bound; and the charged - neutral response per pair.
Usage: python onedim_cross.py <out.json>   (cwd with ./data, ./cal1_train.json); env KIT_PAIRS, KIT_CONFIGS (e.g. "I0 I1 I2 I3 I4")
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

OUT = sys.argv[1] if len(sys.argv) > 1 else "onedim_cross.json"
MODEL = os.environ.get("KIT_MODEL", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_vsolv_fix/models/prod500_vsolv_fix.model")
PAIRS = [int(s) for s in os.environ.get("KIT_PAIRS", "1 52 152").replace("+", " ").split()]
CONFIGS = os.environ.get("KIT_CONFIGS", "I0 I1 I2 I3 I4").replace("+", " ").split()
devname = os.environ.get("KIT_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
dev = torch.device(devname)
if dev.type == "cpu":
    _jl = torch.jit.load
    torch.jit.load = lambda f, map_location=None, **kw: _jl(f, map_location="cpu", **kw)
model = torch.load(MODEL, map_location=dev).to(dev); model.eval()
for p in model.parameters():
    p.requires_grad_(False)
backend = model._get_pb1d_backend(); head = model.pb1d_head
ATOMIC_SIGMAS = [float(s) for s in model.atomic_density_sigmas]
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
print(f"model {os.path.basename(MODEL)}; device {dev}; pairs {PAIRS}; configs {CONFIGS}; atomic sigmas {ATOMIC_SIGMAS}; "
      f"R_B {backend.params['R_B']} A_K {backend.params['A_K']} solve_upsample {backend.solve_upsample}", flush=True)


# ------------------------------------------------------------------ DFT labels
def dft_planes(sid):
    e = ENT_S[sid]
    rb = np.asarray(np.load("data/" + e["path_b"], mmap_mode="r"), dtype=np.float64).mean(axis=(1, 2))
    ri = np.asarray(np.load("data/" + e["path_i"], mmap_mode="r"), dtype=np.float64).mean(axis=(1, 2))
    with np.load("data/" + e["meta_path"]) as m:
        lat = np.asarray(m["lattice"], dtype=np.float64)
    return rb, ri, lat


def dft_net_on_model_grid(sid, grid, mshape):
    """DFT net solute charge (e/A^3, (nz,ny,nx)) -> model grid [nx,ny,nz] values in e (density * volume)."""
    e = ENT_N[sid]
    rho = torch.as_tensor(np.asarray(np.load(e["path"], mmap_mode="r"), dtype=np.float64), device=dev).permute(2, 1, 0).contiguous()
    nx, ny, nz = mshape
    ii, jj, kk = torch.meshgrid(torch.arange(nx, device=dev), torch.arange(ny, device=dev), torch.arange(nz, device=dev), indexing="ij")
    fm = torch.stack([ii.reshape(-1) / nx, jj.reshape(-1) / ny, kk.reshape(-1) / nz], dim=1).double()
    return _interp3_periodic(rho, fm).reshape(nx, ny, nz) * float(grid.volume)


# ------------------------------------------------------------------ interception state
STATE = dict(mode="I0", sid=None, net_dft=None, ne_ml=[], ne_dft=None, dp=[], call_clo=0, call_dp=0, call_gto=0)
_orig_gto = backend._gto_net_density_g
_orig_clo = PB.closure_from_fields
_orig_dp = head.delta_p


def gto_patch(grid, pos_frac, coeffs, sigmas):
    is_solute = [float(s) for s in sigmas] == ATOMIC_SIGMAS
    if is_solute and STATE["mode"] in ("I1", "I3", "I4"):
        STATE["call_gto"] += 1
        return grid.fft(STATE["net_dft"])
    return _orig_gto(grid, pos_frac, coeffs, sigmas)


def clo_patch(n_e, cv, grid, params, tp_):
    k = STATE["call_clo"]; STATE["call_clo"] += 1
    if STATE["mode"] == "I0":
        STATE["ne_ml"].append(n_e.detach().clone())
        STATE["grid"] = grid; STATE["cv_ml"] = cv.detach().clone()
    elif STATE["mode"] == "I1":                                   # DFT source, ML cavity
        n_e = STATE["ne_ml"][min(k, len(STATE["ne_ml"]) - 1)]
    elif STATE["mode"] == "I2":                                   # ML source, DFT cavity
        n_e = STATE["ne_dft"]
    return _orig_clo(n_e, cv, grid, params, tp_)


def dp_patch(c, w_env, u, lz):
    k = STATE["call_dp"]; STATE["call_dp"] += 1
    if STATE["mode"] == "I0":
        out = _orig_dp(c, w_env, u, lz); STATE["dp"].append(out.detach().clone()); return out
    if STATE["mode"] == "I4":
        return torch.zeros_like(_orig_dp(c, w_env, u, lz))
    return STATE["dp"][min(k, len(STATE["dp"]) - 1)]


backend._gto_net_density_g = gto_patch
PB.closure_from_fields = clo_patch
head.delta_p = dp_patch


def forward(sid, mode):
    STATE.update(mode=mode, sid=sid, call_clo=0, call_dp=0, call_gto=0)
    atoms = ATOMS[sid]
    cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(dev)
    hold = {}; orig = backend.solve_graph

    def spy(*a, **kw):
        res = orig(*a, **kw); hold["res"] = res; hold["kw"] = kw; return res
    backend.solve_graph = spy
    try:
        with torch.no_grad():
            model(batch.to_dict(), training=False, compute_force=False)
    finally:
        backend.solve_graph = orig
    r = hold["res"]
    return dict(z=r["z"].detach().cpu().numpy(), rho_b=r["rho_bound_z"].detach().cpu().numpy(), rho_i=r["rho_ion_z"].detach().cpu().numpy(),
                phi_z=r["phi_z"].detach().cpu().numpy(), q_ion=float(r["q_ion"]), mu_bound=float(r["mu_bound"]),
                dp_rms=float(r["delta_p"].pow(2).mean().sqrt()), prior_rms=float(r["prior_solve"].pow(2).mean().sqrt()),
                n_calls=dict(clo=STATE["call_clo"], dp=STATE["call_dp"], gto=STATE["call_gto"]), solver_exit=str(r.get("solver_exit")),
                rms_last=float(r["rms_last"]) if r.get("rms_last") is not None else None, n_outer=int(r["n_outer"]) if r.get("n_outer") is not None else None)


# ------------------------------------------------------------------ 1-D metrics
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
                phi_rms_err=float(np.sqrt(((poisson1d(m, lz) - poisson1d(d, lz)) ** 2).mean())),
                phi_rms_dft=float(np.sqrt((poisson1d(d, lz) ** 2).mean())), net_err_e=None)


def evaluate(prof, D, lat, area):
    rb_d, ri_d = D; lz = lat[2, 2]; nz = rb_d.size; dz = lz / nz
    rb = on_dft_z(prof["rho_b"], prof["z"], lz, nz); ri = on_dft_z(prof["rho_i"], prof["z"], lz, nz)
    out = {"b": m1d(rb, rb_d, lz), "i": m1d(ri, ri_d, lz), "t": m1d(rb + ri, rb_d + ri_d, lz)}
    for ch, m, d in (("b", rb, rb_d), ("i", ri, ri_d), ("t", rb + ri, rb_d + ri_d)):
        out[ch]["net_err_e"] = float((m - d).sum() * dz * area); out[ch]["net_dft_e"] = float(d.sum() * dz * area)
    out["profiles_dftgrid"] = dict(rho_b=rb.tolist(), rho_i=ri.tolist())
    return out


RES = []
for k in PAIRS:
    rec = dict(pair=k, split=ATOMS[k].info["_split"], frames={}, response={})
    profs = {}
    for sid in (k, k + 600):
        rb_d, ri_d, lat = dft_planes(sid); area = abs(np.linalg.det(lat)) / lat[2, 2]
        STATE.update(ne_ml=[], dp=[], net_dft=None, ne_dft=None)
        fr = {}
        P0 = forward(sid, "I0"); grid = STATE["grid"]
        # DFT net charge on the model grid, and the DFT cavity density through the model's own assembly
        net_dft = dft_net_on_model_grid(sid, grid, tuple(STATE["ne_ml"][-1].shape))
        bl_row = backend._bl_index.get(sid)
        fields = torch.from_numpy(np.ascontiguousarray(backend._bl_arr[bl_row][backend._bl_take])).to(dev).double()
        neutral_v = fields[0]
        ne_dft = torch.clamp((neutral_v - net_dft) / float(grid.volume), min=0.0)
        STATE.update(net_dft=net_dft, ne_dft=ne_dft)
        # consistency of the two solute densities (plane means, e/A^3)
        pm_ml = STATE["ne_ml"][-1].mean(dim=(0, 1)); pm_dft = ne_dft.mean(dim=(0, 1))
        chk = dict(ne_plane_L1=float((pm_ml - pm_dft).abs().sum() / pm_dft.abs().sum()), ne_ml_e=float(STATE["ne_ml"][-1].sum() * grid.volume / STATE["ne_ml"][-1].numel()),
                   ne_dft_e=float(ne_dft.sum() * grid.volume / ne_dft.numel()), net_dft_e=float(net_dft.sum() / net_dft.numel()))
        fr["I0"] = dict(evaluate(P0, (rb_d, ri_d), lat, area), q_ion=P0["q_ion"], mu_bound=P0["mu_bound"], dp_rms=P0["dp_rms"], prior_rms=P0["prior_rms"],
                        n_calls=P0["n_calls"], solver=dict(exit=P0["solver_exit"], rms=P0["rms_last"], n_outer=P0["n_outer"]))
        profs[(sid, "I0")] = P0
        for cfgn in [c for c in CONFIGS if c != "I0"]:
            P = forward(sid, cfgn)
            fr[cfgn] = dict(evaluate(P, (rb_d, ri_d), lat, area), q_ion=P["q_ion"], mu_bound=P["mu_bound"], dp_rms=P["dp_rms"], prior_rms=P["prior_rms"],
                            n_calls=P["n_calls"], solver=dict(exit=P["solver_exit"], rms=P["rms_last"], n_outer=P["n_outer"]))
            profs[(sid, cfgn)] = P
        fr["check"] = chk
        rec["frames"][str(sid)] = fr
        print(f"[{time.time() - T0:5.0f}s] sid {sid} (q {float(ATOMS[sid].info['total_charge']):+.3f}) ne plane L1 ML vs DFT {chk['ne_plane_L1']:.3f}; electrons ML {chk['ne_ml_e']:.2f} DFT {chk['ne_dft_e']:.2f}; "
              + " | ".join(f"{c}: b L1 {fr[c]['b']['L1']:.3f} L2 {fr[c]['b']['L2']:.3f} phi {fr[c]['b']['phi_rms_err']:.4f}; i L1 {fr[c]['i']['L1']:.3f} phi {fr[c]['i']['phi_rms_err']:.4f}; t phi {fr[c]['t']['phi_rms_err']:.4f}; q {fr[c]['q_ion']:+.3f} dp {fr[c]['dp_rms']:.2e}" for c in CONFIGS), flush=True)
    # charged - neutral response
    rb_c, ri_c, lat = dft_planes(k); rb_n, ri_n, _ = dft_planes(k + 600); lz = lat[2, 2]; nz = rb_c.size; area = abs(np.linalg.det(lat)) / lz
    for cfgn in CONFIGS:
        Pc, Pn = profs[(k, cfgn)], profs[(k + 600, cfgn)]
        mb = on_dft_z(Pc["rho_b"], Pc["z"], lz, nz) - on_dft_z(Pn["rho_b"], Pn["z"], lz, nz); mi = on_dft_z(Pc["rho_i"], Pc["z"], lz, nz) - on_dft_z(Pn["rho_i"], Pn["z"], lz, nz)
        db, di = rb_c - rb_n, ri_c - ri_n
        rec["response"][cfgn] = {"b": m1d(mb, db, lz), "i": m1d(mi, di, lz), "t": m1d(mb + mi, db + di, lz)}
    print(f"[{time.time() - T0:5.0f}s] pair {k} response: " + " | ".join(f"{c}: b L1 {rec['response'][c]['b']['L1']:.3f} phi {rec['response'][c]['b']['phi_rms_err']:.4f}; i L1 {rec['response'][c]['i']['L1']:.3f}; t phi {rec['response'][c]['t']['phi_rms_err']:.4f}" for c in CONFIGS), flush=True)
    RES.append(rec); json.dump(RES, open(OUT, "w"), indent=1)
print(f"[{time.time() - T0:5.0f}s] done", flush=True)
