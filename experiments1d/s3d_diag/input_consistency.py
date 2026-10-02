"""Input consistency of the production solvent inputs (reviewer 2026-10-02, priority item 1; no training).
(1) Electron bookkeeping on the model grid: the cavity density is clamp((neutral_v - net)/V, 0) while the Poisson source of the
    solute potential is the UNCLAMPED net (pb1d_backend.py lines 428-431). Per frame: NELECT (DFT INCAR), integral of the
    raw density, of the clamped density, of the clipped negative part; the same for the DFT net label pushed through the same
    assembly (control). Where the clipped electrons sit: by DFT cavity class (solute interior s_diel < 0.01, interface, solvent
    > 0.99), inside / outside the slab z-range, and as a plane profile (saved).
(2) Cavity confusion on the model grid (model vs DFT-label cavities, s_diel > 0.5): volume model-solvent/DFT-solute and the
    reverse, plus the plane profile of the difference.
(3) Poisson-source consistency: solute potential from the unclamped source (production) vs from the clamped density
    (consistent with the cavity): rms difference total / lateral / plane mean, against the model-vs-DFT solute error.
(4) Plane-mean ramp and dipole handling (native grid): slope per unit dipole of the production sawtooth cdipol_potential_1d
    (c_unit, indmin, center_z as the backend); dipoles about center_z: DFT solute (net label), DFT solvent (RHOB + RHOION), DFT
    total; model solute (net_values) and the solver's own val_ion_dipole_z; measured slope of <cv_assembly^DFT - phi_sol^DFT>
    and <cv_model - phi_sol^DFT>; residual rms after adding the sawtooth with each candidate dipole (which dipole does the
    DFT potential's correction correspond to; what remains for the network).
Usage: python input_consistency.py <out.json>; env KIT_PAIRS, KIT_DEVICE, KIT_DFT, KIT_MODEL
"""
from __future__ import annotations

import json
import os
import re
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
from mace.modules.pb1d_solver import cdipol_potential_1d
from mace.modules.solvent3d import _interp3_periodic

OUT = sys.argv[1] if len(sys.argv) > 1 else "input_consistency.json"
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
_orig_clo = PB.closure_from_fields; _orig_gto = backend._gto_net_density_g; _orig_solve = PB.Solver1D.solve


def clo_cap(n_e, cv, grid, params_, tp_):
    CAP["ne"] = n_e.detach().clone(); CAP["cv"] = cv.detach().clone(); CAP["grid"] = grid
    return _orig_clo(n_e, cv, grid, params_, tp_)


def gto_cap(grid, pos_frac, coeffs, sigmas):
    out = _orig_gto(grid, pos_frac, coeffs, sigmas)
    if "net_g" not in CAP:                       # the first GTO assembly of a forward is the solute net density
        CAP["net_g"] = out.detach().clone()
    return out


def solve_cap(self, **kw):
    CAP["kw"] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kw.items()}
    return _orig_solve(self, **kw)


PB.closure_from_fields = clo_cap; backend._gto_net_density_g = gto_cap; PB.Solver1D.solve = solve_cap


def forward(sid):
    CAP.pop("net_g", None)
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


def nelect(sid):
    txt = open(f"{dft_dir(sid)}/INCAR").read()
    m = re.search(r"NELECT\s*=\s*([0-9.]+)", txt)
    return float(m.group(1))


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


def line_fit(d, z):
    A = np.stack([z, np.ones_like(z)], 1); c, *_ = np.linalg.lstsq(A, d, rcond=None); r = d - A @ c
    return float(c[0]), float(np.sqrt((r ** 2).mean()))


def classes(s):
    return {"interior": s < 0.01, "interface": (s >= 0.01) & (s <= 0.99), "solvent": s > 0.99}


RES = []
for kpair in PAIRS:
    for sid in (kpair, kpair + 600):
        atoms = ATOMS[sid]; q = float(atoms.info["total_charge"])
        res = forward(sid); grid = CAP["grid"]; ne_cl = CAP["ne"]; cv_prod = CAP["cv"]; net_g = CAP["net_g"]; kw = CAP["kw"]
        V = float(grid.volume); mshape = tuple(ne_cl.shape); nzm = mshape[2]; dVm = V / float(np.prod(mshape))
        cell = np.asarray(atoms.get_cell()); lz = float(cell[2, 2]); zm = np.arange(nzm) * lz / nzm
        bl_row = backend._bl_index.get(sid)
        fields_bl = torch.from_numpy(np.ascontiguousarray(backend._bl_arr[bl_row][backend._bl_take])).to(dev).double()
        neutral_v, phi_base = fields_bl[0], fields_bl[1]; del fields_bl
        out = dict(sid=sid, pair=kpair, q=q, split=atoms.info["_split"])
        with torch.no_grad():
            # ---------------- (1) electron bookkeeping, model grid
            net_values = grid.ifft_real(net_g)
            ne_raw = (neutral_v - net_values) / V
            assert float((torch.clamp(ne_raw, min=0.0) - ne_cl).abs().max()) < 1e-9, "clamped raw density != captured cavity density"
            net_dft = resample_tri(torch.as_tensor(np.asarray(np.load(ENT_N[sid]["path"], mmap_mode="r"), dtype=np.float64), device=dev).permute(2, 1, 0).contiguous(), mshape) * V
            ne_dft_raw = (neutral_v - net_dft) / V
            zpos = atoms.get_positions()[:, 2]; z_lo, z_hi = float(zpos.min()) - 1.0, float(zpos.max()) + 1.0
            in_slab = torch.as_tensor((zm >= z_lo) & (zm <= z_hi), device=dev)[None, None, :].expand(mshape)
            s_i_d, s_d_d, _ = tp.create_cavity_torch(torch.clamp(ne_dft_raw, min=0.0), grid, params); s_d_d = torch.clamp(s_d_d, 0.0, 1.0)
            s_i_m, s_d_m, _ = tp.create_cavity_torch(ne_cl, grid, params); s_d_m = torch.clamp(s_d_m, 0.0, 1.0)
            clipped = torch.clamp(-ne_raw, min=0.0)                      # removed negative density (e/A^3, positive)
            clipped_dft = torch.clamp(-ne_dft_raw, min=0.0)
            e = dict(nelect=nelect(sid), total_charge=q,
                     model_raw=float(ne_raw.sum() * dVm), model_clamped=float(ne_cl.sum() * dVm), model_clipped=float(clipped.sum() * dVm),
                     model_neg_points_frac=float((ne_raw < 0).double().mean()), model_neg_min=float(ne_raw.min()),
                     dft_raw=float(ne_dft_raw.sum() * dVm), dft_clamped=float(torch.clamp(ne_dft_raw, min=0.0).sum() * dVm), dft_clipped=float(clipped_dft.sum() * dVm),
                     net_model_e=float(net_values.sum() / net_values.numel()), net_dft_e=float(net_dft.sum() / net_dft.numel()), neutral_e=float(neutral_v.sum() / neutral_v.numel()))
            loc = {}
            tot = float(clipped.sum()) + 1e-300
            for cname, mask in classes(s_d_d).items():
                loc[f"dft_{cname}"] = float(clipped[mask].sum() / tot)
            for cname, mask in classes(s_d_m).items():
                loc[f"model_{cname}"] = float(clipped[mask].sum() / tot)
            loc["in_slab_zrange"] = float(clipped[in_slab].sum() / tot)
            prof_clip = (clipped.mean((0, 1)) * V / nzm).cpu().numpy()              # electrons per plane
            prof_ne_m = (ne_cl.mean((0, 1)) * V / nzm).cpu().numpy(); prof_ne_d = (torch.clamp(ne_dft_raw, min=0.0).mean((0, 1)) * V / nzm).cpu().numpy()
            out["electrons"] = e; out["clipped_location"] = loc
            # ---------------- (2) cavity confusion
            m_solv, d_solv = s_d_m > 0.5, s_d_d > 0.5
            conf = dict(model_solvent_dft_solute_A3=float((m_solv & ~d_solv).double().sum() * dVm), model_solute_dft_solvent_A3=float((~m_solv & d_solv).double().sum() * dVm),
                        model_solvent_A3=float(m_solv.double().sum() * dVm), dft_solvent_A3=float(d_solv.double().sum() * dVm),
                        s_diel_L1=float((s_d_m - s_d_d).abs().sum() / s_d_d.sum()), s_diel_vol_diff_A3=float((s_d_m - s_d_d).sum() * dVm),
                        s_ion_vol_diff_A3=float((torch.clamp(s_i_m, 0, 1) - torch.clamp(s_i_d, 0, 1)).sum() * dVm))
            prof_sdiff = ((s_d_m - s_d_d).mean((0, 1)) * V / nzm).cpu().numpy()      # A^3 per plane
            out["cavity"] = conf
            # ---------------- (3) Poisson-source consistency
            cv_cl = phi_base - grid.ifft_real(grid.l0_inv_op(grid.fft(neutral_v - ne_cl * V)))     # source consistent with the clamped cavity density
            d3 = cv_cl - cv_prod; dpm = d3.mean((0, 1)).cpu().numpy(); dl = d3 - d3.mean((0, 1))[None, None, :]
            sl, lr = line_fit(dpm, zm)
            out["poisson_source"] = dict(clamped_minus_unclamped_rms=float(d3.pow(2).mean().sqrt()), lateral_rms=float(dl.pow(2).mean().sqrt()),
                                         pa_slope_eV_per_A=sl, pa_rms_lineremoved=lr, cv_prod_lateral_rms=float((cv_prod - cv_prod.mean((0, 1))[None, None, :]).pow(2).mean().sqrt()))
            del d3, dl
            # ---------------- (4) ramp / dipole on the native grid
            d = dft_dir(sid)
            lat, chg = read_grid_fast(f"{d}/CHGCAR"); _, phi3 = read_grid_fast(f"{d}/PHI"); _, rb_raw = read_grid_fast(f"{d}/RHOB"); _, ri_raw = read_grid_fast(f"{d}/RHOION")
            shape = tuple(chg.shape); nz = shape[2]; Vn = float(abs(np.linalg.det(lat))); dV = Vn / chg.numel(); zn = np.arange(nz) * lz / nz
            gd = tp.TorchGrid(lat, shape, device=str(dev), dtype=torch.float64, rspec=True)
            rho_b = -(rb_raw / Vn); rho_i = -(ri_raw / Vn); del rb_raw, ri_raw, chg
            phi_sol_dft = phi3 - gd.ifft_real(gd.l0_inv_op(gd.fft(-Vn * (rho_b + rho_i)))); del phi3
            net_nat = torch.as_tensor(np.asarray(np.load(ENT_N[sid]["path"], mmap_mode="r"), dtype=np.float64), device=dev).permute(2, 1, 0).contiguous()   # e/A^3
            cv_asm = resample_fourier(phi_base, shape) - gd.ifft_real(gd.l0_inv_op(gd.fft(net_nat * Vn)))   # production assembly with the DFT net, native grid
            cv_mod = resample_fourier(cv_prod, shape)
            cell_np = cell.astype(np.float64); c_unit = backend._c_unit(cell_np)
            center_z = 0.5 * float(cell_np[0, 2] + cell_np[1, 2] + cell_np[2, 2]); nouth = nz // 2; indmin = int((nouth + int(0.5 * nz) + 10 * nz) % nz + 1)
            zt = torch.as_tensor(zn, device=dev)
            saw1 = cdipol_potential_1d(nz, lz, torch.tensor(c_unit, device=dev), indmin, dev).cpu().numpy()      # sawtooth per unit dipole
            lin = np.abs(np.abs(((np.arange(1, nz + 1) - indmin + nz) % nz) - nouth) - nouth) > 4.0 + 1e-9        # linear region (outside the cutoff smoothing)
            slope_per_dip, _ = line_fit(saw1[lin], zn[lin])
            dip = lambda rho3: float((rho3 * Vn * zt[None, None, :]).mean() - (rho3 * Vn).mean() * center_z)       # e*A about center_z
            p = dict(dft_solute=dip(net_nat), dft_bound=dip(rho_b), dft_ion=dip(rho_i), model_solute=float((net_values * zt.new_tensor(zm)[None, None, :]).mean() - net_values.mean() * center_z),
                     solver_val_ion_dipole_z=float(kw["val_ion_dipole_z"]) if "val_ion_dipole_z" in kw else None, solver_c_unit=float(kw.get("c_unit", c_unit)), c_unit=c_unit, center_z=center_z, indmin=indmin)
            p["dft_solvent"] = p["dft_bound"] + p["dft_ion"]; p["dft_total"] = p["dft_solute"] + p["dft_solvent"]
            pm_sol = phi_sol_dft.mean((0, 1)).cpu().numpy(); pm_asm = cv_asm.mean((0, 1)).cpu().numpy(); pm_mod = cv_mod.mean((0, 1)).cpu().numpy()
            sl_asm, lr_asm = line_fit(pm_asm - pm_sol, zn); sl_mod, lr_mod = line_fit(pm_mod - pm_sol, zn)
            ramp = dict(slope_per_unit_dipole=slope_per_dip, measured_slope_assembly=sl_asm, measured_slope_model=sl_mod, lineremoved_assembly=lr_asm, lineremoved_model=lr_mod,
                        implied_dipole_assembly=sl_asm / slope_per_dip, implied_dipole_model=sl_mod / slope_per_dip, pa_rms_dft_solute=float(np.sqrt(((pm_sol - pm_sol.mean()) ** 2).mean())))
            after = {}
            for nm, pv in (("dft_total", p["dft_total"]), ("dft_solute", p["dft_solute"]), ("dft_solvent", p["dft_solvent"])):
                saw = cdipol_potential_1d(nz, lz, torch.tensor(c_unit * pv, device=dev), indmin, dev).cpu().numpy()
                r = pm_asm + saw - pm_sol; after[f"assembly+saw({nm})"] = dict(rms_meanremoved=float(np.sqrt(((r - r.mean()) ** 2).mean())), slope_left=line_fit(r, zn)[0])
            for nm, pv in (("model_solute+dft_solvent", p["model_solute"] + p["dft_solvent"]), ("dft_total", p["dft_total"])):
                saw = cdipol_potential_1d(nz, lz, torch.tensor(c_unit * pv, device=dev), indmin, dev).cpu().numpy()
                r = pm_mod + saw - pm_sol; after[f"model+saw({nm})"] = dict(rms_meanremoved=float(np.sqrt(((r - r.mean()) ** 2).mean())), slope_left=line_fit(r, zn)[0])
            lat_asm = float((cv_asm - cv_asm.mean((0, 1))[None, None, :] - (phi_sol_dft - phi_sol_dft.mean((0, 1))[None, None, :])).pow(2).mean().sqrt())
            lat_mod = float((cv_mod - cv_mod.mean((0, 1))[None, None, :] - (phi_sol_dft - phi_sol_dft.mean((0, 1))[None, None, :])).pow(2).mean().sqrt())
            out["dipole"] = p; out["ramp"] = ramp; out["ramp_after_sawtooth"] = after; out["lateral_rms_err"] = dict(assembly=lat_asm, model=lat_mod)
            np.savez_compressed(f"ic_profiles_sid{sid}.npz", z_model=zm, clipped_e_per_plane=prof_clip, ne_model_e_per_plane=prof_ne_m, ne_dft_e_per_plane=prof_ne_d,
                                sdiel_diff_A3_per_plane=prof_sdiff, z_native=zn, pm_phi_sol_dft=pm_sol, pm_cv_assembly=pm_asm, pm_cv_model=pm_mod, saw_unit=saw1)
            del phi_sol_dft, cv_asm, cv_mod, rho_b, rho_i, net_nat
        RES.append(out); json.dump(RES, open(OUT, "w"), indent=1)
        print(f"[{time.time() - T0:5.0f}s] sid {sid} (q {q:+.3f}) NELECT {e['nelect']:.1f} | model raw {e['model_raw']:.3f} clamped {e['model_clamped']:.3f} clipped {e['model_clipped']:.3f} (neg points {e['model_neg_points_frac']*100:.2f} %, min {e['model_neg_min']:.2e}) | "
              f"DFT raw {e['dft_raw']:.3f} clamped {e['dft_clamped']:.3f} clipped {e['dft_clipped']:.4f} | clipped location (DFT cavity) interior {loc['dft_interior']:.2f} interface {loc['dft_interface']:.2f} solvent {loc['dft_solvent']:.2f}; in slab z-range {loc['in_slab_zrange']:.2f}", flush=True)
        print(f"         cavity: model-solvent/DFT-solute {conf['model_solvent_dft_solute_A3']:.1f} A^3, model-solute/DFT-solvent {conf['model_solute_dft_solvent_A3']:.1f} A^3, s_diel vol diff {conf['s_diel_vol_diff_A3']:+.1f}, s_ion vol diff {conf['s_ion_vol_diff_A3']:+.1f} | "
              f"Poisson source clamped-unclamped: rms {out['poisson_source']['clamped_minus_unclamped_rms']:.4f} eV, lateral {out['poisson_source']['lateral_rms']:.4f}, pa slope {out['poisson_source']['pa_slope_eV_per_A']:+.5f}, line-removed {out['poisson_source']['pa_rms_lineremoved']:.4f}", flush=True)
        print(f"         dipoles (e*A): DFT solute {p['dft_solute']:+.3f} bound {p['dft_bound']:+.3f} ion {p['dft_ion']:+.3f} total {p['dft_total']:+.3f} | model solute {p['model_solute']:+.3f} (solver val_dip {p['solver_val_ion_dipole_z']}) | slope/dipole {slope_per_dip:+.5f} eV/A per e*A | "
              f"measured slope assembly {sl_asm:+.5f} (implied dip {ramp['implied_dipole_assembly']:+.3f}) model {sl_mod:+.5f} (implied {ramp['implied_dipole_model']:+.3f}); line-removed asm {lr_asm:.4f} model {lr_mod:.4f}", flush=True)
        print("         after sawtooth: " + " | ".join(f"{k}: rms {v['rms_meanremoved']:.4f} slope left {v['slope_left']:+.5f}" for k, v in after.items()) + f" | lateral rms err assembly {lat_asm:.4f} model {lat_mod:.4f}", flush=True)
print(f"[{time.time() - T0:5.0f}s] done", flush=True)
