"""Standalone checks of the parent solvent effective potential at the stage-1 field
(mace/modules/pb1d_vsolv.py) -- no model change, no training.

  C0  lambda_diel formula against the parent's polarisation: -d lambda_diel / dE = p(E).
  C1  the 1-D field path (plane-uniform phi) equals the 3-D torch_pb._field_quantities path:
      A_diel, A_ion values and |E_loc|(z).
  C2  FD of the partial at fixed phi, per term, in density directions.
  C3  live-phi evaluation equals the fixed-phi value and carries d/dphi*.
  C4  magnitudes: per-term rms/max, plane-mean |v|(z) peak against the cavity, smoothed values at atoms.
  C5  cost and peak memory: grad only; grad with create_graph; per term; node fields + backward
      with and without checkpointing.

Env: KIT_RUN, KIT_MODEL_OBJ, KIT_CKPT, KIT_SIDS (default "28,628"), KIT_OUT.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ.get("KIT_MACE_REPO",
                "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))
from ase.io import read
from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.pb1d_backend as PB
from mace.modules.pb1d_solver import EDEPS
from mace.modules import pb1d_vsolv as VS

RUN = os.environ["KIT_RUN"]; OUT = os.path.abspath(os.environ.get("KIT_OUT", os.path.join(RUN, "logs")))
SIDS = [int(x) for x in os.environ.get("KIT_SIDS", "28,628").split(",")]
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")
os.chdir(RUN)

device = torch_tools.init_device("cuda"); torch_tools.set_default_dtype("float64")
model = torch.load(f=os.environ["KIT_MODEL_OBJ"], map_location=device).to(device)
model.load_state_dict(torch.load(os.environ["KIT_CKPT"], map_location=device)["model"], strict=True)
EPOCH = int(os.environ["KIT_CKPT"].rsplit("epoch-", 1)[1].split(".")[0]); model.eval(); model._pb1d_epoch = EPOCH
for p in model.parameters():
    p.requires_grad_(False)

cap = {}
_clo = PB.closure_from_fields
def _clo_wrap(n_e_density, cvhar3, grid, params, tp):
    if "n_e" not in cap:
        cap.update(n_e=n_e_density.detach().clone(), cvhar3=cvhar3.detach().clone(), grid=grid, params=params, tp=tp)
    return _clo(n_e_density, cvhar3, grid, params, tp)
PB.closure_from_fields = _clo_wrap
_bk = PB.PB1DBackend.solve_graph
def _bk_wrap(self, *a, **k):
    r = _bk(self, *a, **k)
    if "phi_z" not in cap:
        cap.update(phi_z=r["phi_z"].detach().clone(), height=float(r["height"]), backend=self)
    return r
PB.PB1DBackend.solve_graph = _bk_wrap

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin",
                                    "sample_id": "sample_id", "fermi_level": "Fermi", "potential": "potential_diff",
                                    "solvated": "solvated"}, arrays_keys={"forces": "forces"})
atoms_by_sid = {}
for split in ("train", "val"):
    p = os.path.join(RUN, "data", f"{split}.xyz")
    if os.path.exists(p):
        for a in read(p, ":"):
            atoms_by_sid[int(a.info["sample_id"])] = a


def run_frame(sid):
    cap.clear()
    a = atoms_by_sid[sid]
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    with torch.no_grad():
        model(b.to_dict(), compute_force=False, training=False)
    return a


def grads_per_term(n_e, phi_z, grid, params, tp, sigma_b, eps_area, create_graph):
    a = VS.free_energies(n_e, phi_z, grid, params, tp, sigma_b, eps_area)
    dV = grid.volume / float(grid.ngrid)
    out = []
    for x in a:
        (gr,) = torch.autograd.grad(x, n_e, retain_graph=True, create_graph=create_graph, allow_unused=True)
        out.append((gr if gr is not None else torch.zeros_like(n_e)) / dV)
    return out, [float(x) for x in a]


print(f"[vsolv_eval] model @ epoch {EPOCH}; sids {SIDS}")
for sid in SIDS:
    a = run_frame(sid)
    grid, params, tp = cap["grid"], cap["params"], cap["tp"]
    n_e0 = cap["n_e"]; shape = tuple(n_e0.shape); nz = shape[-1]; dV = grid.volume / float(np.prod(shape))
    lz = cap["height"]
    phi_z = VS.fourier_resample_1d(cap["phi_z"], nz)
    sigma_b = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
    eps_area = float(os.environ.get("MACE_PB1D_AREA_EPS", "1e-30"))
    print(f"\n===================== sid {sid} ({'charged' if abs(float(a.info.get('total_charge',0)))>1e-6 else 'neutral'}, {len(a)} atoms) grid {shape} =====================")

    if sid == SIDS[0]:
        worst = 0.0
        for E in (1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 1.0):
            Et = torch.tensor([E], dtype=torch.float64, device=device)
            def lam_of_E(Ev):
                return VS.lambda_diel(tp._local_field_factor(Ev, params) * Ev, params, tp)
            h = 1e-6 * max(E, 1e-3)
            dl = (lam_of_E(Et + h) - lam_of_E(Et - h)) / (2 * h)
            f = tp._local_field_factor(Et, params); Eloc = f * Et
            g = tp._dielectric_g(float(params["PBETA"]) * Eloc, params)
            pE = (float(params["alpha0_rot"]) * g + float(params["alpha_pol"])) * Eloc / EDEPS
            worst = max(worst, float(abs(-dl - pE) / (abs(pE) + 1e-300)))
        print(f"--- C0: -d lambda_diel/dE vs parent p(E): {'PASS' if worst < 1e-4 else 'FAIL'} (worst rel {worst:.2e}) ---")

    # ---- C1: 1-D field path vs 3-D torch_pb path
    with torch.no_grad():
        s_ion, s_diel, s_cav = tp.create_cavity_torch(n_e0, grid, params)
        lam_d, lam_i, emag_loc = VS.field_terms_1d(phi_z, lz, params, tp, sigma_b)
        a_diel_1d = float(params["N_MOL"]) * float((s_diel.sum(dim=(0, 1)) * lam_d).sum()) * dV
        a_ion_1d = float(params["n_max"]) * float((s_ion.sum(dim=(0, 1)) * lam_i).sum()) * dV
        phi3 = phi_z[None, None, :].expand(*shape).contiguous()
        w_b3 = tp._normalized_gaussian_kernel_g(grid, sigma_b)
        fq = tp._field_quantities(phi3, s_ion, s_diel, grid, params, w_b3)
        a_diel_3d = float(params["N_MOL"]) * float((s_diel * VS.lambda_diel(fq["emag"], params, tp)).sum()) * dV
        a_ion_3d = float(params["n_max"]) * float((s_ion * VS.lambda_ion(phi3, params)).sum()) * dV
        d_emag = float((emag_loc - fq["emag"][0, 0, :]).abs().max())
        del phi3, fq, w_b3
    ok = abs(a_diel_1d - a_diel_3d) < 1e-6 and abs(a_ion_1d - a_ion_3d) < 1e-8 and d_emag < 1e-8
    print(f"--- C1: 1-D field path vs 3-D torch_pb path: A_diel {a_diel_1d:+.6f} vs {a_diel_3d:+.6f} (diff {a_diel_1d-a_diel_3d:+.2e}); "
          f"A_ion {a_ion_1d:+.6f} vs {a_ion_3d:+.6f} (diff {a_ion_1d-a_ion_3d:+.2e}); |E_loc(z)| max diff {d_emag:.2e} eV/A [{'PASS' if ok else 'CHECK'}] ---")

    # ---- C2: FD of the partial at fixed phi
    n_e = n_e0.clone().requires_grad_(True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    vs, As = grads_per_term(n_e, phi_z, grid, params, tp, sigma_b, eps_area, create_graph=False)
    torch.cuda.synchronize(); t_grad = time.time() - t0; mem_grad = torch.cuda.max_memory_allocated() / 2**30
    v_cav, v_diel, v_ion = [v.detach() for v in vs]
    print(f"--- values: A_cav {As[0]:+.4f}  A_diel {As[1]:+.4f}  A_ion {As[2]:+.4f} eV ---")
    print("--- C2: FD of the partial at fixed phi (central, per term); directions: blob at max |v_diel|, random smooth ---")
    rng = np.random.default_rng(0)
    c = np.unravel_index(int(torch.argmax(v_diel.abs().flatten())), shape)
    ax = [torch.arange(s, device=device, dtype=torch.float64) for s in shape]
    sp = (grid.volume / np.prod(shape)) ** (1 / 3)
    d2 = ((ax[0][:, None, None] - c[0]) ** 2 + (ax[1][None, :, None] - c[1]) ** 2 + (ax[2][None, None, :] - c[2]) ** 2) * sp ** 2
    blob = torch.exp(-0.5 * d2 / 0.8 ** 2)
    rnd = torch.tensor(rng.standard_normal(shape), device=device, dtype=torch.float64)
    rnd = grid.ifft_real(grid.fft(rnd) * torch.exp(-0.5 * (2 * math.pi) ** 2 * grid.gsq * 1.0 ** 2))
    eps = 1e-3 * float(params["NC_K"])
    for name, dn in (("shell blob", blob), ("random smooth", rnd)):
        dn = dn / (dn.abs().max() + 1e-300)
        with torch.no_grad():
            Ap = VS.free_energies(n_e0 + eps * dn, phi_z, grid, params, tp, sigma_b, eps_area)
            Am = VS.free_energies(n_e0 - eps * dn, phi_z, grid, params, tp, sigma_b, eps_area)
        line = f"   {name:14s}"
        for lbl, v, ap, am in (("cav", v_cav, Ap[0], Am[0]), ("diel", v_diel, Ap[1], Am[1]), ("ion", v_ion, Ap[2], Am[2])):
            fd = float((ap - am) / (2 * eps)); an = float((v * dn).sum() * dV)
            rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-300) * 2
            line += f" | {lbl}: FD {fd:+.4e} AD {an:+.4e} rel {rel:.1e}"
        print(line)
    del n_e, vs; torch.cuda.empty_cache()

    # ---- C3: live phi, and per-term / full-path memory
    phi_live = cap["phi_z"].clone().requires_grad_(True)
    n_e2 = n_e0.clone().requires_grad_(True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    vs2, _ = grads_per_term(n_e2, VS.fourier_resample_1d(phi_live, nz), grid, params, tp, sigma_b, eps_area, create_graph=True)
    torch.cuda.synchronize(); t_cg = time.time() - t0; mem_cg = torch.cuda.max_memory_allocated() / 2**30
    same = max(float((vs2[i].detach() - v).abs().max()) for i, v in enumerate((v_cav, v_diel, v_ion)))
    (gphi,) = torch.autograd.grad((vs2[0] + vs2[1] + vs2[2]).sum(), phi_live, allow_unused=True)
    print(f"--- C3: live-phi values equal fixed-phi to {same:.2e}; d(sum v)/dphi* exists: {gphi is not None} ---")
    del vs2, n_e2, phi_live; torch.cuda.empty_cache()
    mem_terms = []
    for i in range(3):
        n_e3 = n_e0.clone().requires_grad_(True); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        A3 = VS.free_energies(n_e3, phi_z, grid, params, tp, sigma_b, eps_area)
        torch.autograd.grad(A3[i], n_e3, create_graph=True, allow_unused=True); torch.cuda.synchronize()
        mem_terms.append(torch.cuda.max_memory_allocated() / 2**30)
        del n_e3, A3; torch.cuda.empty_cache()
    pos = torch.tensor(a.get_positions(), device=device, dtype=torch.float64)
    cell_t = torch.tensor(np.array(a.get_cell()), device=device, dtype=torch.float64)
    frac = torch.remainder(pos @ torch.linalg.inv(cell_t), 1.0)
    costs = {}
    for ck in (False, True):
        n_e4 = n_e0.clone().requires_grad_(True); ph4 = cap["phi_z"].clone().requires_grad_(True); fr4 = frac.clone().requires_grad_(True)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        nf = VS.vsolv_node_fields(n_e4, ph4, fr4, grid, params, tp, sigma_b, eps_area, list(model.field_feature_widths), checkpoint=ck)
        loss = (nf ** 2).sum()
        gs = torch.autograd.grad(loss, (n_e4, ph4, fr4), allow_unused=True); torch.cuda.synchronize()
        costs[ck] = (time.time() - t0, torch.cuda.max_memory_allocated() / 2**30, [g is not None for g in gs])
        del n_e4, ph4, fr4, nf, loss, gs; torch.cuda.empty_cache()

    # ---- C4: magnitudes
    v_tot = v_cav + v_diel + v_ion
    sym = np.array(a.get_chemical_symbols())
    print("--- C4: magnitudes (eV per electron) ---")
    for lbl, v in (("v_cav", v_cav), ("v_diel", v_diel), ("v_ion", v_ion), ("v_new", v_tot)):
        pz = v.abs().mean(dim=(0, 1)); zpk = int(torch.argmax(pz))
        print(f"   {lbl:6s}: grid rms {float(v.pow(2).mean().sqrt()):.4e} max {float(v.abs().max()):.4e}; plane-mean |v| peak {float(pz.max()):.4e} at z = {zpk*lz/nz:.2f} A")
    with torch.no_grad():
        sm_val, sm_grad = VS.smoothed_value_and_gradient(VS.VSOLV_SIGN * v_tot, grid, frac, float(model.field_feature_widths[0]))
    print(f"   receiver-smoothed (sigma {model.field_feature_widths[0]}) physics-sign value at atoms: rms O {float(sm_val[sym=='O'].pow(2).mean().sqrt()):.4e} "
          f"H {float(sm_val[sym=='H'].pow(2).mean().sqrt()):.4e} Ni {float(sm_val[sym=='Ni'].pow(2).mean().sqrt()):.4e} eV; max |value| {float(sm_val.abs().max()):.4e}; |grad| rms {float(sm_grad.pow(2).mean().sqrt()):.4e} eV/A")
    print(f"--- C5: cost -- grad only {t_grad:.3f} s / {mem_grad:.2f} GiB; grad+create_graph (live phi) {t_cg:.3f} s / {mem_cg:.2f} GiB; "
          f"per-term create_graph peak cav {mem_terms[0]:.2f} diel {mem_terms[1]:.2f} ion {mem_terms[2]:.2f} GiB; "
          f"node fields + backward to (n_e, phi, frac): no-ckpt {costs[False][0]:.3f} s / {costs[False][1]:.2f} GiB grads {costs[False][2]}, "
          f"ckpt {costs[True][0]:.3f} s / {costs[True][1]:.2f} GiB grads {costs[True][2]} ---")
    np.savez(os.path.join(OUT, f"vsolv_eval_sid{sid}.npz"), v_cav=v_cav.cpu().numpy().astype(np.float32),
             v_diel=v_diel.cpu().numpy().astype(np.float32), v_ion=v_ion.cpu().numpy().astype(np.float32),
             phi_z=phi_z.cpu().numpy(), A=np.array(As))
    torch.cuda.empty_cache()
print("\nDONE")
