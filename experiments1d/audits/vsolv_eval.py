"""The parent (VASPsol++) solvent effective potential evaluated at the stage-1 field,
standalone, with its correctness checks -- no model change, no training.

Definition (user 2026-09-21, tightened route):
    v_new(r) = d(A_cav + A_diel + A_ion)/dn_e(r) |_{phi = phi_approx}
with phi_approx = the stage-1 1-D total potential phi*(z) (electron-PE convention, as the
solver and VASP's CVHAR use it) broadcast over the plane, and the three free energies in the
parent's own definitions (paper eqs. 25, 35-38, 48/54):
    A_cav  = TAU * int |grad S_cav[n_e]| dV                         (AREA_EPS-regularised, as in pb1d_backend)
    A_diel = N_MOL * int S_diel[n_e] * lambda_diel(E_loc) dV,
        lambda_diel = lambda_rot + lambda_pol + lambda_sic,
        lambda_rot = -(1/beta) ln( sinh(y)/y ),  y = PBETA * E_loc,
        lambda_pol = -1/2 alpha_pol E_loc^2 / EDEPS,
        lambda_sic = +1/2 invalpha_sic ((alpha0_rot g(y) + alpha_pol) E_loc)^2 / EDEPS,
        E_loc = f_loc(E) * E,  E = |w_b * grad phi|      (torch_pb._field_quantities)
    A_ion  = n_max * int S_ion[n_e] * lambda_ion(phi) dV,
        lambda_ion = -(1/beta) ln(1 - theta_b + theta_b cosh(ZBETA phi))       (lattice gas, theta_b > 0)
The partial derivative is taken with phi an INDEPENDENT input: g(n, phi) = d_n A(n, phi); the
model will call g(n, phi*(n)) so that forces / loss gradients keep the stage-1 dependence.
(autograd.grad on A(n, phi*(n)) would give the TOTAL derivative -- that is exactly what this
structure avoids.)

Checks printed:
  C0  lambda_diel formula against the parent's polarisation: -d lambda_diel / dE = p(E) (envelope
      theorem), scalar FD over a range of fields.
  C1  the partial: FD in density directions at fixed phi, per term.
  C2  outer dependence: v computed with a live phi carries a graph to phi (grad exists) and
      equals the fixed-phi value.
  C3  magnitudes: per-term rms/max on the grid, plane-averaged |v|(z) peak position against the
      cavity, values at O/H/Ni atoms, against the reaction potential's scale.
  C4  cost: wall time and peak GPU memory of one g evaluation (grad, and grad with create_graph).

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

# ---- capture stage 1: closure inputs (n_e, cvhar3, grid, params, tp) and the solver's phi
cap = {}
_clo = PB.closure_from_fields
def _clo_wrap(n_e_density, cvhar3, grid, params, tp):
    if "n_e" not in cap:   # first call of the forward = stage 1 (use_head=False)
        cap.update(n_e=n_e_density.detach().clone(), cvhar3=cvhar3.detach().clone(), grid=grid, params=params, tp=tp)
    return _clo(n_e_density, cvhar3, grid, params, tp)
PB.closure_from_fields = _clo_wrap
_bk = PB.PB1DBackend.solve_graph
def _bk_wrap(self, *a, **k):
    r = _bk(self, *a, **k)
    if "phi_z" not in cap:
        cap.update(phi_z=r["phi_z"].detach().clone(), z=r["z"].detach().clone(), height=float(r["height"]), backend=self)
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


def fourier_resample(f, n_new):
    n = f.shape[0]; F = torch.fft.rfft(f) / n
    G = torch.zeros(n_new // 2 + 1, dtype=F.dtype, device=f.device); m = min(F.shape[0], G.shape[0]); G[:m] = F[:m]
    return torch.fft.irfft(G * n_new, n=n_new)


def lambda_diel(emag_loc, params):
    """Parent molecular dielectric free energy at local field magnitude emag_loc (eV/A)."""
    invb = float(params["invBETA"]); pbeta = float(params["PBETA"])
    a0, ap, isic = float(params["alpha0_rot"]), float(params["alpha_pol"]), float(params["invalpha_sic"])
    y = pbeta * emag_loc
    # ln(sinh y / y): series below 1e-3, direct (log-sum-exp form) above
    small = y < 1.0e-3
    ys = torch.where(small, torch.ones_like(y), y)
    ln_direct = torch.log(torch.sinh(torch.clamp(ys, max=700.0)) / ys)
    ln_series = y * y / 6.0 - y ** 4 / 180.0
    lam_rot = -invb * torch.where(small, ln_series, ln_direct) if bool(params["LNLDIEL"]) else -invb * (y * y / 6.0)
    g = cap["tp"]._dielectric_g(y, params) if bool(params["LNLDIEL"]) else torch.ones_like(y)
    lam_pol = -0.5 * ap * emag_loc ** 2 / EDEPS
    lam_sic = 0.5 * isic * ((a0 * g + ap) * emag_loc) ** 2 / EDEPS
    return lam_rot + lam_pol + lam_sic, g


def lambda_ion(phi, params):
    invb = float(params["invBETA"]); zb = float(params["ZBETA"]); th = float(params["theta_b"])
    assert bool(params["LION"]) and bool(params["LNLION"]) and th > 0.0, "expects the lattice-gas ionic model"
    x = torch.clamp(zb * phi, -100.0, 100.0)
    return -invb * torch.log(1.0 - th + th * torch.cosh(x))


def free_energies(n_e, phi3, grid, params, tp, w_b, eps_area):
    """A_cav, A_diel, A_ion at fixed phi3; n_e is the differentiated input."""
    s_ion, s_diel, s_cav = tp.create_cavity_torch(n_e, grid, params)
    dV = grid.volume / float(np.prod(n_e.shape))
    # pb1d_backend's exact expression: the Stern box mask is applied a second time to s_cav
    # ("Stern mask doubled") before the spectral gradient and the AREA_EPS floor
    if bool(params["LVAC"]) and float(params["SOL_Z1"]) > float(params["SOL_Z0"]):
        m_ion3 = tp._smooth_box(grid, float(params["SOL_Z0"]) + float(params["D_STERN"]),
                                float(params["SOL_Z1"]) - float(params["D_STERN"]), float(params["SOL_SIGMA"]))
        s_cav = s_cav * m_ion3
    gx, gy, gz, _ = grid.grad_from_recip(grid.fft(s_cav))
    a_cav = float(params["TAU"]) * torch.sqrt(gx * gx + gy * gy + gz * gz + eps_area).sum() * dV
    fq = tp._field_quantities(phi3, s_ion, s_diel, grid, params, w_b)   # emag includes f_loc
    lam_d, _ = lambda_diel(fq["emag"], params)
    a_diel = float(params["N_MOL"]) * (s_diel * lam_d).sum() * dV
    a_ion = float(params["n_max"]) * (s_ion * lambda_ion(phi3, params)).sum() * dV
    return a_cav, a_diel, a_ion, (s_ion, s_diel, s_cav, fq["emag"])


def g_partial(n_e, phi3, grid, params, tp, w_b, eps_area, create_graph=False):
    """v_term(r) = dA_term/dn_e(r) at fixed phi3 (eV per electron); n_e must require grad."""
    a_cav, a_diel, a_ion, aux = free_energies(n_e, phi3, grid, params, tp, w_b, eps_area)
    dV = grid.volume / float(np.prod(n_e.shape))
    vs = []
    for a in (a_cav, a_diel, a_ion):
        (gr,) = torch.autograd.grad(a, n_e, retain_graph=True, create_graph=create_graph, allow_unused=True)
        vs.append((gr if gr is not None else torch.zeros_like(n_e)) / dV)
    return vs, (a_cav, a_diel, a_ion), aux


def run_frame(sid):
    cap.clear()
    a = atoms_by_sid[sid]
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
    del pred
    return a


print(f"[vsolv_eval] model @ epoch {EPOCH}; sids {SIDS}")
for sid in SIDS:
    a = run_frame(sid)
    grid, params, tp = cap["grid"], cap["params"], cap["tp"]
    n_e0 = cap["n_e"]                       # (nx, ny, nz) e/A^3, stage-1 input density
    shape = tuple(n_e0.shape); nz = shape[-1]; dV = grid.volume / float(np.prod(shape))
    phi_z_solver = cap["phi_z"]             # solver grid (upsampled), electron-PE eV
    phi_z = fourier_resample(phi_z_solver, nz)
    phi3_fixed = phi_z[None, None, :].expand(*shape).contiguous()
    sigma_b = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
    w_b = tp._normalized_gaussian_kernel_g(grid, sigma_b)
    eps_area = float(os.environ.get("MACE_PB1D_AREA_EPS", "1e-30"))
    print(f"\n===================== sid {sid} ({'charged' if abs(float(a.info.get('total_charge',0)))>1e-6 else 'neutral'}, "
          f"{len(a)} atoms) grid {shape} dV {dV:.4e} A^3 =====================")
    print(f"   params: TAU {params['TAU']} N_MOL {params['N_MOL']} n_max {params['n_max']} theta_b {params['theta_b']} "
          f"PBETA {params['PBETA']} alpha0_rot {params['alpha0_rot']} alpha_pol {params['alpha_pol']} invalpha_sic {params['invalpha_sic']} "
          f"LNLDIEL {params['LNLDIEL']} LNLION {params['LNLION']} sigma_b {sigma_b}")

    # ---- C0: lambda_diel formula vs the parent's polarisation, scalar FD
    if sid == SIDS[0]:
        print("\n--- C0: -d lambda_diel/dE (macroscopic E) against the parent's p(E) = (alpha0 g + alpha_pol) E_loc / EDEPS ---")
        worst = 0.0
        for E in (1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 1.0):
            Et = torch.tensor([E], dtype=torch.float64, device=device)
            def lam_of_E(Ev):
                f = tp._local_field_factor(Ev, params)
                lam, _ = lambda_diel(f * Ev, params); return lam
            h = 1e-6 * max(E, 1e-3)
            dl = (lam_of_E(Et + h) - lam_of_E(Et - h)) / (2 * h)
            f = tp._local_field_factor(Et, params); Eloc = f * Et
            _, g = lambda_diel(Eloc, params)
            pE = (float(params["alpha0_rot"]) * g + float(params["alpha_pol"])) * Eloc / EDEPS
            rel = float(abs(-dl - pE) / (abs(pE) + 1e-300)); worst = max(worst, rel)
            print(f"   E {E:8.4f} eV/A: f_loc {float(f):.5f}  -dlam/dE {float(-dl):+.6e}  p(E) {float(pE):+.6e}  rel diff {rel:.2e}")
        print(f"   C0 {'PASS' if worst < 1e-4 else 'FAIL'} (worst rel {worst:.2e})")

    # ---- C1: the partial by FD in density directions at fixed phi
    n_e = n_e0.clone().requires_grad_(True)
    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    vs, As, aux = g_partial(n_e, phi3_fixed, grid, params, tp, w_b, eps_area)
    torch.cuda.synchronize(); t_grad = time.time() - t0; mem_grad = torch.cuda.max_memory_allocated() / 2**30
    v_cav, v_diel, v_ion = vs
    a_cav, a_diel, a_ion = [float(x) for x in As]
    s_ion, s_diel, s_cav, emag = [x.detach() for x in aux]
    print(f"\n--- values: A_cav {a_cav:+.4f}  A_diel {a_diel:+.4f}  A_ion {a_ion:+.4f} eV  (backend tau*A for this frame is in RUNS; "
          f"E_loc max {float(emag.max()):.3f} eV/A, mean in s_diel>0.5 {float(emag[s_diel>0.5].mean()):.4f})")
    print("--- C1: FD of the partial at fixed phi (central, per term); directions: transition-shell blob, bulk-solvent blob, random smooth ---")
    rng = np.random.default_rng(0)
    # transition shell mask of the dielectric cavity
    shell = ((s_diel > 0.1) & (s_diel < 0.9)).detach()
    idx_shell = torch.nonzero(shell.flatten()).flatten()
    zz = torch.arange(nz, device=device, dtype=torch.float64)
    def blob(center_flat):
        c = np.unravel_index(int(center_flat), shape)
        gxs = [torch.arange(s, device=device, dtype=torch.float64) for s in shape]
        d2 = sum(((gxs[i][(slice(None),) + (None,) * (2 - i)] if i == 0 else gxs[i][(None,) * i + (slice(None),) + (None,) * (2 - i)]) - c[i]) ** 2
                 * (grid.volume / np.prod(shape)) ** (2/3) for i in range(3))
        return torch.exp(-0.5 * d2 / (0.8 ** 2))
    dirs = []
    if idx_shell.numel() > 0:
        dirs.append(("shell blob", blob(idx_shell[rng.integers(idx_shell.numel())])))
    bulk = torch.nonzero(((s_diel > 0.99) & (n_e0 < 1e-5)).flatten()).flatten()
    if bulk.numel() > 0:
        dirs.append(("bulk-solvent blob", blob(bulk[rng.integers(bulk.numel())])))
    rnd = torch.tensor(rng.standard_normal(shape), device=device, dtype=torch.float64)
    rnd = grid.ifft_real(grid.fft(rnd) * torch.exp(-0.5 * (2 * math.pi) ** 2 * grid.gsq * 1.0 ** 2))  # smooth, sigma 1 A
    dirs.append(("random smooth", rnd))
    for name, dn in dirs:
        dn = dn / (dn.abs().max() + 1e-300)
        eps = 1e-3 * float(params["NC_K"])   # small against the cavity cutoff density (the shell is where the terms live)
        with torch.no_grad():
            Ap = free_energies(n_e0 + eps * dn, phi3_fixed, grid, params, tp, w_b, eps_area)[:3]
            Am = free_energies(n_e0 - eps * dn, phi3_fixed, grid, params, tp, w_b, eps_area)[:3]
        line = f"   {name:18s}"
        for (lbl, v, ap, am) in (("cav", v_cav, Ap[0], Am[0]), ("diel", v_diel, Ap[1], Am[1]), ("ion", v_ion, Ap[2], Am[2])):
            fd = float((ap - am) / (2 * eps)); an = float((v.detach() * dn).sum() * dV)
            rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-300) * 2
            line += f" | {lbl}: FD {fd:+.4e} analytic {an:+.4e} rel {rel:.1e}"
        print(line)

    # ---- C2: outer dependence retained when phi is live
    phi_live = phi_z_solver.clone().requires_grad_(True)
    phi3_live = fourier_resample(phi_live, nz)[None, None, :].expand(*shape).contiguous()
    n_e2 = n_e0.clone().requires_grad_(True)
    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    vs2, _, _ = g_partial(n_e2, phi3_live, grid, params, tp, w_b, eps_area, create_graph=True)
    torch.cuda.synchronize(); t_cg = time.time() - t0; mem_cg = torch.cuda.max_memory_allocated() / 2**30
    same = max(float((vs2[i] - vs[i]).abs().max()) for i in range(3))
    probe = (vs2[0] + vs2[1] + vs2[2]).sum()
    (gphi,) = torch.autograd.grad(probe, phi_live, allow_unused=True)
    print(f"\n--- C2: live-phi evaluation equals fixed-phi values to {same:.2e} eV; d(sum v)/d phi* exists: "
          f"{gphi is not None} (|.|max {float(gphi.abs().max()) if gphi is not None else float('nan'):.3e}) ---")

    # ---- C3: magnitudes
    v_tot = (v_cav + v_diel + v_ion).detach()
    pos = torch.tensor(a.get_positions(), device=device, dtype=torch.float64)
    cell = torch.tensor(np.array(a.get_cell()), device=device, dtype=torch.float64)
    frac = torch.linalg.solve(cell.T, pos.T).T % 1.0
    ijk = torch.stack([torch.clamp((frac[:, i] * shape[i]).long(), 0, shape[i] - 1) for i in range(3)], dim=1)
    sym = np.array(a.get_chemical_symbols())
    def at_atoms(v):
        return v[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    phi_solv_z = (phi_z - fourier_resample(cap["cvhar3"].mean(dim=(0, 1)), nz)).detach()   # reaction potential incl. cvdip, 1-D
    print("--- C3: magnitudes (eV per electron) ---")
    for lbl, v in (("v_cav", v_cav), ("v_diel", v_diel), ("v_ion", v_ion), ("v_new", v_tot)):
        v = v.detach(); pz = v.abs().mean(dim=(0, 1)); zpk = int(torch.argmax(pz))
        va = at_atoms(v)
        print(f"   {lbl:6s}: grid rms {float(v.pow(2).mean().sqrt()):.4e} max {float(v.abs().max()):.4e}; plane-avg |v| peak {float(pz.max()):.4e} at z index {zpk} "
              f"(z = {zpk * cap['height'] / nz:.2f} A); at atoms rms O {float(va[sym=='O'].pow(2).mean().sqrt()) if (sym=='O').any() else 0:.4e} "
              f"H {float(va[sym=='H'].pow(2).mean().sqrt()) if (sym=='H').any() else 0:.4e} Ni {float(va[sym=='Ni'].pow(2).mean().sqrt()) if (sym=='Ni').any() else 0:.4e}")
    zpk_s = int(torch.argmax(s_diel.mean(dim=(0, 1)) * (1 - s_diel.mean(dim=(0, 1)))))
    print(f"   dielectric transition (plane-avg S(1-S) peak) at z index {zpk_s} (z = {zpk_s * cap['height'] / nz:.2f} A); "
          f"reaction potential phi_solv(z): rms {float(phi_solv_z.pow(2).mean().sqrt()):.4e}, max {float(phi_solv_z.abs().max()):.4e} eV; at atoms rms {float(phi_solv_z[ijk[:,2]].pow(2).mean().sqrt()):.4e}")
    print(f"--- C4: cost -- g (grad only): {t_grad:.3f} s, peak {mem_grad:.2f} GiB; g with create_graph (live phi): {t_cg:.3f} s, peak {mem_cg:.2f} GiB ---")
    np.savez(os.path.join(OUT, f"vsolv_eval_sid{sid}.npz"), v_cav=v_cav.detach().cpu().numpy().astype(np.float32),
             v_diel=v_diel.detach().cpu().numpy().astype(np.float32), v_ion=v_ion.detach().cpu().numpy().astype(np.float32),
             s_diel=s_diel.cpu().numpy().astype(np.float32), phi_z=phi_z.cpu().numpy(), A=np.array([a_cav, a_diel, a_ion]))
    del vs, vs2, n_e, n_e2; torch.cuda.empty_cache()
print("\nDONE")
