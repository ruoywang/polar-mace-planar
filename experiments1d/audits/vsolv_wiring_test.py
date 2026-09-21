"""Acceptance of the wired stage-1 effective-potential input (solvent_pb1d_vsolv_input) BEFORE
any training (user 2026-09-21): sampler sign/normalisation, cost, forces, loss gradients.
Each section is independent and survives an out-of-memory failure (reported, not fatal).

  S  sampler: the reaction potential of the 1-D solvent charge through the new 3-D sampler
     (pb1d_vsolv.smoothed_value_and_gradient) against the electrostatic channel's own projection
     (_slab_compensation_profile_features) at the atoms: sign convention and normalisation.
  M  magnitude of the new feature rows against the existing compensation rows.
  T  cost: forward + force backward, and forward + force-loss double backward, OFF / ON / ON
     without checkpointing (MACE_PB1D_VSOLV_NOCKPT=1).
  F  forces with the input ON (LIVE_POS path): -dE/dR by autograd against central finite
     differences on a few atoms; the same with the input OFF as the control.
  G  force-loss gradient w.r.t. a charge-head parameter: autograd (double backward) against a
     finite difference in that parameter, ON and OFF.

Env: KIT_RUN, KIT_MODEL_OBJ, KIT_CKPT, KIT_SIDS (default "28,628"), KIT_OUT, KIT_FD_ATOMS (default 2),
     KIT_SECTIONS (default "SMTFG").
"""
from __future__ import annotations

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
from mace.modules.extensions import _slab_compensation_profile_features, _vsolv_projection_features
from mace.modules.pb_solvent import resample_profile_periodic_torch
from mace.modules.pb1d_vsolv import smoothed_value_and_gradient, fourier_resample_1d

RUN = os.environ["KIT_RUN"]; OUT = os.path.abspath(os.environ.get("KIT_OUT", os.path.join(RUN, "logs")))
SIDS = [int(x) for x in os.environ.get("KIT_SIDS", "28,628").split(",")]
N_FD = int(os.environ.get("KIT_FD_ATOMS", "2"))
SECTIONS = os.environ.get("KIT_SECTIONS", "SMTFG")
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")
os.chdir(RUN)
K_COUL = 180.9512816816869

device = torch_tools.init_device("cuda"); torch_tools.set_default_dtype("float64")
model = torch.load(f=os.environ["KIT_MODEL_OBJ"], map_location=device).to(device)
model.load_state_dict(torch.load(os.environ["KIT_CKPT"], map_location=device)["model"], strict=True)
EPOCH = int(os.environ["KIT_CKPT"].rsplit("epoch-", 1)[1].split(".")[0]); model.eval(); model._pb1d_epoch = EPOCH
for p in model.parameters():
    p.requires_grad_(False)
print(f"[vsolv_wiring] model @ epoch {EPOCH}; field_feature_widths {model.field_feature_widths}; "
      f"LIVE_POS={os.environ.get('MACE_PB1D_LIVE_POS','unset')} GRAD_PASSES={os.environ.get('MACE_PB1D_GRAD_PASSES','unset')}; sections {SECTIONS}")

cap = {}
_bk = PB.PB1DBackend.solve_graph
def _bk_wrap(self, *a, **k):
    r = _bk(self, *a, **k)
    if "r" not in cap:
        cap.update(r=r, backend=self)
    return r
PB.PB1DBackend.solve_graph = _bk_wrap
_clo = PB.closure_from_fields
def _clo_wrap(n_e_density, cvhar3, grid, params, tp):
    if "n_e" not in cap:   # first closure call of a forward = stage 1
        cap.update(n_e=n_e_density.detach().clone(), grid=grid, params=params, tp=tp)
    return _clo(n_e_density, cvhar3, grid, params, tp)
PB.closure_from_fields = _clo_wrap

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


def batch_of(a, positions=None):
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    if positions is not None:
        cfg.positions = np.asarray(positions)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    return next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)


def evict():
    b = cap.get("backend")
    if b is not None:
        for nm in ("_bl_ram", "_grids", "_solvers", "_c_units"):
            d = getattr(b, nm, None)
            if isinstance(d, dict):
                d.clear()
    torch.cuda.empty_cache()


def energy_forces(a, flag, positions=None, want_forces=True):
    model.solvent_pb1d_vsolv_input = bool(flag)
    cap.clear()
    b = batch_of(a, positions)
    if want_forces:
        b["positions"].requires_grad_(True)
        with torch.enable_grad():
            pred = model(b.to_dict(), compute_force=False, training=False)
            E = pred["energy"].sum()
            (gE,) = torch.autograd.grad(E, b["positions"])
        return float(E), -gE.detach().cpu().numpy(), pred
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
    return float(pred["energy"].sum()), None, pred


def rms(x): x = np.asarray(x, float); return float(np.sqrt((x ** 2).mean()))


def section(name):
    def deco(fn):
        def run(*args, **kw):
            if name not in SECTIONS:
                return None
            try:
                return fn(*args, **kw)
            except torch.cuda.OutOfMemoryError as exc:
                print(f"--- {name}: OUT OF MEMORY ({str(exc)[:90]}) ---"); evict(); return None
        return run
    return deco


@section("S")
def sec_S(a):
    E_off, F_off, _ = energy_forces(a, False)
    r = cap["r"]; backend = cap["backend"]
    rho_layer = r["rho_layer_z"].detach()
    H = float(r["height"])
    cell_np = np.array(a.get_cell(), dtype=float)
    shape = backend._grid_shape(cell_np); grid = backend._grid_for(cell_np, shape, device); nz = shape[2]
    rho_z = fourier_resample_1d(rho_layer, nz)
    g = 2.0 * np.pi * torch.fft.fftfreq(nz, d=H / nz).to(device=device, dtype=torch.float64)
    rk = torch.fft.fft(rho_z); pk = torch.zeros_like(rk); m = torch.abs(g) > 1e-14
    pk[m] = K_COUL * rk[m] / (g[m] ** 2)
    phi3 = torch.fft.ifft(pk).real[None, None, :].expand(*shape).contiguous()
    pos = torch.tensor(a.get_positions(), device=device, dtype=torch.float64)
    cell_t = torch.tensor(cell_np, device=device, dtype=torch.float64)
    frac = torch.remainder(pos @ torch.linalg.inv(cell_t), 1.0)
    sig0 = float(model.field_feature_widths[0])
    val_new, grad_new = smoothed_value_and_gradient(phi3, grid, frac, sig0)
    prof1024 = resample_profile_periodic_torch(rho_layer, H, 1024, False)[None, :]
    batch_idx = torch.zeros(len(a), dtype=torch.long, device=device)
    pbc = torch.tensor([[True, True, False]], device=device)
    feats_ch, phi_ref, field_ref = _slab_compensation_profile_features(
        model.external_field_contribution, prof1024, cell_t[None], pbc, batch_idx, pos, list(model.field_feature_widths), 2)
    dv = (val_new - phi_ref).cpu(); dg = (grad_new[:, 2] - field_ref[:, 2]).cpu()
    print(f"--- S: new sampler vs electrostatic channel at {len(a)} atoms (sigma {sig0}) ---")
    print(f"   value: channel rms {rms(phi_ref.cpu()):.4f} eV, new rms {rms(val_new.cpu()):.4f}, diff rms {rms(dv):.2e} (rel {rms(dv)/max(rms(phi_ref.cpu()),1e-30):.2e}), "
          f"corr {float(np.corrcoef(val_new.cpu().numpy(), phi_ref.cpu().numpy())[0,1]):+.6f}")
    print(f"   z-gradient: channel rms {rms(field_ref[:,2].cpu()):.4f} eV/A, new rms {rms(grad_new[:,2].cpu()):.4f}, diff rms {rms(dg):.2e} (rel {rms(dg)/max(rms(field_ref[:,2].cpu()),1e-30):.2e}); "
          f"lateral components rms {rms(grad_new[:,:2].cpu()):.2e}")
    evict()
    return E_off, F_off, feats_ch


@section("M")
def sec_M(a, E_off, F_off, feats_ch):
    E_on, F_on, _ = energy_forces(a, True)
    nf = cap["r"].get("vsolv_node_fields")
    if nf is None:
        print("--- M: vsolv_node_fields not returned by the backend -- flag path not active ---"); return None
    sym = np.array(a.get_chemical_symbols())
    vf = _vsolv_projection_features(model.external_field_contribution, nf.detach(), list(model.field_feature_widths))
    print("--- M: new feature rows vs compensation rows ---")
    print(f"   node fields: value rms {rms(nf[:,0,0].detach().cpu()):.4e} max {float(nf[:,0,0].abs().max()):.4e} eV, grad rms {rms(nf[:,0,1:].detach().cpu()):.4e} eV/A; "
          f"value rms by species O {rms(nf[sym=='O',0,0].detach().cpu()):.3e} H {rms(nf[sym=='H',0,0].detach().cpu()):.3e} Ni {rms(nf[sym=='Ni',0,0].detach().cpu()):.3e}")
    print(f"   projected rows rms {rms(vf.cpu()):.4e} vs channel rows rms {rms(feats_ch.cpu()):.4e} (ratio {rms(vf.cpu())/max(rms(feats_ch.cpu()),1e-30):.3e})")
    print(f"   energy OFF {E_off:+.6f}  ON {E_on:+.6f}  (diff {E_on-E_off:+.4e} eV; the head has not been trained on the new rows); forces ON-OFF rms {rms(F_on-F_off)*1e3:.3f} meV/A")
    evict()
    return E_on, F_on


def force_loss(a, flag, fd_ref):
    model.solvent_pb1d_vsolv_input = bool(flag); cap.clear()
    b = batch_of(a); b["positions"].requires_grad_(True)
    with torch.enable_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
        E = pred["energy"].sum()
        (gE,) = torch.autograd.grad(E, b["positions"], create_graph=True)
        L = ((-gE - torch.tensor(fd_ref, device=device, dtype=gE.dtype)) ** 2).sum()
    return L


@section("T")
def sec_T(a, theta):
    fd_ref = np.asarray(a.get_array("forces"))
    print("--- T: cost per frame (forward + force backward | forward + force-loss double backward) ---")
    for label, flag, nockpt in (("OFF", False, False), ("ON", True, False), ("ON no-ckpt", True, True)):
        if nockpt:
            os.environ["MACE_PB1D_VSOLV_NOCKPT"] = "1"
        else:
            os.environ.pop("MACE_PB1D_VSOLV_NOCKPT", None)
        for what in ("E+F", "loss-grad"):
            try:
                evict(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
                if what == "E+F":
                    energy_forces(a, flag)
                else:
                    theta.requires_grad_(True); L = force_loss(a, flag, fd_ref); torch.autograd.grad(L, theta); theta.requires_grad_(False)
                torch.cuda.synchronize()
                print(f"   {label:11s} {what:9s}: {time.time()-t0:6.2f} s, peak {torch.cuda.max_memory_allocated()/2**30:6.2f} GiB")
            except torch.cuda.OutOfMemoryError:
                theta.requires_grad_(False); print(f"   {label:11s} {what:9s}: OUT OF MEMORY"); evict()
    os.environ.pop("MACE_PB1D_VSOLV_NOCKPT", None)
    evict()


@section("F")
def sec_F(a, F_off, F_on):
    sym = np.array(a.get_chemical_symbols()); zpos = a.get_positions()[:, 2]; zc = zpos.mean()
    picks = []
    for s in ("O", "H", "Ni"):
        idx = np.where(sym == s)[0]
        if idx.size:
            picks.append(int(idx[np.argmax(np.abs(zpos[idx] - zc))]))
    picks = picks[:N_FD]
    h = 2.0e-3; fd_ref = np.asarray(a.get_array("forces"))
    for flag, F_ad in ((False, F_off), (True, F_on)):
        if F_ad is None:
            continue
        errs = []; t0 = time.time()
        for i in picks:
            for c in range(3):
                pp = a.get_positions().copy(); pp[i, c] += h
                Ep, _, _ = energy_forces(a, flag, pp, want_forces=False)
                pm = a.get_positions().copy(); pm[i, c] -= h
                Em, _, _ = energy_forces(a, flag, pm, want_forces=False)
                errs.append((i, sym[i], c, -(Ep - Em) / (2 * h), F_ad[i, c]))
        worst = max(abs(e[3] - e[4]) for e in errs)
        print(f"--- F ({'ON ' if flag else 'OFF'}): -dE/dR autograd vs central FD (h {h} A), {len(errs)} components in {time.time()-t0:.0f} s; "
              f"worst |diff| {worst*1e3:.3f} meV/A; AD force error vs DFT rms {rms(F_ad-fd_ref)*1e3:.2f} meV/A ---")
        for e in errs:
            print(f"     atom {e[0]:3d} {e[1]:2s} comp {e[2]}: FD {e[3]*1e3:+9.3f}  AD {e[4]*1e3:+9.3f}  diff {(e[3]-e[4])*1e3:+8.3f} meV/A")
    evict()


@section("G")
def sec_G(a, theta):
    fd_ref = np.asarray(a.get_array("forces"))
    flat_idx = int(torch.argmax(theta.detach().abs().flatten()))
    for flag in (False, True):
        try:
            theta.requires_grad_(True)
            L = force_loss(a, flag, fd_ref)
            (gth,) = torch.autograd.grad(L, theta)
            g_ad = float(gth.flatten()[flat_idx]); theta.requires_grad_(False)
            ht = 1.0e-4 * max(1.0, float(theta.detach().abs().flatten()[flat_idx]))
            with torch.no_grad():
                theta.flatten()[flat_idx] += ht
            Lp = float(force_loss(a, flag, fd_ref))
            with torch.no_grad():
                theta.flatten()[flat_idx] -= 2 * ht
            Lm = float(force_loss(a, flag, fd_ref))
            with torch.no_grad():
                theta.flatten()[flat_idx] += ht
            g_fd = (Lp - Lm) / (2 * ht)
            print(f"--- G ({'ON ' if flag else 'OFF'}): dL_force/dtheta[{flat_idx}] autograd {g_ad:+.6e} vs FD {g_fd:+.6e}, rel diff {abs(g_ad-g_fd)/max(abs(g_ad),abs(g_fd),1e-300):.2e} ---")
        except torch.cuda.OutOfMemoryError:
            theta.requires_grad_(False); print(f"--- G ({'ON ' if flag else 'OFF'}): OUT OF MEMORY ---")
        evict()


@section("X")
def sec_X(a):
    """F2: the worst force component with the input ON: is the AD-FD gap a derivative error or
    solve-tolerance noise in phi* propagated through the new features? Sweep the FD step and the
    solver tolerance; print the solve provenance (n_outer, rms_last) of the displaced forwards."""
    sym = np.array(a.get_chemical_symbols())
    backend = model._get_pb1d_backend()
    tol0 = float(backend.tol)
    comps = [(181, 2), (32, 2)]
    print(f"--- X: AD-FD gap of selected components with the input ON, vs FD step and solver tolerance (default tol {tol0:g}) ---")
    for tol in (tol0, 1.0e-6):
        backend.tol = tol
        E_ad, F_ad, _ = energy_forces(a, True)
        prov_ad = (cap["r"].get("n_outer"), cap["r"].get("rms_last"))
        for (i, c) in comps:
            line = f"   tol {tol:g} atom {i:3d} {sym[i]:2s} comp {c}: AD {F_ad[i,c]*1e3:+9.3f} (n_outer {prov_ad[0]}, rms {prov_ad[1]:.1e})"
            for h in (0.002, 0.005, 0.01, 0.02):
                pp = a.get_positions().copy(); pp[i, c] += h
                Ep, _, _ = energy_forces(a, True, pp, want_forces=False); pr_p = (cap["r"].get("n_outer"), cap["r"].get("rms_last"))
                pm = a.get_positions().copy(); pm[i, c] -= h
                Em, _, _ = energy_forces(a, True, pm, want_forces=False); pr_m = (cap["r"].get("n_outer"), cap["r"].get("rms_last"))
                f_fd = -(Ep - Em) / (2 * h)
                line += f" | h {h}: FD {f_fd*1e3:+9.3f} diff {(f_fd-F_ad[i,c])*1e3:+7.3f} [it {pr_p[0]}/{pr_m[0]}]"
            print(line)
        # the same component with the input OFF at this tolerance (control)
        E_ad0, F_ad0, _ = energy_forces(a, False)
        i, c = comps[0]
        pp = a.get_positions().copy(); pp[i, c] += 0.002; Ep, _, _ = energy_forces(a, False, pp, want_forces=False)
        pm = a.get_positions().copy(); pm[i, c] -= 0.002; Em, _, _ = energy_forces(a, False, pm, want_forces=False)
        print(f"   tol {tol:g} control OFF atom {i} comp {c}: AD {F_ad0[i,c]*1e3:+9.3f} FD {-(Ep-Em)/0.004*1e3:+9.3f} diff {(-(Ep-Em)/0.004-F_ad0[i,c])*1e3:+7.3f} meV/A")
    backend.tol = tol0
    evict()


@section("Y")
def sec_Y(a):
    """Is the AD-FD force gap in the FEATURE computation or downstream? Derivative of a scalar
    functional of the vsolv node fields of atom 181, s = sum_k w_k nf[181, 0, k], w.r.t. the z
    position of atom 181: autograd (input ON, everything live) vs central FD; then autograd with
    one outer dependence cut at a time (phi*, n_e, positions) to attribute the paths."""
    i, c = 181, 2
    w = torch.tensor([1.0, 0.3, -0.7, 0.5], dtype=torch.float64, device=device)

    def scalar_from_forward(positions=None, want_grad=False):
        model.solvent_pb1d_vsolv_input = True; cap.clear()
        b = batch_of(a, positions)
        if want_grad:
            b["positions"].requires_grad_(True)
            with torch.enable_grad():
                model(b.to_dict(), compute_force=False, training=False)
                nf = cap["r"]["vsolv_node_fields"]
                s = (nf[i, 0, :] * w).sum()
                (g,) = torch.autograd.grad(s, b["positions"])
            return float(s), float(g[i, c])
        with torch.no_grad():
            model(b.to_dict(), compute_force=False, training=False)
            nf = cap["r"]["vsolv_node_fields"]
            return float((nf[i, 0, :] * w).sum()), None

    print(f"--- Y: derivative of the vsolv node-field functional of atom {i} w.r.t. its z ---")
    s0, g_ad = scalar_from_forward(want_grad=True)
    fds = []
    for h in (0.002, 0.005, 0.01):
        pp = a.get_positions().copy(); pp[i, c] += h; sp, _ = scalar_from_forward(pp)
        pm = a.get_positions().copy(); pm[i, c] -= h; sm, _ = scalar_from_forward(pm)
        fds.append((h, (sp - sm) / (2 * h)))
    print(f"   s = {s0:+.6e}; AD ds/dz {g_ad:+.6e}; FD " + "  ".join(f"h {h}: {v:+.6e}" for h, v in fds))
    # Y2: is s smooth in z at the FD scale? line scan and a quadratic fit
    ts = np.linspace(-0.01, 0.01, 11); svals = []
    for t in ts:
        pp = a.get_positions().copy(); pp[i, c] += t; sv, _ = scalar_from_forward(pp); svals.append(sv)
    svals = np.array(svals); coef = np.polyfit(ts, svals, 2); resid = svals - np.polyval(coef, ts)
    print(f"   Y2 line scan z +/- 0.01 A (11 pts): quadratic-fit slope at 0 {coef[1]:+.6e} (AD {g_ad:+.6e}), curvature {2*coef[0]:+.3e}, "
          f"fit residual rms {resid.std():.2e} (s scale {abs(s0):.2e}); local slopes " + " ".join(f"{(svals[k+1]-svals[k])/(ts[k+1]-ts[k]):+.4f}" for k in range(len(ts)-1)))
    for det in ("phi", "n", "pos", "phi,n", "phi,pos", "n,pos"):
        os.environ["MACE_PB1D_VSOLV_DETACH"] = det
        try:
            _, g = scalar_from_forward(want_grad=True)
            print(f"   AD with {det:8s} detached: {g:+.6e}   (path contribution of the cut part {g_ad - g:+.6e})")
        except Exception as exc:
            print(f"   AD with {det} detached: FAILED {str(exc)[:80]}")
    os.environ.pop("MACE_PB1D_VSOLV_DETACH", None)
    # the same for the energy: AD force component vs FD with each path cut (the FD is the same energy)
    E0, F0, _ = energy_forces(a, True)
    pp = a.get_positions().copy(); pp[i, c] += 0.002; Ep, _, _ = energy_forces(a, True, pp, want_forces=False)
    pm = a.get_positions().copy(); pm[i, c] -= 0.002; Em, _, _ = energy_forces(a, True, pm, want_forces=False)
    f_fd = -(Ep - Em) / 0.004
    print(f"   force atom {i} comp {c}: FD {f_fd*1e3:+9.3f}  AD all live {F0[i,c]*1e3:+9.3f} meV/A")
    for det in ("phi", "n", "pos"):
        os.environ["MACE_PB1D_VSOLV_DETACH"] = det
        try:
            _, F, _ = energy_forces(a, True)
            print(f"   AD force with {det:4s} detached: {F[i,c]*1e3:+9.3f}  (diff to FD {(F[i,c]-f_fd)*1e3:+8.3f}; cut-path contribution {(F0[i,c]-F[i,c])*1e3:+8.3f} meV/A)")
        except Exception as exc:
            print(f"   AD force with {det} detached: FAILED {str(exc)[:80]}")
    os.environ.pop("MACE_PB1D_VSOLV_DETACH", None)
    evict()


@section("Z")
def sec_Z(a):
    """Which Jacobian of the node fields disagrees with FD? Capture n_e (stage-1 density) and phi*
    from displaced forwards, then evaluate the node-field functional OFFLINE with each input frozen
    or varied (values only), and compare each FD with the matching detached-AD value."""
    from mace.modules import pb1d_vsolv as VS
    i, c = 181, 2
    w = torch.tensor([1.0, 0.3, -0.7, 0.5], dtype=torch.float64, device=device)
    cell_t = torch.tensor(np.array(a.get_cell()), device=device, dtype=torch.float64)

    def capture(positions=None):
        model.solvent_pb1d_vsolv_input = True; cap.clear()
        b = batch_of(a, positions)
        with torch.no_grad():
            model(b.to_dict(), compute_force=False, training=False)
        pos = torch.tensor(a.get_positions() if positions is None else positions, device=device, dtype=torch.float64)
        frac = torch.remainder(pos @ torch.linalg.inv(cell_t), 1.0)
        return cap["n_e"], cap["r"]["phi_z"].detach().clone(), frac, cap["grid"], cap["params"], cap["tp"]

    n0, phi0, fr0, grid, params, tp = capture()
    sigma_b = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
    eps_area = float(os.environ.get("MACE_PB1D_AREA_EPS", "1e-30"))
    sig = list(model.field_feature_widths)

    def s_of(n, phi, frac):
        with torch.no_grad():
            nf = VS.vsolv_node_fields(n, phi, frac, grid, params, tp, sigma_b, eps_area, sig, checkpoint=False)
        return float((nf[i, 0, :] * w).sum())

    s0 = s_of(n0, phi0, fr0)
    print(f"--- Z: node-field functional of atom {i} from captured stage-1 quantities, s0 {s0:+.6e} ---")
    print(f"   {'h':>6} | {'FD all vary':>12} {'FD phi frozen':>13} {'FD n,phi frozen':>15} {'FD n only':>10} {'FD phi only':>11}")
    for h in (0.005, 0.01, 0.02):
        pp = a.get_positions().copy(); pp[i, c] += h; nP, phP, frP, *_ = capture(pp)
        pm = a.get_positions().copy(); pm[i, c] -= h; nM, phM, frM, *_ = capture(pm)
        fd_all = (s_of(nP, phP, frP) - s_of(nM, phM, frM)) / (2 * h)
        fd_phifrozen = (s_of(nP, phi0, frP) - s_of(nM, phi0, frM)) / (2 * h)
        fd_posonly = (s_of(n0, phi0, frP) - s_of(n0, phi0, frM)) / (2 * h)
        fd_nonly = (s_of(nP, phi0, fr0) - s_of(nM, phi0, fr0)) / (2 * h)
        fd_phionly = (s_of(n0, phP, fr0) - s_of(n0, phM, fr0)) / (2 * h)
        print(f"   {h:6.3f} | {fd_all:+12.6f} {fd_phifrozen:+13.6f} {fd_posonly:+15.6f} {fd_nonly:+10.6f} {fd_phionly:+11.6f}")
        dn = float((nP - nM).abs().max()); dphi = float((phP - phM).abs().max())
        print(f"          max|n+ - n-| {dn:.3e} e/A^3, max|phi+ - phi-| {dphi:.3e} eV")
    print("   compare with AD: all live / phi detached / phi,n detached / phi,pos detached / n,pos detached from section Y")
    evict()


for sid in SIDS:
    a = atoms_by_sid[sid]
    print(f"\n===================== sid {sid} ({'charged' if abs(float(a.info.get('total_charge',0)))>1e-6 else 'neutral'}, {len(a)} atoms) =====================")
    head = model.field_dependent_charges_maps[0]
    theta = next(p for p in head.parameters())
    sec_Z(a)
    sec_Y(a)
    sec_X(a)
    res_S = sec_S(a)
    E_off, F_off, feats_ch = res_S if res_S is not None else (None, None, None)
    res_M = sec_M(a, E_off, F_off, feats_ch) if res_S is not None else None
    E_on, F_on = res_M if res_M is not None else (None, None)
    sec_T(a, theta)
    sec_F(a, F_off, F_on)
    sec_G(a, theta)
print("\nDONE")
