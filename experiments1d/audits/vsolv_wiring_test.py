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


for sid in SIDS:
    a = atoms_by_sid[sid]
    print(f"\n===================== sid {sid} ({'charged' if abs(float(a.info.get('total_charge',0)))>1e-6 else 'neutral'}, {len(a)} atoms) =====================")
    head = model.field_dependent_charges_maps[0]
    theta = next(p for p in head.parameters())
    res_S = sec_S(a)
    E_off, F_off, feats_ch = res_S if res_S is not None else (None, None, None)
    res_M = sec_M(a, E_off, F_off, feats_ch) if res_S is not None else None
    E_on, F_on = res_M if res_M is not None else (None, None)
    sec_T(a, theta)
    sec_F(a, F_off, F_on)
    sec_G(a, theta)
print("\nDONE")
