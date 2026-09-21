"""Acceptance of the wired stage-1 effective-potential input (solvent_pb1d_vsolv_input) BEFORE
any training (user 2026-09-21): sampler sign/normalisation, forces, loss gradients, cost.

  S  sampler: the reaction potential of the 1-D solvent charge pushed through the new 3-D
     sampler (pb1d_vsolv.smoothed_value_and_gradient) against the electrostatic channel's own
     projection (_slab_compensation_profile_features -> phi_ref, field_ref) at the atoms:
     fixes sign convention and normalisation (value and z-gradient).
  M  magnitude of the new feature rows against the existing compensation rows.
  F  forces with the input ON (LIVE_POS path): -dE/dR by autograd against central finite
     differences on a few atoms; the same with the input OFF as the control.
  G  force-loss gradient w.r.t. a charge-head parameter: autograd (double backward) against a
     finite difference in that parameter, ON and OFF.
  T  cost: forward + force backward, and forward + force-loss backward, OFF / ON / ON without
     checkpointing (MACE_PB1D_VSOLV_NOCKPT=1 is read by the backend at call time).

Env: KIT_RUN, KIT_MODEL_OBJ, KIT_CKPT, KIT_SIDS (default "28,628"), KIT_OUT, KIT_FD_ATOMS (default 3).
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
N_FD = int(os.environ.get("KIT_FD_ATOMS", "3"))
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
      f"LIVE_POS={os.environ.get('MACE_PB1D_LIVE_POS','unset')} GRAD_PASSES={os.environ.get('MACE_PB1D_GRAD_PASSES','unset')}")

cap = {}
_bk = PB.PB1DBackend.solve_graph
def _bk_wrap(self, *a, **k):
    r = _bk(self, *a, **k)
    if "r" not in cap:
        cap.update(r=r, backend=self, kwargs=k)
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
        out = (float(E), -gE.detach().cpu().numpy(), pred)
    else:
        with torch.no_grad():
            pred = model(b.to_dict(), compute_force=False, training=False)
        out = (float(pred["energy"].sum()), None, pred)
    return out


def rms(x): x = np.asarray(x, float); return float(np.sqrt((x ** 2).mean()))


for sid in SIDS:
    a = atoms_by_sid[sid]; sym = np.array(a.get_chemical_symbols())
    print(f"\n===================== sid {sid} ({'charged' if abs(float(a.info.get('total_charge',0)))>1e-6 else 'neutral'}, {len(a)} atoms) =====================")

    # ---- S: sampler against the electrostatic channel's projection
    E_off, F_off, pred_off = energy_forces(a, False)
    r = cap["r"]; backend = cap["backend"]
    rho_layer = r["rho_layer_z"].detach()                     # solver grid (upsampled), e/A^3, physics sign
    H = float(r["height"]); n_s = rho_layer.shape[0]
    cell_np = np.array(a.get_cell(), dtype=float)
    shape = backend._grid_shape(cell_np); grid = backend._grid_for(cell_np, shape, device)
    nz = shape[2]
    # 1-D reaction potential (physics sign) of the layer on the PB grid, broadcast
    rho_z = fourier_resample_1d(rho_layer, nz)
    g = 2.0 * np.pi * torch.fft.fftfreq(nz, d=H / nz).to(device=device, dtype=torch.float64)
    rk = torch.fft.fft(rho_z); pk = torch.zeros_like(rk); m = torch.abs(g) > 1e-14
    pk[m] = K_COUL * rk[m] / (g[m] ** 2)
    phi_z = torch.fft.ifft(pk).real
    phi3 = phi_z[None, None, :].expand(*shape).contiguous()
    pos = torch.tensor(a.get_positions(), device=device, dtype=torch.float64)
    cell_t = torch.tensor(cell_np, device=device, dtype=torch.float64)
    frac = torch.remainder(pos @ torch.linalg.inv(cell_t), 1.0)
    sig0 = float(model.field_feature_widths[0])
    val_new, grad_new = smoothed_value_and_gradient(phi3, grid, frac, sig0)
    # the channel's own projection of the same layer
    prof1024 = resample_profile_periodic_torch(rho_layer, H, 1024, False)[None, :]
    batch_idx = torch.zeros(len(a), dtype=torch.long, device=device)
    pbc = torch.tensor([[True, True, False]], device=device)
    feats_ch, phi_ref, field_ref = _slab_compensation_profile_features(
        model.external_field_contribution, prof1024, cell_t[None], pbc, batch_idx, pos, list(model.field_feature_widths), 2)
    dv = (val_new - phi_ref); dg = (grad_new[:, 2] - field_ref[:, 2])
    corr_v = float(np.corrcoef(val_new.cpu().numpy(), phi_ref.cpu().numpy())[0, 1])
    print(f"--- S: new sampler vs electrostatic channel at {len(a)} atoms (sigma {sig0}) ---")
    print(f"   value: channel rms {rms(phi_ref.cpu()):.4f} eV, new rms {rms(val_new.cpu()):.4f}, diff rms {rms(dv.cpu()):.2e} (rel {rms(dv.cpu())/max(rms(phi_ref.cpu()),1e-30):.2e}), corr {corr_v:+.6f}")
    print(f"   z-gradient: channel rms {rms(field_ref[:,2].cpu()):.4f} eV/A, new rms {rms(grad_new[:,2].cpu()):.4f}, diff rms {rms(dg.cpu()):.2e} (rel {rms(dg.cpu())/max(rms(field_ref[:,2].cpu()),1e-30):.2e}); "
          f"lateral components new rms {rms(grad_new[:,:2].cpu()):.2e} (should be ~0 for a 1-D potential)")
    print(f"   => the channel's potential is the PHYSICS-sign potential of the solvent charge (k rho/G^2); "
          f"v_new (energy per electron) enters as VSOLV_SIGN * v_new with VSOLV_SIGN = -1.")

    # ---- M: magnitude of the new rows
    E_on, F_on, pred_on = energy_forces(a, True)
    nf = cap["r"].get("vsolv_node_fields")
    if nf is None:
        print("   !! vsolv_node_fields not returned by the backend -- flag path not active"); continue
    vf = _vsolv_projection_features(model.external_field_contribution, nf.detach(), list(model.field_feature_widths))
    print(f"--- M: new feature rows vs compensation rows ---")
    print(f"   v_new node fields: value rms {rms(nf[:,0,0].detach().cpu()):.4e} max {float(nf[:,0,0].abs().max()):.4e} eV; grad rms {rms(nf[:,0,1:].detach().cpu()):.4e} eV/A; "
          f"by species value rms O {rms(nf[sym=='O',0,0].detach().cpu()):.3e} H {rms(nf[sym=='H',0,0].detach().cpu()):.3e} Ni {rms(nf[sym=='Ni',0,0].detach().cpu()):.3e}")
    print(f"   projected rows rms {rms(vf.cpu()):.4e} vs channel rows rms {rms(feats_ch.cpu()):.4e} (ratio {rms(vf.cpu())/max(rms(feats_ch.cpu()),1e-30):.3e})")
    print(f"   energy OFF {E_off:+.6f}  ON {E_on:+.6f}  (diff {E_on-E_off:+.4e} eV; head untrained on the new rows); forces ON-OFF rms {rms(F_on-F_off)*1e3:.3f} meV/A")

    # ---- F: force FD, ON and OFF, on N_FD atoms (outermost O, an H, a Ni)
    zpos = a.get_positions()[:, 2]; zc = zpos.mean()
    picks = []
    for s in ("O", "H", "Ni"):
        idx = np.where(sym == s)[0]
        if idx.size:
            picks.append(int(idx[np.argmax(np.abs(zpos[idx] - zc))]))
    picks = picks[:N_FD]
    h = 2.0e-3
    fd_ref = np.asarray(a.get_array("forces"))
    for flag, F_ad in ((False, F_off), (True, F_on)):
        errs = []; t0 = time.time()
        for i in picks:
            for c in range(3):
                pp = a.get_positions().copy(); pp[i, c] += h
                Ep, _, _ = energy_forces(a, flag, pp, want_forces=False)
                pm = a.get_positions().copy(); pm[i, c] -= h
                Em, _, _ = energy_forces(a, flag, pm, want_forces=False)
                f_fd = -(Ep - Em) / (2 * h)
                errs.append((i, sym[i], c, f_fd, F_ad[i, c]))
        worst = max(abs(e[3] - e[4]) for e in errs)
        print(f"--- F ({'ON ' if flag else 'OFF'}): -dE/dR autograd vs central FD (h {h} A), {len(errs)} components in {time.time()-t0:.0f} s; "
              f"worst |diff| {worst*1e3:.3f} meV/A; AD force error vs DFT rms {rms(F_ad-fd_ref)*1e3:.2f} meV/A ---")
        for e in errs:
            print(f"     atom {e[0]:3d} {e[1]:2s} comp {e[2]}: FD {e[3]*1e3:+9.3f}  AD {e[4]*1e3:+9.3f}  diff {(e[3]-e[4])*1e3:+8.3f} meV/A")

    # ---- G: force-loss parameter gradient, ON and OFF (first parameter tensor of the charge head)
    head = model.field_dependent_charges_maps[0]
    theta = next(p for p in head.parameters())
    flat_idx = int(torch.argmax(theta.detach().abs().flatten()))
    def force_loss(flag):
        model.solvent_pb1d_vsolv_input = bool(flag); cap.clear()
        b = batch_of(a); b["positions"].requires_grad_(True)
        with torch.enable_grad():
            pred = model(b.to_dict(), compute_force=False, training=False)
            E = pred["energy"].sum()
            (gE,) = torch.autograd.grad(E, b["positions"], create_graph=True)
            F = -gE
            L = ((F - torch.tensor(fd_ref, device=device, dtype=F.dtype)) ** 2).sum()
        return L
    for flag in (False, True):
        theta.requires_grad_(True)
        L = force_loss(flag)
        (gth,) = torch.autograd.grad(L, theta)
        g_ad = float(gth.flatten()[flat_idx]); theta.requires_grad_(False)
        ht = 1.0e-4 * max(1.0, float(theta.detach().abs().flatten()[flat_idx]))
        with torch.no_grad():
            theta.flatten()[flat_idx] += ht
        Lp = float(force_loss(flag))
        with torch.no_grad():
            theta.flatten()[flat_idx] -= 2 * ht
        Lm = float(force_loss(flag))
        with torch.no_grad():
            theta.flatten()[flat_idx] += ht
        g_fd = (Lp - Lm) / (2 * ht)
        print(f"--- G ({'ON ' if flag else 'OFF'}): dL_force/dtheta[{flat_idx}] autograd {g_ad:+.6e} vs FD {g_fd:+.6e}, rel diff {abs(g_ad-g_fd)/max(abs(g_ad),abs(g_fd),1e-300):.2e} ---")
    evict()

    # ---- T: cost
    print("--- T: cost per frame (forward + force backward; forward + force-loss double backward) ---")
    for label, flag, nockpt in (("OFF", False, False), ("ON", True, False), ("ON no-ckpt", True, True)):
        if nockpt:
            os.environ["MACE_PB1D_VSOLV_NOCKPT"] = "1"
        else:
            os.environ.pop("MACE_PB1D_VSOLV_NOCKPT", None)
        try:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            energy_forces(a, flag); torch.cuda.synchronize(); t1 = time.time() - t0; m1 = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            theta.requires_grad_(True); L = force_loss(flag); torch.autograd.grad(L, theta); theta.requires_grad_(False)
            torch.cuda.synchronize(); t2 = time.time() - t0; m2 = torch.cuda.max_memory_allocated() / 2**30
            print(f"   {label:11s}: E+F {t1:.2f} s, peak {m1:.2f} GiB | force-loss grad {t2:.2f} s, peak {m2:.2f} GiB")
        except RuntimeError as exc:
            print(f"   {label:11s}: FAILED ({str(exc)[:80]})")
        evict()
    os.environ.pop("MACE_PB1D_VSOLV_NOCKPT", None)
print("\nDONE")
