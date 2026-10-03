"""Pre-flight checks of the charge-state-input + full-field-repair change (reviewer plan 2026-10-03, item 5), on the baseline
model prod500_vsolv_fix with the NEW head architecture transplanted (production weights copied into the first linear block,
the augmented block zero, ion gate off, repair on) -- structural checks do not need trained augmented weights.
Frames: charged/neutral pairs (KIT_PAIRS) + one NiN88 frame (339 atoms, first in val).
  (1) forbidden-zone constraint: |rho| mass where M < 1e-3 before / after the repair (both channels), per frame
  (2) layer means: max |mean_xy(rho_new) - B(z)| on kept layers (must be ~1e-15); dropped layers (Mbar <= min_layer): count,
      1-D charge dropped (e), max layer-mean change; net charge before / after
  (3) consistency: the loss-facing field  base(z_p) + interp3(d_sup_new)  vs the energy-side field  B + delta_new  at the
      sampled label points (max abs / rms) and the exported field (same object); old vs new field difference
  (4) forces: difference protocol (repair+inputs ON minus OFF) autograd vs central FD (h = 0.01 A) for three atoms (z);
      and the full-model autograd-vs-FD gap with the new path on (tier i, cached baseline), as test_s3d_forces did
  (5) parameter gradient of a force loss through the new path: directional derivative v.grad_theta L_F (AD) vs central FD
      in theta (eps), theta = the head's parameters (linear + linear_aug); L_F = sum F^2 / n_atoms
Usage: python preflight_repair.py <out.json>; env KIT_PAIRS (default "1 52"), KIT_SCAL_JSON (stats json), KIT_MASK_SIGMA/T0/T1/MINL
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")
os.environ["MACE_S3D_EXPORT_DELTA"] = "1"

import numpy as np
import torch
from ase.io import read
from e3nn import o3

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils
import mace.modules.pb1d_backend as PB
from mace.modules.solvent3d import Solvent3DChargeHead, _interp1_periodic, _interp3_periodic

OUT = sys.argv[1] if len(sys.argv) > 1 else "preflight_repair.json"
MODEL = os.environ.get("KIT_MODEL", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_vsolv_fix/models/prod500_vsolv_fix.model")
PAIRS = [int(s) for s in os.environ.get("KIT_PAIRS", "1 52").replace("+", " ").split()]
MASK = dict(sigma=float(os.environ.get("KIT_MASK_SIGMA", "0.25")), t0=float(os.environ.get("KIT_MASK_T0", "1e-4")),
            t1=float(os.environ.get("KIT_MASK_T1", "1e-2")), min_layer=float(os.environ.get("KIT_MASK_MINL", "1e-3")))
SCAL_JSON = os.environ.get("KIT_SCAL_JSON", "")
H = 0.01; EPS = 1.0e-3
torch.set_default_dtype(torch.float64)
dev = torch.device(os.environ.get("KIT_DEVICE", "cuda:0"))
model = torch.load(MODEL, map_location=dev).to(dev); model.eval()
for p in model.parameters():
    p.requires_grad_(False)
backend = model._get_pb1d_backend()
old_head = model.solvent3d_head
# ---- transplant: new head architecture with the production weights in the first block
hidden = o3.Irreps(old_head.linear.irreps_in)
new_head = Solvent3DChargeHead(hidden, old_head.sigmas, charge_state_input=True, ion_gate=False).to(dev)
new_head.linear.load_state_dict(old_head.linear.state_dict())
if SCAL_JSON and os.path.exists(SCAL_JSON):
    st = json.load(open(SCAL_JSON)); new_head.set_scalar_standardisation(st["mean"], st["std"]); scal_src = SCAL_JSON
else:
    new_head.set_scalar_standardisation([-0.536, -10.97, -0.0026], [0.546, 10.05, 0.096]); scal_src = "16 training pairs (arms run)"
for p in new_head.parameters():
    p.requires_grad_(False)
ATOMS = {}
for sp in ("train", "val", "test"):
    for a in read(f"data/{sp}.xyz", ":"):
        a.info["_split"] = sp; ATOMS[int(a.info["sample_id"])] = a
big = [s for s, a in ATOMS.items() if a.info["_split"] == "val" and len(a) > 300][:1]
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin", "sample_id": "sample_id",
                                    "solvated": "solvated", "fermi_level": "Fermi", "potential": "potential_diff"}, arrays_keys={"forces": "forces"})
man = json.load(open("data/solvent3d_grid_manifest.json")); ENT = {int(k): v for k, v in man["entries"].items()}
T0 = time.time()
RES = dict(meta=dict(model=MODEL, mask=MASK, scal_source=scal_src, pairs=PAIRS, big=big, code=os.popen("git -C /scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy log --oneline -1").read().strip()), frames={}, forces={}, param_grad={})
print(f"model {os.path.basename(MODEL)}; mask {MASK}; scal {scal_src} mean {new_head.scal_mean.tolist()} std {new_head.scal_std.tolist()}; frames {PAIRS} + big {big}", flush=True)


def configure(new: bool):
    model.solvent3d_head = new_head if new else old_head
    model.solvent3d_repair = bool(new); model.solvent3d_mask = dict(MASK)


def batch_of(atoms):
    cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    return next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(dev)


def forward(atoms, force=False, want_res=True):
    b = batch_of(atoms); hold = {}; orig = backend.solve_graph

    def spy(*a, **kw):
        res = orig(*a, **kw); hold["res"] = res; hold["kw"] = kw; return res
    backend.solve_graph = spy
    try:
        if force:
            out = model(b.to_dict(), training=False, compute_force=True, compute_stress=False)
        else:
            with torch.no_grad():
                out = model(b.to_dict(), training=False, compute_force=False)
    finally:
        backend.solve_graph = orig
    return out, hold.get("res"), hold.get("kw"), b


def label_points(sid, n=50000, seed=0):
    e = ENT[sid]; rb = np.load("data/" + e["path_b"], mmap_mode="r"); shape = rb.shape; nz, ny, nx = shape
    rng = np.random.default_rng(seed); lin = rng.permutation(nz * ny * nx)[:n]
    iz, rem = np.divmod(lin, ny * nx); iy, ix = np.divmod(rem, nx)
    frac = np.stack([ix / nx, iy / ny, iz / nz], axis=1)
    return torch.as_tensor(frac, device=dev)


# ------------------------------------------------------------------ (1)-(3) field checks
frames = [s for k in PAIRS for s in (k, k + 600)] + big
for sid in frames:
    atoms = ATOMS[sid]; rec = {}
    for tag, new in (("old", False), ("new", True)):
        configure(new)
        out, res, kw, b = forward(atoms)
        obs = res["s3d_obs"]; grid_shape = tuple(obs["d_sup_b"].shape)
        cell = kw["cell"].detach().double().reshape(3, 3); V = float(abs(torch.det(cell))); lz = float(cell[2, 2]); nz = grid_shape[2]
        zfrac = torch.arange(nz, device=dev, dtype=torch.float64) / nz
        Bb = _interp1_periodic(res["rho_bound_z"].detach().double(), zfrac); Bi = _interp1_periodic(res["rho_ion_z"].detach().double(), zfrac)
        rho_b = Bb[None, None, :] + obs["d_sup_b"].detach(); rho_i = Bi[None, None, :] + obs["d_sup_i"].detach()
        dV = V / rho_b.numel()
        r = dict(e_s3d=float(res["e_s3d"]) if res.get("e_s3d") is not None else None, e_xsol=obs.get("e_xsol"), e_self=obs.get("e_self"),
                 mu_delta=float(obs["mu_delta"]), delta_b_pl_absmax=float(obs["delta_b_pl"].abs().max()), delta_i_pl_absmax=float(obs["delta_i_pl"].abs().max()),
                 net_b_e=float(rho_b.sum() * dV), net_i_e=float(rho_i.sum() * dV), q_total=float(kw["total_charge"]),
                 layer_mean_resid_b=float((rho_b.mean((0, 1)) - Bb).abs().max()), layer_mean_resid_i=float((rho_i.mean((0, 1)) - Bi).abs().max()),
                 energy=float(out["energy"].detach()), c_absmax=float(res["s3d_coeffs"].detach().abs().max()) if res.get("s3d_coeffs") is not None else None)
        if "repair" in obs:
            r["repair"] = obs["repair"]
        # energy-side exported field vs the loss-facing field at sampled points
        dg_b = obs.get("delta_b_grid"); dg_i = obs.get("delta_i_grid")
        if dg_b is not None and dg_b.numel() > 1:
            pts = label_points(sid); pb_loss = _interp1_periodic(res["rho_bound_z"].detach().double(), pts[:, 2]) + _interp3_periodic(obs["d_sup_b"].detach(), pts)
            pb_en = _interp3_periodic(Bb[None, None, :] + dg_b.detach(), pts)
            r["loss_vs_energy_field_b_maxabs"] = float((pb_loss - pb_en).abs().max()); r["loss_vs_energy_field_b_rms"] = float((pb_loss - pb_en).pow(2).mean().sqrt())
            r["field_b_rms"] = float(pb_en.pow(2).mean().sqrt())
            pi_loss = _interp1_periodic(res["rho_ion_z"].detach().double(), pts[:, 2]) + _interp3_periodic(obs["d_sup_i"].detach(), pts)
            pi_en = _interp3_periodic(Bi[None, None, :] + dg_i.detach(), pts)
            r["loss_vs_energy_field_i_maxabs"] = float((pi_loss - pi_en).abs().max())
        rec[tag] = r
        # forbidden-zone mass with the SAME mask definition, computed here independently (model cavity from the final solve)
        cav = getattr(res and backend._grid_for(cell.cpu().numpy(), backend._grid_shape(cell.cpu().numpy()), dev), "_solv3d_cavity", None)
        if cav is not None:
            Mb = PB._s3d_allowed_weight(cav[1], backend._grid_for(cell.cpu().numpy(), backend._grid_shape(cell.cpu().numpy()), dev), MASK)
            Mi = PB._s3d_allowed_weight(torch.clamp(cav[0], 0, 1), backend._grid_for(cell.cpu().numpy(), backend._grid_shape(cell.cpu().numpy()), dev), MASK)
            r["zone_mass_b_e"] = float((rho_b.abs() * (Mb < 1e-3)).sum() * dV); r["zone_mass_i_e"] = float((rho_i.abs() * (Mi < 1e-3)).sum() * dV)
            r["total_abs_b_e"] = float(rho_b.abs().sum() * dV); r["total_abs_i_e"] = float(rho_i.abs().sum() * dV)
            r["Mbar_b_min"] = float(Mb.mean((0, 1)).min()); r["layers_Mbar_b_le_minl"] = int((Mb.mean((0, 1)) <= MASK["min_layer"]).sum())
            r["Mbar_i_min"] = float(Mi.mean((0, 1)).min()); r["layers_Mbar_i_le_minl"] = int((Mi.mean((0, 1)) <= MASK["min_layer"]).sum())
            r["B_b_in_dropped_layers_e"] = float((Bb * (Mb.mean((0, 1)) <= MASK["min_layer"])).sum() * V / nz); r["B_i_in_dropped_layers_e"] = float((Bi * (Mi.mean((0, 1)) <= MASK["min_layer"])).sum() * V / nz)
    RES["frames"][str(sid)] = dict(n_atoms=len(atoms), q=float(atoms.info["total_charge"]), **{f"{t}": rec[t] for t in rec})
    o, n = rec["old"], rec["new"]
    print(f"[{time.time() - T0:5.0f}s] sid {sid} ({len(atoms)} atoms, q {float(atoms.info['total_charge']):+.2f}): zone mass b {o.get('zone_mass_b_e', float('nan')):.4f} -> {n.get('zone_mass_b_e', float('nan')):.2e} e (of {o.get('total_abs_b_e', 0):.3f}), i {o.get('zone_mass_i_e', float('nan')):.4f} -> {n.get('zone_mass_i_e', float('nan')):.2e} | "
          f"layer-mean resid b {o['layer_mean_resid_b']:.1e} -> {n['layer_mean_resid_b']:.1e}, i {n['layer_mean_resid_i']:.1e} | dropped layers b {n.get('layers_Mbar_b_le_minl')} (B there {n.get('B_b_in_dropped_layers_e', 0):+.2e} e) i {n.get('layers_Mbar_i_le_minl')} ({n.get('B_i_in_dropped_layers_e', 0):+.2e} e) | "
          f"net b {o['net_b_e']:+.4f} -> {n['net_b_e']:+.4f}, i {o['net_i_e']:+.4f} -> {n['net_i_e']:+.4f} (Q {n['q_total']:+.3f}) | loss-vs-energy field b maxabs {n.get('loss_vs_energy_field_b_maxabs', float('nan')):.1e} (rms field {n.get('field_b_rms', 0):.2e}) | E {o['energy']:+.4f} -> {n['energy']:+.4f} (e_s3d {o['e_s3d']} -> {n['e_s3d']}) | repair diag {n.get('repair')}", flush=True)
    json.dump(RES, open(OUT, "w"), indent=1, default=float)

# ------------------------------------------------------------------ (4) forces: difference protocol + full gap
sid_f = PAIRS[0]; atoms = ATOMS[sid_f]; nat = len(atoms); test_atoms = [0, nat // 2, nat - 1]; syms = atoms.get_chemical_symbols()
fres = {}
for tag, new in (("off", False), ("on", True)):
    configure(new)
    out, res, kw, b = forward(atoms, force=True); e0 = float(out["energy"].detach()); f0 = out["forces"].detach().cpu().numpy()
    fd = {}
    for ia in test_atoms:
        ap = atoms.copy(); ap.positions[ia, 2] += H; am = atoms.copy(); am.positions[ia, 2] -= H
        ep = float(forward(ap)[0]["energy"]); em = float(forward(am)[0]["energy"]); fd[ia] = -(ep - em) / (2 * H)
    fres[tag] = (e0, f0, fd)
rows = []
for ia in test_atoms:
    dfa = fres["on"][1][ia, 2] - fres["off"][1][ia, 2]; dff = fres["on"][2][ia] - fres["off"][2][ia]
    rows.append(dict(atom=ia, sym=syms[ia], dF_autograd=float(dfa), dF_fd=float(dff), gap=float(dfa - dff), F_on=float(fres["on"][1][ia, 2]), FD_on=float(fres["on"][2][ia]), full_gap_on=float(fres["on"][1][ia, 2] - fres["on"][2][ia]), full_gap_off=float(fres["off"][1][ia, 2] - fres["off"][2][ia])))
RES["forces"] = dict(sid=sid_f, h=H, dE_new=float(fres["on"][0] - fres["off"][0]), rows=rows)
print(f"[{time.time() - T0:5.0f}s] forces sid {sid_f}: dE(new path) {RES['forces']['dE_new']:+.5f} eV | " + " | ".join(f"atom {r['atom']} {r['sym']}: dF_auto {r['dF_autograd']:+.5f} dF_fd {r['dF_fd']:+.5f} gap {r['gap']:+.1e}; full gap on {r['full_gap_on']:+.1e} off {r['full_gap_off']:+.1e}" for r in rows), flush=True)
json.dump(RES, open(OUT, "w"), indent=1, default=float)

# ------------------------------------------------------------------ (5) parameter gradient of a force loss through the head (new path on)
configure(True)
params = [new_head.linear.weight] + ([new_head.linear_aug.weight] if new_head.linear_aug is not None else [])
for p in params:
    p.requires_grad_(True)
# perturb the augmented block away from zero so its gradient path is exercised (random small weights, restored afterwards)
saved = [p.detach().clone() for p in params]
g = torch.Generator(device="cpu").manual_seed(7)
with torch.no_grad():
    if new_head.linear_aug is not None:
        new_head.linear_aug.weight.copy_(0.05 * torch.randn(new_head.linear_aug.weight.shape, generator=g).to(dev))
b = batch_of(atoms)
out = model(b.to_dict(), training=True, compute_force=True, compute_stress=False)
LF = (out["forces"] ** 2).sum() / nat
grads = torch.autograd.grad(LF, params, allow_unused=True)
v = [torch.randn(p.shape, generator=g).to(dev) for p in params]
for vi in v:
    vi /= vi.norm()
ad = sum(float((gi * vi).sum()) for gi, vi in zip(grads, v) if gi is not None)
with torch.no_grad():
    for p, vi in zip(params, v): p.add_(EPS * vi)
    out_p = model(b.to_dict(), training=False, compute_force=True, compute_stress=False); LFp = float((out_p["forces"] ** 2).sum() / nat)
    for p, vi in zip(params, v): p.add_(-2 * EPS * vi)
    out_m = model(b.to_dict(), training=False, compute_force=True, compute_stress=False); LFm = float((out_m["forces"] ** 2).sum() / nat)
    for p, s0 in zip(params, saved): p.copy_(s0)
fdv = (LFp - LFm) / (2 * EPS)
RES["param_grad"] = dict(sid=sid_f, eps=EPS, LF=float(LF), v_dot_grad_AD=ad, FD=fdv, rel_gap=float((ad - fdv) / max(abs(fdv), 1e-30)), params=[tuple(p.shape) for p in params],
                         grad_norm_linear=float(grads[0].norm()) if grads[0] is not None else None, grad_norm_aug=float(grads[1].norm()) if len(grads) > 1 and grads[1] is not None else None)
print(f"[{time.time() - T0:5.0f}s] force-loss parameter gradient (head linear + linear_aug, random direction): AD {ad:+.6e} FD {fdv:+.6e} rel gap {RES['param_grad']['rel_gap']:+.2e}; |grad| linear {RES['param_grad']['grad_norm_linear']} aug {RES['param_grad']['grad_norm_aug']}", flush=True)
json.dump(RES, open(OUT, "w"), indent=1, default=float)
print(f"[{time.time() - T0:5.0f}s] done", flush=True)
