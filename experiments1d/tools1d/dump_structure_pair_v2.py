"""Dump charged + neutral twin of one geometry for the pair report page (v2, 2026-09-21).

Same content as dump_structure_pair.py (charges, Phi1D, plane-averaged net density vs the DFT
grid, solvent profiles, scalars, charged-minus-neutral difference), adapted to the mix800 data
set and the pb-s3d-energy code:
  * the neutral twin is SOLVATED too (5-44_neutral_withsolv), so its 1-D PB solve is captured
    as well and stored with the suffix _n;
  * the solver is captured by wrapping backend.solve_graph (kwargs + returned dict of the LAST
    call of a forward = the stage-2 solve), nothing is re-solved;
  * the model's plane-averaged net density is evaluated on the DFT grid's own planes with the
    training GTO evaluator (all 500 planes, so the curve covers the whole cell); the DFT target
    exists on its valid planes only;
  * Phi1D residuals through the training loss function with the run's config values.

Usage: python dump_structure_pair_v2.py <model.model> <xyz> <idx_charged> <idx_neutral> <out_npz>
Run from a dir with ./data (the training data dir) and ./config_pb1d.yaml (or set KIT_CONFIG).
"""
from __future__ import annotations

import json
import os
import sys

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")

import numpy as np
import torch
import yaml
from ase.io import read

model_path, xyz_path = sys.argv[1], sys.argv[2]
idx_c, idx_n, out_path = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
torch.set_default_dtype(torch.float64)
device = "cuda"
model = torch.load(model_path, map_location=device).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils
import mace.modules.loss as L
from mace.modules.loss import _gto_density_at_points_axis2_pbc, _parse_density_sigmas

cfg = yaml.safe_load(open(os.environ.get("KIT_CONFIG", "config_pb1d.yaml")))
SIGMAS = _parse_density_sigmas(cfg["density_3d_sigma"])
PHI_ALIGN = cfg.get("potential_1d_profile_align", "upper")
PHI_USE_SOLV = bool(cfg.get("potential_1d_profile_use_solvent_profile", True))
SOLV_SIGMA_G = float(cfg.get("solvent_sigma_g", 0.85))
manifest = json.load(open(cfg["density_3d_file"]))["entries"]
targets_phi = L._load_potential_1d_profiles_npz(cfg["potential_1d_profile_file"])

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin",
               "sample_id": "sample_id", "solvated": "solvated", "fermi_level": "Fermi",
               "potential": "potential_diff"},
    arrays_keys={"forces": "forces"},
)
backend = model._get_pb1d_backend()
print(f"model {os.path.basename(model_path)}; vsolv_input={getattr(model, 'solvent_pb1d_vsolv_input', False)}; "
      f"sigmas {SIGMAS}; phi align {PHI_ALIGN} use_solvent_profile {PHI_USE_SOLV} sigma_g {SOLV_SIGMA_G}")


def forward_frame(idx):
    atoms = read(xyz_path, str(idx))
    sid = int(atoms.info["sample_id"])
    cfgm = mace_data.config_from_atoms(atoms, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfgm, z_table=z_table, cutoff=float(model.r_max))]
    batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    holder = {}
    orig = backend.solve_graph

    def spy(*a, **kw):
        res = orig(*a, **kw)
        holder["kw"] = kw; holder["res"] = res; holder["n_calls"] = holder.get("n_calls", 0) + 1
        return res
    backend.solve_graph = spy
    try:
        with torch.no_grad():
            out = model(batch.to_dict(), training=False, compute_force=False)
    finally:
        backend.solve_graph = orig
    assert holder, f"sid {sid}: solvent solve was never invoked"
    diag = dict(getattr(backend, "last_diagnostics", {}) or {})
    return atoms, sid, batch, out, holder, diag


def nbar_and_phi(atoms, sid, batch, out):
    """Plane-averaged model net density on the DFT grid planes + DFT target + Phi1D pair."""
    e = manifest[str(sid)]
    meta = np.load(e["meta_path"]); lattice = np.asarray(meta["lattice"], float); valid_iz = np.asarray(meta["valid_iz"], np.int64)
    mm = np.load(e["path"], mmap_mode="r"); nz, ny, nx = mm.shape
    nbar_dft = np.array([float(mm[i].mean()) for i in valid_iz])
    del mm
    z_dft = valid_iz / nz * float(lattice[2, 2])
    coeffs = out["charge_density_radial_coefficients"].detach()
    cell = torch.as_tensor(lattice, dtype=torch.float64, device=device)
    pos = batch["positions"].detach(); zs = batch["atomic_numbers"].to(torch.long)
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    fx = (ix.ravel() / nx).astype(float); fy = (iy.ravel() / ny).astype(float)
    nbar_model = np.zeros(nz)
    with torch.no_grad():
        for c0 in range(0, nz, 4):
            c1 = min(c0 + 4, nz)
            frac = np.concatenate([np.column_stack([fx, fy, np.full(fx.size, j / nz)]) for j in range(c0, c1)])
            pts = torch.as_tensor(frac @ lattice, dtype=torch.float64, device=device)
            rho = _gto_density_at_points_axis2_pbc(points=pts, density_coefficients=coeffs, positions=pos,
                                                   atomic_numbers=zs, cell=cell, sigma=SIGMAS).view(c1 - c0, -1)
            nbar_model[c0:c1] = rho.mean(dim=1).cpu().numpy()
    z_grid = np.arange(nz) / nz * float(lattice[2, 2])
    lz = float(lattice[2, 2])
    # Phi1D through the training loss
    t = targets_phi[sid]
    pred = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}
    with torch.no_grad():
        residuals = L.potential_1d_profile_residuals(
            ref=batch, pred=pred, potential_targets=targets_phi, density_smearing_width=SIGMAS,
            axis=2, solvent_sigma_g=SOLV_SIGMA_G, align=PHI_ALIGN, use_solvent_profile=PHI_USE_SOLV)
    z_phi = np.asarray(t["z_A"], float); phi_ref = np.asarray(t["phi_eV"], float)
    align_mask = np.asarray(t["align_mask"], float)
    res_np = residuals.detach().cpu().numpy().reshape(-1)
    phi_ref_cmp = phi_ref - (phi_ref * align_mask).sum() / max(align_mask.sum(), 1.0)
    return dict(z_grid=z_grid, nbar_model=nbar_model, z_dft=z_dft, nbar_dft=nbar_dft, valid_iz=valid_iz, lz=lz,
                z_phi=z_phi, phi_ref_cmp=phi_ref_cmp, phi_pred_cmp=phi_ref_cmp + res_np, phi_residual=res_np)


def solvent_arrays(holder):
    r = holder["res"]
    def g(k):
        v = r.get(k)
        return v.detach().cpu().numpy().astype(float) if torch.is_tensor(v) else (np.asarray(v, float) if v is not None else np.array([]))
    return dict(z_solve=g("z"), rho_ion=g("rho_ion_z"), rho_bound=g("rho_bound_z"), prior=g("prior_solve"), delta_p=g("delta_p"),
                phi_z=g("phi_z"), n_calls=int(holder.get("n_calls", 0)))


def fermi_of(out):
    v = out.get("fermi_level_pred")
    if v is None:
        v = out["fermi_level"]
    return float(v.detach().cpu().reshape(-1)[0])


# ---- charged frame -------------------------------------------------------
atoms_c, sid_c, batch_c, out_c, hold_c, diag_c = forward_frame(idx_c)
pc = nbar_and_phi(atoms_c, sid_c, batch_c, out_c)
sc = solvent_arrays(hold_c)
charges_c = out_c["charges"].detach().cpu().numpy().reshape(-1)
# ---- neutral frame -------------------------------------------------------
atoms_n, sid_n, batch_n, out_n, hold_n, diag_n = forward_frame(idx_n)
assert np.abs(atoms_n.positions - atoms_c.positions).max() < 1e-8, "geometry mismatch"
assert abs(float(atoms_n.info["total_charge"])) < 1e-9
pn = nbar_and_phi(atoms_n, sid_n, batch_n, out_n)
sn = solvent_arrays(hold_n)
charges_n = out_n["charges"].detach().cpu().numpy().reshape(-1)

# ---- charged minus neutral density difference ---------------------------
assert pc["z_grid"].shape == pn["z_grid"].shape and np.allclose(pc["z_grid"], pn["z_grid"])
dn_model = pc["nbar_model"] - pn["nbar_model"]
iz_common = np.intersect1d(pc["valid_iz"], pn["valid_iz"])
sel_c = np.searchsorted(pc["valid_iz"], iz_common); sel_n = np.searchsorted(pn["valid_iz"], iz_common)
dn_dft = pc["nbar_dft"][sel_c] - pn["nbar_dft"][sel_n]
z_dft_diff = pc["z_dft"][sel_c]

sfx = lambda d, s: {k + s: v for k, v in d.items()}
np.savez(
    out_path,
    sid=sid_c, sid_n=sid_n, symbols=np.array(atoms_c.get_chemical_symbols()), z_atoms=atoms_c.positions[:, 2],
    charges=charges_c, charges_n=charges_n, total_charge=float(atoms_c.info["total_charge"]),
    pot_pred=float(out_c["potential"].detach().cpu().reshape(-1)[0]), pot_ref=float(atoms_c.info["potential_diff"]),
    fermi_pred=fermi_of(out_c), fermi_ref=float(atoms_c.info["Fermi"]),
    pot_pred_n=float(out_n["potential"].detach().cpu().reshape(-1)[0]), pot_ref_n=float(atoms_n.info["potential_diff"]),
    fermi_pred_n=fermi_of(out_n), fermi_ref_n=float(atoms_n.info["Fermi"]),
    energy_pred=float(out_c["energy"].detach().cpu().reshape(-1)[0]), energy_ref=float(atoms_c.info["energy"]),
    energy_pred_n=float(out_n["energy"].detach().cpu().reshape(-1)[0]), energy_ref_n=float(atoms_n.info["energy"]),
    q_ion=float(diag_c.get("q_ion", np.nan)), layer_mean=float(diag_c.get("layer_mean", np.nan)), mu_bound=float(diag_c.get("mu_bound", np.nan)),
    q_ion_n=float(diag_n.get("q_ion", np.nan)), layer_mean_n=float(diag_n.get("layer_mean", np.nan)), mu_bound_n=float(diag_n.get("mu_bound", np.nan)),
    rc_match=0.0, model_file=os.path.basename(model_path),
    vsolv_input=bool(getattr(model, "solvent_pb1d_vsolv_input", False)),
    dn_model=dn_model, z_dft_diff=z_dft_diff, dn_dft=dn_dft,
    **sfx({k: v for k, v in pc.items() if k != "valid_iz"}, ""), **sfx(sc, ""),
    **sfx({k: v for k, v in pn.items() if k != "valid_iz"}, "_n"), **sfx(sn, "_n"),
)
print(f"dumped charged sid={sid_c} + neutral sid={sid_n} -> {out_path}  (solver calls per forward: {sc['n_calls']} / {sn['n_calls']})")
for tag, p in (("charged", pc), ("neutral", pn)):
    print(f"{tag}: phi residual rms = {float(np.sqrt((p['phi_residual'] ** 2).mean())):.4f} eV")
print(f"pot  pred/ref charged = {float(out_c['potential']):+.4f}/{atoms_c.info['potential_diff']:+.4f} | neutral = {float(out_n['potential']):+.4f}/{atoms_n.info['potential_diff']:+.4f}")
print(f"fermi pred/ref charged = {fermi_of(out_c):+.4f}/{atoms_c.info['Fermi']:+.4f} | neutral = {fermi_of(out_n):+.4f}/{atoms_n.info['Fermi']:+.4f}")
A = abs(float(np.cross(atoms_c.cell[0], atoms_c.cell[1])[2])); dz = pc["lz"] / pc["z_grid"].size
print(f"q_ion charged {diag_c.get('q_ion', np.nan)}; mu_bound {diag_c.get('mu_bound', np.nan)}; sum(dn_model)*A*dz = {dn_model.sum() * A * dz:+.4f} e (expect {atoms_c.info['total_charge']:+.4f})")
