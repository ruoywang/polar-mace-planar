"""Representation-ceiling test for the 1-D bound charge (one sid).

Bypasses training entirely: captures the trained model's solver inputs for
one structure, then optimizes (a) the 8 head coefficients directly and
(b) a free P_off profile (sigma_z-blurred), both against the DFT RHOB
target through the same differentiable solve. Outputs the ceiling curves.

Usage: python fit_rhob_ceiling.py <model> <xyz> <sid> <ref_npz> <out_npz>
Run from a dir with ./data and ./cache (enc counters past warmup).
"""
from __future__ import annotations

import os
import sys

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"

import numpy as np
import torch
from ase.io import read

model_path, xyz_path, sid_want, ref_npz, out_path = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
torch.set_default_dtype(torch.float64)
device = "cuda"
model = torch.load(model_path, map_location=device).to(device)
model.eval()

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils
from mace.modules.loss import _load_solvent_rhob_1d_npz, _gaussian_smear_periodic_1d
from mace.modules.pb_solvent import resample_profile_periodic_torch
from mace.modules.pb1d_head import gaussian_blur_periodic

kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff"},
    arrays_keys={"forces": "forces", "charges": "REF_charges"},
)
atoms = next(a for a in read(xyz_path, ":") if int(a.info["sample_id"]) == sid_want)
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)

# ---- capture solver call + head basis from one eval forward ---------------
backend = model._get_pb1d_backend()
head = model.pb1d_head
cap = {}
orig_dp = head.delta_p
def spy_dp(c, w_env, u, lz):
    cap["coeff"], cap["w_s"], cap["u_s"], cap["lz"] = c, w_env, u, lz
    return orig_dp(c, w_env, u, lz)
head.delta_p = spy_dp
orig_sg = backend.solve_graph
def spy_sg(**kw):
    cap["cell"] = kw["cell"]
    return orig_sg(**kw)
backend.solve_graph = spy_sg
# capture the inner solver.solve kwargs
import mace.modules.pb1d_solver as PS
solver_cls = None
for nm in dir(PS):
    o = getattr(PS, nm)
    if isinstance(o, type) and hasattr(o, "solve") and hasattr(o, "bound_matrix"):
        solver_cls = o
orig_solve = solver_cls.solve
def spy_solve(self, **kw):
    cap["solver"], cap["solve_kw"] = self, dict(kw)
    return orig_solve(self, **kw)
solver_cls.solve = spy_solve

with torch.no_grad():
    out0 = model(batch.to_dict(), training=False, compute_force=False)
head.delta_p = orig_dp
backend.solve_graph = orig_sg
solver_cls.solve = orig_solve
assert float(out0["solvent_rho_bound_1d_mask"].view(-1)[0]) > 0.5

solver, kw0 = cap["solver"], cap["solve_kw"]
coeff0, w_s, u_s, lz = cap["coeff"].detach(), cap["w_s"].detach(), cap["u_s"].detach(), cap["lz"]
volume = float(abs(torch.linalg.det(cap["cell"].reshape(3, 3))))
p_off0 = kw0["p_off"].detach()
dp0 = orig_dp(coeff0, w_s, u_s, lz).detach()
prior = p_off0 - dp0
H = lz

refs = _load_solvent_rhob_1d_npz(ref_npz)
target = torch.tensor(refs["targets"][sid_want], device=device)
tgt_sm = _gaussian_smear_periodic_1d(target.unsqueeze(0), 0.25, refs["lz_A"])[0]


def rho_b_512(p_off):
    kw = dict(kw0)
    kw["p_off"] = p_off
    out = solver.solve(**kw)
    rb = -(out["n_b"] / volume)
    return resample_profile_periodic_torch(rb, H, 512, False)


def loss_of(p_off):
    m = _gaussian_smear_periodic_1d(rho_b_512(p_off).unsqueeze(0), 0.25, refs["lz_A"])[0]
    return torch.mean((m - tgt_sm) ** 2)


def optimize(make_poff, params, steps, lr, tag):
    opt = torch.optim.Adam(params, lr=lr)
    best, best_state = float("inf"), None
    for it in range(steps):
        opt.zero_grad()
        l = loss_of(make_poff())
        l.backward()
        opt.step()
        lv = float(l)
        if lv < best:
            best = lv
            best_state = [p.detach().clone() for p in params]
        if it % 50 == 0 or it == steps - 1:
            print(f"{tag} step {it}: rmse {lv**0.5:.6e}", flush=True)
    for p, s in zip(params, best_state):
        p.data.copy_(s)
    return best ** 0.5


print(f"trained coeffs: {coeff0.cpu().numpy().round(4).tolist()}  (clamp {model.pb1d_head.c_max})")
sat = float(coeff0.abs().max()) / float(model.pb1d_head.c_max)
print(f"max|c|/c_max = {sat:.3f} {'<-- SATURATED' if sat > 0.97 else '(not saturated)'}")
with torch.no_grad():
    rb_trained = rho_b_512(p_off0)
    print(f"trained model rmse (smeared): {float(loss_of(p_off0))**0.5:.6e}")

# ---- ceiling A: free coefficients (no tanh clamp) --------------------------
c_free = coeff0.clone().requires_grad_(True)
r8 = optimize(lambda: prior + orig_dp(c_free, w_s, u_s, lz), [c_free], 300, 0.02, "coeff8")
with torch.no_grad():
    rb_c8 = rho_b_512(prior + orig_dp(c_free, w_s, u_s, lz))
print(f"ceiling-8coeff rmse {r8:.6e}; coeffs {c_free.detach().cpu().numpy().round(4).tolist()}")

# ---- ceiling B: free P_off profile (sigma_z floor kept) --------------------
raw = torch.zeros_like(prior, requires_grad=True)
rF = optimize(lambda: prior + gaussian_blur_periodic(raw, model.pb1d_head.sigma_z, lz),
              [raw], 400, 0.005, "freeP")
with torch.no_grad():
    rb_free = rho_b_512(prior + gaussian_blur_periodic(raw, model.pb1d_head.sigma_z, lz))
print(f"ceiling-freeP rmse {rF:.6e}")

sm = lambda t: _gaussian_smear_periodic_1d(t.unsqueeze(0), 0.25, refs["lz_A"])[0]
np.savez(out_path,
         z=np.arange(512) * refs["lz_A"] / 512.0, lz=refs["lz_A"], sid=sid_want,
         ref_raw=target.cpu().numpy(), ref_smeared=tgt_sm.cpu().numpy(),
         trained_raw=rb_trained.cpu().numpy(), trained_smeared=sm(rb_trained).cpu().numpy(),
         ceiling8_raw=rb_c8.cpu().numpy(), ceiling8_smeared=sm(rb_c8).cpu().numpy(),
         freeP_raw=rb_free.cpu().numpy(), freeP_smeared=sm(rb_free).cpu().numpy(),
         coeff_trained=coeff0.cpu().numpy(), coeff_ceiling=c_free.detach().cpu().numpy())
print(f"wrote {out_path}")
