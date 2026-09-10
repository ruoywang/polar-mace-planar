"""Hold a1 fixed, get the compensation exactly right: does the WHOLE result improve?
(user-specified staged design, 2026-09-10. Steps 1 and 2; step 3 waits on step 2.)

WHY THIS IS THE RIGHT MOVE, and it dissolves the problem that stopped the last
round. With a1 held at the model's value there is only ONE unknown left in
P = a1 E + p_off, and it is EXACTLY determined:

  P_off*   = plane_mean(P_DFT) - a1_model * plane_mean(E_DFT)
  delta_p* = P_off* - prior_model

So the coupled-pair obstruction is gone. The previous round's 2x2 failed
because a1 and p_off are complementary halves of one decomposition and
substituting either alone breaks a cancellation the reference itself relies
on; defining the target WITH the model's own a1 makes the pair exact by
construction instead, and there is nothing left to break. It also means the
earlier target prior_ref = Pz_ref - A_ref*Ez_ref was the wrong target for this
purpose: it is the correct p_off for the REFERENCE a1, not for the model's, so
it made delta_p answer for a1's error too. Step 1 quantifies how much the
target moves.

Explicitly NOT done here, per the user: no grid change, and no scaling up of
the correction coefficients. This asks only whether getting the compensation
right, with everything else as it is, improves the aggregate.

  STEP 1  recompute the correct learning target from arrays already in hand,
          and re-decide whether the current delta_p is short in AMPLITUDE or
          wrong in SHAPE -- the earlier answer was against the wrong target.
  STEP 2  put that exact P_off* in and RE-SOLVE self-consistently. Reproducing
          the reference at the fixed DFT field is only a wiring check, since
          P_off* was built at that field; the test is whether the improvement
          survives self-consistency, judged on aggregate charge, potential,
          cross energy and the FULL self-energy including the bound-ion mutual
          term (computed on the SUM field, never as a sum of separate self
          energies).
  STEP 3  only if step 2 improves: can the existing head, with its own basis
          and its actual coefficient clipping, produce delta_p*? Not run here.

One thing step 3 needs is already free here and may settle it in advance: the
head is delta_p = G_sigma_z * [w_env(z) * sum_k c_k B_k(u)] with K = 8 and
c_k = c_max * tanh(...) at c_max = 0.25, a HARD bound. If the current
c_absmax already sits at 0.25 the head is saturated and cannot supply a
larger correction whatever the target says, so c_absmax is printed here.
"""
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get(
    "KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get(
    "KIT_MACE_REPO",
    "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))

from ase.io import read

from pure_python import torch_pb as tp
from pure_python.pb import derived_params

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.pb1d_backend as PB
from mace.modules.pb1d_backend import fourier_upsample
from mace.modules.pb1d_solver import Solver1D

DFT = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single")
SID, DFTDIR, TAG = 1, f"{DFT}/1-44_GCE/cal_1", "NiN44 q=-1.00"
CFG = os.environ.get(
    "KIT_PB_CONFIG",
    "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/cal1_train.json")
CKDIR = os.environ.get("KIT_CKPT_DIR", "checkpoints")
DATADIR = os.environ.get("KIT_DATA_DIR", "data")
SHIFTS = np.arange(-2.0, 2.0001, 0.025)

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
with open(CFG) as fh:
    params = derived_params(json.load(fh).get("solvation", {}))
SIGMA_B = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
print(f"=================== {TAG} ===================")
print(f"  a1 HELD FIXED at the model's value; only the compensation changes. "
      f"No grid change, no scaling of the correction.", flush=True)


def read_grid(path):
    f = open(path); f.readline(); s = float(f.readline())
    lat = np.array([[float(x) for x in f.readline().split()]
                    for _ in range(3)]) * s
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
    need = nx * ny * nz; vals = []
    while len(vals) < need:
        vals.extend(float(x) for x in f.readline().split())
    return lat, torch.tensor(np.ascontiguousarray(
        np.array(vals[:need]).reshape(nz, ny, nx).transpose(2, 1, 0)))


def fresample(p, n):
    m = p.shape[0]
    if m == n:
        return p
    sp = torch.fft.rfft(p)
    keep = min(sp.shape[0], n // 2 + 1)
    out = torch.zeros(n // 2 + 1, dtype=sp.dtype, device=sp.device)
    out[:keep] = sp[:keep]
    return torch.fft.irfft(out, n=n) * (n / m)


def fshift(p, d, lz_):
    n = p.shape[0]
    k = torch.fft.rfftfreq(n, d=lz_ / n, device=p.device) * (2 * math.pi)
    return torch.fft.irfft(torch.fft.rfft(p) * torch.exp(-1j * k * d), n=n)


def wb1d_spec(nz, lz_, sigma):
    z = torch.arange(nz, dtype=torch.float64, device=device) * lz_ / nz
    r = torch.minimum(z, lz_ - z)
    real = torch.exp(-0.5 * (r / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)) ** 3
    real = real * (nz / real.sum())
    return torch.fft.rfft(real) / nz


def d1d_spec(nz, lz_):
    hz = torch.arange(nz // 2 + 1, dtype=torch.float64, device=device)
    if nz % 2 == 0:
        hz[-1] = 0.0
    return 1j * 2 * math.pi * hz / lz_


def apply_spec(f, spec, nz):
    return torch.fft.irfft(torch.fft.rfft(f) * spec, n=nz)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / max(float(a.norm() * b.norm()), 1e-30))


def split(m_, r_):
    """best single scale, error before, error after -- amplitude vs shape."""
    sc = float((m_ * r_).sum() / torch.clamp((r_ * r_).sum(), min=1e-300))
    e0 = float((m_ - r_).pow(2).mean().sqrt())
    e1 = float((m_ - sc * r_).pow(2).mean().sqrt())
    return sc, e0, e1


# ---- the DFT side: 3-D polarization, bound and ionic reference ----
lat, chg = read_grid(f"{DFTDIR}/CHGCAR")
shd = tuple(chg.shape); Vd = float(abs(np.linalg.det(lat)))
lz_d = float(np.linalg.norm(lat[2])); nzd = shd[2]
gd = tp.TorchGrid(lat, shd, device=str(device), dtype=torch.float64, rspec=True)
ne_d = torch.clamp(chg.to(device) / Vd, min=0.0); del chg
s_ion3, s_diel3, _ = tp.create_cavity_torch(ne_d, gd, params)
s_diel3 = torch.clamp(s_diel3, 0.0, 1.0)
del ne_d, s_ion3
torch.cuda.empty_cache()
_, phi_raw = read_grid(f"{DFTDIR}/PHI")
phi3 = phi_raw.to(device); del phi_raw
phi_dft_z = phi3.mean(dim=(0, 1)).clone()
w_b = tp._normalized_gaussian_kernel_g(gd, SIGMA_B)
ex, ey, ez, emag = gd.grad_from_recip(-torch.conj(w_b) * gd.fft(phi3))
del phi3
f_loc = tp._local_field_factor(emag, params)
g = tp._dielectric_g(float(params["PBETA"]) * (emag * f_loc), params)
poe = (float(params["alpha0_rot"]) / tp.EDEPS * g
       + float(params["alpha_pol"]) / tp.EDEPS)
p_over_e = float(params["N_MOL"]) * s_diel3 * poe
a3_ref = p_over_e * f_loc
Pz_ref_n = (a3_ref * ez).mean(dim=(0, 1))
A_ref_n = a3_ref.mean(dim=(0, 1))
Ez_ref_n = ez.mean(dim=(0, 1))
del a3_ref, ez, ex, ey, emag, f_loc, g, poe, p_over_e, s_diel3
torch.cuda.empty_cache()
_, rb = read_grid(f"{DFTDIR}/RHOB")
rho_b_ref_n = (-(rb.to(device)) / Vd).mean(dim=(0, 1)); del rb
_, ri = read_grid(f"{DFTDIR}/RHOION")
rho_i_ref_n = (-(ri.to(device)) / Vd).mean(dim=(0, 1)); del ri
torch.cuda.empty_cache()

# ---- one model call ----
model = torch.load(f=os.path.join(CKDIR, "s3d_gate_bl2_run-123.model"),
                   map_location=device).to(device)
ck = torch.load(os.path.join(CKDIR, "s3d_gate_bl2_run-123_epoch-33.pt"),
                map_location=device)
model.load_state_dict(ck["model"] if "model" in ck else ck.model)
model.eval()
for p in model.parameters():
    p.requires_grad = False
model.solvent3d_energy = True
model.solvent_cavity_energy = True
model.solvent_baseline_coupling = True

cap = {}
_sv = Solver1D.solve
def wrap_sv(self, **kw):
    out = _sv(self, **kw)
    cap["solver"] = self; cap["kw"] = dict(kw)
    cap["phi_tot"] = out["phi"].detach().clone()
    cap["n_b0"] = out["n_b"].detach().clone()
    cap["n_i0"] = out["n_ion"].detach().clone()
    cap["exit0"] = out.get("solver_exit")
    return out
Solver1D.solve = wrap_sv
_sg = PB.PB1DBackend.solve_graph
def wrap_sg(self, *a, **k):
    out = _sg(self, *a, **k)
    for k2 in ("prior_solve", "delta_p", "rho_bound_z", "rho_ion_z",
               "w_env_solve", "u_solve", "c_absmax", "dp_rms"):
        v = out.get(k2)
        if torch.is_tensor(v):
            cap[k2] = v.detach().clone()
        elif v is not None:
            cap[k2] = v
    return out
PB.PB1DBackend.solve_graph = wrap_sg
_clo = PB.closure_from_fields
def wrap_clo(n_e, cv, grid, prm, tp_):
    o = _clo(n_e, cv, grid, prm, tp_)
    cap["w_env_closure"] = o["w_env"].detach().clone()
    return o
PB.closure_from_fields = wrap_clo

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff",
               "solvated": "solvated"},
    arrays_keys={"forces": "forces"},
)
frame = None
for spl in ("train", "val"):
    for a in read(os.path.join(DATADIR, f"{spl}.xyz"), ":"):
        if int(a.info.get("sample_id", -1)) == SID:
            frame = a; break
    if frame is not None:
        break
cfg = mace_data.config_from_atoms(frame, key_specification=kspec)
ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                       cutoff=float(model.r_max))]
b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
with torch.no_grad():
    model(b.to_dict(), compute_force=False, training=False)

solver, kw = cap["solver"], cap["kw"]
nz_s = kw["cvhar_z"].shape[0]
lz, volume = solver.lz, solver.volume
dz, area = lz / nz_s, volume / lz
a1_m, p_off_m = kw["a1"], kw["p_off"]
prior_m, dp_m = cap["prior_solve"], cap["delta_p"]
wbs, dsp = wb1d_spec(nz_s, lz, SIGMA_B), d1d_spec(nz_s, lz)
# rebuilt exactly as solve_graph does: clamp(fourier_upsample(w_env, f)) then
# the normalised cumulative sum. Saved for step 3; not used here.
_f = nz_s // cap["w_env_closure"].shape[0]
w_s = torch.clamp(fourier_upsample(cap["w_env_closure"], _f), min=0.0)
u_s = torch.cumsum(w_s, dim=0)
u_s = u_s / torch.clamp(u_s[-1], min=1.0e-30)

# everything onto the solve grid. 500 -> 600 is band-limited zero-padding, so
# the 1-D reference profiles transfer without loss; the driving field is built
# on 600 with the solver's own operators so a1_m multiplies the field the
# solver would actually give it.
Pz_ref = fresample(Pz_ref_n, nz_s)
rho_b_ref = fresample(rho_b_ref_n, nz_s)
rho_i_ref = fresample(rho_i_ref_n, nz_s)
E_dft = -apply_spec(apply_spec(fresample(phi_dft_z, nz_s), dsp, nz_s), wbs, nz_s)
prior_ref_old = fresample(Pz_ref_n - A_ref_n * Ez_ref_n, nz_s)

# ======================================================================
# STEP 1 -- the correct target, with a1 held fixed
# ======================================================================
P_off_star = Pz_ref - a1_m * E_dft
dp_star = P_off_star - prior_m
print(f"\n[STEP 1] the correct compensation with a1 held at the model's value")
print(f"  P_off*   = plane_mean(P_DFT) - a1_model * plane_mean(E_DFT)")
print(f"  delta_p* = P_off* - prior_model")
print(f"\n  {'quantity':>32} {'rms':>12}")
for lbl, v in (("P_off* (the exact target)", P_off_star),
               ("prior_model", prior_m),
               ("p_off_model = prior + delta_p", p_off_m),
               ("delta_p* (what is needed)", dp_star),
               ("delta_p (what the head gives)", dp_m)):
    print(f"  {lbl:>32} {float(v.pow(2).mean().sqrt()):12.4e}")
print(f"\n  HOW MUCH THE TARGET MOVED by holding a1 fixed. The earlier round "
      f"scored delta_p against prior_ref - prior_model, which is the correct "
      f"p_off for the REFERENCE a1, so it charged delta_p for a1's error too.")
old_need = prior_ref_old - prior_m
print(f"    old target rms {float(old_need.pow(2).mean().sqrt()):.4e}, new "
      f"target rms {float(dp_star.pow(2).mean().sqrt()):.4e}, ratio "
      f"{float(dp_star.pow(2).mean().sqrt())/max(float(old_need.pow(2).mean().sqrt()),1e-30):.3f}, "
      f"correlation between them {corr(dp_star, old_need):+.4f}")
print(f"\n  AMPLITUDE OR SHAPE -- re-decided against the correct target")
print(f"  {'delta_p vs':>26} {'best scale':>11} {'err rms':>11} "
      f"{'after rescale':>14} {'removed':>8} {'corr':>8}")
for lbl, tgt in (("delta_p* (correct)", dp_star),
                 ("the old target", old_need)):
    sc, e0, e1 = split(dp_m, tgt)
    print(f"  {lbl:>26} {sc:11.4f} {e0:11.4e} {e1:14.4e} "
          f"{100*(1-e1/max(e0,1e-30)):7.1f}% {corr(dp_m,tgt):+8.4f}")
sc_s = float((dp_star * dp_m).sum() / torch.clamp((dp_m * dp_m).sum(), min=1e-300))
res_s = float((dp_star - sc_s * dp_m).pow(2).mean().sqrt())
print(f"  rescaling delta_p by {sc_s:.2f}x would leave rms {res_s:.4e} of "
      f"{float(dp_star.pow(2).mean().sqrt()):.4e} needed, i.e. a pure gain "
      f"change removes {100*(1-res_s/max(float(dp_star.pow(2).mean().sqrt()),1e-30)):.1f}%")
c_am = cap.get("c_absmax")
if c_am is not None:
    print(f"\n  HEAD SATURATION, free here and decisive for step 3: c_absmax "
          f"{float(c_am):.6f} against the hard bound c_max = 0.25 "
          f"({100*float(c_am)/0.25:.1f}% of it). delta_p = "
          f"G_0.2 * [w_env * sum_k c_k B_k(u)], K = 8, "
          f"c_k = 0.25*tanh(...).\n  If this is already at the bound, no "
          f"training can supply a larger correction and step 3 is decided in "
          f"advance; if it is far below, the amplitude is a learning outcome "
          f"and not a representational limit.", flush=True)

# ======================================================================
# STEP 2 -- put it in and re-solve
# ======================================================================
def rho_of(a1_, poff_, E_):
    return apply_spec(apply_spec(a1_ * E_ + poff_, dsp, nz_s), wbs, nz_s)


w_star = float((rho_of(a1_m, P_off_star, E_dft) - rho_b_ref).abs().sum() * dz * area)
w_base = float((rho_of(a1_m, p_off_m, E_dft) - rho_b_ref).abs().sum() * dz * area)
print(f"\n[STEP 2a] WIRING CHECK ONLY, at the fixed DFT field. P_off* was "
      f"built at this field, so reproducing the reference here proves the "
      f"algebra and nothing physical.")
print(f"    bound-charge L1 with P_off*: {w_star:.5f} e, against "
      f"{w_base:.5f} e for the model's own p_off and an int|.| of "
      f"{float(rho_b_ref.abs().sum()*dz*area):.4f} e")
print(f"    (the residual is the 500->600 operator discretisation plus the "
      f"0.18% by which the 3-D reconstruction differs from stored RHOB, not "
      f"a modelling error)", flush=True)

l0i = solver.l0_inv
phi_score = -(kw["cvhar_z"] - kw["cvhar_z"].mean())


def phi_of(rho):
    return torch.fft.irfft(torch.fft.rfft(rho * volume) * l0i, n=nz_s)


def selfE(rho):
    return 0.5 * float((rho * phi_of(rho)).sum() * dz * area)


def resolve(poff, lbl):
    kw2 = dict(kw); kw2["p_off"] = poff
    with torch.no_grad():
        o = solver.solve(**kw2)
    rb = -(o["n_b"] / volume); rio = -(o["n_ion"] / volume)
    return dict(lbl=lbl, rb=rb, ri=rio, phi=-o["phi"], exit=o.get("solver_exit"))


base = resolve(p_off_m, "baseline, model p_off")
gB = float((base["rb"] - cap["rho_bound_z"]).abs().max())
print(f"\n[STEP 2b] SELF-CONSISTENT RE-SOLVE, everything else identical")
print(f"   [{'PASS' if gB < 1e-12 else 'FAIL'}] the baseline re-solve "
      f"reproduces the model's own bound charge: max abs {gB:.3e} e/A^3")
star = resolve(P_off_star, "exact P_off*, re-solved")


def whole(d):
    rt, rt_ref = d["rb"] + d["ri"], rho_b_ref + rho_i_ref
    cross = float((rt * phi_score).sum() * dz * area)
    se = selfE(rt)
    dphi = d["phi"] - (-fresample(phi_dft_z, nz_s))
    return dict(
        lbl=d["lbl"], net=float(rt.sum() * dz * area),
        l1=float((rt - rt_ref).abs().sum() * dz * area),
        mx=float((rt - rt_ref).abs().max()),
        l1b=float((d["rb"] - rho_b_ref).abs().sum() * dz * area),
        l1i=float((d["ri"] - rho_i_ref).abs().sum() * dz * area),
        cross=cross, self=se, tot=cross + se,
        mutual=se - selfE(d["rb"]) - selfE(d["ri"]),
        pl1=float(dphi.abs().sum() * dz), pmx=float(dphi.abs().max()),
        prms=float(dphi.pow(2).mean().sqrt()))


refd = dict(lbl="DFT reference", rb=rho_b_ref, ri=rho_i_ref,
            phi=-fresample(phi_dft_z, nz_s))
rows = [whole(refd), whole(base), whole(star)]
r0 = rows[0]
print(f"\n  {'case':>24} {'net (e)':>9} {'chg L1':>8} {'bound L1':>9} "
      f"{'ion L1':>8} {'cross':>9} {'self':>9} {'total':>9} {'d total':>9}")
for r in rows:
    print(f"  {r['lbl']:>24} {r['net']:+9.4f} {r['l1']:8.5f} {r['l1b']:9.5f} "
          f"{r['l1i']:8.5f} {r['cross']:+9.4f} {r['self']:+9.4f} "
          f"{r['tot']:+9.4f} {r['tot']-r0['tot']:+9.4f}")
print(f"\n  mutual bound-ion term (self energy of the SUM minus the two "
      f"separate self energies, so the interaction is included):")
for r in rows:
    print(f"    {r['lbl']:>24} {r['mutual']:+9.4f} eV")
print(f"  potential profile error against the DFT total potential:")
for r in rows[1:]:
    print(f"    {r['lbl']:>24} L1 {r['pl1']:.4f}, max {r['pmx']:.4f}, rms "
          f"{r['prms']:.5f} eV")
for k in ("fix_exit", "newton_exit"):
    print(f"  solver exit {k}: baseline {(base['exit'] or {}).get(k)}, "
          f"P_off* {(star['exit'] or {}).get(k)}")
b_, s_ = rows[1], rows[2]
print(f"\n  VERDICT on step 2, on the aggregate and not on the bound channel "
      f"alone:")
for nm, kk, unit in (("charge sum L1", "l1", "e"), ("bound L1", "l1b", "e"),
                     ("ion L1", "l1i", "e"), ("potential rms", "prms", "eV")):
    d0, d1 = b_[kk], s_[kk]
    print(f"    {nm:>16}: {d0:.5f} -> {d1:.5f} {unit} "
          f"({'BETTER' if d1 < d0 else 'worse'}, "
          f"{100*(d1-d0)/max(abs(d0),1e-30):+.1f}%)")
for nm, kk in (("cross gap", "cross"), ("self gap", "self"),
               ("total gap", "tot"), ("mutual", "mutual")):
    g0, g1 = b_[kk] - r0[kk], s_[kk] - r0[kk]
    print(f"    {nm:>16}: {g0:+.4f} -> {g1:+.4f} eV "
          f"({'BETTER' if abs(g1) < abs(g0) else 'worse'})")
print(f"\n  If the aggregate does not improve here, STOP: the compensation is "
      f"not the binding problem and the next place to look is the other "
      f"inputs and the self-consistent coupling, not training. If it does "
      f"improve,\n  step 3 asks whether the existing head can produce "
      f"delta_p* within its own basis and its c_max = 0.25 clipping.",
      flush=True)
np.savez(os.path.join(os.environ.get("KIT_OUT", "."),
                      "poff_exact_target_arrays.npz"),
         z=np.arange(nz_s) * dz, P_off_star=P_off_star.cpu().numpy(),
         dp_star=dp_star.cpu().numpy(), dp_model=dp_m.cpu().numpy(),
         prior_model=prior_m.cpu().numpy(), p_off_model=p_off_m.cpu().numpy(),
         old_need=old_need.cpu().numpy(), a1_model=a1_m.cpu().numpy(),
         E_dft=E_dft.cpu().numpy(), Pz_ref=Pz_ref.cpu().numpy(),
         rho_b_ref=rho_b_ref.cpu().numpy(), rho_i_ref=rho_i_ref.cpu().numpy(),
         rb_base=base["rb"].cpu().numpy(), ri_base=base["ri"].cpu().numpy(),
         rb_star=star["rb"].cpu().numpy(), ri_star=star["ri"].cpu().numpy(),
         phi_base=base["phi"].cpu().numpy(), phi_star=star["phi"].cpu().numpy(),
         phi_dft=(-fresample(phi_dft_z, nz_s)).cpu().numpy(),
         w_env_solve=w_s.cpu().numpy(), u_solve=u_s.cpu().numpy())
print(f"\n  arrays saved to poff_exact_target_arrays.npz")
print("DONE")
