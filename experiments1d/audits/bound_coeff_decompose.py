"""Mean response, covariance background, or learned correction -- which is wrong?
(user-specified design, 2026-09-10, charged frame, no re-solve anywhere.)

WHY THE PREVIOUS CANDIDATE WAS REPLACED. I proposed comparing the polarization
itself, P_model = a1*E + p_off, against the integral of the DFT plane-averaged
RHOB. The user rejected it on two counts, both correct.

  One result cannot fix two unknowns. P = a1*E + p_off with P and E known
  still admits many (a1, p_off) pairs, so plotting the model's two terms does
  not give the right value of either. And the code's prior is ALREADY a
  covariance -- prior = plane_mean(a3*E) - plane_mean(a3)*plane_mean(E), read
  straight out of closure_from_fields -- whose job is to repair exactly the
  "average first, then multiply" error. So a1*E + prior = plane_mean(a3*E) by
  construction: the two terms are SUPPOSED to cancel strongly, and the size of
  that cancellation says the covariance is comparable to the mean product,
  which is a statement about lateral inhomogeneity, not a defect. My reading
  of the 52-fold cancellation as pointing at p_off is withdrawn.

  The integral gives W_B P, not P. The charge is the divergence of the
  SMOOTHED polarization, n_b = -V * WB @ D @ P, so integrating RHOB recovers
  W_B P and cannot be set against an unsmoothed a1*E + p_off. Fixing the
  constant in the vacuum does not remove that difference.

WHAT REPLACES IT, in the user's three steps: build an INDEPENDENT reference
for the two coefficients out of the DFT fields themselves, then compare the
model's pieces against it one at a time.

  1. Compute the 3-D polarization from the DFT native density and the DFT
     total potential and confirm it reproduces RHOB. No DFT re-solve. If this
     fails, the 3-D response, the cavity or the parameters are wrong and
     nothing downstream is readable.
  2. From the same 3-D fields take the plane-averaged response coefficient and
     the product-mean-minus-mean-product covariance. These are the CORRECT
     values of a1 and p_off, and exactly so: plane-averaging the 3-D
     construction gives n_b = -V * WB_1d @ D @ plane_mean(P_z), the lateral
     derivatives dropping out, so the 1-D reduction is exact when
     a1 = plane_mean(a3) and p_off = cov(a3, E_z). Both identities are gated
     numerically rather than asserted, and so is the claim that the 1-D
     driving field equals the plane average of the 3-D one.
  3. On one grid, one sign convention and one smoothing, compare the model's
     a1, its prior and its learned delta_p separately, and check that together
     they give the right polarization and the right charge.

That separates a wrong mean response from a wrong covariance background from a
correction that failed to repair the difference. The polarization integral
rides along as a cross-check only and attributes nothing.

STANDING LIMITS carried in. The robust result so far is that the model's bound
charge has close to the right TOTAL but a clearly wrong SHAPE, and that
swapping in the DFT total potential alone does not repair it. That is all --
the generation of the total potential is NOT excluded. The top few frequencies
contributing nothing excluded only that band of the potential difference; grid
error in the cavity and in the a1 construction is still open, so the band
content of both coefficients is reported: the model's come from a 100x100x300
grid and the reference from 168x168x500, and the model cannot represent what
its own grid cannot carry.
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
from mace.modules.pb1d_closure import response_a3
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
print(f"  params from the PB config directly; sigma_b {SIGMA_B:.6f} A, "
      f"LNLDIEL {params['LNLDIEL']}, N_MOL {params['N_MOL']:.6e}, "
      f"alpha0_rot {params['alpha0_rot']:.6e}, alpha_pol "
      f"{params['alpha_pol']:.6e}", flush=True)


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
    """The 1-D reduction of the 3-D Gaussian, built exactly as
    _gaussian_wb_matrix builds it, applied spectrally instead of densely."""
    z = torch.arange(nz, dtype=torch.float64, device=device) * lz_ / nz
    r = torch.minimum(z, lz_ - z)
    real = torch.exp(-0.5 * (r / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)) ** 3
    real = real * (nz / real.sum())
    return torch.fft.rfft(real) / nz


def d1d_spec(nz, lz_):
    """The derivative operator's spectrum, Nyquist zeroed for even nz exactly
    as _rfft_matrix_ops does."""
    hz = torch.arange(nz // 2 + 1, dtype=torch.float64, device=device)
    if nz % 2 == 0:
        hz[-1] = 0.0
    return 1j * 2 * math.pi * hz / lz_


def apply_spec(f, spec, nz):
    return torch.fft.irfft(torch.fft.rfft(f) * spec, n=nz)


def hi_share(p, cut):
    """Fraction of a profile's spectral energy above mode `cut` (mode 0 out)."""
    sp = torch.fft.rfft(p)
    e = (sp[1:].real ** 2 + sp[1:].imag ** 2)
    return float(e[cut:].sum() / torch.clamp(e.sum(), min=1e-300))


# ======================================================================
# STEP 1 -- the 3-D reference
# ======================================================================
lat, chg = read_grid(f"{DFTDIR}/CHGCAR")
shd = tuple(chg.shape)
Vd = float(abs(np.linalg.det(lat)))
lz_d = float(np.linalg.norm(lat[2]))
gd = tp.TorchGrid(lat, shd, device=str(device), dtype=torch.float64,
                  rspec=True)
ne_d = torch.clamp(chg.to(device) / Vd, min=0.0)
del chg
print(f"\n  DFT native grid {shd}, cell volume {Vd:.4f} A^3, lz {lz_d:.4f} A",
      flush=True)
s_ion3, s_diel3, _ = tp.create_cavity_torch(ne_d, gd, params)
s_diel3 = torch.clamp(s_diel3, 0.0, 1.0)
del ne_d, s_ion3
torch.cuda.empty_cache()

_, phi_raw = read_grid(f"{DFTDIR}/PHI")
phi3 = phi_raw.to(device)   # electron-energy convention, same as the solver's
del phi_raw                 # phi -- confirmed by the reproduction below
phi_dft_z = phi3.mean(dim=(0, 1)).clone()
w_b = tp._normalized_gaussian_kernel_g(gd, SIGMA_B)
ex, ey, ez, emag = gd.grad_from_recip(-torch.conj(w_b) * gd.fft(phi3))
del phi3
f_loc = tp._local_field_factor(emag, params)
y = float(params["PBETA"]) * (emag * f_loc)
g = tp._dielectric_g(y, params)
poe = (float(params["alpha0_rot"]) / tp.EDEPS * g
       + float(params["alpha_pol"]) / tp.EDEPS)
p_over_e = float(params["N_MOL"]) * s_diel3 * poe
# a3 multiplies the RAW smoothed field, matching closure_from_fields'
# response_a3, which folds f_loc into the coefficient instead of the field
a3_ref = p_over_e * f_loc
div_p_g = gd.div_real_vector(p_over_e * (ex * f_loc), p_over_e * (ey * f_loc),
                             p_over_e * (ez * f_loc))
# n_b = ifft(-w_b * div P) * V and rho = -(n_b/V), so rho = ifft(w_b * div P)
rho_b_3d_z = gd.ifft_real(w_b * div_p_g).mean(dim=(0, 1))
del div_p_g, y, g, poe, emag, f_loc, ex, ey, p_over_e
torch.cuda.empty_cache()

A_ref_n = a3_ref.mean(dim=(0, 1))
Ez_ref_n = ez.mean(dim=(0, 1))
Pz_ref_n = (a3_ref * ez).mean(dim=(0, 1))
prior_ref_n = Pz_ref_n - A_ref_n * Ez_ref_n
sd_ref_n = s_diel3.mean(dim=(0, 1))
nzd = shd[2]
del a3_ref, ez, s_diel3
torch.cuda.empty_cache()

_, rb_raw = read_grid(f"{DFTDIR}/RHOB")
rho_b_ref_z = (-(rb_raw.to(device)) / Vd).mean(dim=(0, 1))
del rb_raw
torch.cuda.empty_cache()

dz_d, area_d = lz_d / nzd, Vd / lz_d
l1_3d = float((rho_b_3d_z - rho_b_ref_z).abs().sum() * dz_d * area_d)
ref_abs = float(rho_b_ref_z.abs().sum() * dz_d * area_d)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / max(float(a.norm() * b.norm()), 1e-30))


cc = corr(rho_b_3d_z, rho_b_ref_z)
g1 = l1_3d / max(ref_abs, 1e-30) < 0.05
print(f"\n[STEP 1] 3-D polarization from the DFT density and the DFT total "
      f"potential, against DFT RHOB (plane-averaged, native grid)")
print(f"   reconstructed  net {float(rho_b_3d_z.sum()*dz_d*area_d):+.6f} e, "
      f"int|.| {float(rho_b_3d_z.abs().sum()*dz_d*area_d):.4f} e")
print(f"   DFT RHOB       net {float(rho_b_ref_z.sum()*dz_d*area_d):+.6f} e, "
      f"int|.| {ref_abs:.4f} e")
print(f"   [{'PASS' if g1 else 'FAIL'}] L1 {l1_3d:.5f} e = "
      f"{100*l1_3d/max(ref_abs,1e-30):.2f}% of the reference int|.|, "
      f"correlation {cc:+.6f}")
print(f"   (this reproduction is also what confirms the sign pair: phi enters "
      f"as +PHI_raw and the density as -(n_b/V), the backend's own pair.\n"
      f"    A failure here means the 3-D response, the cavity or the "
      f"parameters are wrong and NOTHING below can be attributed.)",
      flush=True)

# ---- step 2: the reference decomposition, three identities gated ----
wbs_d, ds_d = wb1d_spec(nzd, lz_d, SIGMA_B), d1d_spec(nzd, lz_d)
rho_from_1d = apply_spec(apply_spec(Pz_ref_n, ds_d, nzd), wbs_d, nzd)
e_red = float((rho_from_1d - rho_b_3d_z).abs().max())
e_dec = float((A_ref_n * Ez_ref_n + prior_ref_n - Pz_ref_n).abs().max())
# threshold must be RELATIVE: an absolute 1e-18 is unachievable for terms of
# order 0.2 in float64, where one epsilon is already 4e-17. And note what this
# gate can and cannot show -- prior_ref is DEFINED here as
# Pz_ref - A_ref*Ez_ref, so the identity holds by construction and this
# measures float64 re-association only. The SUBSTANTIVE identity is the next
# gate, which compares two independently computed quantities.
eps_dec = 2.220446049250313e-16 * float((A_ref_n * Ez_ref_n).abs().max())
Ez_from_1d = -apply_spec(apply_spec(phi_dft_z, ds_d, nzd), wbs_d, nzd)
e_fld = float((Ez_from_1d - Ez_ref_n).abs().max())
scl = max(float(rho_b_3d_z.abs().max()), 1e-30)
g2a, g2b = e_dec < 10.0 * eps_dec, e_red / scl < 1e-3
g2c = e_fld / max(float(Ez_ref_n.abs().max()), 1e-30) < 1e-3
print(f"\n[STEP 2] the reference decomposition -- three identities MEASURED, "
      f"not assumed")
print(f"   [{'PASS' if g2a else 'FAIL'}] A_ref*<E_z> + prior_ref = "
      f"plane_mean(P_z): max abs {e_dec:.3e}, i.e. {e_dec/max(eps_dec,1e-300):.2f} "
      f"float64 epsilon on the terms being summed. The covariance makes the "
      f"1-D product EXACT, so these two are the correct a1 and p_off --\n"
      f"       but prior_ref is DEFINED here as Pz_ref - A_ref*Ez_ref, so this "
      f"holds by construction and the gate measures round-off, nothing "
      f"physical. The next gate is the substantive one.")
print(f"   [{'PASS' if g2b else 'FAIL'}] the 1-D operator on plane_mean(P_z) "
      f"equals the plane average of the 3-D charge: max abs {e_red:.3e} "
      f"against a profile max of {scl:.3e} e/A^3.\n       This is the "
      f"smoothing the user flagged -- the charge is div of W_B P, and the 1-D "
      f"and 3-D forms agree only because both carry it.")
print(f"   [{'PASS' if g2c else 'FAIL'}] the 1-D driving field equals the "
      f"plane average of the 3-D one: max abs {e_fld:.3e} against "
      f"{float(Ez_ref_n.abs().max()):.3e} eV/A", flush=True)

# ======================================================================
# STEP 3 -- the model's three pieces against that reference
# ======================================================================
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
    return out
Solver1D.solve = wrap_sv
_sg = PB.PB1DBackend.solve_graph
def wrap_sg(self, *a, **k):
    out = _sg(self, *a, **k)
    for k2 in ("prior_solve", "delta_p", "rho_bound_z"):
        if k2 in out and torch.is_tensor(out[k2]):
            cap[k2] = out[k2].detach().clone()
    return out
PB.PB1DBackend.solve_graph = wrap_sg
_clo = PB.closure_from_fields
def wrap_clo(n_e, cv, grid, prm, tp_):
    o = _clo(n_e, cv, grid, prm, tp_)
    cap["sd_mean_model"] = o["s_diel_mean"].detach().clone()
    cap["A_native"] = o["A_scr"].detach().clone()
    cap["prior_native"] = o["prior"].detach().clone()
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
for split in ("train", "val"):
    for a in read(os.path.join(DATADIR, f"{split}.xyz"), ":"):
        if int(a.info.get("sample_id", -1)) == SID:
            frame = a; break
    if frame is not None:
        break
cfg = mace_data.config_from_atoms(frame, key_specification=kspec)
ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                       cutoff=float(model.r_max))]
b = next(iter(torch_geometric.dataloader.DataLoader(
    ds, batch_size=1))).to(device)
with torch.no_grad():
    model(b.to_dict(), compute_force=False, training=False)   # ONCE

solver = cap["solver"]; kw = cap["kw"]
nz_s = kw["cvhar_z"].shape[0]
lz, volume = solver.lz, solver.volume
dz, area = lz / nz_s, volume / lz
a1_m, p_off_m = kw["a1"], kw["p_off"]
prior_m, dp_m = cap["prior_solve"], cap["delta_p"]
nz_c = cap["A_native"].shape[0]
print(f"\n  one model call. Solve grid nz_s {nz_s}, closure grid {nz_c}, "
      f"lz {lz:.4f} A, volume {volume:.4f} A^3")
gcell = abs(volume - Vd) / Vd < 1e-6
print(f"   [{'PASS' if gcell else 'FAIL'}] the two cells agree: model "
      f"{volume:.4f} vs DFT {Vd:.4f} A^3, relative {abs(volume-Vd)/Vd:.2e}")
gdp = float((prior_m + dp_m - p_off_m).abs().max())
print(f"   [{'PASS' if gdp < 1e-18 else 'FAIL'}] p_off = prior + delta_p as "
      f"captured: max abs {gdp:.3e}", flush=True)

wbs, dsp = wb1d_spec(nz_s, lz, SIGMA_B), d1d_spec(nz_s, lz)
A_ref = fresample(A_ref_n, nz_s)
prior_ref = fresample(prior_ref_n, nz_s)
Pz_ref = fresample(Pz_ref_n, nz_s)
rho_b_ref = fresample(rho_b_ref_z, nz_s)
sd_ref = fresample(sd_ref_n, nz_s)
E_dft = -apply_spec(apply_spec(fresample(phi_dft_z, nz_s), dsp, nz_s),
                    wbs, nz_s)
E_mod = -apply_spec(apply_spec(cap["phi_tot"], dsp, nz_s), wbs, nz_s)


def rho_of(a1_, poff_, E_):
    """rho_bound = +WB @ D @ P, since n_b = -V WB D P and rho = -(n_b/V)."""
    return apply_spec(apply_spec(a1_ * E_ + poff_, dsp, nz_s), wbs, nz_s)


g3 = float((rho_of(a1_m, p_off_m, E_mod)
            - fresample(cap["rho_bound_z"], nz_s)).abs().max())
print(f"   [{'PASS' if g3 < 1e-12 else 'FAIL'}] this script's 1-D chain "
      f"reproduces the model's own bound charge from a1, p_off and its own "
      f"potential: max abs {g3:.3e} e/A^3", flush=True)
allok = g1 and g2a and g2b and g2c and gcell and gdp < 1e-18 and g3 < 1e-12
if not allok:
    print(f"\n   A GATE FAILED -- results below are printed but NOT to be "
          f"attributed.", flush=True)

# ---- the coefficients, one at a time ----
resp_unit = float(response_a3(torch.zeros(1, dtype=torch.float64,
                                          device=device),
                              torch.ones(1, dtype=torch.float64,
                                         device=device), params, tp)[0])
sd_m = fresample(cap["sd_mean_model"], nz_s)
A_cav = resp_unit * (sd_m - sd_ref)      # the part of a1's error from s_diel
print(f"\n[STEP 3a] the two coefficients against their reference "
      f"(rms over z, and relative to the reference's own rms)")
print(f"  {'quantity':>34} {'ref rms':>11} {'model rms':>11} "
      f"{'err rms':>11} {'rel':>8} {'corr':>8}")
for lbl, m_, r_ in (("a1  (mean response)", a1_m, A_ref),
                    ("prior  (covariance background)", prior_m, prior_ref),
                    ("p_off = prior + delta_p", p_off_m, prior_ref)):
    er = float((m_ - r_).pow(2).mean().sqrt())
    rr = float(r_.pow(2).mean().sqrt())
    print(f"  {lbl:>34} {rr:11.4e} {float(m_.pow(2).mean().sqrt()):11.4e} "
          f"{er:11.4e} {er/max(rr,1e-30):8.3f} {corr(m_, r_):+8.4f}")
print(f"  a1's error split: total rms "
      f"{float((a1_m-A_ref).pow(2).mean().sqrt()):.4e}, of which the pure "
      f"cavity part resp_unit*(<s_diel>_model - <s_diel>_ref) is "
      f"{float(A_cav.pow(2).mean().sqrt()):.4e}\n    "
      f"({100*float(A_cav.pow(2).mean().sqrt())/max(float((a1_m-A_ref).pow(2).mean().sqrt()),1e-30):.1f}% "
      f"in rms; the remainder is the saturation/pairing heuristic, which "
      f"evaluates the response at a screened VACUUM field estimate rather "
      f"than at the self-consistent one)")

def best_scale(m_, r_):
    """The single factor s minimising |m - s r|, plus what is left after it.
    This is what separates a WRONG AMPLITUDE from a WRONG SHAPE, and it cannot
    be inferred from an rms and a mean-removed correlation: a1 carries a large
    positive mean, so those two numbers are not on the same footing."""
    sc = float((m_ * r_).sum() / torch.clamp((r_ * r_).sum(), min=1e-300))
    e0 = float((m_ - r_).pow(2).mean().sqrt())
    e1 = float((m_ - sc * r_).pow(2).mean().sqrt())
    return sc, e0, e1


print(f"\n[STEP 3a-2] wrong amplitude or wrong shape? The single best scale "
      f"factor, and the residual it cannot remove")
print(f"  {'quantity':>34} {'best scale':>11} {'err rms':>11} "
      f"{'after rescale':>14} {'removed':>8}")
for lbl, m_, r_ in (("a1  (mean response)", a1_m, A_ref),
                    ("prior  (covariance background)", prior_m, prior_ref),
                    ("p_off = prior + delta_p", p_off_m, prior_ref)):
    sc, e0, e1 = best_scale(m_, r_)
    print(f"  {lbl:>34} {sc:11.4f} {e0:11.4e} {e1:14.4e} "
          f"{100*(1-e1/max(e0,1e-30)):7.1f}%")
print(f"  (a scale far from 1 with most of the error removed = the shape is "
      f"right and the gain is wrong. A scale near 1 with little removed = the "
      f"shape itself is wrong.)", flush=True)

need = prior_ref - prior_m
sup = float((dp_m * need).sum() / torch.clamp((need * need).sum(), min=1e-300))
print(f"\n[STEP 3b] did the learned correction move prior toward the "
      f"reference?")
print(f"  needed   = prior_ref - prior_model : rms "
      f"{float(need.pow(2).mean().sqrt()):.4e}")
print(f"  supplied = delta_p                 : rms "
      f"{float(dp_m.pow(2).mean().sqrt()):.4e}")
print(f"  projection of delta_p onto what was needed: {sup:+.4f} "
      f"(1 = exactly the right correction, 0 = orthogonal, negative = the "
      f"wrong direction), correlation {corr(dp_m, need):+.4f}")
print(f"  residual after the correction: rms "
      f"{float((need - dp_m).pow(2).mean().sqrt()):.4e} against "
      f"{float(need.pow(2).mean().sqrt()):.4e} before "
      f"({'reduced' if float((need-dp_m).pow(2).mean().sqrt()) < float(need.pow(2).mean().sqrt()) else 'NOT reduced'})")
_sc = float((need * dp_m).sum() / torch.clamp((dp_m * dp_m).sum(), min=1e-300))
_e1 = float((need - _sc * dp_m).pow(2).mean().sqrt())
print(f"  IF delta_p were simply rescaled by {_sc:.2f}x: residual rms "
      f"{_e1:.4e}, i.e. {100*(1-_e1/max(float(need.pow(2).mean().sqrt()),1e-30)):.1f}% "
      f"of what is needed removed by a pure gain change.")
print(f"  This rescale residual is the decisive number, not the correlation: "
      f"two profiles both localised at the dielectric interface correlate "
      f"highly whatever their detail, so +{corr(dp_m, need):.4f} alone could "
      f"be a\n  localisation artefact. The rescale residual is not -- if it "
      f"is small, the learned correction has the right shape and only the "
      f"wrong gain.", flush=True)
print(f"\n[cavity plateau, same recipe and parameters, different grid] "
      f"plane_mean(s_diel) max: model {float(sd_m.max()):.7f} on its "
      f"{nz_c}-in-z grid, reference {float(sd_ref.max()):.7f} on the DFT "
      f"native {nzd}\n  (the earlier density test found the model grid gives "
      f"the same plateau for BOTH densities, so density is not the "
      f"difference here -- that leaves resolution, and this is the first "
      f"direct evidence for it)", flush=True)
print(f"\n[STEP 3c] grid reach -- the model's closure grid is {nz_c} in z, so "
      f"its coefficients carry no mode above {nz_c//2}. Share of the "
      f"reference's spectral energy above that cut:")
print(f"    A_ref     {100*hi_share(A_ref, nz_c//2):6.2f}%")
print(f"    prior_ref {100*hi_share(prior_ref, nz_c//2):6.2f}%")
print(f"  (a large share is content the model's grid cannot represent at all, "
      f"which is the cavity/a1 grid error the user left open.)", flush=True)

# ---- do they give the right polarization and charge together? ----
phi_score = -(kw["cvhar_z"] - kw["cvhar_z"].mean())


def metrics(p, lbl):
    cross = float((p * phi_score).sum() * dz * area)
    cross_ref = float((rho_b_ref * phi_score).sum() * dz * area)
    rows = []
    for d in SHIFTS:
        ps = fshift(p, float(d), lz)
        den = float((ps * ps).sum())
        sc = float((rho_b_ref * ps).sum()) / max(den, 1e-300)
        rows.append((float(d), sc, float((rho_b_ref - sc * ps).norm()
                                         / max(float(rho_b_ref.norm()), 1e-30))))
    bst = min(rows, key=lambda r: r[2])
    return dict(lbl=lbl, net=float(p.sum() * dz * area),
                absq=float(p.abs().sum() * dz * area), gap=cross - cross_ref,
                l1=float((p - rho_b_ref).abs().sum() * dz * area),
                mx=float((p - rho_b_ref).abs().max()),
                shift=bst[0], resid=bst[2],
                edge=abs(abs(bst[0]) - float(SHIFTS[-1])) < 1e-9)


print(f"\n[STEP 3d] together: swap one coefficient at a time, all driven by "
      f"the DFT total potential (the field the reference was built from), no "
      f"re-solve")
print(f"  {'a1':>8} {'p_off':>8} {'net (e)':>9} {'int|.|':>8} {'gap':>9} "
      f"{'L1':>9} {'max dev':>9} {'shift':>8} {'resid':>7}")
_rf = metrics(rho_b_ref, "ref")
print(f"  {'-':>8} {'-':>8} {_rf['net']:+9.4f} {_rf['absq']:8.4f} "
      f"{_rf['gap']:+9.4f} {_rf['l1']:9.5f} {_rf['mx']:9.2e} "
      f"{_rf['shift']:+8.3f} {100*_rf['resid']:6.1f}%   <- DFT reference RHOB")
# the rescale-only row exists because a bound is available for it but an
# EXACT number is one line away. Rescaling prior by its own best inverse gain
# removes 85.9% of its rms error; converting that into an L1 effect by
# assuming the residual keeps the same shape is an assumption, and WB@D is
# band-pass (peak k about 8.4/A) so |WB D x| depends on x's SPECTRUM, not its
# rms -- the best-scale residual is the component orthogonal to prior_ref and
# is plausibly higher-frequency than the difference it came from. The bound
# from the triangle inequality is 3.8x to 5.1x worse than the baseline and the
# operator's gain on the residual would have to be 5.0x its gain on the full
# error to overturn the sign, so the direction is safe either way -- but there
# is no reason to report a bound when the row itself can be evaluated.
_sc_pr = float((prior_m * prior_ref).sum()
               / torch.clamp((prior_ref * prior_ref).sum(), min=1e-300))
cells = (("model", "model", a1_m, p_off_m), ("ref", "model", A_ref, p_off_m),
         ("model", "ref", a1_m, prior_ref), ("ref", "ref", A_ref, prior_ref),
         ("model", f"prior/{_sc_pr:.3f}", a1_m, prior_m / _sc_pr))
out = {}
for la, lp, a_, p_ in cells:
    m_ = metrics(rho_of(a_, p_, E_dft), f"{la}/{lp}")
    out[(la, lp)] = m_
    fl = "  <- shift at scan edge" if m_["edge"] else ""
    if (la, lp) == ("ref", "ref"):
        fl += "  <- must reproduce the reference"
    if la == "model" and lp.startswith("prior/"):
        fl += "  <- the covariance GAIN fixed alone, exact"
    print(f"  {la:>8} {lp:>8} {m_['net']:+9.4f} {m_['absq']:8.4f} "
          f"{m_['gap']:+9.4f} {m_['l1']:9.5f} {m_['mx']:9.2e} "
          f"{m_['shift']:+8.3f} {100*m_['resid']:6.1f}%{fl}")
mm, rm = out[("model", "model")], out[("ref", "model")]
mr, rr_ = out[("model", "ref")], out[("ref", "ref")]
print(f"\n  by L1 (e): both model {mm['l1']:.5f}; a1 from the reference "
      f"-> {rm['l1']:.5f}; p_off from the reference -> {mr['l1']:.5f}; "
      f"both -> {rr_['l1']:.5f}")
print(f"  The 'ref/ref' row is a closure check, not a result: it must come "
      f"back at the reference. If it does not, the chain is broken and no "
      f"attribution holds.")
print(f"  WHAT THE SINGLE SWAPS ARE, and what they cannot say. Each row's "
      f"deviation from ref/ref is exactly one error array: ref/model isolates "
      f"the p_off error, model/ref the a1 error, model/model both. But\n  "
      f"a1*E + prior = plane_mean(a3*E) holds by construction on BOTH sides, "
      f"so the two coefficients are complementary halves of one decomposition "
      f"and a reference half paired with a model half breaks a cancellation\n"
      f"  the reference itself relies on -- measured on the reference side as "
      f"a1*E rms against plane_mean(P_z) rms below. So a single swap being "
      f"worse does NOT indict that coefficient, and 'whichever swap helps "
      f"most'\n  is not an available reading. What the rows DO give is the "
      f"size of each error's uncompensated effect and how much the two errors "
      f"cancel when both are present.")
print(f"  Plane-averaged 1-D only; no lateral 3-D error is in any of it.",
      flush=True)
_ea, _ep = rm['l1'], mr['l1']
print(f"  uncompensated: p_off error alone {_ea:.5f} e, a1 error alone "
      f"{_ep:.5f} e, both together {mm['l1']:.5f} e -> the two coefficient "
      f"errors cancel {0.5*(_ea+_ep)/max(mm['l1'],1e-30):.1f}-fold in L1.")
_rs = out[("model", f"prior/{_sc_pr:.3f}")]
print(f"  IS THE DIAGNOSIS A REPAIR LIST? No, and this is the exact number "
      f"rather than a bound. Dividing prior by its own best gain "
      f"{_sc_pr:.3f} -- the single change that removes 85.9% of its rms "
      f"error -- gives charge L1 {_rs['l1']:.5f} e against the baseline's "
      f"{mm['l1']:.5f} e,\n  a factor of "
      f"{_rs['l1']/max(mm['l1'],1e-30):.2f}. Fixing the covariance gain ALONE "
      f"{'makes the bound charge WORSE' if _rs['l1'] > mm['l1'] else 'helps'}"
      f", because it removes a compensation the wrong a1 currently relies on. "
      f"The two halves have to move together.")
print(f"  The two halves are NOT equally tractable, though, and the binding "
      f"constraint is a1: prior's error is 85.9% removable by one scalar, so a "
      f"joint change has a cheap well-specified handle on that half, while\n"
      f"  a1's is 7.2% removable by any scalar and 77.7% pure cavity, so its "
      f"half has no scalar handle at all and needs the cavity or the grid it "
      f"is computed on.")
print(f"  That cancellation is partly structural, not an independent finding: "
      f"the model's own closure satisfies the same identity with its own "
      f"fields, and the model's bound-charge total is close to the "
      f"reference,\n  so the two errors are largely forced to oppose. It is "
      f"reported as a magnitude, not read as a correlation between "
      f"independent errors.", flush=True)

# ---- the polarization integral, cross-check only ----
# rho_b = +WB @ D @ P (since n_b = -V WB D P and rho = -(n_b/V)), so the
# running integral of rho_b is +W_B P, NOT -W_B P. The earlier minus made the
# two arrays exact negatives -- ratio 1.99986 with correlation -0.999910.
P_int = torch.cumsum(rho_b_ref, dim=0) * dz
P_int = P_int - P_int.mean()
Pz_wb = apply_spec(Pz_ref, wbs, nz_s)
Pz_wb = Pz_wb - Pz_wb.mean()
print(f"\n[cross-check, attributes nothing] the polarization integral. "
      f"-cumsum(RHOB)*dz recovers W_B P, not P, so it is set against "
      f"W_B*plane_mean(P_z):")
print(f"    rms of the difference {float((P_int-Pz_wb).pow(2).mean().sqrt()):.4e} "
      f"against a W_B P rms of {float(Pz_wb.pow(2).mean().sqrt()):.4e}, "
      f"correlation {corr(P_int, Pz_wb):+.6f}")
print(f"    (the mean is removed from BOTH because the integration constant "
      f"is undetermined here -- that constant is precisely why this cannot "
      f"attribute anything, and it is why it is a cross-check only.)")
np.savez(os.path.join(os.environ.get("KIT_OUT", "."),
                      "bound_coeff_decompose_arrays.npz"),
         z=np.arange(nz_s) * dz, A_ref=A_ref.cpu().numpy(),
         a1_model=a1_m.cpu().numpy(), prior_ref=prior_ref.cpu().numpy(),
         prior_model=prior_m.cpu().numpy(), delta_p=dp_m.cpu().numpy(),
         p_off_model=p_off_m.cpu().numpy(), Pz_ref=Pz_ref.cpu().numpy(),
         E_dft=E_dft.cpu().numpy(), E_model=E_mod.cpu().numpy(),
         rho_b_ref=rho_b_ref.cpu().numpy(),
         rho_b_3d_native=rho_b_3d_z.cpu().numpy(),
         sd_mean_model=sd_m.cpu().numpy(), sd_mean_ref=sd_ref.cpu().numpy(),
         phi_dft_z=phi_dft_z.cpu().numpy(),
         phi_model=cap["phi_tot"].cpu().numpy())
print(f"\n  arrays saved to bound_coeff_decompose_arrays.npz")
print("DONE")
