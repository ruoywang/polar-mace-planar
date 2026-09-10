"""What does fixing the lateral bound charge actually buy the total energy?
(user-specified single experiment, 2026-09-10, six frames, no training.)

THE MAIN LINE THIS RETURNS TO. The bound coupling gap splits by channel as
plane-average 0.525 eV (28.7%) and LATERAL 1.307 eV (71.3%) -- RUNS.md, the
T2 channel decomposition. The 3-D fit's only responsibility is that lateral
shortfall, and every round since has been in the 1-D plane-averaged channel,
which is the other 28.7%. This experiment asks the question that line was
opened for: substitute the DFT reference lateral bound charge and measure the
change in the FINAL TOTAL ENERGY error, per atom, on the frames already
checked.

WHAT IS HELD AND WHAT MOVES. Held: the solute electron density, the 1-D
background, the ionic charge and every other network output. Moved: only the
lateral part of the bound charge, replaced by

    lat_DFT(r) = rho_b_DFT(r) - plane_mean(rho_b_DFT)(z)

so every z-plane sums to zero and the net charge, the 1-D profile and the
z-dipole are all left exactly as they were. That is the same per-plane
conserving form the model's own residual takes -- d_sup_b already has zero
plane means by construction -- so the swap is like-for-like.

WHY THE ENERGY UPDATE IS A ONE-TERM CHANGE. The reconciliation identity
established that

    (3) int delta*phi                = e_xsol
    (4) 0.5 int delta*phi[delta]     = e_self
    (5) int rho_1d*phi[delta]        = 0   by the per-plane projection

and solvent3d_energy_g = e_xsol + e_self enters the total energy additively.
Since lat_DFT also has zero plane means, (5) still holds and the 1-D terms
(comp, E_bl) are untouched, so the whole effect on the total energy is

    dE = [cross + self](delta_i + lat_DFT) - [cross + self](d_sup_b + d_sup_i)

with the self-energy taken on the SUM, which is what makes the bound-ion
interaction part of it rather than a term dropped between two separate self
energies.

ONE LIMITATION THAT MUST TRAVEL WITH THE NUMBER. The model's energy grid is
coarser laterally than the DFT one, so the substitution can only inject the
lateral structure that grid can represent. The share of the native lateral
bound charge's power above the model grid's lateral Nyquist is therefore
reported, and the cross energy is computed on BOTH grids: if they agree, the
truncation does not matter for the energy; if they do not, the measured
improvement is a LOWER BOUND on what a perfect lateral fix would give.

THE READING, set by the user in advance:
  the charged frames' deviation drops clearly -> there is a basis for
      concentrating on how the 3-D bound charge is represented and trained.
  the improvement is small or negative -> that part of the coupling gap can no
      longer be used to explain the total-energy plateau, and the remaining
      energy terms and their compensation have to be examined.
  it improves partly -> record exactly how much is recovered and how much is
      left.

ONE CAVEAT ON THE ONE-TERM CLAIM, from the workstation: e_xsol = int delta*phi
is a one-term change only because there is NO re-solve, so phi is fixed. If
anyone later re-solves after the swap, dE stops being one term and the
derivation has to be redone -- and the s_ion round showed how much a re-solve
moves things.

This does NOT show that the network can learn the reference charge. It settles
whether the line is worth the work.
"""
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

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.extensions as ext
import mace.modules.pb1d_backend as PB
from mace.modules.solvent3d import poisson_phi_periodic

DFT = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single")
GCE44, GCE88 = f"{DFT}/1-44_GCE", f"{DFT}/2-88_GCE"
NEU = f"{DFT}/5-44_neutral_withsolv"
CKDIR = os.environ.get("KIT_CKPT_DIR", "checkpoints")
DATADIR = os.environ.get("KIT_DATA_DIR", "data")
_ALL = [
    (61, f"{GCE44}/cal_61", "NiN44 q=-0.80"),
    (1, f"{GCE44}/cal_1", "NiN44 q=-1.00"),
    (28, f"{GCE44}/cal_28", "NiN44 q=-1.32"),
    (201, f"{GCE88}/cal_1", "NiN88 q=-1.00"),
    (353, f"{GCE88}/cal_153", "NiN88 q=-1.31"),
    (601, f"{NEU}/cal_1", "neutral"),
]
# KIT_FRAMES selects a subset, because the NiN88 DFT payload is not on every
# machine that can run this: "n44" is the three NiN44 charged frames plus the
# neutral one, "n88" the two NiN88 frames, "all" (the default) everything.
# The n44 subset keeps the charge sweep (-0.80, -1.00, -1.32) and the neutral
# isolation but has NO cell-size variation, so a subset run must say so.
_SEL = os.environ.get("KIT_FRAMES", "all").lower()
if _SEL == "n44":
    FRAMES = [f for f in _ALL if f[0] in (61, 1, 28, 601)]
elif _SEL == "n88":
    FRAMES = [f for f in _ALL if f[0] in (201, 353)]
else:
    FRAMES = _ALL
_missing = [f[2] for f in FRAMES if not os.path.exists(f"{f[1]}/RHOB")]

print(f"  KIT_FRAMES={_SEL}: {len(FRAMES)} of {len(_ALL)} frames -- "
      f"{', '.join(f[2] for f in FRAMES)}")
if _missing:
    raise SystemExit(f"  STOP: RHOB missing for {_missing}. This machine does "
                     f"not hold that DFT payload; run those frames where it "
                     f"does rather than dropping them from the table.")
if len(FRAMES) < len(_ALL):
    print(f"  SUBSET RUN -- the table below covers {len(FRAMES)} frames, not "
          f"the six the user named. Report it as a subset and say which "
          f"variation is absent (n44 has no cell-size variation).")

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
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
_comp = ext._slab_compensation_periodic_1d_energy_radial
ext._slab_compensation_periodic_1d_energy_radial = (
    lambda *a, **k: cap.__setitem__("comp", _comp(*a, **k).detach().clone())
    or cap["comp"])
_clo = PB.closure_from_fields
def wrap_clo(n_e, cv, grid, params, tp_):
    cap["cvhar3"] = cv.detach().clone()
    cap["grid"] = grid
    return _clo(n_e, cv, grid, params, tp_)
PB.closure_from_fields = wrap_clo
_solve = PB.PB1DBackend.solve_graph
def wrap_solve(self, *a, **k):
    out = _solve(self, *a, **k)
    cap["rho_layer_z"] = out["rho_layer_z"].detach().clone()
    o = out.get("s3d_obs")
    if o is not None:
        cap["d_b"] = o["d_sup_b"].detach().clone()
        cap["d_i"] = o["d_sup_i"].detach().clone()
        cap["e_xsol"] = float(o["e_xsol"])
        cap["e_self"] = float(o["e_self"])
    return out
PB.PB1DBackend.solve_graph = wrap_solve

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff",
               "solvated": "solvated"},
    arrays_keys={"forces": "forces"},
)
want = {sid: None for sid, _, _ in FRAMES}
for split in ("train", "val"):
    for a in read(os.path.join(DATADIR, f"{split}.xyz"), ":"):
        sid = int(a.info.get("sample_id", -1))
        if sid in want and want[sid] is None:
            want[sid] = a
    if all(v is not None for v in want.values()):
        break


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


def resample_axis(f, n_out, ax):
    """Band-limited periodic resample along one axis. The (g=0) component is
    always kept, so a field with zero plane means keeps them."""
    n_in = f.shape[ax]
    if n_in == n_out:
        return f
    g = torch.fft.fft(f, dim=ax)
    sh = list(f.shape); sh[ax] = n_out
    out = torch.zeros(sh, dtype=g.dtype, device=g.device)
    h = min(n_in, n_out) // 2
    lo = [slice(None)] * 3; lo[ax] = slice(0, h)
    hi = [slice(None)] * 3; hi[ax] = slice(-h, None)
    out[tuple(lo)] = g[tuple(lo)]
    out[tuple(hi)] = g[tuple(hi)]
    del g
    return torch.fft.ifft(out, dim=ax).real * (n_out / n_in)


def resample3d(f, shape_out):
    # z first: it shrinks the array most, so the later axes cost less
    f = resample_axis(f, shape_out[2], 2)
    f = resample_axis(f, shape_out[0], 0)
    return resample_axis(f, shape_out[1], 1)


def resample_fourier(prof, n_out):
    n_in = prof.shape[0]
    if n_in == n_out:
        return prof
    g = torch.fft.fft(prof)
    out = torch.zeros(n_out, dtype=g.dtype, device=g.device)
    h = min(n_in, n_out) // 2
    out[:h] = g[:h]; out[-h:] = g[-h:]
    return torch.fft.ifft(out).real * (n_out / n_in)


def lateral_power_above(f, nx_keep, ny_keep):
    """Share of a field's lateral power above a cut, at fixed z-content:
    what the model's energy grid cannot represent."""
    g = torch.fft.fftn(f, dim=(0, 1))
    p = (g.real ** 2 + g.imag ** 2).sum(dim=2)
    del g                      # ~1 GB complex128 on the native grid
    torch.cuda.empty_cache()
    tot = float(p.sum())
    nx, ny = f.shape[0], f.shape[1]
    fx = torch.fft.fftfreq(nx, device=f.device).abs() * nx
    fy = torch.fft.fftfreq(ny, device=f.device).abs() * ny
    keep = (fx[:, None] <= nx_keep // 2) & (fy[None, :] <= ny_keep // 2)
    return 1.0 - float(p[keep].sum()) / max(tot, 1e-300)


rows, notes = [], []
for sid, dftdir, tag in FRAMES:
    a = want[sid]
    nat = len(a)
    e_dft = float(a.info["energy"])
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(
        ds, batch_size=1))).to(device)
    cap.clear()
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
    grid = cap["grid"]
    cell64 = torch.as_tensor(grid.cell, dtype=torch.float64, device=device)
    shape = tuple(grid.shape); V = grid.volume
    dV = V / float(np.prod(shape))
    nz = shape[2]; lz = float(cell64[2, 2]); dz = lz / nz; area = V / lz
    e_model = float(pred["energy"].sum())
    e3d = float(pred["solvent3d_energy_g"].sum())
    phi = -(cap["cvhar3"] - cap["cvhar3"].mean())
    d_b, d_i = cap["d_b"].to(device), cap["d_i"].to(device)
    rho1 = resample_fourier(cap["rho_layer_z"].to(device),
                            nz)[None, None, :].expand(shape).contiguous()

    # ---- the model's own 3-D term, recomputed through THE Coulomb function
    dl_old = d_b + d_i
    c_old = float((dl_old * phi).sum() * dV)
    s_old = 0.5 * float((dl_old * poisson_phi_periodic(dl_old, cell64)).sum() * dV)
    g_id = abs((c_old + s_old) - e3d)

    # ---- the DFT lateral bound charge, native then resampled
    lat, rb = read_grid(f"{dftdir}/RHOB")
    Vd = float(abs(np.linalg.det(lat)))
    rb = -(rb.to(device)) / Vd
    lat_nat = rb - rb.mean(dim=(0, 1), keepdim=True)
    del rb
    shd = tuple(lat_nat.shape)
    trunc = lateral_power_above(lat_nat, shape[0], shape[1])
    # cross energy on the NATIVE grid, phi upsampled, to bound the truncation
    phi_nat = resample3d(phi.clone(), shd)
    c_nat = float((lat_nat * phi_nat).sum() * (Vd / float(np.prod(shd))))
    del phi_nat
    torch.cuda.empty_cache()
    lat_m = resample3d(lat_nat, shape)
    del lat_nat
    torch.cuda.empty_cache()
    g_pm = float(lat_m.mean(dim=(0, 1)).abs().max())
    c_lat = float((lat_m * phi).sum() * dV)

    # ---- the substituted 3-D term: self-energy on the SUM
    dl_new = d_i + lat_m
    c_new = float((dl_new * phi).sum() * dV)
    p_new = poisson_phi_periodic(dl_new, cell64)
    s_new = 0.5 * float((dl_new * p_new).sum() * dV)
    g_x5 = abs(float((rho1 * p_new).sum() * dV))
    mut_new = s_new - 0.5 * float((d_i * poisson_phi_periodic(d_i, cell64)).sum() * dV) \
        - 0.5 * float((lat_m * poisson_phi_periodic(lat_m, cell64)).sum() * dV)
    mut_old = s_old - 0.5 * float((d_i * poisson_phi_periodic(d_i, cell64)).sum() * dV) \
        - 0.5 * float((d_b * poisson_phi_periodic(d_b, cell64)).sum() * dV)
    del p_new
    corr_b = float(((d_b - d_b.mean()) * (lat_m - lat_m.mean())).sum()
                   / max(float((d_b - d_b.mean()).norm()
                               * (lat_m - lat_m.mean()).norm()), 1e-30))
    e3d_new = c_new + s_new
    dE = e3d_new - e3d
    err0 = (e_model - e_dft) / nat * 1000.0
    err1 = (e_model + dE - e_dft) / nat * 1000.0
    rows.append((tag, nat, err0, err1, err1 - err0, dE, e3d, e3d_new,
                 c_old, c_new, s_old, s_new, mut_old, mut_new, c_lat, c_nat,
                 trunc, g_id, g_pm, g_x5, corr_b, shd, shape))
    print(f"  {tag:>16}: err {err0:+9.3f} -> {err1:+9.3f} meV/atom "
          f"(dE {dE:+.4f} eV over {nat} atoms), truncated power "
          f"{100*trunc:.1f}%, corr(model, DFT lateral) {corr_b:+.4f}",
          flush=True)
    del d_b, d_i, dl_old, dl_new, lat_m, rho1, phi
    torch.cuda.empty_cache()

print(f"\n[GATES]")
gid = max(r[17] for r in rows); gpm = max(r[18] for r in rows)
gx5 = max(r[19] for r in rows)
print(f"   [{'PASS' if gid < 1e-6 else 'FAIL'}] the model's own 3-D term is "
      f"reproduced through THE Coulomb function: worst |(cross+self) - "
      f"solvent3d_energy_g| {gid:.3e} eV -- so dE really is a one-term change")
print(f"   [{'PASS' if gpm < 1e-15 else 'FAIL'}] the substituted lateral "
      f"charge has zero plane means: worst max |plane mean| {gpm:.3e} e/A^3 "
      f"-- net charge, 1-D profile and z-dipole are untouched by construction")
print(f"   [{'PASS' if gx5 < 1e-9 else 'FAIL'}] the 1-D/3-D cross term stays "
      f"zero after the swap: worst |int rho_1d*phi[delta_new]| {gx5:.3e} eV "
      f"-- so comp and E_bl are genuinely unaffected")

print(f"\n[RESULT] the only table: final total energy error, meV/atom")
print(f"  {'frame':>16} {'atoms':>6} {'original':>10} {'substituted':>12} "
      f"{'improvement':>12}")
for r in rows:
    print(f"  {r[0]:>16} {r[1]:6d} {r[2]:+10.3f} {r[3]:+12.3f} "
          f"{r[4]:+12.3f}")
chg = [r for r in rows if r[0] != "neutral"]
if chg:
    m0 = sum(abs(r[2]) for r in chg) / len(chg)
    m1 = sum(abs(r[3]) for r in chg) / len(chg)
    print(f"  {'charged mean |err|':>16} {'':>6} {m0:10.3f} {m1:12.3f} "
          f"{m1-m0:+12.3f}")

print(f"\n[SUPPORTING] the energy pieces that produced it (eV)")
print(f"  {'frame':>16} {'E_3d old':>9} {'E_3d new':>9} {'cross old':>10} "
      f"{'cross new':>10} {'self old':>9} {'self new':>9} {'mutual old':>11} "
      f"{'mutual new':>11}")
for r in rows:
    print(f"  {r[0]:>16} {r[6]:+9.4f} {r[7]:+9.4f} {r[8]:+10.4f} "
          f"{r[9]:+10.4f} {r[10]:+9.4f} {r[11]:+9.4f} {r[12]:+11.4f} "
          f"{r[13]:+11.4f}")

print(f"\n[TRUNCATION] what the model's energy grid cannot carry, and whether "
      f"it matters to the energy")
print(f"  {'frame':>16} {'native grid':>16} {'model grid':>14} "
      f"{'power above':>12} {'cross model':>12} {'cross native':>13} "
      f"{'ratio':>7}")
for r in rows:
    print(f"  {r[0]:>16} {str(r[21]):>16} {str(r[22]):>14} "
          f"{100*r[16]:11.1f}% {r[14]:+12.4f} {r[15]:+13.4f} "
          f"{r[14]/max(abs(r[15]),1e-30)*(1 if r[15]>0 else -1):7.3f}")
print(f"  If the two cross energies agree the truncation does not matter for "
      f"the energy. If they do not, the improvement above is a LOWER BOUND on "
      f"what a perfect lateral fix would give, and that is how it must be\n"
      f"  reported -- the substitution can only inject what the model's grid "
      f"can represent.")
print(f"\n  This does NOT show the network can learn the reference charge. It "
      f"settles whether the line is worth the work.", flush=True)
np.savez(os.path.join(os.environ.get("KIT_OUT", "."),
                      "lateral_bound_swap.npz"),
         tags=np.array([r[0] for r in rows]),
         err0=np.array([r[2] for r in rows]),
         err1=np.array([r[3] for r in rows]),
         dE=np.array([r[5] for r in rows]),
         trunc=np.array([r[16] for r in rows]),
         corr=np.array([r[20] for r in rows]))
print(f"\n  summary saved to lateral_bound_swap.npz")
print("DONE")
