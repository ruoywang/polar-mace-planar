"""Where does the 0.9444 plateau come from? (step 1, user-fixed plan 2026-09-09)

The model's dielectric switch plateaus at 0.9447 (0.9444 inside the mask used
below) across the solvent region on both frames, while the DFT switch reaches
1.0000 -- a 5.5% systematic deficit, frame-independent and stable to four
digits. The candidate explanation is that the PREDICTED electron density does
not decay to zero out there, so the switch never saturates.

CRITERIA FIXED BY THE USER, and they change what counts as support:

  - Use the actual parameters, the actual grid and the FULL cavity generation
    pipeline, computing POINTWISE and aggregating afterwards. Substituting a
    mean density into a switch formula is not a test: the pipeline may be
    non-local, in which case s_diel is not a pointwise function of n_e at all.
  - Reproducing 0.9444 proves only that the computation path matches. It is
    not evidence for the hypothesis.
  - Support requires that replacing ONLY the suspect density tail restores the
    plateau. Nothing weaker.
  - If the plateau is not reproduced, find which step differs. Do NOT declare
    the recipe or the parameters wrong.
  - This must not become a large waiting project.

So, four parts, all on fields already loaded:

A. PATH CONSISTENCY, self-calibrated. Recompute the cavity from the model's
   own captured n_e through create_cavity_torch with the captured params on
   the model grid, and compare POINTWISE against the cavity the model actually
   used. No fixed tolerance is chosen: the full pipeline is called TWICE on
   the identical input and the call-to-call difference is the noise floor the
   comparison is judged against. This matters because the code is known to be
   irreproducible run to run at up to 2.2e-09 relative on energy scalars, so
   any fixed criterion at or below about 1e-8 could not separate a real path
   difference from noise -- and a single run would then be unable to say
   which. Note the 2.2e-09 was measured BETWEEN runs on energy scalars, not
   on two cavity fields inside one run, which is why the floor is measured
   here rather than borrowed. Three verdicts are possible: below the measured
   floor (path matches as far as one run can tell), above the floor but under
   1e-8 (INDETERMINATE from one run, reported as neither agreement nor
   finding), or clearly above (localised by z and by s_diel value, which is
   the "which step differs" diagnostic and not a verdict).

B. IS THE SWITCH ITSELF THE SAME MAP? Bin s_diel by n_e for both fields on
   their own grids and report the mean AND the spread within each bin. If the
   same n_e gives the same s_diel, the switch is identical and any plateau
   difference is the density's. If not, the switch or its parameters differ.
   The spread also answers whether the map is pointwise at all -- a non-local
   pipeline shows scatter inside every bin, and then no pointwise reasoning
   about "plugging in a density" is valid in either direction.

C. THE TAIL SWAP, in both directions so the interpolation direction cannot be
   the explanation. Define the suspect region as the mask where the DFT switch
   sits within 1e-4 of its own ceiling -- the same mask that produced 0.9444.
   Then rebuild the cavity through the full pipeline from a hybrid density:
     on the model grid : model n_e outside the mask, DFT n_e inside it
     on the native grid: DFT n_e outside the mask, model n_e inside it
   If the tail is the cause, the first should rise towards 1.0000 and the
   second should fall towards 0.9444. Interpolation smooths whichever density
   is carried across, so running both directions is what keeps that from being
   the answer.

D. THE DENSITIES THEMSELVES, as distributions and not means: percentiles of
   both inside the mask, against the switch parameters actually in use.

Paths come from env vars with the LS6 values as defaults.
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
import mace.modules.pb1d_backend as PB

DFT = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single")
FRAMES = [(1, f"{DFT}/1-44_GCE/cal_1", "NiN44 q=-1.00"),
          (601, f"{DFT}/5-44_neutral_withsolv/cal_1", "neutral")]
CKDIR = os.environ.get("KIT_CKPT_DIR", "checkpoints")
DATADIR = os.environ.get("KIT_DATA_DIR", "data")
NEB = [0.0, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1.5e-2, 3e-2, 1e9]
PCT = [1, 5, 25, 50, 75, 95, 99, 100]

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
_clo = PB.closure_from_fields
def wrap_clo(n_e, cv, grid, params, tp_):
    cap["n_e"] = n_e.detach().clone()      # the density the cavity is built on
    cap["grid"] = grid
    cap["params"] = params
    return _clo(n_e, cv, grid, params, tp_)
PB.closure_from_fields = wrap_clo

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


def to_shape(t, shape):
    if tuple(t.shape) == tuple(shape):
        return t
    return torch.nn.functional.interpolate(
        t[None, None].to(torch.float64), size=tuple(shape), mode="trilinear",
        align_corners=False)[0, 0].contiguous()


def pcts(x):
    q = torch.tensor([p / 100.0 for p in PCT], dtype=torch.float64,
                     device=x.device)
    return torch.quantile(x.flatten().float(), q.float()).double().tolist()


from pure_python import torch_pb as tp

for sid, dftdir, tag in FRAMES:
    a = want[sid]
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(
        ds, batch_size=1))).to(device)
    cap.clear()
    with torch.no_grad():
        model(b.to_dict(), compute_force=False, training=False)
    gm = cap["grid"]; params = cap["params"]
    shm = tuple(gm.shape)
    ne_m = cap["n_e"].to(device)
    s_used = torch.clamp(gm._solv3d_cavity[1], 0.0, 1.0)
    print(f"\n=================== {tag} ===================", flush=True)
    ks = [k for k in ("NC_K", "NC_SIGMA", "SIGMA_K", "TAU", "EB", "NELECT")
          if hasattr(params, "get") and params.get(k) is not None]
    print(f"  switch parameters in use: "
          + ", ".join(f"{k}={params.get(k)}" for k in ks)
          + f"  (all keys: {sorted(list(params.keys()))[:14]}...)", flush=True)

    # ---------------- A. path consistency, pointwise ----------------
    # SELF-CALIBRATED, no chosen tolerance. The workstation pointed out that a
    # fixed threshold here is unsafe: this code is known not to be reproducible
    # run to run (up to 2.2e-09 relative on e_bl), so any criterion at or below
    # about 1e-8 cannot separate a real path difference from noise, and a
    # single run then cannot say which it is. But that 2.2e-09 was measured
    # BETWEEN runs on energy scalars, which is not the quantity compared here:
    # part A compares two cavity fields built from the SAME captured n_e inside
    # ONE run. That noise floor has never been measured, so measure it instead
    # of guessing: call the full pipeline twice on the identical input and use
    # the call-to-call difference as the floor the comparison is judged against.
    _si, s_re, _x = tp.create_cavity_torch(ne_m, gm, params)
    s_re = torch.clamp(s_re, 0.0, 1.0)
    _si_b, s_re_b, _x_b = tp.create_cavity_torch(ne_m, gm, params)
    s_re_b = torch.clamp(s_re_b, 0.0, 1.0)
    noise = float((s_re - s_re_b).abs().max())
    d = (s_re - s_used).abs()
    dmax = float(d.max())
    print(f"\n[A] recomputed the cavity from the model's own captured n_e "
          f"through the full pipeline, same grid and params")
    print(f"  call-to-call noise floor on IDENTICAL input, same run: "
          f"max |s_a - s_b| {noise:.3e}")
    print(f"  pointwise |s_recomputed - s_used|: max {dmax:.3e}, "
          f"mean {float(d.mean()):.3e}, 99th pct "
          f"{float(torch.quantile(d.flatten().float(), 0.99)):.3e}")
    print(f"  ceilings: recomputed {float(s_re.max()):.4f}, "
          f"as used {float(s_used.max()):.4f}")
    if dmax <= max(10.0 * noise, 1.0e-14):
        print(f"  path difference is INDISTINGUISHABLE from the call-to-call "
              f"floor ({dmax:.3e} against {noise:.3e}). The path matches as "
              f"far as one run can tell, which proves the computation path "
              f"only and is NOT evidence for any hypothesis.")
    elif dmax <= 1.0e-8:
        print(f"  path difference {dmax:.3e} is above the call-to-call floor "
              f"{noise:.3e} but within the band where this code is known to "
              f"be irreproducible between runs (2.2e-09 observed on energy "
              f"scalars). Reported as INDETERMINATE from a single run -- not "
              f"agreement, and not a finding. Localisation follows for "
              f"information only.")
    if dmax > max(10.0 * noise, 1.0e-14):
        print(f"  PATH DIFFERS -- localising rather than concluding:")
        lz = float(torch.linalg.norm(torch.as_tensor(
            gm.cell, dtype=torch.float64, device=device)[2]))
        nz = shm[2]
        for k in range(0, nz, max(1, nz // 12)):
            sl = (slice(None), slice(None), slice(k, min(k + nz // 12, nz)))
            print(f"    z {k*lz/nz:6.2f}: max diff {float(d[sl].max()):.3e}, "
                  f"<s_used> {float(s_used[sl].mean()):.4f}", flush=True)
        for lo, hi in ((0.0, 0.01), (0.01, 0.5), (0.5, 0.9), (0.9, 1.01)):
            m = (s_used >= lo) & (s_used < hi)
            if bool(m.any()):
                print(f"    s_used in [{lo},{hi}): max diff "
                      f"{float(d[m].max()):.3e} over "
                      f"{100.0*float(m.sum())/float(np.prod(shm)):.1f}% of vol",
                      flush=True)

    # ---------------- DFT side, native ----------------
    latd, chg = read_grid(f"{dftdir}/CHGCAR")
    Vd = float(abs(np.linalg.det(latd)))
    shd = tuple(chg.shape)
    ne_d = torch.clamp(chg.to(device) / Vd, min=0.0)
    gd = tp.TorchGrid(latd, shd, device=str(device), dtype=torch.float64,
                      rspec=True)
    _si2, s_d, _x2 = tp.create_cavity_torch(ne_d, gd, params)
    s_d = torch.clamp(s_d, 0.0, 1.0)
    del chg
    mx_d = float(s_d.max())
    M_d = s_d > mx_d - 1.0e-4                       # the suspect region
    M_m = to_shape(M_d.to(torch.float64), shm) > 0.5
    print(f"\n  DFT cavity ceiling {mx_d:.4f}; suspect mask = within 1e-4 of "
          f"it, {100.0*float(M_d.sum())/float(np.prod(shd)):.1f}% of the "
          f"native cell ({100.0*float(M_m.sum())/float(np.prod(shm)):.1f}% "
          f"after mapping to the model grid)")
    print(f"  plateau inside the mask: model {float(s_used[M_m].mean()):.4f} "
          f"+- {float(s_used[M_m].std()):.4f}, DFT "
          f"{float(s_d[M_d].mean()):.4f} +- {float(s_d[M_d].std()):.4f}",
          flush=True)

    # ---------------- B. is the switch the same map? ----------------
    print(f"\n[B] s_diel binned by n_e, each field on its OWN grid. Mean and "
          f"spread; spread also says whether the map is pointwise at all.")
    print(f"  {'n_e bin (e/A^3)':>22} | {'model <s>':>10} {'sd':>8} {'vol%':>7} "
          f"| {'DFT <s>':>9} {'sd':>8} {'vol%':>7}")
    for k in range(len(NEB) - 1):
        mm = (ne_m >= NEB[k]) & (ne_m < NEB[k + 1])
        md = (ne_d >= NEB[k]) & (ne_d < NEB[k + 1])
        if not (bool(mm.any()) or bool(md.any())):
            continue
        hi = "inf" if NEB[k + 1] > 1e8 else f"{NEB[k+1]:.1e}"
        f1 = (f"{float(s_used[mm].mean()):10.4f} {float(s_used[mm].std()):8.4f} "
              f"{100.0*float(mm.sum())/float(np.prod(shm)):7.2f}"
              if bool(mm.any()) else f"{'-':>10} {'-':>8} {0.0:7.2f}")
        f2 = (f"{float(s_d[md].mean()):9.4f} {float(s_d[md].std()):8.4f} "
              f"{100.0*float(md.sum())/float(np.prod(shd)):7.2f}"
              if bool(md.any()) else f"{'-':>9} {'-':>8} {0.0:7.2f}")
        print(f"  {NEB[k]:9.1e}-{hi:<11} | {f1} | {f2}", flush=True)
    print(f"  (same n_e -> same s_diel in both columns means the switch is "
          f"identical and any plateau difference is the density's. A large sd "
          f"inside a bin means the pipeline is NOT a pointwise function of "
          f"n_e, and then no 'plug in a density' argument is valid either way.)")

    # ---------------- C. the tail swap, both directions ----------------
    print(f"\n[C] tail swap through the FULL pipeline, both directions")
    ne_d_m = to_shape(ne_d, shm)
    hyb_m = torch.where(M_m, ne_d_m, ne_m)
    _s1, s_hyb_m, _y1 = tp.create_cavity_torch(hyb_m, gm, params)
    s_hyb_m = torch.clamp(s_hyb_m, 0.0, 1.0)
    ne_m_d = to_shape(ne_m, shd)
    hyb_d = torch.where(M_d, ne_m_d, ne_d)
    _s2, s_hyb_d, _y2 = tp.create_cavity_torch(hyb_d, gd, params)
    s_hyb_d = torch.clamp(s_hyb_d, 0.0, 1.0)
    print(f"  model grid, DFT tail inside the mask : plateau "
          f"{float(s_hyb_m[M_m].mean()):.4f} (was {float(s_used[M_m].mean()):.4f}, "
          f"target {float(s_d[M_d].mean()):.4f}), ceiling "
          f"{float(s_hyb_m.max()):.4f}")
    print(f"  native grid, model tail inside the mask: plateau "
          f"{float(s_hyb_d[M_d].mean()):.4f} (was {float(s_d[M_d].mean()):.4f}, "
          f"target {float(s_used[M_m].mean()):.4f}), ceiling "
          f"{float(s_hyb_d.max()):.4f}")
    r1 = ((float(s_hyb_m[M_m].mean()) - float(s_used[M_m].mean()))
          / max(float(s_d[M_d].mean()) - float(s_used[M_m].mean()), 1e-30))
    r2 = ((float(s_hyb_d[M_d].mean()) - float(s_d[M_d].mean()))
          / max(float(s_used[M_m].mean()) - float(s_d[M_d].mean()), 1e-30))
    print(f"  fraction of the gap closed by swapping the tail alone: "
          f"{100.0*r1:.1f}% (model grid), {100.0*r2:.1f}% (native grid)")
    print(f"  (both directions are reported because interpolation smooths "
          f"whichever density is carried across. Only if BOTH move most of "
          f"the way does the tail explain the plateau; the numbers above are "
          f"the measurement, not an attribution.)", flush=True)

    # ---------------- D. the densities, as distributions ----------------
    print(f"\n[D] n_e inside the suspect mask, percentiles {PCT}")
    print(f"  model : " + "  ".join(f"{v:.3e}" for v in pcts(ne_m[M_m])))
    print(f"  DFT   : " + "  ".join(f"{v:.3e}" for v in pcts(ne_d[M_d])))
    thr = params.get("NC_K", None) if hasattr(params, "get") else None
    print(f"  switch threshold NC_K = {thr}; model mean "
          f"{float(ne_m[M_m].mean()):.3e}, DFT mean "
          f"{float(ne_d[M_d].mean()):.3e} e/A^3", flush=True)
    del ne_m, ne_d, s_used, s_re, s_d, s_hyb_m, s_hyb_d, hyb_m, hyb_d
    del ne_d_m, ne_m_d, M_d, M_m, gd, d
    import gc; gc.collect(); torch.cuda.empty_cache()
print("DONE")
