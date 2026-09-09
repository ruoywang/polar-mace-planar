"""How much of the error is unreachable by any 3-D refit, and two anomalies.
(follow-up to the full T1/T4 tables from the workstation, 2026-09-09)

The full per-shell tables made one pattern unmistakable. Sorting the charged
frame's bound shells by how much residual weight the envelope gives them:

  shell     int|bg|  int|delta|   bg/del   model|q| / DFT|q|   gap (eV)
  0-1 A       0.169     0.016      10.6          6.07           +0.785
  3-3.5 A     0.064     0.005      12.8          9.83           -0.167
  3.5-5 A     0.189     0.000        inf         3.44           +0.295
  5-inf       0.809     0.000        inf         0.87           -0.235
  1-1.5 A     0.238     0.374       0.64         1.27           -0.567
  1.5-2 A     0.239     0.892       0.27         0.84           +1.362
  2-2.5 A     0.194     0.333       0.58         0.88           +0.508

Where the residual has weight, the magnitude lands within 27%. Where it has
almost none, the model's charge is 3 to 10 times the reference and is almost
purely 1-D background. That is the representation defect the user predicted
on 2026-09-09: the 1-D background is broadcast over the whole plane including
inside the solute, and where the envelope is zero the residual CANNOT cancel
it, however well the coefficients are fitted.

RESULT, and it corrected the estimate that motivated the script. Summing the
shells above where bg/del was large gave "about +0.53 eV", suspiciously close
to the 0.525 eV plane-average deficit. That estimate is WITHDRAWN: binning by
env_b properly shows the cumulative gap is non-monotonic and changes sign
across thresholds (+0.278, +0.239, +0.004, -0.159 eV at env_b < 1e-4, 1e-3,
1e-2, 3e-2), so the closeness was an artefact of an ad-hoc shell cut. The
defensible floor is the 1e-4 threshold, where the residual holds 1e-4 e
against 1.318 e of background: +0.278 eV, 15.2% of the 1.832 eV bound gap.
No other threshold may be quoted as a floor without naming it.

The reframing matters more than the correction. The strongest-envelope bin
(env_b 0.3-1.0) carries +1.896 eV, 103% of the whole bound gap, on 4.06% of
the volume holding 78.67% of the envelope weight, with the model already at
91% of the reference charge there. So the dominant part of the gap sits where
the residual IS free to act, which makes it a fitting, basis or objective
question rather than a representation impossibility. The representation
defect is real and it is 15%, not the majority.

Two anomalies from the same run are also measured here, because both are
cheap on fields already loaded and neither can be a basis-capacity effect:

A. env_b puts 45.8% (charged) and 46.4% (neutral) of its weight beyond 5 A
   from any atom, and the native |grad s_diel| agrees at 44.1% / 44.0%, so it
   is genuine weight and not the weighted mean being dragged by the 59% of
   the cell out there. RESOLVED, and the answer is a defect: the model's
   switch never saturates. It plateaus at about 0.9447 in the bulk on every
   bin from 19.5 to 36 A, both frames, while the DFT switch reaches exactly
   1.0000 -- a 5.5% systematic bulk dielectric deficit, structural and
   frame-independent. A switch still creeping instead of saturated has a
   small non-vanishing gradient across half the cell, which is where the
   far-field envelope weight comes from. Note that a genuinely FLAT plateau
   would have zero gradient however far below unity it sat, so the mechanism
   is the creep, not the offset. Upstream hypothesis, not established: the
   predicted electron density does not decay to zero in the solvent region
   (cavity_compare measured model n_e ~1e-4 e/A^3 in the open region against
   DFT ~0), and with a switch that is a function of log(n/n_c) a residual
   1e-4 against n_c = 0.015 lands short of saturation exactly like this. The
   same table shows the model missing the soft outer tail: over z 10.5-15.0 A
   the DFT gradient carries 5.07% of its total against the model's 0.81%, and
   at 12.0-13.5 A the model is 4.44e-06 against DFT 2.43e-02.

B. Inside the slab at z 6.0-7.5 A the DFT plane-averaged bound charge is zero
   to 3e-13 while the model puts -9.2e-06 (charged) / -8.8e-06 (neutral)
   e/A^3 there, and the mean potential in that bin is +15.3 / +16.5 V, so it
   becomes -0.074 / -0.067 eV of spurious coupling -- nearly identical in
   both frames, hence structural. The neighbouring bins carry the opposite
   sign, so it is charge-neutral, converted into energy by the huge interior
   potential. GIBBS RINGING FROM BAND-LIMITED UPSAMPLING IS REFUTED: the
   spectrum this script prints shows the model carrying 280x (charged) and
   340x (neutral) LESS power in the top quarter of the band than the DFT
   reference, so the model profile is smoother up there, not ringing. Cause
   open. Threshold-free restatement of the artefact, from the interior slice:
   1.241e-02 e of plane-averaged charge over 97 of 300 planes where DFT has
   3.346e-08 e, nearly charge-neutral, costing -0.0674 eV against a 2.090 V
   mean interior potential, and -0.0550 eV on the neutral frame.

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
from mace.modules.solvent3d import (
    _grad_mag_periodic, normalized_gradient_envelope)

DFT = os.environ.get("KIT_DFT", "/scratch/08384/tg876840/tmp/2-NiN_single")
FRAMES = [(1, f"{DFT}/1-44_GCE/cal_1", "NiN44 q=-1.00"),
          (601, f"{DFT}/5-44_neutral_withsolv/cal_1", "neutral")]
CKDIR = os.environ.get("KIT_CKPT_DIR", "checkpoints")
DATADIR = os.environ.get("KIT_DATA_DIR", "data")
# env_b bins: the first is "effectively zero envelope"
EB = [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.01]
NZB = 30

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
_solve = PB.PB1DBackend.solve_graph
def wrap_solve(self, *a, **k):
    out = _solve(self, *a, **k)
    cap["rho_bound_z"] = out["rho_bound_z"].detach().clone()
    cap["rho_ion_z"] = out["rho_ion_z"].detach().clone()
    o = out.get("s3d_obs")
    if o is not None and "delta_b_grid" in o:
        cap["delta_b"] = o["delta_b_grid"].detach().clone()
        cap["delta_i"] = o["delta_i_grid"].detach().clone()
    return out
PB.PB1DBackend.solve_graph = wrap_solve
_clo = PB.closure_from_fields
def wrap_clo(n_e, cv, grid, params, tp_):
    cap["cvhar3"] = cv.detach().clone(); cap["grid"] = grid
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


def res1d(prof, n):
    if prof.shape[0] == n:
        return prof
    return torch.nn.functional.interpolate(
        prof[None, None].to(torch.float64), size=(n,), mode="linear",
        align_corners=False)[0, 0]


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
    grid = cap["grid"]
    shm = tuple(grid.shape)
    cellm = torch.as_tensor(grid.cell, dtype=torch.float64, device=device)
    lz = float(torch.linalg.norm(cellm[2])); nz = shm[2]; dz = lz / nz
    Vm = grid.volume; dVm = Vm / float(np.prod(shm)); area = Vm / lz
    phi_m = -(cap["cvhar3"] - cap["cvhar3"].mean())
    cavd = grid._solv3d_cavity
    s_diel_m = torch.clamp(cavd[1], 0.0, 1.0)
    gm_m = _grad_mag_periodic(cavd[1], cellm)
    env_b = normalized_gradient_envelope(cavd[1], cellm)
    db = cap["delta_b"].to(device)
    bg_b = res1d(cap["rho_bound_z"].to(device).to(torch.float64),
                 nz)[None, None, :].expand(shm).contiguous()
    mb = bg_b + db
    print(f"\n=================== {tag} ===================", flush=True)

    latd, rb = read_grid(f"{dftdir}/RHOB")
    _, chg = read_grid(f"{dftdir}/CHGCAR")
    Vd = float(abs(np.linalg.det(latd)))
    shd = tuple(rb.shape)
    celld = torch.tensor(latd, device=device)
    rb_d = -(rb.to(device)) / Vd
    ne_d = torch.clamp(chg.to(device) / Vd, min=0.0)
    from pure_python import torch_pb as tp
    gd = tp.TorchGrid(latd, shd, device=str(device), dtype=torch.float64,
                      rspec=True)
    _si_d, s_diel_d, _ = tp.create_cavity_torch(ne_d, gd, cap["params"])
    s_diel_d = torch.clamp(s_diel_d, 0.0, 1.0)
    gm_d = _grad_mag_periodic(s_diel_d, celld)
    rb_dm = to_shape(rb_d, shm)
    del rb, chg, ne_d

    # ---- 1. how much of the gap sits where the envelope cannot reach ----
    gap = (mb - rb_dm) * phi_m
    gtot = float(gap.sum() * dVm)
    ebtot = float(env_b.sum() * dVm)
    print(f"\n[V1] bound cross-energy gap {gtot:+.4f} eV, binned by the "
          f"envelope value itself (no threshold imposed).")
    print(f"  {'env_b range':>14} {'vol%':>6} {'env%':>6} {'int|bg|':>8} "
          f"{'int|del|':>9} {'DFT|q|':>8} {'mdl|q|':>8} {'gap eV':>8} "
          f"{'cum gap':>8}")
    cum = 0.0
    for k in range(len(EB) - 1):
        m = (env_b >= EB[k]) & (env_b < EB[k + 1])
        if not bool(m.any()):
            continue
        gk = float((gap * m).sum() * dVm); cum += gk
        print(f"  {EB[k]:6.0e}-{EB[k+1]:<7.0e} "
              f"{100.0*float(m.sum())/float(np.prod(shm)):6.2f} "
              f"{100.0*float((env_b*m).sum()*dVm)/max(ebtot,1e-30):6.2f} "
              f"{float((bg_b.abs()*m).sum()*dVm):8.3f} "
              f"{float((db.abs()*m).sum()*dVm):9.3f} "
              f"{float((rb_dm.abs()*m).sum()*dVm):8.3f} "
              f"{float((mb.abs()*m).sum()*dVm):8.3f} "
              f"{gk:+8.3f} {cum:+8.3f}", flush=True)
    for thr in (1e-4, 1e-3, 1e-2, 3e-2):
        m = env_b < thr
        print(f"  where env_b < {thr:.0e}: {100.0*float(m.sum())/float(np.prod(shm)):5.2f}% "
              f"of volume holds {float((bg_b.abs()*m).sum()*dVm):.3f} e of "
              f"background against {float((db.abs()*m).sum()*dVm):.4f} e of "
              f"residual, and {float((gap*m).sum()*dVm):+.4f} eV of the gap",
              flush=True)
    print(f"  (the last column is a FLOOR on what no refit of the 3-D "
          f"coefficients can recover: the residual has essentially no\n"
          f"   weight there, so those regions are 1-D background whatever the "
          f"coefficients are. It is not a claim about cause.)")

    # ---- 2. anomaly A: is s_diel saturated in the far field? ----
    print(f"\n[V2] s_diel and |grad s_diel| along z (model vs DFT), to see "
          f"where the far-field envelope weight comes from")
    s_dm = to_shape(s_diel_d, shm); g_dm = to_shape(gm_d, shm)
    edge = torch.linspace(0, nz, NZB + 1).round().long()
    gm_tot = float(gm_m.sum() * dVm); gd_tot = float(g_dm.sum() * dVm)
    print(f"  {'z (A)':>13} {'<s^m>':>7} {'<s^D>':>7} {'<|gs|^m>':>9} "
          f"{'<|gs|^D>':>9} {'gs^m %':>7} {'gs^D %':>7} {'<phi>':>8}")
    for k in range(NZB):
        i0, i1 = int(edge[k]), int(edge[k + 1])
        if i1 <= i0:
            continue
        sl = (slice(None), slice(None), slice(i0, i1))
        print(f"  {i0*dz:6.2f}-{i1*dz:<6.2f} {float(s_diel_m[sl].mean()):7.4f} "
              f"{float(s_dm[sl].mean()):7.4f} {float(gm_m[sl].mean()):9.2e} "
              f"{float(g_dm[sl].mean()):9.2e} "
              f"{100.0*float(gm_m[sl].sum()*dVm)/max(gm_tot,1e-30):7.2f} "
              f"{100.0*float(g_dm[sl].sum()*dVm)/max(gd_tot,1e-30):7.2f} "
              f"{float(phi_m[sl].mean()):8.3f}", flush=True)
    # A fixed 0.99 threshold gave an EMPTY set and a nan on the workstation,
    # and the reason is the finding: the model's switch never reaches 0.99. It
    # plateaus at about 0.9447 in the bulk while DFT saturates at 1.0000, both
    # frames. So report each switch's own ceiling and use a relative cut.
    mx_m = float(s_diel_m.max()); mx_d = float(s_dm.max())
    # This block tests the plateau LEVEL only. It is NOT a test of the
    # far-field envelope weight: that is accounted for in full by a second
    # dielectric interface at the far face of the solvent slab (z ~ 39-42 A),
    # which both fields have and which lies beyond 5 A from every atom simply
    # because the solvent layer is about 20 A thick. An earlier draft asserted
    # here that a switch short of saturation must carry non-vanishing gradient
    # where it is still creeping; the two lines below REFUTE that -- the model
    # carries about 1.06% of its gradient weight inside its own plateau
    # region against DFT's 1.15%, i.e. less, not more.
    print(f"  switch ceiling: model max s_diel {mx_m:.4f}, DFT max {mx_d:.4f}"
          f"  -> model plateau deficit {100.0*(mx_d-mx_m)/max(mx_d,1e-30):.2f}%"
          f" of unity. This is a level difference; the plateau region carries "
          f"almost no gradient weight on either field, so it does not explain "
          f"the far-field envelope share (the second interface does).",
          flush=True)
    for nm2, fld, mx, tot in (("model", s_diel_m, mx_m, gm_tot),
                              ("DFT", s_dm, mx_d, gd_tot)):
        g = gm_m if nm2 == "model" else g_dm
        sat = fld > 0.99 * mx
        if not bool(sat.any()):
            print(f"  {nm2}: no point exceeds 0.99 x its own ceiling", flush=True)
            continue
        print(f"  {nm2}: where s_diel > 0.99 x its ceiling "
              f"({100.0*float(sat.sum())/float(np.prod(shm)):.1f}% of volume): "
              f"mean |grad s| {float(g[sat].mean()):.3e} /A, carrying "
              f"{100.0*float((g*sat).sum()*dVm)/max(tot,1e-30):.2f}% of that "
              f"field's total", flush=True)
    nb = (dm > 5.0) if False else None
    print(f"  bulk check: mean s_diel where the DFT switch is within 1e-4 of "
          f"its ceiling -- model {float(s_diel_m[s_dm > mx_d - 1.0e-4].mean()):.4f} "
          f"vs DFT {float(s_dm[s_dm > mx_d - 1.0e-4].mean()):.4f}", flush=True)

    # ---- 3. anomaly B: ringing in the 1-D bound profile ----
    print(f"\n[V3] spectrum of the model's rho_bound_z on its own grid "
          f"(Gibbs ringing from band-limited upsampling would show as power "
          f"near the Nyquist)")
    pb_z = cap["rho_bound_z"].to(device).to(torch.float64)
    npl = pb_z.shape[0]
    sp = torch.fft.rfft(pb_z).abs()
    sp = sp / torch.clamp(sp.max(), min=1e-30)
    nq = sp.shape[0]
    bands = [(0, nq // 8), (nq // 8, nq // 4), (nq // 4, nq // 2),
             (nq // 2, 3 * nq // 4), (3 * nq // 4, nq)]
    print(f"  profile length {npl}, model 3-D grid nz {nz}")
    for i0, i1 in bands:
        if i1 <= i0:
            continue
        print(f"    modes {i0:4d}-{i1:<4d} ({i0/nq:.2f}-{i1/nq:.2f} Nyquist): "
              f"mean |amp| {float(sp[i0:i1].mean()):.3e}, max "
              f"{float(sp[i0:i1].max()):.3e}", flush=True)
    rbz_d = res1d(rb_d.mean(dim=(0, 1)), npl)
    spd = torch.fft.rfft(rbz_d).abs(); spd = spd / torch.clamp(spd.max(), min=1e-30)
    print(f"    DFT <RHOB>_xy on the same length, top-quarter modes: mean "
          f"{float(spd[3*nq//4:nq].mean()):.3e} (model "
          f"{float(sp[3*nq//4:nq].mean()):.3e})", flush=True)
    # interior: where the DFT plane average is essentially zero
    pz = phi_m.mean(dim=(0, 1))
    mbz = res1d(pb_z, nz); rdz = rb_dm.mean(dim=(0, 1))
    interior = rdz.abs() < 1.0e-9
    if bool(interior.any()):
        print(f"  interior slice (DFT plane average below 1e-9 e/A^3, "
              f"{int(interior.sum())} of {nz} planes):")
        print(f"    DFT  int|.| {float((rdz*interior).abs().sum()*dz*area):.3e} e"
              f"  coupling {float((rdz*pz*interior).sum()*dz*area):+.4f} eV")
        print(f"    model int|.| {float((mbz*interior).abs().sum()*dz*area):.3e} e"
              f"  coupling {float((mbz*pz*interior).sum()*dz*area):+.4f} eV")
        print(f"    net model charge there "
              f"{float((mbz*interior).sum()*dz*area):+.3e} e, mean |phi| "
              f"{float(pz[interior].abs().mean()):.3f} V -- a near-neutral "
              f"profile times a large interior potential", flush=True)
    del (phi_m, s_diel_m, gm_m, env_b, db, bg_b, mb, rb_d, rb_dm, s_diel_d,
         gm_d, s_dm, g_dm, gap, gd, pb_z)
    import gc; gc.collect(); torch.cuda.empty_cache()
print("DONE")
