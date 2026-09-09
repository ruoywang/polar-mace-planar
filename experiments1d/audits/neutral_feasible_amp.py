"""Neutral frame: finish the verdict with ONE synthesis pass (user directive
2026-09-09).

Correction being applied: the previous run reported "S_ref above the family's
attainable range" for basis A. That was a limit of the nu-parametrisation, not
of the subspace. On the hyperplane {cross = cross_ref} the self-energy is a
convex quadratic, so its range is [min_self, infinity): any self >= min_self is
attainable. Confirmed offline from the saved matrices — the line from the
min-self point through the previous pick reaches cross 1.0000x and self 1.0000x
at t = +1.105, with point rms/ref 0.3155 against the basis-A cap of 0.320.

What is left is the amplitude, which is not a quadratic form and needs the
field. So: generate many feasible points per basis (cross = ref and self = ref
exactly, by construction), keep those inside the point-error cap, and rebuild
their fields in a single synthesis pass to measure |q|. No 111-minute rerun.

Feasible-point generator: alpha(t) = alpha_min + t*d with d in null(x) keeps
cross fixed; self(alpha(t)) = S_ref is a scalar quadratic in t, so each
direction d gives up to two exactly feasible points.
cwd = gate_bl (reads jss_601_*.npz written by joint_subspace_solve).
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, "/work/08384/tg876840/ls6/repos/cep-dip-python-pb")
sys.path.insert(0, "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy")

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.pb1d_backend as PB
from mace.modules.solvent3d import (
    normalized_gradient_envelope, poisson_phi_periodic)

SID = 601
NPTS = 80000          # must match the run that wrote the npz
EPS_SQ = 1.0e-3
NDIR = 24             # random null-space directions per basis
KEEP = 4              # candidates rebuilt per basis (lowest point error)
FILES = [("A", "jss_601_A.npz", [0]),
         ("A+s_diel", "jss_601_Aps_diel.npz", [0, 1]),
         ("A+env^0.5", "jss_601_Apenv0.5.npz", [0, 2])]

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
model = torch.load(f="checkpoints/s3d_gate_bl2_run-123.model", map_location=device).to(device)
ck = torch.load("checkpoints/s3d_gate_bl2_run-123_epoch-33.pt", map_location=device)
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
    cap["cvhar3"] = cv.detach().clone(); cap["grid"] = grid
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
frame = None
for split in ("train", "val"):
    for a in read(f"data/{split}.xyz", ":"):
        if int(a.info.get("sample_id", -1)) == SID:
            frame = a; break
    if frame is not None:
        break
a = frame
cfg = mace_data.config_from_atoms(a, key_specification=kspec)
ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                       cutoff=float(model.r_max))]
b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
with torch.no_grad():
    model(b.to_dict(), compute_force=False, training=False)
grid = cap["grid"]
cell64 = torch.as_tensor(grid.cell, dtype=torch.float64, device=device)
shape = tuple(grid.shape)
V = grid.volume; dV = V / float(np.prod(shape))
phi = -(cap["cvhar3"] - cap["cvhar3"].mean())
cavd = grid._solv3d_cavity
env_b = normalized_gradient_envelope(cavd[1], cell64)
s_diel = torch.clamp(cavd[1], 0.0, 1.0)
env_sq = env_b / torch.sqrt(env_b + EPS_SQ)
env_sq = env_sq / torch.clamp(env_sq.max(), min=1e-30)
CH = [env_b, s_diel, env_sq]        # neutral frame: ion channel gated off
CHm = [torch.clamp(w.mean(dim=(0, 1)), min=1e-12) for w in CH]
sigmas = [float(x) for x in (model.solvent3d_sigmas.tolist()
                             if torch.is_tensor(model.solvent3d_sigmas)
                             else model.solvent3d_sigmas)]
nat = len(a); K1 = nat * len(sigmas) * 9
pos_frac = torch.remainder(
    torch.as_tensor(a.get_positions(), device=device, dtype=torch.float64)
    @ torch.linalg.inv(cell64), 1.0)
beobj = model._pb1d_backend

cands = []          # (basis, label, coeff vector over the full column space,
                    #  cross/ref, self/ref, rms/ref, cap)
rng = np.random.default_rng(5)
for bname, fn, chans in FILES:
    d = np.load(fn)
    M, bb, S, x = (d[k].astype(float) for k in ("M", "b", "S", "x"))
    al_prev = d["alpha"].astype(float)
    coeffs = d["coeffs"].astype(float)
    X_ref, S_ref, Q_ref, rms_all_ref, rms_deep_ref = d["refs"]
    Sinv = np.linalg.pinv(S, rcond=1e-10)
    xSx = float(x @ Sinv @ x)
    a_min = Sinv @ x * (X_ref / xSx)
    # y^2 recovered from the previous run's reported quantities is not needed:
    # compare point error through the quadratic part, and convert with the
    # previous pick as the anchor (its rms/ref was printed by the earlier run)
    quad = lambda v: float(v @ M @ v - 2 * v @ bb)
    # plain L2 solution = family vector 0 -> its quadratic value sets the cap
    e0 = np.zeros(len(x)); e0[0] = 1.0
    # y^2 from the plain fit: rms_plain/ref is recoverable as sqrt((y2+quad)/N)
    # but y2 is unknown here; solve for it using the identity that the plain
    # fit is the minimiser of the unweighted objective -> its rms was printed.
    # Instead of relying on printed text, recompute y2 from the ridge-free
    # normal equations: the minimiser is M^-1 b, and rms_min^2*N = y2 + quad_min
    Mi = np.linalg.pinv(M, rcond=1e-12)
    a_ls = Mi @ bb
    # y2 is common to all candidates, so differences in quad are what matter;
    # anchor on the *previous* pick, whose rms/ref the earlier run printed:
    ANCHOR = {"A": 0.286, "A+s_diel": 0.218, "A+env^0.5": 0.168}[bname]
    y2 = (ANCHOR * rms_all_ref) ** 2 * NPTS - quad(al_prev)
    def rms_ratio(v):
        return math.sqrt(max(y2 + quad(v), 0.0) / NPTS) / rms_all_ref
    rms_plain = rms_ratio(a_ls)
    capv = 1.5 * rms_plain
    # feasible points: alpha_min + t d, d in null(x), self = S_ref
    n = len(x)
    Q, _ = np.linalg.qr(np.column_stack([x] + [rng.normal(size=n)
                                               for _ in range(n - 1)]))
    null = Q[:, 1:]
    found = []
    dirs = [al_prev - a_min] + [null @ rng.normal(size=n - 1) for _ in range(NDIR)]
    for k, dv in enumerate(dirs):
        A2 = 0.5 * float(dv @ S @ dv)
        B2 = float(a_min @ S @ dv)
        C2 = 0.5 * float(a_min @ S @ a_min) - S_ref
        if abs(A2) < 1e-30:
            continue
        disc = B2 * B2 - 4 * A2 * C2
        if disc < 0:
            continue
        for t in ((-B2 + math.sqrt(disc)) / (2 * A2),
                  (-B2 - math.sqrt(disc)) / (2 * A2)):
            v = a_min + t * dv
            found.append((rms_ratio(v), k, t, v))
    found.sort(key=lambda z: z[0])
    print(f"\n{bname}: plain rms/ref {rms_plain:.3f}, cap {capv:.3f}; "
          f"{len(found)} exactly feasible points, best point-error "
          f"{found[0][0]:.3f}", flush=True)
    for r, k, t, v in found[:KEEP]:
        full = np.zeros(coeffs.shape[1])
        for j in range(len(v)):
            full += v[j] * coeffs[j]
        lbl = f"dir{k}{'+' if t > 0 else '-'}"
        cands.append((bname, lbl, torch.tensor(full, device=device),
                      float(x @ v) / X_ref,
                      0.5 * float(v @ S @ v) / S_ref, r, capv,
                      float(Q_ref)))
        print(f"    keep {lbl}: cross {float(x@v)/X_ref:.4f} self "
              f"{0.5*float(v@S@v)/S_ref:.4f} rms/ref {r:.3f}", flush=True)

print(f"\nrebuilding {len(cands)} candidate fields in one synthesis pass",
      flush=True)
flds = [torch.zeros(shape, dtype=torch.float64, device=device) for _ in cands]
cvec = torch.zeros(nat, len(sigmas), 9, dtype=torch.float64, device=device)
col = 0
for ia in range(nat):
    for isig in range(len(sigmas)):
        for lm in range(9):
            todo = []
            for ci in range(len(CH)):
                for j, c in enumerate(cands):
                    val = float(c[2][ci * K1 + col])
                    if abs(val) > 1e-16:
                        todo.append((j, ci, val))
            if todo:
                cvec.zero_(); cvec[ia, isig, lm] = 1.0
                Gk = grid.ifft_real(beobj._gto_net_density_g(
                    grid, pos_frac, cvec, sigmas)) / V
                projs = {}
                for j, ci, val in todo:
                    if ci not in projs:
                        w = CH[ci]; wm = CHm[ci]
                        raw = w * Gk
                        r = raw.mean(dim=(0, 1)) / wm
                        projs[ci] = raw - r[None, None, :] * w
                    flds[j] += val * projs[ci]
            col += 1
    if ia % 40 == 0:
        print(f"   rebuild {ia}/{nat}", flush=True)

print(f"\n{'basis':>11} {'cand':>9} {'cross/ref':>10} {'self/ref':>9} "
      f"{'rms/ref':>8} {'cap':>6} {'|q|/ref':>8} {'verdict':>8}")
for (bname, lbl, _, crr, ser, r, capv, Q_ref), f in zip(cands, flds):
    cr = float((f * phi).sum() * dV)
    se = 0.5 * float((f * poisson_phi_periodic(f, cell64)).sum() * dV)
    am = float(f.abs().sum() * dV) / Q_ref
    ok = (abs(crr - 1) < 0.05 and abs(ser - 1) < 0.15
          and abs(am - 1) < 0.20 and r <= capv)
    print(f"{bname:>11} {lbl:>9} {crr:10.4f} {ser:9.4f} {r:8.3f} {capv:6.3f} "
          f"{am:8.3f} {'PASS' if ok else 'FAIL':>8}", flush=True)
print("DONE  (cross and self are exact by construction; the amplitude is the "
      "quantity this pass was for)")
