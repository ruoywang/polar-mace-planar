"""A/B energy re-fit, phase 1: cache the frozen physics and features ONCE.
(user-specified design, 2026-09-10. NiN44 charged/neutral is the primary criterion.)

WHY THIS EXPERIMENT. The lateral-swap run showed that substituting the DFT
lateral bound charge recovers 26.4% of the NiN44 charged energy error but makes
NiN88 2.2x worse, with dE of uniform NEGATIVE sign on every charged frame. The
user's reading: NiN88 is a separate system whose bias may already have been
LEARNED, so the swap's damage there says little -- and the question that
matters is whether a more accurate lateral charge makes the energy EASIER TO
LEARN once the readout is allowed to re-adapt. A one-shot substitution with a
frozen readout cannot answer that, because it charges the physics change with
whatever the readout had absorbed.

  arm   lateral bound charge      trainable
  A     current model prediction  existing energy readout
  B     DFT reference             the same energy readout

Both arms: same checkpoint, same train/val data, same budget. Density, P, the
ionic channel and every other feature frozen, and NO per-system compensation
constant added.

WHAT THE COHORT LOOKS LIKE, measured rather than assumed:
  sid   1-200  NiN44 charged   207 atoms  solvated=1   200 frames, RHOB present
  sid 201-400  NiN88 charged   339 atoms  solvated=1   200 frames, RHOB present
  sid 401-600  unsolvated      207 atoms  solvated=0   200 frames, NO solvent
  sid 601-800  neutral(44)     207 atoms  solvated=1   200 frames, RHOB present
Split 640/80/80, and it is PAIR-AWARE: all 200 charging pairs k <-> 600+k share
geometry to <1e-6 A, and every pair sits wholly within one split (160 train, 20
val, 20 test). So the same-geometry charging check the user asked for is
available on 20 val pairs with no cross-split leakage, and it needs no new DFT.

THE COHORT IS PRESERVED IN FULL, which was not obvious. Arm B needs a DFT RHOB
and only 600 exist -- but the 200 frames without one are exactly the unsolvated
group, where extensions.py skips the residual-3D solvent charge outright
("unsolvated frame: no solvent anywhere"), so E_3d = 0 and arm A is IDENTICALLY
arm B there. No frame is dropped and no arm is a hybrid.

WHAT IS CACHED. Per frame: the inputs to every readout (per atom), the frozen
physics scalars the readout does not produce (comp, E_bl, E_cav, and E_3d for
BOTH arms), the E0 sum, the model's own predicted energy for a gate, the DFT
reference energy, the atom count, the split and the group. After this, fitting
the readout needs no PB solve at all.

TWO GATES, because a cache that misrepresents the model is worse than no cache:
  the captured readout inputs, pushed back through the readouts, must reproduce
  the model's own node_energy sum -- this validates the capture regardless of
  how the feature plumbing works;
  and E_3d^A recomputed through THE Coulomb function must equal
  solvent3d_energy_g, which is what makes the A/B difference a one-term change
  in the target.

Arm B's E_3d is built exactly as lateral_bound_swap.py builds it, from
lat_DFT = rho_b_DFT - plane_mean(rho_b_DFT), so every z-plane sums to zero and
the net charge, the 1-D profile and the z-dipole are untouched. The self-energy
is taken on the SUM of the ionic and bound residuals so the bound-ion
interaction stays inside it.
"""
import math
import os
import sys
import time

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
CKDIR = os.environ.get("KIT_CKPT_DIR", "checkpoints")
DATADIR = os.environ.get("KIT_DATA_DIR", "data")
OUT = os.environ.get("KIT_OUT", ".")
LIMIT = int(os.environ.get("KIT_LIMIT", "0"))       # 0 = all frames


def dftdir(sid):
    """The DFT directory for a sid, or None when the frame is unsolvated."""
    if 1 <= sid <= 200:
        return f"{DFT}/1-44_GCE/cal_{sid}"
    if 201 <= sid <= 400:
        return f"{DFT}/2-88_GCE/cal_{sid - 200}"
    if 601 <= sid <= 800:
        return f"{DFT}/5-44_neutral_withsolv/cal_{sid - 600}"
    return None


def group(sid):
    if 1 <= sid <= 200:
        return "nin44_charged"
    if 201 <= sid <= 400:
        return "nin88_charged"
    if 401 <= sid <= 600:
        return "unsolvated"
    return "neutral44"


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

# ---- capture the readout inputs by pre-hook, and gate that it worked --------
grab = {"feats": [], "outs": []}


def mk_hook(idx):
    def _h(mod, args):
        grab["feats"].append((idx, args[0].detach().to(torch.float32).cpu()))
        return None
    return _h


hooks = [r.register_forward_pre_hook(mk_hook(i))
         for i, r in enumerate(model.readouts)]
n_ro = len(model.readouts)
print(f"  {n_ro} readouts hooked: "
      f"{', '.join(type(r).__name__ for r in model.readouts)}", flush=True)
emb = getattr(model, "embedding_readout", None)
if emb is not None:
    hooks.append(emb.register_forward_pre_hook(mk_hook(-1)))
    print(f"  plus an embedding_readout ({type(emb).__name__}) at index -1",
          flush=True)

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


def read_grid_fast(path):
    """Same layout as the other audits but parsed with np.fromstring: 4 s
    against 15 s per 245 MB file, which is what makes 600 frames feasible."""
    f = open(path)
    f.readline(); s = float(f.readline())
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
    a = np.fromstring(f.read(), dtype=np.float64, sep=' ')[:nx * ny * nz]
    f.close()
    return lat, torch.tensor(np.ascontiguousarray(
        a.reshape(nz, ny, nx).transpose(2, 1, 0)))


def resample_axis(f, n_out, ax):
    n_in = f.shape[ax]
    if n_in == n_out:
        return f
    g = torch.fft.fft(f, dim=ax)
    sh = list(f.shape); sh[ax] = n_out
    out = torch.zeros(sh, dtype=g.dtype, device=g.device)
    h = min(n_in, n_out) // 2
    lo = [slice(None)] * 3; lo[ax] = slice(0, h)
    hi = [slice(None)] * 3; hi[ax] = slice(-h, None)
    out[tuple(lo)] = g[tuple(lo)]; out[tuple(hi)] = g[tuple(hi)]
    del g
    return torch.fft.ifft(out, dim=ax).real * (n_out / n_in)


def resample3d(f, shape_out):
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


frames = []
for split in ("train", "val", "test"):
    for a in read(os.path.join(DATADIR, f"{split}.xyz"), ":"):
        frames.append((int(a.info["sample_id"]), split, a))
frames.sort(key=lambda t: t[0])
if LIMIT:
    sel, keep = [], {}
    for sid, split, a in frames:
        k = (group(sid), split)
        keep[k] = keep.get(k, 0)
        if keep[k] < LIMIT:
            keep[k] += 1
            sel.append((sid, split, a))
    frames = sel
    print(f"  KIT_LIMIT={LIMIT}: {len(frames)} frames "
          f"(up to {LIMIT} per group per split) -- a SMOKE subset, not the "
          f"experiment", flush=True)
print(f"  {len(frames)} frames to cache", flush=True)

recs = []
g_feat, g_e3d = [], []
t0 = time.time()
for n, (sid, split, atoms) in enumerate(frames):
    grab["feats"].clear(); cap.clear()
    cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(
        ds, batch_size=1))).to(device)
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
    nat = int(b["ptr"][1] - b["ptr"][0])
    solv = float(atoms.info.get("solvated", 1.0))
    e_dft = float(atoms.info["energy"])
    e_model = float(pred["energy"].sum())
    e3d_a = float(pred["solvent3d_energy_g"].sum()) if solv >= 0.5 else 0.0
    ecav = float(pred.get("cavity_energy_g", torch.zeros(1)).sum())
    ebl = float(pred.get("baseline_coupling_energy_g", torch.zeros(1)).sum())
    comp = float(cap["comp"].sum()) if "comp" in cap else 0.0
    node_e = float(pred["node_energy"].sum())

    # ---- arm B: only the lateral bound charge changes ----
    e3d_b = e3d_a
    dE = 0.0
    if solv >= 0.5 and "d_b" in cap:
        grid = cap["grid"]
        cell64 = torch.as_tensor(grid.cell, dtype=torch.float64, device=device)
        shape = tuple(grid.shape); Vc = grid.volume
        dV = Vc / float(np.prod(shape))
        phi = -(cap["cvhar3"] - cap["cvhar3"].mean())
        d_b, d_i = cap["d_b"].to(device), cap["d_i"].to(device)
        dl_old = d_b + d_i
        c_old = float((dl_old * phi).sum() * dV)
        s_old = 0.5 * float((dl_old * poisson_phi_periodic(dl_old, cell64)
                             ).sum() * dV)
        g_e3d.append(abs((c_old + s_old) - e3d_a))
        lat, rb = read_grid_fast(f"{dftdir(sid)}/RHOB")
        Vd = float(abs(np.linalg.det(lat)))
        rb = -(rb.to(device)) / Vd
        lat_m = resample3d(rb - rb.mean(dim=(0, 1), keepdim=True), shape)
        del rb
        dl_new = d_i + lat_m
        c_new = float((dl_new * phi).sum() * dV)
        s_new = 0.5 * float((dl_new * poisson_phi_periodic(dl_new, cell64)
                             ).sum() * dV)
        e3d_b = c_new + s_new
        dE = e3d_b - e3d_a
        del d_b, d_i, dl_old, dl_new, lat_m, phi
        torch.cuda.empty_cache()

    # ---- readout inputs, and the gate that the capture is faithful ----
    feats = {}
    for idx, t in grab["feats"]:
        feats[idx] = t if idx not in feats else torch.cat([feats[idx], t], 0)
    with torch.no_grad():
        chk = 0.0
        for idx, t in sorted(feats.items()):
            ro = model.readouts[idx] if idx >= 0 else model.embedding_readout
            o = ro(t.to(device=device, dtype=torch.float64), None)
            chk += float(o[:, 0].sum() if o.dim() > 1 else o.sum())
    g_feat.append(abs(chk - node_e))
    recs.append(dict(sid=sid, split=split, group=group(sid), nat=nat,
                     solv=solv, e_dft=e_dft, e_model=e_model, node_e=node_e,
                     e3d_a=e3d_a, e3d_b=e3d_b, dE=dE, ecav=ecav, ebl=ebl,
                     comp=comp,
                     feats={k: v.numpy() for k, v in feats.items()}))
    if (n + 1) % 25 == 0 or n == len(frames) - 1:
        el = time.time() - t0
        print(f"  [{n+1}/{len(frames)}] sid {sid} {group(sid):>14} "
              f"dE {dE:+.4f} eV | {el/(n+1):.2f} s/frame, eta "
              f"{el/(n+1)*(len(frames)-n-1)/60:.1f} min", flush=True)

print(f"\n[GATES]")
gf = max(g_feat) if g_feat else float("nan")
ge = max(g_e3d) if g_e3d else float("nan")
print(f"   [{'PASS' if gf < 1e-6 else 'FAIL'}] captured readout inputs "
      f"reproduce the model's own node_energy sum: worst |diff| {gf:.3e} eV "
      f"over {len(g_feat)} frames")
print(f"   [{'PASS' if ge < 1e-6 else 'FAIL'}] E_3d^A recomputed through THE "
      f"Coulomb function equals solvent3d_energy_g: worst |diff| {ge:.3e} eV "
      f"over {len(g_e3d)} solvated frames")
if not (gf < 1e-6 and ge < 1e-6):
    print(f"   A GATE FAILED -- the cache misrepresents the model, so it is "
          f"written but must not be fitted.", flush=True)

path = os.path.join(OUT, "ab_refit_cache.pt")
torch.save(dict(recs=recs, n_readouts=n_ro,
                has_embedding=emb is not None,
                gate_feat=gf, gate_e3d=ge), path)
sz = os.path.getsize(path) / 1e9
print(f"\n  cache written: {path} ({sz:.2f} GB, {len(recs)} frames)")
import collections
cg = collections.Counter((r["group"], r["split"]) for r in recs)
for k in sorted(cg):
    dd = [r["dE"] for r in recs if (r["group"], r["split"]) == k]
    print(f"    {k[0]:>14} {k[1]:>5}: {cg[k]:4d} frames, dE mean "
          f"{np.mean(dd):+.4f} eV, |dE| max {np.max(np.abs(dd)):.4f}")
for h in hooks:
    h.remove()
print("DONE")
