"""Can the native local_electron_energy head FIT the charging residual when it
is the only thing being trained? (user-specified design, 2026-09-11)

WHAT THIS SEPARATES. The 40-epoch joint run left the head contributing
d(le) = -0.040 eV against a residual of +4.87 eV. That measures what joint
training PRODUCED, not what the head CAN produce, and it is not a licence to
multiply the output by 121. Freezing everything else and training only this
head separates two very different conclusions: "the head cannot express the
charging difference" from "joint training never handed it the task".

A MECHANISM CORRECTION THIS SCRIPT CHECKS WITH THE RIGHT METRIC. I previously
wrote that charge reaches the head only through the solvent field feedback at
extensions.py:2562. That is wrong. At extensions.py:2415-2436 a Fukui-weighted
normalisation drives the monopole channels to the requested total charge,

    Q_p_S = (total_charge + (total_spin - 1))[batch]
    spin_charge_density[:, 0, :n_sig] += fukui * ((Q_p_S/2) - pred_total_0) / n_sig

and field_independent_spin_charge_density is cloned at 2444, i.e. AFTER that.
So charges_0 carries the total charge from the start. My "0.36% per-coefficient
max difference" was also the wrong statistic: one electron spread over 207
atoms and 3 radial channels is ~0.0016 e per coefficient, which against
coefficients of order 0.5 is ~0.3% -- exactly what was measured, and consistent
with the whole electron being PRESENT and distributed rather than lost. The
right statistic is the monopole sum, which must equal the requested charge, and
it is checked here for charges_0 and charges_induced separately.

WHY CACHING IS EXACT, NOT AN APPROXIMATION. The head's output is added to
total_energy at extensions.py:2597-2613 and is not consumed by anything
upstream, and reading field_blocks.py:723 the head uses only node_feats,
field_feats, charges_0 and charges_induced -- node_attrs, edge_attrs,
edge_feats and edge_index are accepted and ignored. So its inputs do not depend
on its weights and can be cached once. Both halves of that claim are gated:
GATE A re-runs the head on the cached tensors and must reproduce the captured
output, and GATE B perturbs the head's weights, re-runs the FULL model, and
requires the cached inputs to come back unchanged.

THE OBJECTIVE, as specified:

    E_rest    = E_current - E_LE,current
    target    = Delta E_DFT - Delta E_rest
              = -eps_Delta + Delta E_LE,current
    train     min over head params of (Delta E_LE_pred - target)^2

Only the PAIRED difference is constrained, so a constant shift of the head
output is unidentifiable here -- it cancels between the two frames. That is
intended, and it means a head fitted this way is not a drop-in for the full
model until the absolute energies are checked, which is deliberately a later
step.

THE OPTIMISER MUST BE SHOWN TO HAVE ACTED. A non-zero output or non-zero
weights prove nothing about training: the probe that produced d(le) ran under
no_grad. So the head's weights are snapshotted before and after, per-tensor
max|dw| is reported, and the gradient norm at the first step is printed. If
those are zero the fit is not interpretable regardless of the loss curve.

SCALE, STATED UP FRONT. 160 training pairs against 76545 trainable parameters
is 478x over-parameterised, so fitting the training pairs is expected and
carries little information on its own. Train and val are therefore always
reported side by side, and a re-fitted a*dN baseline is scored on the same
residual for comparison.
"""
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
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.pb1d_backend as PB

RUN = os.environ.get("KIT_RUN",
                     "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/gate_le")
CACHE = os.environ.get(
    "KIT_CACHE",
    "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/exp_le_headfit/"
    "le_head_cache.pt")
A_CONST = float(os.environ.get("KIT_A_CONST", "-5.2745"))
STEPS = int(os.environ.get("KIT_STEPS", "4000"))
LR = float(os.environ.get("KIT_LR", "1e-3"))
WD = float(os.environ.get("KIT_WD", "0.0"))
EVAL_EVERY = int(os.environ.get("KIT_EVAL_EVERY", "50"))
NPAIR_LIMIT = int(os.environ.get("KIT_NPAIR_LIMIT", "0"))   # 0 = all

os.chdir(RUN)
device = torch_tools.init_device(os.environ.get("KIT_DEVICE", "cuda"))
torch_tools.set_default_dtype("float64")


# ------------------------------------------------------------------ model ----
def find_model(run):
    import glob
    c = ([p for p in sorted(glob.glob(os.path.join(run, "models", "*.model")))
          if "compiled" not in os.path.basename(p)]
         + [p for p in sorted(glob.glob(os.path.join(run, "checkpoints", "*.model")))
            if "compiled" not in os.path.basename(p)])
    if not c:
        raise SystemExit(f"no .model under {run}")
    return c[-1]


def find_ckpt(run):
    import glob
    pts = sorted(glob.glob(os.path.join(run, "checkpoints", "*_epoch-*.pt")),
                 key=lambda p: int(p.rsplit("epoch-", 1)[1].split(".")[0]))
    return pts[-1] if pts else None


mpath, cpath = find_model(RUN), os.environ.get("KIT_CKPT") or find_ckpt(RUN)
model = torch.load(f=mpath, map_location=device).to(device)
if cpath:
    ck = torch.load(cpath, map_location=device)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck.model
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss or unexp:
        raise SystemExit(f"state_dict mismatch: {len(miss)} missing, "
                         f"{len(unexp)} unexpected")
model.eval()
model.solvent3d_energy = True
model.solvent_cavity_energy = True
model.solvent_baseline_coupling = True
head = getattr(model, "local_electron_energy", None)
if head is None:
    raise SystemExit("this model has local_electron_energy = None")

print("=" * 78)
print(f"model      {os.path.basename(mpath)}")
print(f"weights    {os.path.basename(cpath) if cpath else '(as saved)'}")
npar = sum(p.numel() for p in head.parameters())
ntot = sum(p.numel() for p in model.parameters())
print(f"head       {npar} trainable parameters, {100.0 * npar / ntot:.2f}% of "
      f"the model's {ntot}")
print(f"n_sigmas   {int(model.atomic_density_num_sigmas)} radial monopole "
      f"channels per spin")

# ------------------------------------------------------------------- data ----
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "sample_id": "sample_id", "Fermi": "Fermi",
               "fermi_level": "Fermi", "solvated": "solvated",
               "potential": "potential_diff", "head": "head"},
    arrays_keys={"forces": "forces"})
atoms, split_of = {}, {}
for sp in ("train", "val", "test"):
    f = os.path.join(RUN, "data", f"{sp}.xyz")
    if not os.path.exists(f):
        continue
    for a in read(f, ":"):
        sid = a.info.get("sample_id")
        if sid is None:
            continue
        atoms[int(sid)] = a
        split_of[int(sid)] = sp

PAIRS = [(k, k + 600) for k in range(1, 401)
         if k in atoms and (k + 600) in atoms]
straddle = [p for p in PAIRS if split_of[p[0]] != split_of[p[1]]]
if straddle:
    raise SystemExit(f"{len(straddle)} pairs straddle splits: {straddle[:5]}")
# TASK_4080: no dense linear algebra on the NiN88 cell (sid 201, 353). Those
# frames are 339-atom singles and are not members of any pair, which is
# asserted rather than assumed.
nat_of = {s: len(a) for s, a in atoms.items()}
bad = [p for p in PAIRS if nat_of[p[0]] != 207 or nat_of[p[1]] != 207]
if bad:
    raise SystemExit(f"non-NiN44 pairs present, refusing: {bad[:5]}")
for s in (201, 353):
    if any(s in p for p in PAIRS):
        raise SystemExit(f"sid {s} is in a pair; forbidden by TASK_4080")
bysp = {}
for p in PAIRS:
    bysp.setdefault(split_of[p[0]], []).append(p)
if NPAIR_LIMIT:
    # take N pairs FROM EACH SPLIT, not the first N overall -- the first N
    # overall are all train and would leave the val set empty
    bysp = {k: v[:NPAIR_LIMIT] for k, v in bysp.items()}
    PAIRS = [p for v in bysp.values() for p in v]
print(f"pairs      {len(PAIRS)} NiN44 pairs, "
      + ", ".join(f"{k} {len(v)}" for k, v in sorted(bysp.items())))
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])


# ------------------------------------------------------------------ cache ----
cap = {}


def pre_hook(m, args, kwargs):
    cap["node_feats"] = kwargs["node_feats"].detach().clone()
    cap["field_feats"] = kwargs["field_feats"].detach().clone()
    cap["charges_0"] = kwargs["charges_0"].detach().clone()
    cap["charges_induced"] = kwargs["charges_induced"].detach().clone()
    cap["unused"] = {k: (None if kwargs.get(k) is None
                         else tuple(kwargs[k].shape))
                     for k in ("node_attrs", "edge_attrs", "edge_feats",
                               "edge_index")}
    return None


def post_hook(m, args, kwargs, out):
    cap["le"] = out.detach().clone()
    return None


head.register_forward_pre_hook(pre_hook, with_kwargs=True)
head.register_forward_hook(post_hook, with_kwargs=True)

_backend = {}
_bk = PB.PB1DBackend.solve_graph


def _wrap(self, *a, **k):
    _backend["b"] = self
    return _bk(self, *a, **k)


PB.PB1DBackend.solve_graph = _wrap


def _evict():
    b = _backend.get("b")
    if b is None:
        return
    for at in ("_bl_ram", "_grids", "_solvers", "_c_units"):
        d = getattr(b, at, None)
        if isinstance(d, dict):
            d.clear()


def forward_frame(sid):
    a = atoms[sid]
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(
        ds, batch_size=1))).to(device)
    cap.clear()
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
    return b, pred, {k: v for k, v in cap.items()}


def build_cache():
    """Incremental on purpose: a partially built cache is resumed rather than
    discarded. Measured pace is 62 s for the first frame (baseline-cache
    warmup) and ~5.8 s/frame after, so 400 frames is ~40 min -- long enough
    that losing it to a wall-clock kill would cost a whole job, which has
    already happened once in this project."""
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    out = {}
    if os.path.exists(CACHE):
        try:
            out = torch.load(CACHE, map_location="cpu")
            print(f"   resuming a partial cache with {len(out)} frames")
        except Exception as e:
            print(f"   existing cache unreadable ({e}), starting over")
            out = {}
    t0 = time.time()
    sids = [s for p in PAIRS for s in p]
    todo = [s for s in sids if s not in out]
    print(f"   {len(todo)} of {len(sids)} frames to compute")
    for i, sid in enumerate(todo):
        _, pred, got = forward_frame(sid)
        out[sid] = dict(
            node_feats=got["node_feats"].cpu(),
            field_feats=got["field_feats"].cpu(),
            charges_0=got["charges_0"].cpu(),
            charges_induced=got["charges_induced"].cpu(),
            le=got["le"].cpu(),
            e_total=float(pred["energy"].sum()),
            label=float(atoms[sid].info["energy"]),
            nat=len(atoms[sid]),
            q=float(atoms[sid].info.get("total_charge", 0.0)),
            split=split_of[sid],
            unused=got["unused"])
        if i % 10 == 0 or i == len(todo) - 1:
            el = time.time() - t0
            rate = el / (i + 1)
            print(f"   cached {i + 1}/{len(todo)}  {el:6.1f} s  "
                  f"({rate:.2f} s/frame, eta "
                  f"{rate * (len(todo) - i - 1) / 60:.1f} min)", flush=True)
        if (i + 1) % 50 == 0:
            torch.save(out, CACHE)
            print(f"   checkpointed the cache at {len(out)} frames", flush=True)
        _evict()
    torch.save(out, CACHE)
    print(f"   cache written: {CACHE} "
          f"({os.path.getsize(CACHE) / 1e6:.0f} MB, {len(out)} frames)")
    return out


need = {s for p in PAIRS for s in p}
if os.path.exists(CACHE) and not os.environ.get("KIT_FORCE_CACHE"):
    C = torch.load(CACHE, map_location="cpu")
    if need.issubset(set(C)):
        print(f"cache      reusing {CACHE} ({len(C)} frames)")
    else:
        print(f"cache      {len(need - set(C))} frames missing, rebuilding")
        C = build_cache()
else:
    print(f"cache      building")
    C = build_cache()

# ------------------------------------------------------- the two cache gates --
print("\n" + "=" * 78)
print("GATE A -- the head is a pure function of the cached tensors")
s0 = PAIRS[0][0]
d0 = C[s0]
print(f"   arguments the head accepts but does not read: "
      + ", ".join(f"{k}={v}" for k, v in d0["unused"].items()))


def run_head(d):
    return head(node_attrs=None,
                node_feats=d["node_feats"].to(device),
                edge_attrs=None, edge_feats=None, edge_index=None,
                field_feats=d["field_feats"].to(device),
                charges_0=d["charges_0"].to(device),
                charges_induced=d["charges_induced"].to(device))


with torch.no_grad():
    rep = run_head(d0)
dev = float((rep.cpu() - d0["le"]).abs().max())
sc = float(d0["le"].abs().max())
print(f"   sid {s0}: max |replay - captured| {dev:.3e} eV, scale {sc:.6f} eV, "
      f"relative {dev / max(sc, 1e-30):.3e}")
if dev / max(sc, 1e-30) > 1e-12:
    raise SystemExit("GATE A FAILED: the cached tensors do not reproduce the "
                     "head's output, so the cache is not the head's input")
print(f"   gate A passed")

print("\nGATE B -- does perturbing the head move the head's own inputs?")
# THIS GATE NEEDS A CONTROL. The first version asserted bit-equality of the
# inputs after a weight perturbation and failed at 2.6e-11 relative on
# charges_induced and 5.6e-12 on field_feats (node_feats and charges_0 sat at
# 1e-16, pure float noise). But the PB solve is ITERATIVE and STATEFUL -- it
# carries a phi cache and an encounter counter -- so simply running the same
# frame a second time can move those two quantities by a similar amount for
# reasons that have nothing to do with the head. Without a control arm the gate
# cannot tell the two apart, so a repeat forward at UNCHANGED weights is run
# first and reported alongside.
#
# The criterion is also wrong if it asks for bit-equality. What matters is
# whether the input wobble, from whatever cause, changes the HEAD'S OUTPUT by
# anything comparable to the quantity being fitted -- d(le) is order 1e-2 eV
# and the target order 1 eV. So both arms are pushed back through the head at
# fixed weights and judged in eV.
print("   control arm: same frame again, weights UNCHANGED")
_, _, got_c = forward_frame(s0)
_evict()
lastw = [p for p in head.parameters()][-1]
keep = lastw.detach().clone()
with torch.no_grad():
    lastw.add_(torch.ones_like(lastw) * 7.0)
print("   test arm: same frame again, last head weight shifted by +7.0")
_, pred_p, got_p = forward_frame(s0)
with torch.no_grad():
    lastw.copy_(keep)
_evict()
print(f"   {'input':>16} {'control':>12} {'test':>12}   (relative to scale)")
for k in ("node_feats", "field_feats", "charges_0", "charges_induced"):
    base = C[s0][k]
    scl = max(float(base.abs().max()), 1e-30)
    dc = float((got_c[k].cpu() - base).abs().max()) / scl
    dp = float((got_p[k].cpu() - base).abs().max()) / scl
    print(f"   {k:>16} {dc:12.3e} {dp:12.3e}")


def le_on(d):
    with torch.no_grad():
        return float(run_head(d).sum())


base_in = {k: C[s0][k] for k in ("node_feats", "field_feats", "charges_0",
                                 "charges_induced")}
ctl_in = {k: got_c[k].cpu() for k in base_in}
tst_in = {k: got_p[k].cpu() for k in base_in}
le_b, le_c, le_t = le_on(base_in), le_on(ctl_in), le_on(tst_in)
# caveat kept explicit: the control is the 2nd forward of this frame and the
# test the 3rd, so the test arm has one more step of solver-state evolution
# behind it. That asymmetry is why the criterion is in eV rather than a
# control/test ratio.
print(f"   pushing each arm's inputs through the head at FIXED weights:")
print(f"     cached inputs  le = {le_b:+.9f} eV")
print(f"     control inputs le = {le_c:+.9f} eV   (moved {abs(le_c - le_b):.3e} eV)")
print(f"     test inputs    le = {le_t:+.9f} eV   (moved {abs(le_t - le_b):.3e} eV)")
print(f"   (the perturbation did reach the head's output: with the shifted "
      f"weight the full model reported le sum "
      f"{float(got_p['le'].sum()):+.4f} vs {float(C[s0]['le'].sum()):+.4f} eV)")
TOL_EV = 1e-6
worst_ev = max(abs(le_c - le_b), abs(le_t - le_b))
if worst_ev > TOL_EV:
    raise SystemExit(
        f"GATE B FAILED: input variation moves the head's output by "
        f"{worst_ev:.3e} eV, which is not negligible against the ~1e-2 eV "
        f"quantity being fitted, so the cache is not a sound stand-in")
excess = abs(le_t - le_b) - abs(le_c - le_b)
print(f"   gate B passed: the largest effect on the head's output is "
      f"{worst_ev:.3e} eV, under the {TOL_EV:.0e} eV tolerance and "
      f"{abs(0.0401) / max(worst_ev, 1e-30):.1e}x smaller than the d(le) being "
      f"fitted; the perturbation adds {excess:+.3e} eV over the control, so "
      f"{'the head does not measurably influence its own inputs' if excess <= abs(le_c - le_b) + 1e-12 else 'part of the wobble tracks the perturbation'}")

# ------------------------------ the mechanism check with the right statistic --
print("\n" + "=" * 78)
print("charges_0 CARRIES THE TOTAL CHARGE (the corrected mechanism claim)")
nsig = int(model.atomic_density_num_sigmas)
print(f"   monopole channels are the first {nsig} of each spin block "
      f"(irreps order scalars first); per-spin blocks of "
      f"{C[s0]['charges_0'].shape[1] // 2}")


def monopole_sums(x, nsig):
    half = x.shape[1] // 2
    a = float(x[:, :nsig].sum())
    b = float(x[:, half:half + nsig].sum())
    return a, b


print(f"   {'sid':>5} {'q':>7} {'charges_0 up/dn':>24} "
      f"{'charges_induced up/dn':>24} {'q_in up/dn':>22}")
for sid in (PAIRS[0][0], PAIRS[0][1]):
    d = C[sid]
    a0, b0 = monopole_sums(d["charges_0"], nsig)
    ai, bi = monopole_sums(d["charges_induced"], nsig)
    aq, bq = monopole_sums(d["charges_0"] + d["charges_induced"], nsig)
    print(f"   {sid:>5} {d['q']:+7.3f} {a0:+11.6f} {b0:+11.6f} "
          f"{ai:+11.6f} {bi:+11.6f} {aq:+10.6f} {bq:+10.6f}")
d1, d6 = C[PAIRS[0][0]], C[PAIRS[0][1]]
for nm, ka in (("charges_0", "charges_0"), ("charges_induced", "charges_induced")):
    x, y = d1[ka], d6[ka]
    a1, b1 = monopole_sums(x, nsig)
    a6, b6 = monopole_sums(y, nsig)
    print(f"   {nm:>16}: monopole sum changes by {a1 - a6:+.6f} / {b1 - b6:+.6f} "
          f"across the pair (dN = {d6['q'] - d1['q']:+.3f}); "
          f"per-coefficient max |diff| {float((x - y).abs().max()):.3e}, "
          f"L1 of the difference {float((x - y).abs().sum()):.6f}")

# ---------------------------------------------------------------- the fit ----
print("\n" + "=" * 78)
print("THE OBJECTIVE")


def stack(pairs):
    """One flat batch of all frames in these pairs, with an index that scatters
    per-atom head output back to its frame."""
    frames = [s for p in pairs for s in p]
    nf = torch.cat([C[s]["node_feats"] for s in frames], 0).to(device)
    ff = torch.cat([C[s]["field_feats"] for s in frames], 0).to(device)
    c0 = torch.cat([C[s]["charges_0"] for s in frames], 0).to(device)
    ci = torch.cat([C[s]["charges_induced"] for s in frames], 0).to(device)
    idx = torch.cat([torch.full((C[s]["node_feats"].shape[0],), i,
                                dtype=torch.long)
                     for i, s in enumerate(frames)]).to(device)
    return dict(frames=frames, nf=nf, ff=ff, c0=c0, ci=ci, idx=idx,
                n=len(frames))


def head_le_totals(B):
    le = head(node_attrs=None, node_feats=B["nf"], edge_attrs=None,
              edge_feats=None, edge_index=None, field_feats=B["ff"],
              charges_0=B["c0"], charges_induced=B["ci"])
    tot = torch.zeros(B["n"], dtype=le.dtype, device=le.device)
    tot = tot.index_add(0, B["idx"], le)
    return tot


def targets(pairs):
    """target = dE_DFT - dE_rest = -eps + dE_LE,current, per pair."""
    t, dn, eps0, dle0 = [], [], [], []
    for sc, sn in pairs:
        dc, dnn = C[sc], C[sn]
        de_mod = dc["e_total"] - dnn["e_total"]
        de_dft = dc["label"] - dnn["label"]
        dle_cur = float(dc["le"].sum()) - float(dnn["le"].sum())
        de_rest = de_mod - dle_cur
        t.append(de_dft - de_rest)
        dn.append(dnn["q"] - dc["q"])
        eps0.append(de_mod - de_dft)
        dle0.append(dle_cur)
    return (torch.tensor(t, dtype=torch.float64, device=device),
            torch.tensor(dn, dtype=torch.float64, device=device),
            np.array(eps0), np.array(dle0))


TR, VA = bysp.get("train", []), bysp.get("val", [])
BTR, BVA = stack(TR), stack(VA)
yTR, dnTR, epsTR, dle0TR = targets(TR)
yVA, dnVA, epsVA, dle0VA = targets(VA)
for nm, y, dn, e0, l0 in (("train", yTR, dnTR, epsTR, dle0TR),
                          ("val", yVA, dnVA, epsVA, dle0VA)):
    print(f"   {nm:>5}: {len(y):3d} pairs   target mean {float(y.mean()):+.4f} "
          f"rmse {float((y ** 2).mean() ** 0.5):.4f} eV   "
          f"current d(le) mean {l0.mean():+.4f}   eps mean {e0.mean():+.4f}   "
          f"dN mean {float(dn.mean()):+.3f}")


def paired_pred(B):
    tot = head_le_totals(B)
    return tot[0::2] - tot[1::2]      # frames are [charged, neutral] per pair


def report(tag, pTR, pVA):
    for nm, p, y in (("train", pTR, yTR), ("val", pVA, yVA)):
        err = (p - y).detach()
        print(f"   {tag:>22} {nm:>5}: rmse {float((err ** 2).mean() ** .5):.4f} "
              f"bias {float(err.mean()):+.4f}  mean-removed "
              f"{float(((err - err.mean()) ** 2).mean() ** .5):.4f} eV")


print("\n" + "=" * 78)
print("BEFORE TRAINING")
with torch.no_grad():
    p0TR, p0VA = paired_pred(BTR), paired_pred(BVA)
# self-consistency with the probe: this must equal the cached d(le)
chk = float((p0TR.cpu().numpy() - dle0TR).__abs__().max())
print(f"   replay of the current head reproduces the cached d(le) to "
       f"{chk:.3e} eV")
report("current head", p0TR, p0VA)

print("\na*dN BASELINE refitted on the same residual")
a_fit = float((yTR * dnTR).sum() / (dnTR ** 2).sum())
print(f"   a fitted on train = {a_fit:+.6f} eV/electron "
      f"(previously measured {A_CONST:+.4f})")
report("a*dN refit", a_fit * dnTR, a_fit * dnVA)
report(f"a*dN a={A_CONST}", A_CONST * dnTR, A_CONST * dnVA)

print("\n" + "=" * 78)
print("TRAINING THE HEAD ALONE")
for p in model.parameters():
    p.requires_grad = False
for p in head.parameters():
    p.requires_grad = True
w_before = {k: v.detach().clone() for k, v in head.state_dict().items()}
opt = torch.optim.Adam([p for p in head.parameters() if p.requires_grad],
                       lr=LR, weight_decay=WD)
best = (float("inf"), None, -1)
g0 = None
t0 = time.time()
for step in range(STEPS + 1):
    opt.zero_grad(set_to_none=True)
    loss = ((paired_pred(BTR) - yTR) ** 2).mean()
    loss.backward()
    if step == 0:
        g0 = max(float(p.grad.abs().max()) for p in head.parameters()
                 if p.grad is not None)
        ng = sum(1 for p in head.parameters()
                 if p.grad is not None and float(p.grad.abs().max()) > 0)
        nt = sum(1 for p in head.parameters())
        print(f"   step 0: loss {float(loss):.6f}, max |grad| {g0:.3e}, "
              f"{ng}/{nt} parameter tensors have a non-zero gradient")
        if g0 == 0.0:
            raise SystemExit("the gradient is identically zero; the fit would "
                             "be meaningless")
    opt.step()
    if step % EVAL_EVERY == 0 or step == STEPS:
        with torch.no_grad():
            vr = float((((paired_pred(BVA) - yVA)) ** 2).mean() ** 0.5)
            tr = float(loss) ** 0.5
        if vr < best[0]:
            best = (vr, {k: v.detach().clone()
                         for k, v in head.state_dict().items()}, step)
        if step % (EVAL_EVERY * 8) == 0 or step == STEPS:
            print(f"   step {step:5d}  train rmse {tr:8.4f}  val rmse "
                  f"{vr:8.4f} eV   [{time.time() - t0:5.1f} s]", flush=True)

print(f"   best val rmse {best[0]:.4f} eV at step {best[2]}")
head.load_state_dict(best[1])

print("\n   DID THE OPTIMISER ACTUALLY MOVE THE WEIGHTS?")
tot_mv, nz = 0.0, 0
for k, v in head.state_dict().items():
    d = float((v - w_before[k]).abs().max())
    s = max(float(w_before[k].abs().max()), 1e-30)
    if d > 0:
        nz += 1
    tot_mv = max(tot_mv, d / s)
    if "mlp" in k or d == 0:
        print(f"     {k:>62} max|dw| {d:.3e} rel {d / s:.3e}")
print(f"     {nz}/{len(w_before)} tensors changed, worst relative move "
      f"{tot_mv:.3e}")
if nz == 0:
    raise SystemExit("no weight changed; the fit did not happen")

print("\n" + "=" * 78)
print("AFTER TRAINING")
with torch.no_grad():
    p1TR, p1VA = paired_pred(BTR), paired_pred(BVA)
report("current head", p0TR, p0VA)
report("a*dN refit", a_fit * dnTR, a_fit * dnVA)
report("head trained alone", p1TR, p1VA)

print("\n   per-pair val errors, trained head vs a*dN refit")
print(f"   {'pair':>6} {'dN':>7} {'target':>9} {'head':>9} {'a*dN':>9} "
      f"{'err_head':>9} {'err_a':>9}")
for i, (sc, sn) in enumerate(VA):
    print(f"   {sc:>6} {float(dnVA[i]):+7.3f} {float(yVA[i]):+9.4f} "
          f"{float(p1VA[i]):+9.4f} {a_fit * float(dnVA[i]):+9.4f} "
          f"{float(p1VA[i] - yVA[i]):+9.4f} "
          f"{a_fit * float(dnVA[i]) - float(yVA[i]):+9.4f}")
print("\nNOTE: only the PAIRED difference was fitted, so the absolute level of "
      "the head is unidentified by this objective and the fitted head is not a "
      "drop-in for the full model. Energy and force of the full model are a "
      "separate, later step by design.")
