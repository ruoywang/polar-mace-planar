"""Why did the ORIGINAL joint training leave the native head at 0.04 eV when the
head alone can reach 0.1359 eV on the pair? Gradient audit of the real training
setup. (user-specified priorities, 2026-09-11)

NOTHING HERE IS REBUILT BY HAND. The loss comes from get_loss_fn, the optimizer
from get_params_options + get_optimizer, the scheduler from LRScheduler, all fed
the run's own config through mace's own arg parser, and the optimizer and
scheduler states are loaded from the checkpoint. The earlier "20/20 tensors have
a non-zero gradient" came from a FRESH Adam created by an audit script and says
nothing about the original updates, which is why it is not reused.

ONE THING CANNOT BE RECOVERED, and it bounds everything below. Every checkpoint
save in tools/train.py is wrapped in `with ema.average_parameters()`, so the
stored weights are the EMA average while the stored optimizer moments belong to
the raw (non-EMA) trajectory. The raw weights were never written to disk. So a
step taken here starts from EMA weights carrying the raw trajectory's Adam
state -- the closest available state, not the exact one. Reported explicitly
rather than glossed.

VERIFIED IN SOURCE, NOT INFERRED. loss.py:83-87 is
    square((ref.energy - pred.energy) / num_atoms)
so the energy term is per-atom normalised with weight 1 while the force term is
a plain MSE with weight 100. For 207 atoms that is a factor 1/207^2 = 1/42849
on the energy term relative to an unnormalised total-energy MSE. That is a
reason to MEASURE, not a conclusion: the parameter derivatives of the two
outputs differ, and Adam rescales by its own second moment, so the ratio of
weights does not predict the ratio of updates.

WHAT IS MEASURED
  priority 1: the WEIGHTED gradient each loss term puts on the head -- its norm,
    and its component along the direction that reduces the paired error. That
    direction is -grad of (dE_LE - target)^2 taken on the cached head inputs,
    which is exact for this head (its inputs do not depend on its weights:
    measured 1.198e-11 eV against a 6.222e-12 eV control). Alignment is
    reported as cos(-g, d_pair), i.e. of the UPDATE a term would produce, so
    positive means the term pushes toward a better pair and negative means it
    pushes away.
  priority 2: total-model gradient norm before clipping, the clip factor, the
    head's share of that norm, and then ONE step of the loaded Adam -- measuring
    the change in the LIVE model's paired dE, not a cached proxy, since a step
    moves every parameter and therefore the head's inputs too. Four arms from
    the same restored state: energy only, forces only, energy+forces, and the
    full loss.
  priority 3: the same at an earlier checkpoint. Note that only epoch-25
    (warmup stage, before solvent_pb1d_warmup_encounters=30) and epoch-39 (late
    full-PB) survive: the rolling save_latest_every=5 policy overwrote the
    epoch-30..37 files, so there is no EARLY-full-PB checkpoint on disk and that
    half of priority 3 cannot be answered from what exists.

THE DISCRIMINATING OUTCOME the user named: from one state, an energy-only step
lowers the paired error while adding the force term removes or reverses that
gain. If instead the combined direction is right but the realised step is tiny,
the cause is clipping or the Adam state, which priority 2 separates.
"""
import copy
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get(
    "KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get(
    "KIT_MACE_REPO",
    "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")

from ase.io import read

from mace import data as mace_data, modules, tools
from mace.tools import torch_geometric, torch_tools
from mace.tools.scripts_utils import (LRScheduler, get_loss_fn, get_optimizer,
                                      get_params_options)
from mace.tools.train import (attach_density_3d_samples_to_batch,
                              attach_solvent3d_samples_to_batch)
import mace.modules.pb1d_backend as PB

RUN = os.environ.get("KIT_RUN",
                     "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/gate_le")
CFG = os.environ.get("KIT_CFG", "config_pb1d.yaml")
NAME = os.environ.get("KIT_NAME", "s3d_gate_le")
CKPT = os.environ.get("KIT_CKPT", "")      # "" = latest
CACHE = os.environ.get(
    "KIT_CACHE",
    "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/exp_le_headfit/"
    "le_head_cache.pt")
NPAIRS = int(os.environ.get("KIT_NPAIRS", "4"))

os.chdir(RUN)
device = torch_tools.init_device(os.environ.get("KIT_DEVICE", "cuda"))
torch_tools.set_default_dtype("float64")

# ------------------------------------------------- the run's own args --------
argv = ["--config", CFG, "--name", NAME, "--seed", "123",
        "--work_dir", ".", "--log_dir", "logs", "--model_dir", "models",
        "--checkpoints_dir", "checkpoints", "--results_dir", "results",
        "--device", "cuda"]
args = tools.build_default_arg_parser().parse_args(argv)
# run_train mutates args between parsing and building the loss/optimizer. I
# listed every `args.X =` in run() before line 1286 and the only ones that
# apply here (foundation_model is None, so that whole branch is dead) are the
# key specification and the default head; element_charge_residual_scale feeds
# the model constructor, and we load the model from its pickle.
from mace.data.utils import KeySpecification, update_keyspec_from_kwargs
from mace.tools.scripts_utils import prepare_default_head
args.key_specification = KeySpecification()
update_keyspec_from_kwargs(args.key_specification, vars(args))
if args.heads is None:
    args.heads = prepare_default_head(args)
print("=" * 78)
print(f"config  {CFG}   loss={args.loss}  optimizer={args.optimizer}  "
      f"amsgrad={args.amsgrad}")
print(f"weights energy {args.energy_weight}  forces {args.forces_weight}  "
      f"potential {args.potential_weight}  fermi {args.fermi_level_weight}  "
      f"density_3d {args.density_3d_weight}  occ_aug {args.occ_aug_weight}")
print(f"        pot1d {args.potential_1d_profile_weight}  "
      f"rho1d {args.charge_density_1d_weight}  "
      f"rhob1d {args.solvent_rhob_1d_weight}  solv3d {args.solvent3d_weight}")
print(f"clip_grad {args.clip_grad}   lr {args.lr}   ema {args.ema} "
      f"decay {args.ema_decay}   batch_size {args.batch_size}")

# -------------------------------------------------------------- model --------
mp = [p for p in sorted(glob.glob(os.path.join(RUN, "models", "*.model")))
      if "compiled" not in os.path.basename(p)][-1]
cpath = CKPT or sorted(
    glob.glob(os.path.join(RUN, "checkpoints", "*_epoch-*.pt")),
    key=lambda p: int(p.rsplit("epoch-", 1)[1].split(".")[0]))[-1]
model = torch.load(f=mp, map_location=device).to(device)
ck = torch.load(cpath, map_location=device)
model.load_state_dict(ck["model"] if "model" in ck else ck.model, strict=False)
model.solvent3d_energy = True
model.solvent_cavity_energy = True
model.solvent_baseline_coupling = True
head = model.local_electron_energy
ep = int(os.path.basename(cpath).rsplit("epoch-", 1)[1].split(".")[0])
stage = ("WARMUP stage (epoch < solvent_pb1d_warmup_encounters="
         f"{args.solvent_pb1d_warmup_encounters})" if ep < args.solvent_pb1d_warmup_encounters
         else "full-PB stage")
print(f"\ncheckpoint {os.path.basename(cpath)}  epoch {ep}  -> {stage}")
print(f"WEIGHTS ARE EMA-AVERAGED: every save in tools/train.py runs inside "
      f"`with ema.average_parameters()`, and the raw training weights were "
      f"never written to disk. The Adam moments in the same file belong to the "
      f"raw trajectory, so the state below is the closest available, not exact.")

# ------------------------------------------------- loss / optimizer ----------
# run_train.py:1058-1075 -- for model=PolarMACE with this loss it sets
# dipole_only=False, compute_dipole=True, and derives compute_energy /
# compute_forces from whether the ORIGINAL weights are non-zero. Taken from
# source rather than guessed: passing compute_dipole=False tripped the
# assertion at scripts_utils.py:811 (job 3430262).
assert args.model == "PolarMACE" and args.loss == "energy_forces_electrostatics"
dipole_only = False
args.compute_dipole = True
args.compute_energy = bool(float(args.energy_weight) != 0.0)
args.compute_forces = bool(float(args.forces_weight) != 0.0)
args.compute_virials = False
args.compute_stress = False
args.compute_polarizability = False
print(f"\nderived as run_train does: dipole_only={dipole_only} "
      f"compute_dipole={args.compute_dipole} "
      f"compute_energy={args.compute_energy} "
      f"compute_forces={args.compute_forces}")
loss_fn = get_loss_fn(args, dipole_only, args.compute_dipole).to(device)
param_options = get_params_options(args, model)
optimizer = get_optimizer(args, param_options)
lr_sched = LRScheduler(optimizer, args)
if "optimizer" in ck:
    optimizer.load_state_dict(ck["optimizer"])
if "lr_scheduler" in ck:
    try:
        lr_sched.load_state_dict(ck["lr_scheduler"])
    except Exception as e:
        print(f"   lr_scheduler state not loaded: {e}")
print(f"\nloss_fn {loss_fn}"[:400])
head_ids = {id(p) for p in head.parameters()}
grp_of_head = [gi for gi, g in enumerate(optimizer.param_groups)
               for p in g["params"] if id(p) in head_ids]
print(f"\nOPTIMIZER: {len(optimizer.param_groups)} param groups; the head's "
      f"{len(head_ids)} tensors sit in group(s) {sorted(set(grp_of_head))}")
for gi in sorted(set(grp_of_head)):
    g = optimizer.param_groups[gi]
    print(f"   group {gi}: name={g.get('name')} lr={g['lr']:.6g} "
          f"weight_decay={g.get('weight_decay')} amsgrad={g.get('amsgrad')}")
nreg = sum(1 for g in optimizer.param_groups for p in g["params"]
           if id(p) in head_ids)
print(f"   head tensors registered in the optimizer: {nreg} of {len(head_ids)}"
      + ("  <- all registered" if nreg == len(head_ids) else "  <- MISSING"))
steps = [int(optimizer.state[p]["step"]) for g in optimizer.param_groups
         for p in g["params"] if p in optimizer.state
         and "step" in optimizer.state[p]]
hsteps = [int(optimizer.state[p]["step"]) for g in optimizer.param_groups
          for p in g["params"] if id(p) in head_ids and p in optimizer.state
          and "step" in optimizer.state[p]]
print(f"   Adam step counts: model min/max {min(steps) if steps else '-'}/"
      f"{max(steps) if steps else '-'}, head min/max "
      f"{min(hsteps) if hsteps else '-'}/{max(hsteps) if hsteps else '-'}")
print(f"   tensors with Adam state: {len(steps)} of "
      f"{sum(len(g['params']) for g in optimizer.param_groups)}; head "
      f"{len(hsteps)} of {len(head_ids)}")

# ---------------------------------------------------------------- data -------
kspec = args.key_specification
atoms, split_of = {}, {}
for sp in ("train", "val", "test"):
    f = os.path.join(RUN, "data", f"{sp}.xyz")
    if not os.path.exists(f):
        continue
    for a in read(f, ":"):
        s = a.info.get("sample_id")
        if s is not None:
            atoms[int(s)] = a
            split_of[int(s)] = sp
CA = torch.load(CACHE, map_location="cpu")
allp = [(k, k + 600) for k in range(1, 401)
        if k in atoms and k + 600 in atoms and split_of[k] == "train"
        and k in CA and k + 600 in CA]
dn = {p: atoms[p[1]].info.get("total_charge", 0.0)
         - atoms[p[0]].info.get("total_charge", 0.0) for p in allp}
allp.sort(key=lambda p: dn[p])
# span the charge range rather than take the first N
idx = np.linspace(0, len(allp) - 1, NPAIRS).round().astype(int)
PAIRS = [allp[i] for i in idx]
print(f"\nSAMPLES: {NPAIRS} TRAIN pairs spanning dN "
      + ", ".join(f"{p[0]}:{dn[p]:+.3f}" for p in PAIRS))
z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
# the forward is IDENTICAL in every arm -- only the loss weights change -- so
# the term isolation cannot be confounded by a different forward
output_args = {"energy": bool(args.compute_energy),
               "forces": bool(args.compute_forces),
               "virials": False, "stress": False,
               "dipoles": bool(args.compute_dipole)}
print(f"output_args {output_args}")


def make_batch(sid):
    cfg = mace_data.config_from_atoms(atoms[sid], key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1)))
    b = b.to(device)
    attach_density_3d_samples_to_batch(b, loss_fn)
    attach_solvent3d_samples_to_batch(b, loss_fn)
    return b


BATCH = {s: make_batch(s) for p in PAIRS for s in p}
_bd = {}
_bk = PB.PB1DBackend.solve_graph


def _wrap(self, *a, **k):
    _bd["b"] = self
    return _bk(self, *a, **k)


PB.PB1DBackend.solve_graph = _wrap


def _evict():
    b = _bd.get("b")
    if b is not None:
        for at in ("_bl_ram", "_grids", "_solvers", "_c_units"):
            d = getattr(b, at, None)
            if isinstance(d, dict):
                d.clear()


# ------------------------------- the direction that helps the pair -----------
W_KEYS = [k for k, _ in head.named_parameters()]


def head_vec(getter):
    return torch.cat([getter(p).reshape(-1) for p in head.parameters()])


def cached_dle(w_state=None):
    """paired dE_LE on the cached head inputs, differentiable in the head's
    parameters. Exact for this head: its inputs do not depend on its weights."""
    outs = []
    for sc, sn in PAIRS:
        tot = []
        for s in (sc, sn):
            d = CA[s]
            le = head(node_attrs=None,
                      node_feats=d["node_feats"].to(device),
                      edge_attrs=None, edge_feats=None, edge_index=None,
                      field_feats=d["field_feats"].to(device),
                      charges_0=d["charges_0"].to(device),
                      charges_induced=d["charges_induced"].to(device))
            tot.append(le.sum())
        outs.append(tot[0] - tot[1])
    return torch.stack(outs)


def pair_targets():
    t = []
    for sc, sn in PAIRS:
        de_dft = CA[sc]["label"] - CA[sn]["label"]
        rest = ((CA[sc]["e_total"] - float(CA[sc]["le"].sum()))
                - (CA[sn]["e_total"] - float(CA[sn]["le"].sum())))
        t.append(de_dft - rest)
    return torch.tensor(t, dtype=torch.float64, device=device)


Y = pair_targets()
model.zero_grad(set_to_none=True)
for p in model.parameters():
    p.requires_grad_(True)
lp = ((cached_dle() - Y) ** 2).mean()
lp.backward()
d_pair = -head_vec(lambda p: p.grad.detach()).clone()
d_pair_n = d_pair / d_pair.norm()
print(f"\nDESCENT DIRECTION for the paired error at this state: "
      f"L_pair {float(lp):.4f} eV^2, ||d_pair|| {float(d_pair.norm()):.4e}")
model.zero_grad(set_to_none=True)

# ------------------------------------------------------------ priority 1 -----
ALL_W = {k: float(getattr(loss_fn, k).item())
         for k in dir(loss_fn)
         if k.endswith("_weight") and torch.is_tensor(getattr(loss_fn, k, None))}


def set_terms(keep):
    """keep = set of weight-attribute names to leave at their original value;
    everything else is zeroed. The weights are registered buffers, so this
    isolates terms inside the ORIGINAL loss implementation."""
    for k, v in ALL_W.items():
        getattr(loss_fn, k).fill_(v if k in keep else 0.0)


ARMS = [("energy only", {"energy_weight"}),
        ("forces only", {"forces_weight"}),
        ("energy+forces", {"energy_weight", "forces_weight"}),
        ("everything else", set(ALL_W) - {"energy_weight", "forces_weight"}),
        ("full loss", set(ALL_W))]

print("\n" + "=" * 78)
print("PRIORITY 1 -- the weighted gradient each term puts on the head")
print(f"   gradients are averaged over the {2 * len(PAIRS)} frames, mimicking "
      f"the DDP mean over ranks (the run used {args.batch_size} per rank and 3 "
      f"ranks)")
print(f"\n   {'arm':>17} {'loss':>12} {'||g_head||':>12} {'||g_model||':>12} "
      f"{'head share':>11} {'cos(-g,d_pair)':>15} {'proj eV^-1':>12}")
res1 = {}
for nm, keep in ARMS:
    set_terms(keep)
    gh = torch.zeros_like(d_pair)
    gm2, lsum, nfr = 0.0, 0.0, 0
    for s in BATCH:
        model.zero_grad(set_to_none=True)
        out = model(BATCH[s].to_dict(), training=True,
                    compute_force=output_args["forces"],
                    compute_virials=False, compute_stress=False)
        L = loss_fn(pred=out, ref=BATCH[s])
        L.backward()
        gh += head_vec(lambda p: (p.grad.detach() if p.grad is not None
                                  else torch.zeros_like(p)))
        gm2 += float(sum((p.grad.detach() ** 2).sum() for p in model.parameters()
                         if p.grad is not None))
        lsum += float(L)
        nfr += 1
        _evict()
    gh /= nfr
    gm = (gm2 / nfr) ** 0.5
    cs = float((-gh / max(float(gh.norm()), 1e-300)) @ d_pair_n) if gh.norm() > 0 else 0.0
    pr = float((-gh) @ d_pair_n)
    res1[nm] = dict(gh=gh.clone(), norm=float(gh.norm()), gm=gm, cos=cs, proj=pr,
                    loss=lsum / nfr)
    print(f"   {nm:>17} {lsum / nfr:12.6f} {float(gh.norm()):12.4e} "
          f"{gm:12.4e} {float(gh.norm()) / max(gm, 1e-300):11.3e} "
          f"{cs:+15.4f} {pr:+12.4e}")
set_terms(set(ALL_W))
print(f"\n   cos > 0 means the update that term would produce moves the paired "
      f"error DOWN; cos < 0 means it moves it up.")
if "energy only" in res1 and "forces only" in res1:
    e, f = res1["energy only"], res1["forces only"]
    print(f"   energy vs forces on the head: norms {e['norm']:.3e} against "
          f"{f['norm']:.3e}, ratio {f['norm'] / max(e['norm'], 1e-300):.1f}x; "
          f"alignments {e['cos']:+.4f} against {f['cos']:+.4f}")
    ef = res1.get("energy+forces")
    if ef is not None:
        print(f"   combined: norm {ef['norm']:.3e}, alignment {ef['cos']:+.4f} "
              f"-- {'the force term flips the sign of the useful component' if e['cos'] > 0 and ef['cos'] < 0 else 'the sign survives combination' if ef['cos'] > 0 else 'the combined direction does not help the pair'}")

# ------------------------------------------------------------ priority 2 -----
print("\n" + "=" * 78)
print("PRIORITY 2 -- clipping, and what ONE step of the loaded Adam does")
set_terms(set(ALL_W))
model.zero_grad(set_to_none=True)
s0 = PAIRS[0][0]
out = model(BATCH[s0].to_dict(), training=True,
            compute_force=output_args["forces"], compute_virials=False,
            compute_stress=False)
L = loss_fn(pred=out, ref=BATCH[s0])
L.backward()
tot_before = float(torch.sqrt(sum((p.grad.detach() ** 2).sum()
                                  for p in model.parameters()
                                  if p.grad is not None)))
head_before = float(head_vec(lambda p: (p.grad.detach() if p.grad is not None
                                        else torch.zeros_like(p))).norm())
clipped = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=args.clip_grad))
head_after = float(head_vec(lambda p: (p.grad.detach() if p.grad is not None
                                       else torch.zeros_like(p))).norm())
factor = min(1.0, args.clip_grad / max(tot_before, 1e-300))
print(f"   one frame, full loss: total ||g|| before clipping {tot_before:.4e}, "
      f"clip_grad {args.clip_grad}")
print(f"   clip factor {factor:.4e}  ({'CLIPPED' if tot_before > args.clip_grad else 'not clipped'})")
print(f"   head ||g||: {head_before:.4e} -> {head_after:.4e} "
      f"(share of total {head_before / max(tot_before, 1e-300):.3e})")
model.zero_grad(set_to_none=True)
_evict()

W0 = copy.deepcopy(model.state_dict())
O0 = copy.deepcopy(optimizer.state_dict())


def live_paired():
    out = []
    for sc, sn in PAIRS:
        es = []
        for s in (sc, sn):
            with torch.no_grad():
                p = model(BATCH[s].to_dict(), training=False,
                          compute_force=False)
            es.append(float(p["energy"].sum()))
            _evict()
        out.append(es[0] - es[1])
    return np.array(out)


de_dft = np.array([CA[sc]["label"] - CA[sn]["label"] for sc, sn in PAIRS])
base = live_paired()
print(f"\n   paired dE before any step: model {base.round(4)}")
print(f"   {'':>28} DFT   {de_dft.round(4)}")
print(f"   |eps| mean before = {np.abs(base - de_dft).mean():.4f} eV")

print(f"\n   {'arm':>17} {'|eps| after':>12} {'change':>10} "
      f"{'max|dw| head':>13} {'rel':>10} {'d(paired dE)':>13}")
for nm, keep in ARMS:
    model.load_state_dict(W0)
    optimizer.load_state_dict(copy.deepcopy(O0))
    set_terms(keep)
    wb = {k: v.detach().clone() for k, v in head.state_dict().items()}
    for s in BATCH:                      # accumulate like a 3-rank DDP mean
        model.zero_grad(set_to_none=True)
        out = model(BATCH[s].to_dict(), training=True,
                    compute_force=output_args["forces"],
                    compute_virials=False, compute_stress=False)
        (loss_fn(pred=out, ref=BATCH[s]) / len(BATCH)).backward()
        _evict()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
    optimizer.step()
    aft = live_paired()
    mdw = max(float((head.state_dict()[k] - wb[k]).abs().max()) for k in wb)
    rel = mdw / max(max(float(wb[k].abs().max()) for k in wb), 1e-300)
    print(f"   {nm:>17} {np.abs(aft - de_dft).mean():12.4f} "
          f"{np.abs(aft - de_dft).mean() - np.abs(base - de_dft).mean():+10.4f} "
          f"{mdw:13.3e} {rel:10.3e} {(aft - base).mean():+13.3e}")
model.load_state_dict(W0)
optimizer.load_state_dict(copy.deepcopy(O0))
set_terms(set(ALL_W))
print(f"\n   'change' negative means one step of the ORIGINAL optimiser reduced "
      f"the paired error. The head-only Adam needed thousands of steps to reach "
      f"0.1359 eV, so a single step is expected to be small; what matters is "
      f"the SIGN and how the arms differ.")
