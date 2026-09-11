"""Why did the ORIGINAL joint training leave the native head at 0.04 eV on the
pair? Gradient audit of the real training setup. (user-specified, 2026-09-11;
this is a rewrite -- the first version had three defects that invalidated its
conclusions, all listed below.)

DEFECT 1, FATAL TO THE PREVIOUS ONE-STEP TABLE. The step loops zeroed the
gradient INSIDE the frame loop, so optimizer.step() saw only the last frame's
gradient divided by the frame count, while the gradient table above it was a
genuine multi-frame average. The two were not the same gradient, so the
observation "the gradient points the wrong way yet the Adam step improves the
pair" could not be attributed to Adam at all. Fixed: zero once, accumulate, and
CHECK the accumulated head gradient against the per-frame average before any
step is taken.

DEFECT 2, FATAL TO THE EPOCH COMPARISON. The PB warm-up gate at
extensions.py:1703-1710 reads `int(getattr(self, "_pb1d_epoch", 0))`, and
nothing set it, so every forward ran with enc = 0 < warmup = 30 and took the
planar warm-up path. The epoch-25 arm was right by accident; the epoch-39 arm
was measuring warm-up-path gradients while claiming to be the full-PB stage.
The flag itself was fine -- `_pb1d_training_flag` is set from the `training`
kwarg at extensions.py:2147 -- but the epoch was not. Fixed: set _pb1d_epoch
from the checkpoint and MEASURE that crossing the threshold changes the
forward.

DEFECT 3, FATAL TO THE REFERENCE DIRECTIONS AT EPOCH 25. d_pair, u_sum and
u_diff were built from le_head_cache.pt, which was generated once from the
epoch-39 weights in the validation path. Gate B established only that the
head's inputs are insensitive to the HEAD's own weights; the rest of the model
changes between checkpoints, and so do those inputs, and so does the target,
which contains the rest-energy. Fixed: the cache is generated on the spot for
whatever checkpoint and PB stage is loaded, and gated against the live model.

TWO CLAIMS FROM THE PREVIOUS ROUND, NOW STATED CORRECTLY.
  The 0.5817 share of the energy gradient on mlp.net.9.bias is a ratio of
  NORMS. The additive quantity is the squared share, 33.8%, and net.9.weight's
  0.8121 is 65.9%, so those two tensors together hold 99.7% of the energy
  gradient's squared norm. "Most of the energy gradient sits on the output
  bias" was wrong; a third of it does, and the bias is invisible to the forces
  and cancels exactly in a same-atom-count paired difference.
  The 9657x is a ratio of two LOCAL gradient norms in the current parameter
  coordinates. It says the sensitivities are very unequal. It does not prove
  that learning the pair must produce a huge common mode or bad forces, and
  since cos(u_sum, u_diff) = +0.1187 is not zero, moving along u_sum does
  change the pair -- only a genuinely identical energy offset cancels exactly.

THE CRITERION, unchanged: an energy-only update improves the pair, adding the
force term removes the improvement, AND the force error itself falls. Only all
three together support a real energy-versus-force competition in this training.
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

from mace import data as mace_data, tools
from mace.data.utils import KeySpecification, update_keyspec_from_kwargs
from mace.tools import torch_geometric, torch_tools
from mace.tools.multihead_tools import prepare_default_head
from mace.tools.scripts_utils import (LRScheduler, get_loss_fn, get_optimizer,
                                      get_params_options)
from mace.tools.train import (attach_density_3d_samples_to_batch,
                              attach_solvent3d_samples_to_batch)
import mace.modules.pb1d_backend as PB

RUN = os.environ.get("KIT_RUN",
                     "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/gate_le")
CFG = os.environ.get("KIT_CFG", "config_pb1d.yaml")
NAME = os.environ.get("KIT_NAME", "s3d_gate_le")
CKPT = os.environ.get("KIT_CKPT", "")
NPAIRS = int(os.environ.get("KIT_NPAIRS", "4"))

os.chdir(RUN)
device = torch_tools.init_device(os.environ.get("KIT_DEVICE", "cuda"))
torch_tools.set_default_dtype("float64")

argv = ["--config", CFG, "--name", NAME, "--seed", "123", "--work_dir", ".",
        "--log_dir", "logs", "--model_dir", "models",
        "--checkpoints_dir", "checkpoints", "--results_dir", "results",
        "--device", "cuda"]
args = tools.build_default_arg_parser().parse_args(argv)
args.key_specification = KeySpecification()
update_keyspec_from_kwargs(args.key_specification, vars(args))
if args.heads is None:
    args.heads = prepare_default_head(args)
assert args.model == "PolarMACE" and args.loss == "energy_forces_electrostatics"
dipole_only = False
args.compute_dipole = True
args.compute_energy = bool(float(args.energy_weight) != 0.0)
args.compute_forces = bool(float(args.forces_weight) != 0.0)
args.compute_virials = args.compute_stress = args.compute_polarizability = False

print("=" * 78)
print(f"loss {args.loss}  energy {args.energy_weight}  forces "
      f"{args.forces_weight}  clip_grad {args.clip_grad}  lr {args.lr}  "
      f"amsgrad {args.amsgrad}  warmup_encounters "
      f"{args.solvent_pb1d_warmup_encounters}  fresh_stage1 "
      f"{args.solvent_pb1d_fresh_stage1}")

mp = [p for p in sorted(glob.glob(os.path.join(RUN, "models", "*.model")))
      if "compiled" not in os.path.basename(p)][-1]
ALL_CK = sorted(glob.glob(os.path.join(RUN, "checkpoints", "*_epoch-*.pt")),
                key=lambda p: int(p.rsplit("epoch-", 1)[1].split(".")[0]))
cpath = CKPT or ALL_CK[-1]
EP = int(os.path.basename(cpath).rsplit("epoch-", 1)[1].split(".")[0])
model = torch.load(f=mp, map_location=device).to(device)
ck = torch.load(cpath, map_location=device)
model.load_state_dict(ck["model"] if "model" in ck else ck.model, strict=False)
model.solvent3d_energy = True
model.solvent_cavity_energy = True
model.solvent_baseline_coupling = True
head = model.local_electron_energy
head_ids = {id(p) for p in head.parameters()}

# ------------------------- defect 2: set the PB stage, then prove it matters --
model._pb1d_epoch = int(EP)
WARM = int(args.solvent_pb1d_warmup_encounters)
print(f"\ncheckpoint {os.path.basename(cpath)}  epoch {EP}")
print(f"_pb1d_epoch set to {EP}; warm-up gate fires while _pb1d_epoch < {WARM} "
      f"AND training=True, so this state is "
      f"{'the WARM-UP path' if EP < WARM else 'the FULL-PB path'}")
print(f"weights are EMA-averaged (every save in tools/train.py runs inside "
      f"`with ema.average_parameters()`); the Adam moments in the same file "
      f"belong to the raw trajectory, which was never written to disk")

loss_fn = get_loss_fn(args, dipole_only, args.compute_dipole).to(device)
optimizer = get_optimizer(args, get_params_options(args, model))
lr_sched = LRScheduler(optimizer, args)
if "optimizer" in ck:
    optimizer.load_state_dict(ck["optimizer"])
if "lr_scheduler" in ck:
    try:
        lr_sched.load_state_dict(ck["lr_scheduler"])
    except Exception as e:
        print(f"   lr_scheduler not loaded: {e}")
nreg = sum(1 for g in optimizer.param_groups for p in g["params"]
           if id(p) in head_ids)
print(f"\noptimizer: head tensors registered {nreg}/{len(head_ids)}; groups "
      + ", ".join(f"{gi}:{g.get('name')} lr={g['lr']:.4g}"
                  for gi, g in enumerate(optimizer.param_groups)
                  if any(id(p) in head_ids for p in g["params"])))

# ----------------- defect 4: Adam step counts BY NAME, across checkpoints ----
opt_order = [p for g in optimizer.param_groups for p in g["params"]]
name_of_param = {id(p): n for n, p in model.named_parameters()}
idx_name = [name_of_param.get(id(p), "?") for p in opt_order]
print(f"\nADAM STEP COUNTS BY PARAMETER NAME, both checkpoints")
print(f"   the optimizer's parameter order is deterministic, so the integer "
      f"keys in a saved optimizer state map to these names. step() only ever "
      f"increments, so the SAME name cannot go down between two checkpoints "
      f"unless the state was lost.")
tbl = {}
for cp in ALL_CK:
    e = int(os.path.basename(cp).rsplit("epoch-", 1)[1].split(".")[0])
    st = torch.load(cp, map_location="cpu").get("optimizer", {}).get("state", {})
    tbl[e] = {idx_name[i]: int(v["step"]) for i, v in st.items()
              if isinstance(i, int) and i < len(idx_name) and "step" in v}
eps_sorted = sorted(tbl)
if len(eps_sorted) >= 2:
    a, b = eps_sorted[0], eps_sorted[-1]
    common = sorted(set(tbl[a]) & set(tbl[b]))
    went_down = [(n, tbl[a][n], tbl[b][n]) for n in common
                 if tbl[b][n] < tbl[a][n]]
    print(f"   epoch {a}: {len(tbl[a])} tensors, steps "
          f"{min(tbl[a].values())}..{max(tbl[a].values())}")
    print(f"   epoch {b}: {len(tbl[b])} tensors, steps "
          f"{min(tbl[b].values())}..{max(tbl[b].values())}")
    print(f"   names whose count DECREASED from epoch {a} to {b}: "
          f"{len(went_down)} of {len(common)}")
    for n, x, y in went_down[:8]:
        print(f"     {n:>58} {x} -> {y}")
    if went_down:
        print(f"   -> a decrease cannot come from 'received gradients less "
              f"often'. The resume job (3430182) rebuilt the optimizer and "
              f"loaded state from the epoch-37 file, so this points at "
              f"resume-time state restoration, and it degrades any one-step "
              f"test taken from the later checkpoint.")
    else:
        print(f"   -> no name decreased; the earlier min/max difference was "
              f"tensors that receive gradients at different rates, not lost "
              f"state.")

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
allp = [(k, k + 600) for k in range(1, 401)
        if k in atoms and k + 600 in atoms and split_of[k] == "train"
        and len(atoms[k]) == 207]
dN = {p: atoms[p[1]].info.get("total_charge", 0.0)
         - atoms[p[0]].info.get("total_charge", 0.0) for p in allp}
allp.sort(key=lambda p: dN[p])
PAIRS = [allp[i] for i in
         np.linspace(0, len(allp) - 1, NPAIRS).round().astype(int)]
print(f"\nsamples: {NPAIRS} TRAIN pairs spanning dN "
      + ", ".join(f"{p[0]}:{dN[p]:+.3f}" for p in PAIRS))
z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])


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
FRAMES = [s for p in PAIRS for s in p]
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


cap = {}
head.register_forward_pre_hook(
    lambda m, a, k: cap.update(
        {kk: k[kk].detach().clone() for kk in
         ("node_feats", "field_feats", "charges_0", "charges_induced")}),
    with_kwargs=True)
head.register_forward_hook(
    lambda m, a, k, o: cap.__setitem__("le", o.detach().clone()),
    with_kwargs=True)


def fwd(sid, training):
    cap.clear()
    with torch.no_grad():
        out = model(BATCH[sid].to_dict(), training=training,
                    compute_force=False)
    r = dict(e=float(out["energy"].sum()), le=cap["le"].clone(),
             **{k: cap[k].clone() for k in
                ("node_feats", "field_feats", "charges_0", "charges_induced")})
    _evict()
    return r


# ------------- defect 2 continued: prove the stage flag changes the forward --
print("\n" + "=" * 78)
print("DOES _pb1d_epoch ACTUALLY CHANGE THE FORWARD? (defect 2, measured)")
s0 = FRAMES[0]
model._pb1d_epoch = 0
a0 = fwd(s0, True)
model._pb1d_epoch = WARM + 5
a1 = fwd(s0, True)
model._pb1d_epoch = int(EP)
aE = fwd(s0, True)
aV = fwd(s0, False)
print(f"   sid {s0}, training=True : _pb1d_epoch 0 -> E {a0['e']:.6f} eV,  "
      f"{WARM + 5} -> E {a1['e']:.6f} eV,  difference {a1['e'] - a0['e']:+.6f} eV")
print(f"   sid {s0}, training=False: E {aV['e']:.6f} eV  (the flag is off, so "
      f"validation always takes the full-PB path)")
if abs(a1["e"] - a0["e"]) < 1e-9:
    print(f"   the stage flag changes nothing here, so defect 2 had no effect "
          f"on the numbers -- but it is still set correctly from now on")
else:
    print(f"   the stage flag DOES change the forward by "
          f"{abs(a1['e'] - a0['e']):.6f} eV, so the previous epoch-39 run, "
          f"which left _pb1d_epoch at its default 0, measured warm-up-path "
          f"gradients while reporting the full-PB stage. Those numbers are "
          f"withdrawn.")
print(f"   this audit now runs at _pb1d_epoch = {EP} "
      f"({'warm-up' if EP < WARM else 'full-PB'}), matching the checkpoint")

# --------- defect 3: build the cache HERE, in this state, and gate it --------
print("\n" + "=" * 78)
print("CACHE BUILT IN THIS STATE (defect 3), then gated against the live model")
C = {s: fwd(s, True) for s in FRAMES}


def run_head(d):
    return head(node_attrs=None, node_feats=d["node_feats"],
                edge_attrs=None, edge_feats=None, edge_index=None,
                field_feats=d["field_feats"], charges_0=d["charges_0"],
                charges_induced=d["charges_induced"])


with torch.no_grad():
    g1 = max(float((run_head(C[s]) - C[s]["le"]).abs().max()) for s in FRAMES)
print(f"   gate A, head replayed on the cached inputs: worst |diff| "
      f"{g1:.3e} eV")
assert g1 < 1e-10, "cached inputs do not reproduce the head's output"
live = np.array([C[sc]["e"] - C[sn]["e"] for sc, sn in PAIRS])
pred = np.array([((C[sc]["e"] - float(C[sc]["le"].sum()))
                  - (C[sn]["e"] - float(C[sn]["le"].sum())))
                 + (float(C[sc]["le"].sum()) - float(C[sn]["le"].sum()))
                 for sc, sn in PAIRS])
print(f"   gate B, cached rest + cached le reproduces the live paired dE: "
      f"worst |diff| {np.abs(live - pred).max():.3e} eV")
assert np.abs(live - pred).max() < 1e-9
DE_DFT = np.array([atoms[sc].info["energy"] - atoms[sn].info["energy"]
                   for sc, sn in PAIRS])
Y = torch.tensor(
    [DE_DFT[i] - ((C[sc]["e"] - float(C[sc]["le"].sum()))
                  - (C[sn]["e"] - float(C[sn]["le"].sum())))
     for i, (sc, sn) in enumerate(PAIRS)], dtype=torch.float64, device=device)
print(f"   paired dE, model {live.round(4)}")
print(f"              DFT   {DE_DFT.round(4)}")
print(f"   |eps| mean {np.abs(live - DE_DFT).mean():.4f} eV;  target for the "
      f"head, mean {float(Y.mean()):+.4f} eV;  current d(le) mean "
      f"{np.mean([float(C[sc]['le'].sum()) - float(C[sn]['le'].sum()) for sc, sn in PAIRS]):+.4f} eV")


def head_vec(getter):
    return torch.cat([getter(p).reshape(-1) for p in head.parameters()])


def dle_t():
    return torch.stack([run_head(C[sc]).sum() - run_head(C[sn]).sum()
                        for sc, sn in PAIRS])


def dir_of(fn):
    model.zero_grad(set_to_none=True)
    fn().backward()
    v = head_vec(lambda q: (q.grad.detach() if q.grad is not None
                            else torch.zeros_like(q))).clone()
    model.zero_grad(set_to_none=True)
    return v


d_pair = -dir_of(lambda: ((dle_t() - Y) ** 2).mean())
d_pair_n = d_pair / d_pair.norm()
u_diff = dir_of(lambda: dle_t().sum())
u_sum = dir_of(lambda: torch.stack([run_head(C[s]).sum()
                                    for s in FRAMES]).sum())


def cosv(a, b):
    return float(a @ b) / max(float(a.norm()) * float(b.norm()), 1e-300)


print(f"\n   reference directions in THIS state: ||d_pair|| "
      f"{float(d_pair.norm()):.4e}, ||u_diff|| {float(u_diff.norm()):.4e}, "
      f"||u_sum|| {float(u_sum.norm()):.4e}")
print(f"   ||u_sum||/||u_diff|| = "
      f"{float(u_sum.norm()) / max(float(u_diff.norm()), 1e-300):.0f}x -- a "
      f"ratio of two LOCAL gradient norms in the current coordinates, i.e. "
      f"unequal sensitivity. It does not by itself imply that fixing the pair "
      f"must produce a large common mode, and since cos(u_sum,u_diff) = "
      f"{cosv(u_sum, u_diff):+.4f} is non-zero, moving along u_sum does change "
      f"the pair; only an exactly identical offset cancels.")

# ------------------------------------------------------------ priority 1 -----
ALL_W = {k: float(getattr(loss_fn, k).item()) for k in dir(loss_fn)
         if k.endswith("_weight") and torch.is_tensor(getattr(loss_fn, k, None))}


def set_terms(keep):
    for k, v in ALL_W.items():
        getattr(loss_fn, k).fill_(v if k in keep else 0.0)


ARMS = [("energy only", {"energy_weight"}),
        ("forces only", {"forces_weight"}),
        ("energy+forces", {"energy_weight", "forces_weight"}),
        ("everything else", set(ALL_W) - {"energy_weight", "forces_weight"}),
        ("full loss", set(ALL_W))]


def accumulate(keep):
    """zero ONCE, then backward each frame with weight 1/N -- what a DDP step
    over N samples produces. Defect 1 was zeroing inside this loop."""
    set_terms(keep)
    model.zero_grad(set_to_none=True)
    lsum = 0.0
    for s in FRAMES:
        out = model(BATCH[s].to_dict(), training=True,
                    compute_force=args.compute_forces)
        L = loss_fn(pred=out, ref=BATCH[s])
        (L / len(FRAMES)).backward()
        lsum += float(L) / len(FRAMES)
        _evict()
    gh = head_vec(lambda p: (p.grad.detach().clone() if p.grad is not None
                             else torch.zeros_like(p)))
    gm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum()
                              for p in model.parameters()
                              if p.grad is not None)))
    return gh, gm, lsum


def per_frame_average(keep):
    """the same quantity computed the old way, one frame at a time, as a check
    that the accumulation is what it claims to be"""
    set_terms(keep)
    acc = None
    for s in FRAMES:
        model.zero_grad(set_to_none=True)
        out = model(BATCH[s].to_dict(), training=True,
                    compute_force=args.compute_forces)
        loss_fn(pred=out, ref=BATCH[s]).backward()
        v = head_vec(lambda p: (p.grad.detach().clone() if p.grad is not None
                                else torch.zeros_like(p)))
        acc = v if acc is None else acc + v
        _evict()
    model.zero_grad(set_to_none=True)
    return acc / len(FRAMES)


print("\n" + "=" * 78)
print("PRIORITY 1 -- the weighted gradient each term puts on the head")
print(f"   accumulated over {len(FRAMES)} frames with weight 1/N each")
res = {}
print(f"\n   {'arm':>17} {'loss':>11} {'||g_head||':>11} {'||g_model||':>11} "
      f"{'cos(-g,d_pair)':>15} {'proj':>11} {'accum vs avg':>13}")
for nm, keep in ARMS:
    gh, gm, ls = accumulate(keep)
    avg = per_frame_average(keep)
    chk = float((gh - avg).abs().max()) / max(float(avg.abs().max()), 1e-300)
    cs = -cosv(gh, d_pair) if float(gh.norm()) > 0 else 0.0
    res[nm] = dict(gh=gh.clone(), gm=gm, cos=cs, loss=ls)
    print(f"   {nm:>17} {ls:11.6f} {float(gh.norm()):11.4e} {gm:11.4e} "
          f"{cs:+15.4f} {float((-gh) @ d_pair_n):+11.3e} {chk:13.3e}")
print(f"   'accum vs avg' is the relative difference between the accumulated "
      f"gradient and the per-frame average; it must be ~0 or defect 1 is back")
print(f"\n   {'arm':>17} {'cos(g,u_diff)':>14} {'cos(g,u_sum)':>13} "
      f"{'net.9.bias sq share':>20} {'net.9.weight sq share':>22}")
names = [k for k, _ in head.named_parameters()]
sizes = [q.numel() for q in head.parameters()]
offs = np.cumsum([0] + sizes)
for nm, _ in ARMS:
    g = res[nm]["gh"]
    if float(g.norm()) == 0.0:
        print(f"   {nm:>17} {'--':>14} {'--':>13} {'--':>20} {'--':>22}")
        continue
    sq = {}
    for j, nt in enumerate(names):
        sq[nt] = (float(g[offs[j]:offs[j + 1]].norm()) / float(g.norm())) ** 2
    print(f"   {nm:>17} {cosv(g, u_diff):+14.4f} {cosv(g, u_sum):+13.4f} "
          f"{sq.get('mlp.net.9.bias', 0.0):20.4f} "
          f"{sq.get('mlp.net.9.weight', 0.0):22.4f}")
print(f"   shares are SQUARED norm fractions, which are the additive ones")

# --------------------------------------------------- clipping + one step -----
print("\n" + "=" * 78)
print("PRIORITY 2 -- clipping, and one step in which ONLY the head moves")
gh, gm, _ = accumulate(set(ALL_W))
fac = min(1.0, args.clip_grad / max(gm, 1e-300))
print(f"   accumulated full-loss ||g_model|| {gm:.4e} vs clip_grad "
      f"{args.clip_grad} -> factor {fac:.4e} "
      f"({'CLIPPED' if gm > args.clip_grad else 'not clipped'}); head ||g|| "
      f"{float(gh.norm()):.4e}")
print(f"   clipping only ever SHORTENS the gradient -- one scalar for the whole "
      f"model, direction unchanged -- so it can make an update small but never "
      f"point it the wrong way. Adam also divides by its own second moment, so "
      f"a uniform factor largely cancels unless it varies step to step.")

W0 = copy.deepcopy(model.state_dict())
O0 = copy.deepcopy(optimizer.state_dict())


def live_eps():
    v = []
    for sc, sn in PAIRS:
        v.append(fwd(sc, True)["e"] - fwd(sn, True)["e"])
    return np.array(v)


base = live_eps()
print(f"\n   |eps| before any step {np.abs(base - DE_DFT).mean():.4f} eV")
print(f"\n   {'arm':>17} {'|eps| after':>12} {'change':>10} {'max|dw|':>10} "
      f"{'cos(dw,d_pair)':>15} {'F rmse before':>14} {'F rmse after':>13}")


def force_rmse():
    out, tot, n = [], 0.0, 0
    for s in FRAMES:
        p = model(BATCH[s].to_dict(), training=True, compute_force=True)
        fm = p["forces"].detach().cpu().numpy()
        fd = np.asarray(atoms[s].get_array("forces"))
        tot += float(((fm - fd) ** 2).sum())
        n += fm.size
        _evict()
    return (tot / n) ** 0.5


f_before = force_rmse()
ARMS2 = ARMS + [("zero grad, Adam history", None)]
for nm, keep in ARMS2:
    model.load_state_dict(W0)
    optimizer.load_state_dict(copy.deepcopy(O0))
    wb = {k: v.detach().clone() for k, v in head.state_dict().items()}
    if keep is None:
        # CONTROL: the current gradient is exactly zero but the Adam moments
        # are kept, so whatever moves is pure momentum. Zeros, not None --
        # None makes Adam skip the tensor entirely and measures nothing.
        model.zero_grad(set_to_none=True)
        for prm in model.parameters():
            prm.grad = (torch.zeros_like(prm) if id(prm) in head_ids else None)
    else:
        accumulate(keep)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=args.clip_grad)
        for prm in model.parameters():
            if id(prm) not in head_ids:
                prm.grad = None
    optimizer.step()
    aft = live_eps()
    dw = torch.cat([(head.state_dict()[k] - wb[k]).reshape(-1) for k in names])
    mdw = float(dw.abs().max())
    cdw = (float(dw @ d_pair_n) / max(float(dw.norm()), 1e-300)
           if float(dw.norm()) > 0 else 0.0)
    f_after = force_rmse()
    print(f"   {nm:>17} {np.abs(aft - DE_DFT).mean():12.4f} "
          f"{np.abs(aft - DE_DFT).mean() - np.abs(base - DE_DFT).mean():+10.4f} "
          f"{mdw:10.3e} {cdw:+15.4f} {f_before:14.4f} {f_after:13.4f}")
model.load_state_dict(W0)
optimizer.load_state_dict(copy.deepcopy(O0))
set_terms(set(ALL_W))
print(f"\n   CRITERION: energy-only lowers |eps|, energy+forces does not, AND "
      f"the force rmse falls. All three are needed before calling this an "
      f"energy-versus-force competition.")
