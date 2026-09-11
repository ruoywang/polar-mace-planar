"""Does weakening the force constraint actually make the charging energy easier
to learn, and at what cost? Three-arm short controlled experiment.
(user-specified design, 2026-09-11.)

  arm A: energy 1, forces 100   the run's own weights, the control
  arm B: energy 1, forces  20   a decisive intervention, below BOTH local
                                crossover points measured in the gradient
                                audit (forces_weight 32.4 at epoch 25 and 87.6
                                at epoch 39, where the net projection on the
                                pair-improving direction changes sign)
  arm C: energy 1, forces   0   can the energy supervision alone move it?

20 is chosen to be clearly past the crossover, NOT as a proposed training
weight.

WHAT IS HELD FIXED. Only local_electron_energy is updated; every other
parameter keeps its gradient for the clipping norm, exactly as training
computes it, and is then skipped by setting its grad to None before the step.
The forward keeps the full coordinate-derivative path with compute_force=True
-- forces cannot be trained from a cache of position-detached features, so
nothing here is cached. All three arms start from the same weights, the same
loaded Adam state and the same learning rate, and consume the same samples in
the same order. The original per-atom energy normalisation and the original
whole-model clipping rule are untouched, and NO paired loss is added: changing
the weights and the objective at once would confound the two.

WHAT THIS IS NOT. The checkpoint stores EMA-averaged weights (every save in
tools/train.py runs inside `with ema.average_parameters()`) while the Adam
moments belong to the raw trajectory, which was never written to disk. So this
is a controlled short experiment from the closest available state, not a replay
of the original trajectory, and 100 steps is not a training run.

READING, fixed in advance:
  B improves the paired energy while absolute energies and forces hold   ->
      supports lowering the force weight, then a full joint run to confirm
  B or C improves the pair but the forces clearly degrade   ->  a real
      trade-off exists and simply lowering the force weight is not a fix
  even C mostly moves the COMMON energy level with the pair barely changing ->
      the problem is not only the force weight, and the next target is how the
      energy head distinguishes the two charge states
  the three arms do not separate within the budget  ->  report that the short
      experiment did not resolve it; that is not evidence that the weight is
      irrelevant
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
SMOKE = bool(int(os.environ.get("KIT_SMOKE", "0")))
NTRAIN_PAIRS = 2 if SMOKE else int(os.environ.get("KIT_NTRAIN_PAIRS", "16"))
STEPS = 4 if SMOKE else int(os.environ.get("KIT_STEPS", "100"))
MARKS = [0, 2, 4] if SMOKE else [0, 20, 50, 100]
ARM_W = [("A", 1.0, 100.0), ("B", 1.0, 20.0), ("C", 1.0, 0.0)]

os.chdir(RUN)
device = torch_tools.init_device(os.environ.get("KIT_DEVICE", "cuda"))
torch_tools.set_default_dtype("float64")

argv = ["--config", "config_pb1d.yaml", "--name", "s3d_gate_le", "--seed",
        "123", "--work_dir", ".", "--log_dir", "logs", "--model_dir", "models",
        "--checkpoints_dir", "checkpoints", "--results_dir", "results",
        "--device", "cuda"]
args = tools.build_default_arg_parser().parse_args(argv)
args.key_specification = KeySpecification()
update_keyspec_from_kwargs(args.key_specification, vars(args))
if args.heads is None:
    args.heads = prepare_default_head(args)
args.compute_dipole = True
args.compute_energy = True
args.compute_forces = True
args.compute_virials = args.compute_stress = args.compute_polarizability = False

mp = [p for p in sorted(glob.glob(os.path.join(RUN, "models", "*.model")))
      if "compiled" not in os.path.basename(p)][-1]
cpath = sorted(glob.glob(os.path.join(RUN, "checkpoints", "*_epoch-*.pt")),
               key=lambda p: int(p.rsplit("epoch-", 1)[1].split(".")[0]))[-1]
EP = int(os.path.basename(cpath).rsplit("epoch-", 1)[1].split(".")[0])
model = torch.load(f=mp, map_location=device).to(device)
ck = torch.load(cpath, map_location=device)
model.load_state_dict(ck["model"] if "model" in ck else ck.model, strict=False)
model.solvent3d_energy = True
model.solvent_cavity_energy = True
model.solvent_baseline_coupling = True
model._pb1d_epoch = int(EP)
head = model.local_electron_energy
head_ids = {id(p) for p in head.parameters()}
for p in model.parameters():
    p.requires_grad_(True)

loss_fn = get_loss_fn(args, False, True).to(device)
optimizer = get_optimizer(args, get_params_options(args, model))
lr_sched = LRScheduler(optimizer, args)
if "optimizer" in ck:
    optimizer.load_state_dict(ck["optimizer"])
if "lr_scheduler" in ck:
    try:
        lr_sched.load_state_dict(ck["lr_scheduler"])
    except Exception as e:
        print(f"lr_scheduler not loaded: {e}")

print("=" * 78)
print(f"start   {os.path.basename(cpath)} (epoch {EP}, _pb1d_epoch {EP}, "
      f"EMA-averaged weights; Adam moments are the raw trajectory's)")
print(f"steps   {STEPS}, recorded at {MARKS};  train pairs {NTRAIN_PAIRS}")
print(f"frozen  everything except local_electron_energy "
      f"({sum(p.numel() for p in head.parameters())} parameters)")
print(f"fixed   lr {optimizer.param_groups[0]['lr']:.4g}, clip_grad "
      f"{args.clip_grad}, per-atom energy normalisation, no added paired loss")
print(f"arms    " + ",  ".join(f"{a}: energy {e:g} forces {f:g}"
                               for a, e, f in ARM_W))

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
pairs_of = {sp: [] for sp in ("train", "val", "test")}
for k in range(1, 401):
    if k in atoms and k + 600 in atoms and len(atoms[k]) == 207 \
            and split_of[k] == split_of[k + 600]:
        pairs_of[split_of[k]].append((k, k + 600))
dN = {p: atoms[p[1]].info.get("total_charge", 0.0)
         - atoms[p[0]].info.get("total_charge", 0.0)
      for v in pairs_of.values() for p in v}
tr = sorted(pairs_of["train"], key=lambda p: dN[p])
TRAIN = [tr[i] for i in
         np.linspace(0, len(tr) - 1, NTRAIN_PAIRS).round().astype(int)]
VAL = pairs_of["val"][:2] if SMOKE else pairs_of["val"]
print(f"\ntrain pairs, spanning dN {dN[TRAIN[0]]:+.3f}..{dN[TRAIN[-1]]:+.3f}: "
      + " ".join(str(p[0]) for p in TRAIN))
print(f"val pairs ({len(VAL)}): " + " ".join(str(p[0]) for p in VAL))
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


BATCH = {s: make_batch(s) for p in TRAIN + VAL for s in p}
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


ALL_W = {k: float(getattr(loss_fn, k).item()) for k in dir(loss_fn)
         if k.endswith("_weight") and torch.is_tensor(getattr(loss_fn, k, None))}


def set_weights(e, f):
    for k, v in ALL_W.items():
        getattr(loss_fn, k).fill_(v)
    loss_fn.energy_weight.fill_(e)
    loss_fn.forces_weight.fill_(f)


# ------------------------------------------------------------- evaluation ----
def evaluate():
    """the fixed 20 val pairs: paired dE stats, and per-state absolute energy
    and force errors. Forces need the coordinate derivative, so this is a full
    forward with compute_force=True."""
    per = {}
    for sc, sn in VAL:
        for s in (sc, sn):
            out = model(BATCH[s].to_dict(), training=False, compute_force=True)
            e = float(out["energy"].sum())
            fm = out["forces"].detach().cpu().numpy()
            fd = np.asarray(atoms[s].get_array("forces"))
            per[s] = dict(e=e, err=e - float(atoms[s].info["energy"]),
                          nat=len(atoms[s]),
                          frmse=float(np.sqrt(((fm - fd) ** 2).mean())))
            _evict()
    de_m = np.array([per[sc]["e"] - per[sn]["e"] for sc, sn in VAL])
    de_d = np.array([float(atoms[sc].info["energy"])
                     - float(atoms[sn].info["energy"]) for sc, sn in VAL])
    r = de_m - de_d
    out = dict(
        pair_rmse=float(np.sqrt((r ** 2).mean())), pair_bias=float(r.mean()),
        pair_mr=float(np.sqrt(((r - r.mean()) ** 2).mean())))
    for tag, i in (("chg", 0), ("neu", 1)):
        v = [per[p[i]] for p in VAL]
        pa = np.array([1000.0 * x["err"] / x["nat"] for x in v])
        out[f"E_{tag}_mean"] = float(pa.mean())
        out[f"E_{tag}_rms"] = float(np.sqrt((pa ** 2).mean()))
        out[f"F_{tag}"] = float(np.mean([x["frmse"] for x in v]))
    return out


KEYS = ["pair_rmse", "pair_bias", "pair_mr", "E_chg_mean", "E_chg_rms",
        "F_chg", "E_neu_mean", "E_neu_rms", "F_neu"]
UNITS = {"pair_rmse": "eV", "pair_bias": "eV", "pair_mr": "eV",
         "E_chg_mean": "meV/at", "E_chg_rms": "meV/at", "F_chg": "eV/A",
         "E_neu_mean": "meV/at", "E_neu_rms": "meV/at", "F_neu": "eV/A"}

W0 = copy.deepcopy(model.state_dict())
O0 = copy.deepcopy(optimizer.state_dict())
# the sample order is fixed once and shared by all three arms
ORDER = [TRAIN[i % len(TRAIN)] for i in range(STEPS)]

print("\n" + "=" * 78)
print("step 0 (identical for all arms, computed once)")
set_weights(1.0, 100.0)
base = evaluate()
for k in KEYS:
    print(f"   {k:>11} {base[k]:+.6e} {UNITS[k]}")

RES = {}
for arm, ew, fw in ARM_W:
    model.load_state_dict(W0)
    optimizer.load_state_dict(copy.deepcopy(O0))
    set_weights(ew, fw)
    print("\n" + "=" * 78)
    print(f"ARM {arm}: energy {ew:g}, forces {fw:g}")
    RES[arm] = {0: base}
    for st in range(1, STEPS + 1):
        sc, sn = ORDER[st - 1]
        model.zero_grad(set_to_none=True)
        for s in (sc, sn):
            out = model(BATCH[s].to_dict(), training=True, compute_force=True)
            (loss_fn(pred=out, ref=BATCH[s]) / 2.0).backward()
            _evict()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=args.clip_grad)
        for prm in model.parameters():
            if id(prm) not in head_ids:
                prm.grad = None
        optimizer.step()
        if st in MARKS:
            RES[arm][st] = evaluate()
            print(f"   step {st:>4}: pair rmse {RES[arm][st]['pair_rmse']:+.6e} "
                  f"bias {RES[arm][st]['pair_bias']:+.6e}  "
                  f"F_chg {RES[arm][st]['F_chg']:+.6e}", flush=True)
    print(f"   arm {arm} done")
model.load_state_dict(W0)
optimizer.load_state_dict(copy.deepcopy(O0))

print("\n" + "=" * 78)
print("CHANGES FROM STEP 0, all in scientific notation")
for k in KEYS:
    print(f"\n   {k} [{UNITS[k]}]   step 0 = {base[k]:+.6e}")
    print(f"   {'arm':>5} " + " ".join(f"{('step ' + str(m)):>16}"
                                       for m in MARKS[1:]))
    for arm, ew, fw in ARM_W:
        row = []
        for m in MARKS[1:]:
            v = RES[arm].get(m)
            row.append("--" if v is None else f"{v[k] - base[k]:+16.6e}")
        print(f"   {arm:>5} " + " ".join(f"{x:>16}" for x in row))
print(f"\n   arm A is the control at the run's own weights; B and C differ from "
      f"it ONLY in forces_weight. Any column where the three arms do not "
      f"separate by more than A's own drift is unresolved at this budget, not "
      f"evidence that the weight is irrelevant.")
