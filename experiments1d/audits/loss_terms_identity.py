"""Does MACE_PB1D_LIVE_POS change any LOSS-TERM value? The energy gate says the
energies and energy terms are identical, but the full training loss rose
(2.699734 -> 3.737508 on the first cost frame) once stage 1 carried a graph.
The expected cause is the force term -- the reconnected force is farther from
DFT, as anticipated -- but the observable terms (potential, fermi, density_3d,
occ_aug, solvent3d, the 1-D profiles) were not individually gated, and the
joint A/B is confounded if any of them changed value. Same frames, switch off
vs on in one process, training=True, GRAD_PASSES=0, each term isolated by
zeroing the loss's registered weight buffers; the force term is expected to
differ and is reported, every other term must not.
"""
import glob, os, sys
import torch
sys.path.insert(0, os.environ.get("KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get("KIT_MACE_REPO", "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1"); os.environ["MACE_PB1D_GRAD_PASSES"] = "0"
assert not os.environ.get("MACE_PB1D_DFORCE"); os.environ.pop("MACE_PB1D_LIVE_POS", None)
from ase.io import read
from mace import data as mace_data, tools
from mace.data.utils import KeySpecification, update_keyspec_from_kwargs
from mace.tools import torch_geometric, torch_tools
from mace.tools.multihead_tools import prepare_default_head
from mace.tools.scripts_utils import get_loss_fn
from mace.tools.train import attach_density_3d_samples_to_batch, attach_solvent3d_samples_to_batch
import mace.modules.pb1d_backend as PB
RUN = "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/gate_le"; os.chdir(RUN)
device = torch_tools.init_device("cuda"); torch_tools.set_default_dtype("float64")
args = tools.build_default_arg_parser().parse_args(["--config", "config_pb1d.yaml", "--name", "s3d_gate_le", "--seed", "123", "--work_dir", ".", "--device", "cuda"])
args.key_specification = KeySpecification(); update_keyspec_from_kwargs(args.key_specification, vars(args))
if args.heads is None: args.heads = prepare_default_head(args)
args.compute_dipole = True; args.compute_energy = True; args.compute_forces = True
args.compute_virials = args.compute_stress = args.compute_polarizability = False
mp = [p for p in sorted(glob.glob(os.path.join(RUN, "models", "*.model"))) if "compiled" not in p][-1]
cp = sorted(glob.glob(os.path.join(RUN, "checkpoints", "*_epoch-*.pt")), key=lambda p: int(p.rsplit("epoch-",1)[1].split(".")[0]))[-1]
model = torch.load(f=mp, map_location=device).to(device)
model.load_state_dict(torch.load(cp, map_location=device)["model"], strict=False)
for p in model.parameters(): p.requires_grad = False
model.solvent3d_energy = model.solvent_cavity_energy = model.solvent_baseline_coupling = True
model._pb1d_epoch = 39
loss_fn = get_loss_fn(args, False, True).to(device)
z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
_bd = {}; _bk = PB.PB1DBackend.solve_graph
def _wrap(self, *a, **k): _bd["b"] = self; return _bk(self, *a, **k)
PB.PB1DBackend.solve_graph = _wrap
def _evict():
    b = _bd.get("b")
    if b is not None:
        for at in ("_bl_ram", "_grids", "_solvers", "_c_units"):
            d = getattr(b, at, None)
            if isinstance(d, dict): d.clear()
atoms = {}
for a in read(os.path.join(RUN, "data", "train.xyz"), ":"):
    s = a.info.get("sample_id")
    if s is not None and len(a) == 207: atoms[int(s)] = a
FR = sorted(atoms)[:3]
def batch(sid):
    cfg = mace_data.config_from_atoms(atoms[sid], key_specification=args.key_specification)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    attach_density_3d_samples_to_batch(b, loss_fn); attach_solvent3d_samples_to_batch(b, loss_fn); return b
W = {k: float(getattr(loss_fn, k).item()) for k in dir(loss_fn)
     if k.endswith("_weight") and torch.is_tensor(getattr(loss_fn, k, None)) and float(getattr(loss_fn, k).item()) > 0}
def set_only(k):
    for kk, v in W.items(): getattr(loss_fn, kk).fill_(v if kk == k else 0.0)
def terms(sid):
    b = batch(sid); out = {}
    with torch.enable_grad():
        pred = model(b.to_dict(), training=True, compute_force=True)
        for k in W:
            set_only(k); out[k] = float(loss_fn(pred=pred, ref=b).detach())
        for kk, v in W.items(): getattr(loss_fn, kk).fill_(v)
        out["TOTAL"] = float(loss_fn(pred=pred, ref=b).detach())
    _evict(); return out
print(f"model {os.path.basename(mp)} @ {os.path.basename(cp)}  training=True  GRAD_PASSES=0  frames {FR}")
worst_nonforce = 0.0
for sid in FR:
    os.environ.pop("MACE_PB1D_LIVE_POS", None); off = terms(sid)
    os.environ["MACE_PB1D_LIVE_POS"] = "1";   on = terms(sid)
    os.environ.pop("MACE_PB1D_LIVE_POS", None)
    print(f"\nframe {sid}:  {'loss term':>28} {'LIVE_POS off':>14} {'LIVE_POS on':>14} {'|diff|':>11}")
    for k in list(W) + ["TOTAL"]:
        d = abs(on[k] - off[k]); tag = ""
        if k == "forces_weight": tag = "  <- expected to change (reconnected force vs DFT)"
        elif k != "TOTAL":
            worst_nonforce = max(worst_nonforce, d)
            if d > 1e-9: tag = "  <- CHANGED: not a gradient-only switch"
        print(f"          {k:>28} {off[k]:14.8f} {on[k]:14.8f} {d:11.3e}{tag}")
print(f"\nworst non-force loss-term difference: {worst_nonforce:.3e}  ->  "
      + ("ALL NON-FORCE TERMS UNCHANGED" if worst_nonforce < 1e-9 else "A NON-FORCE TERM CHANGED"))
