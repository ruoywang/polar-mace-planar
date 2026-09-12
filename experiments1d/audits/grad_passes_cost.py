"""Cost of the exact unrolled solver graph (GRAD_PASSES=0) against the analytic
adjoint (=1) in a real training step: wall time and peak GPU memory.

Decision input, not a physics result. The second-order check showed the
analytic adjoint gives a force-loss gradient to solve-reaching parameters that
is wrong in sign and size, while the unrolled graph recovers it. Whether the
unrolled graph is affordable at training scale decides whether it is the
remedy or whether the adjoint must be made second-order correct instead.

One training-style step per frame: forward with training=True and
compute_force=True (create_graph), the run's own loss, backward. Three frames
stand in for the three DDP ranks. Everything else as in training at epoch 39
under LIVE_POS=1. Peak memory is torch.cuda.max_memory_allocated, reset per
setting; times are synchronised.
"""
import glob, os, sys, time
import torch
sys.path.insert(0, os.environ.get("KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get("KIT_MACE_REPO", "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")
assert os.environ.get("MACE_PB1D_LIVE_POS")
from ase.io import read
from mace import data as mace_data, tools
from mace.data.utils import KeySpecification, update_keyspec_from_kwargs
from mace.tools import torch_geometric, torch_tools
from mace.tools.multihead_tools import prepare_default_head
from mace.tools.scripts_utils import get_loss_fn
from mace.tools.train import attach_density_3d_samples_to_batch, attach_solvent3d_samples_to_batch
import mace.modules.pb1d_backend as PB
RUN = "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/gate_le"
os.chdir(RUN)
device = torch_tools.init_device("cuda"); torch_tools.set_default_dtype("float64")
args = tools.build_default_arg_parser().parse_args(
    ["--config", "config_pb1d.yaml", "--name", "s3d_gate_le", "--seed", "123", "--work_dir", ".", "--device", "cuda"])
args.key_specification = KeySpecification(); update_keyspec_from_kwargs(args.key_specification, vars(args))
if args.heads is None: args.heads = prepare_default_head(args)
args.compute_dipole = True; args.compute_energy = True; args.compute_forces = True
args.compute_virials = args.compute_stress = args.compute_polarizability = False
mp = [p for p in sorted(glob.glob(os.path.join(RUN, "models", "*.model"))) if "compiled" not in p][-1]
cp = sorted(glob.glob(os.path.join(RUN, "checkpoints", "*_epoch-*.pt")), key=lambda p: int(p.rsplit("epoch-",1)[1].split(".")[0]))[-1]
model = torch.load(f=mp, map_location=device).to(device)
ck = torch.load(cp, map_location=device); model.load_state_dict(ck["model"], strict=False)
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
B = {s: batch(s) for s in FR}
print(f"model {os.path.basename(mp)} @ {os.path.basename(cp)}  LIVE_POS=1  frames {FR}")
print(f"{'GRAD_PASSES':>12} {'frame':>6} {'fwd+loss+bwd s':>15} {'peak GiB':>9} {'loss':>14}")
for gp in (1, 0, 1):   # 1 twice: the first pass pays warm-up costs
    os.environ["MACE_PB1D_GRAD_PASSES"] = str(gp)
    for s in FR:
        model.zero_grad(set_to_none=True); torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            out = model(B[s].to_dict(), training=True, compute_force=True)
            L = loss_fn(pred=out, ref=B[s]); L.backward()
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            pk = torch.cuda.max_memory_allocated() / 2**30
            print(f"{gp:>12} {s:>6} {dt:15.2f} {pk:9.2f} {float(L):14.6f}", flush=True)
        except RuntimeError as e:
            torch.cuda.synchronize()
            print(f"{gp:>12} {s:>6} {'FAILED':>15} {'-':>9}  {str(e)[:80]}", flush=True)
        _evict(); torch.cuda.empty_cache()
print("DONE")
