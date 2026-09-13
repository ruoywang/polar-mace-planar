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
    if s is not None and len(a) == int(os.environ.get("KIT_NATOMS", "207")): atoms[int(s)] = a
FR = [int(x) for x in os.environ["KIT_FRAMES"].split(",")] if os.environ.get("KIT_FRAMES") else sorted(atoms)[:3]
def batch(sid):
    cfg = mace_data.config_from_atoms(atoms[sid], key_specification=args.key_specification)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    attach_density_3d_samples_to_batch(b, loss_fn); attach_solvent3d_samples_to_batch(b, loss_fn); return b
B = {s: batch(s) for s in FR}
print(f"model {os.path.basename(mp)} @ {os.path.basename(cp)}  LIVE_POS=1  frames {FR}  atoms/frame {[len(atoms[s]) for s in FR]}  total_charge {[round(float(atoms[s].info.get('total_charge', 0.0)), 3) for s in FR]}")
print(f"{'GRAD_PASSES':>12} {'frame':>6} {'fwd+loss+bwd s':>15} {'peak GiB':>9} {'loss':>14}")
REC = {}   # (gp, frame) -> (E, F, loss) of the first pass at that gp: value comparison across gp settings
for gp in [int(x) for x in os.environ.get("KIT_GPS", "1,0,1").split(",")]:   # first entry pays warm-up costs
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
            if (gp, s) not in REC:
                # the FULL training-loss gradient, per top-level parameter group
                # (what the optimizer step actually uses); compared across gp
                G = {}
                for n, prm in model.named_parameters():
                    if prm.grad is not None:
                        G.setdefault(n.split(".")[0], []).append(prm.grad.detach().flatten().clone())
                G = {k: torch.cat(v) for k, v in G.items()}
                REC[(gp, s)] = (float(out["energy"].sum()), out["forces"].detach().clone(), float(L), G)
        except RuntimeError as e:
            torch.cuda.synchronize()
            print(f"{gp:>12} {s:>6} {'FAILED':>15} {'-':>9}  {str(e)[:80]}", flush=True)
        _evict(); torch.cuda.empty_cache()
gps = sorted({g for g, _ in REC})
if 0 in gps and len(gps) > 1:
    print(f"\nVALUES vs GRAD_PASSES=0 (the unrolled reference), same frame: |dE| eV, max|dF| eV/A, |dloss|")
    for g in gps:
        if g == 0: continue
        for s in FR:
            if (g, s) in REC and (0, s) in REC:
                E1, F1, L1, G1 = REC[(g, s)]; E0, F0, L0, G0 = REC[(0, s)]
                print(f"  gp={g} frame {s}: |dE| {abs(E1-E0):.3e}   max|dF| {float((F1-F0).abs().max()):.3e}   rms dF {float(((F1-F0)**2).mean().sqrt()):.3e}   |dloss| {abs(L1-L0):.3e}")
    print(f"\nTRAINING-LOSS GRADIENT vs GRAD_PASSES=0, per parameter group: rel err ||g-g0||/||g0||, cosine, ||g0||")
    for g in gps:
        if g == 0: continue
        for s in FR:
            if (g, s) not in REC or (0, s) not in REC: continue
            G1, G0 = REC[(g, s)][3], REC[(0, s)][3]
            allk = sorted(set(G0) | set(G1)); tot1 = []; tot0 = []
            print(f"  gp={g} frame {s}:")
            for k in allk:
                a = G1.get(k); b = G0.get(k)
                if a is None or b is None or a.numel() != b.numel():
                    print(f"    {k:>32}: present in gp={g}: {a is not None}, in gp=0: {b is not None}"); continue
                tot1.append(a); tot0.append(b)
                n0 = float(b.norm()); rel = float((a - b).norm()) / max(n0, 1e-300)
                cos = float((a * b).sum() / max(a.norm() * b.norm(), 1e-300))
                print(f"    {k:>32}: rel err {rel:.3e}   cos {cos:.8f}   ||g0|| {n0:.3e}   ({a.numel()} params)")
            if tot1:
                a = torch.cat(tot1); b = torch.cat(tot0)
                print(f"    {'ALL':>32}: rel err {float((a-b).norm()/b.norm()):.3e}   cos {float((a*b).sum()/(a.norm()*b.norm())):.8f}   ||g0|| {float(b.norm()):.3e}")
print("DONE")
