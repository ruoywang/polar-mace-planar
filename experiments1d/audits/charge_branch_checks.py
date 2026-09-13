"""Pre-training checks of the charge-scalar residual branch (user-specified
2026-09-13): on a few charged / neutral frames of both cell sizes,
  (a) zero-initialised branch: energies and forces identical to the plain model;
  (b) the branch INPUT carries the charge-state difference: per-atom input blocks
      compared between a charged frame and its neutral partner (same geometry);
  (c) with a non-zero output layer the branch contributes energy, its parameters
      receive gradient from the training loss, and one Adam step moves them.
Model: arm B's final model (EMA weights) with the branch attached post hoc.
The FD checks of the branch's force and force-loss gradient are done by
term_derivative_split / second_order_check with KIT_ATTACH_CHARGE_BRANCH.
"""
import glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.environ.get("KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get("KIT_MACE_REPO", "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")
from ase.io import read
from mace import data as mace_data, tools
from mace.data.utils import KeySpecification, update_keyspec_from_kwargs
from mace.tools import torch_geometric, torch_tools
from mace.tools.scripts_utils import get_loss_fn
from mace.modules.loss import attach_density_3d_samples_to_batch
from mace.modules.solvent3d import attach_solvent3d_samples_to_batch
from mace.modules.field_blocks import attach_charge_branch
import mace.modules.pb1d_backend as PB
RUN = os.environ.get("KIT_RUN", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/ab_deriv_B"); NAME = os.environ.get("KIT_NAME", "ab_deriv_B")
os.chdir(RUN)
SIDS = [int(x) for x in os.environ.get("KIT_SIDS", "28,628,30,630,202,203").split(",")]
STD = float(os.environ.get("KIT_STD", "0.05"))
device = torch_tools.init_device("cuda"); torch_tools.set_default_dtype("float64")
args = tools.build_default_arg_parser().parse_args(["--config", "config_pb1d.yaml", "--name", NAME, "--seed", "123", "--work_dir", ".", "--device", "cuda"])
args.key_specification = KeySpecification(); update_keyspec_from_kwargs(args.key_specification, vars(args))
args.compute_virials = args.compute_stress = args.compute_polarizability = False
mp = [p for p in sorted(glob.glob(os.path.join(RUN, "models", "*.model"))) if "compiled" not in p][-1]
cp = sorted(glob.glob(os.path.join(RUN, "checkpoints", "*_epoch-*.pt")), key=lambda p: int(p.rsplit("epoch-",1)[1].split(".")[0]))[-1]
model = torch.load(f=mp, map_location=device).to(device)
model.load_state_dict(torch.load(cp, map_location=device)["model"], strict=False)
model.eval(); model._pb1d_epoch = 39
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
    torch.cuda.empty_cache()
atoms = {}
for sp in ("train", "val", "test"):
    f = os.path.join(RUN, "data", f"{sp}.xyz")
    if os.path.exists(f):
        for a in read(f, ":"):
            s = a.info.get("sample_id")
            if s is not None: atoms[int(s)] = a
def batch(sid):
    cfg = mace_data.config_from_atoms(atoms[sid], key_specification=args.key_specification)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    attach_density_3d_samples_to_batch(b, loss_fn); attach_solvent3d_samples_to_batch(b, loss_fn); return b
def ef(sid):
    b = batch(sid); b["positions"].requires_grad_(True)
    with torch.enable_grad():
        out = model(b.to_dict(), compute_force=False, training=False); E = out["energy"].sum()
        F = -torch.autograd.grad(E, b["positions"])[0]
    r = (float(E), F.detach().cpu().numpy(), float(out["charge_branch_energy"].sum()), float(out["electron_energy"].sum()))
    del out, E, F, b; _evict(); return r

print(f"model {os.path.basename(mp)} @ {os.path.basename(cp)}   env LIVE_POS={os.environ.get('MACE_PB1D_LIVE_POS')} GRAD_PASSES={os.environ.get('MACE_PB1D_GRAD_PASSES')} AREA_EPS={os.environ.get('MACE_PB1D_AREA_EPS')}")
print(f"frames {SIDS}  atoms {[len(atoms[s]) for s in SIDS]}  total_charge {[round(float(atoms[s].info.get('total_charge', 0.0)), 3) for s in SIDS]}")
for p in model.parameters(): p.requires_grad_(False)
base = {s: ef(s) for s in SIDS}
cb = attach_charge_branch(model, int(os.environ.get("KIT_HIDDEN", "64")))
for p in cb.parameters(): p.requires_grad_(False)
print(f"\nbranch attached: n_in {cb.n_in} = {len(cb.node_scalar_idx)} node scalars + {len(cb.charge_l0_idx)} total-charge l0 + {len(cb.charge_l0_idx)} induced l0 + {len(cb.field_l0_idx)} potential + {cb.n_field_l1} |field|; {sum(p.numel() for p in cb.parameters())} params, output layer zero")
print("\n(a) ZERO-INIT IDENTITY, branch attached vs plain model")
for s in SIDS:
    E, F, Eb, Ele = ef(s); E0, F0, _, Ele0 = base[s]
    print(f"  sid {s:>4}: |dE| {abs(E-E0):.3e} eV   max|dF| {np.abs(F-F0).max():.3e} eV/A   branch energy {Eb:.3e}   electron_energy {Ele:+.6f} (plain {Ele0:+.6f})")
print("\n(b) BRANCH INPUTS: charged frame vs neutral partner (same geometry), per-atom differences by block")
cap = {}
def hook(mod, inp, out): cap["x"] = mod.inputs(*inp).detach().cpu().numpy()
h = cb.register_forward_hook(hook)
X = {}
for s in SIDS:
    ef(s); X[s] = cap["x"]
h.remove()
nq = len(cb.charge_l0_idx); nv0 = len(cb.field_l0_idx); nv1 = cb.n_field_l1
blocks = {"q_total l0": slice(0, nq), "q_induced l0": slice(nq, 2*nq), "potential": slice(2*nq, 2*nq+nv0), "|field|": slice(2*nq+nv0, 2*nq+nv0+nv1), "node scalars": slice(2*nq+nv0+nv1, None)}
for s in SIDS:
    x = X[s]; qt = x[:, blocks["q_total l0"]].sum(axis=1); qi = x[:, blocks["q_induced l0"]].sum(axis=1)
    print(f"  sid {s:>4}: sum_atoms q_total l0 {qt.sum():+.4f}  sum_atoms q_induced l0 {qi.sum():+.4f}  (frame total_charge {float(atoms[s].info.get('total_charge', 0.0)):+.4f});  per-atom |q_induced| mean {np.abs(qi).mean():.4f} max {np.abs(qi).max():.4f}")
for k in [s for s in SIDS if s < 600 and s + 600 in X]:
    dx = X[k] - X[k+600]
    print(f"  pair {k}/{k+600}: " + "  ".join(f"{nm}: mean|d| {np.abs(dx[:, sl]).mean():.3e} max|d| {np.abs(dx[:, sl]).max():.3e} (|x| {np.abs(X[k][:, sl]).mean():.3e})" for nm, sl in blocks.items()))
print(f"\n(c) NON-ZERO OUTPUT LAYER: N(0, {STD}) weights, seed 11")
g = torch.Generator(device="cpu").manual_seed(11)
with torch.no_grad():
    cb.out.weight.copy_(torch.randn(cb.out.weight.shape, generator=g, dtype=torch.float64).to(device) * STD); cb.out.bias.zero_()
for s in SIDS:
    E, F, Eb, Ele = ef(s); E0, F0, _, _ = base[s]
    print(f"  sid {s:>4}: branch energy {Eb:+.5f} eV ({1e3*Eb/len(atoms[s]):+.3f} meV/atom)   dE vs plain {E-E0:+.5f}   max|dF| {np.abs(F-F0).max():.3e} eV/A")
for k in [s for s in SIDS if s < 600 and s + 600 in base]:
    print(f"  pair {k}/{k+600}: branch dE(charged - neutral) = {ef(k)[2] - ef(k+600)[2]:+.5f} eV")
print("\n  gradients of the TRAINING LOSS (training=True, compute_force) on the branch, one frame each:")
for p in cb.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam(cb.parameters(), lr=0.01, amsgrad=True)
for s in SIDS[:4]:
    b = batch(s); model.zero_grad(set_to_none=True); opt.zero_grad(set_to_none=True)
    out = model(b.to_dict(), training=True, compute_force=True); L = loss_fn(pred=out, ref=b); L.backward()
    gn = {n: float(p.grad.norm()) if p.grad is not None else float("nan") for n, p in cb.named_parameters()}
    w0 = [p.detach().clone() for p in cb.parameters()]
    opt.step()
    dw = float(torch.sqrt(sum(((p.detach() - w) ** 2).sum() for p, w in zip(cb.parameters(), w0))))
    print(f"  sid {s:>4}: loss {float(L):.6f}  |grad| lin1.w {gn['lin1.weight']:.3e} lin2.w {gn['lin2.weight']:.3e} out.w {gn['out.weight']:.3e} out.b {gn['out.bias']:.3e}   one Adam step |dw| {dw:.3e}")
    del out, L, b; _evict()
print("\nDONE")
