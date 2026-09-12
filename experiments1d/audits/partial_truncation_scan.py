"""Which CONNECTED output carries a partial position truncation, per path?

The detached-but-moving scan flagged only small diagnostics (solvent_layer_mean,
the density-threshold crossings, charge sums, charges_history, and on the
training path cavity_energy_g). The three energy terms that still disagree
with their finite differences on the training path -- E_bl 32-92, solute
electrostatics 152-154, slab dipole 52-63 meV/A -- all HAVE a grad_fn, so the
truncation sits inside a connected quantity and the earlier scan cannot see it.

This goes one level deeper. For a fixed random unit vector v per output and
the same displacement (largest-|F| atom, z, h = 0.01 A):
    autograd:  d(v . out)/dR_z  from one grad-enabled forward
    FD:        [v . out(+h) - v . out(-h)] / 2h
An output with grad_fn whose autograd and FD disagree beyond the FD floor is a
connected quantity with a detached dependence inside it. Whatever is shared by
the three residual terms -- the SCF charges, the solvent potential, the profile
features -- should appear here, on the training path and not on deployment.
LIVE_POS=1, DFORCE unset.
"""
import glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.environ.get("KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get("KIT_MACE_REPO", "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")
assert not os.environ.get("MACE_PB1D_DFORCE") and os.environ.get("MACE_PB1D_LIVE_POS")
from ase.io import read
from mace import data as mace_data, tools
from mace.data.utils import KeySpecification, update_keyspec_from_kwargs
from mace.tools import torch_geometric, torch_tools
import mace.modules.pb1d_backend as PB
RUN = "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/gate_le"; H = 0.01
os.chdir(RUN)
device = torch_tools.init_device("cuda"); torch_tools.set_default_dtype("float64")
args = tools.build_default_arg_parser().parse_args(["--config", "config_pb1d.yaml", "--name", "s3d_gate_le", "--seed", "123", "--work_dir", ".", "--device", "cuda"])
args.key_specification = KeySpecification(); update_keyspec_from_kwargs(args.key_specification, vars(args))
mp = [p for p in sorted(glob.glob(os.path.join(RUN, "models", "*.model"))) if "compiled" not in p][-1]
cp = sorted(glob.glob(os.path.join(RUN, "checkpoints", "*_epoch-*.pt")), key=lambda p: int(p.rsplit("epoch-",1)[1].split(".")[0]))[-1]
model = torch.load(f=mp, map_location=device).to(device)
model.load_state_dict(torch.load(cp, map_location=device)["model"], strict=False)
model.eval()
for p in model.parameters(): p.requires_grad = False
model.solvent3d_energy = model.solvent_cavity_energy = model.solvent_baseline_coupling = True
model._pb1d_epoch = 39
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
atoms_by_sid, split_of = {}, {}
for sp in ("train", "val", "test"):
    f = os.path.join(RUN, "data", f"{sp}.xyz")
    if os.path.exists(f):
        for a in read(f, ":"):
            s = a.info.get("sample_id")
            if s is not None: atoms_by_sid[int(s)] = a; split_of[int(s)] = sp
pair = [(k, k+600) for k in range(1, 401) if k in atoms_by_sid and k+600 in atoms_by_sid and split_of.get(k) == "val" and len(atoms_by_sid[k]) == 207][0]
gen = torch.Generator(device="cpu").manual_seed(11)

def build(at, keep_sid):
    a = at.copy()
    if not keep_sid: a.info.pop("sample_id", None)
    cfg = mace_data.config_from_atoms(a, key_specification=args.key_specification)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    return next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)

print(f"model {os.path.basename(mp)} @ {os.path.basename(cp)}  LIVE_POS=1  h={H} A along z")
for sid in pair:
    at = atoms_by_sid[sid]
    ia = int(np.argmax(np.linalg.norm(np.asarray(at.get_array("forces")), axis=1)))
    for keep_sid, tag in ((True, "training-cache"), (False, "deployment")):
        b = build(at, keep_sid); b["positions"].requires_grad_(True)
        pred = model(b.to_dict(), compute_force=False, training=False)
        outs = {k: v for k, v in pred.items() if torch.is_tensor(v) and v.numel() > 0
                and v.dtype.is_floating_point and v.grad_fn is not None}
        V = {k: torch.randn(v.shape, generator=gen, dtype=torch.float64).to(device) for k, v in outs.items()}
        for k in V: V[k] = V[k] / V[k].norm()
        auto = {}
        keys = list(outs)
        for i, k in enumerate(keys):
            g = torch.autograd.grad((outs[k] * V[k]).sum(), b["positions"],
                                    retain_graph=(i < len(keys) - 1), allow_unused=True)[0]
            auto[k] = float(g[ia, 2]) if g is not None else 0.0
        _evict()
        vals = {}
        for sign in (+1.0, -1.0):
            pp = at.copy(); pp.positions[ia, 2] += sign * H
            bb = build(pp, keep_sid)
            with torch.no_grad(): pr = model(bb.to_dict(), compute_force=False, training=False)
            vals[sign] = {k: float((pr[k] * V[k]).sum()) for k in keys if k in pr and pr[k].shape == outs[k].shape}
            _evict()
        print("\n" + "=" * 78)
        print(f"sid {sid} [{tag}], atom {ia}, z.  Connected outputs: autograd d(v.out)/dR vs FD")
        print(f"{'output':>34} {'shape':>16} {'autograd':>13} {'FD':>13} {'|diff|':>11} {'rel':>9}  flag")
        rows = []
        for k in keys:
            if k not in vals[1.0] or k not in vals[-1.0]: continue
            fd = (vals[1.0][k] - vals[-1.0][k]) / (2 * H)
            d = abs(auto[k] - fd); rel = d / max(abs(fd), 1e-30)
            rows.append((k, tuple(outs[k].shape), auto[k], fd, d, rel))
        rows.sort(key=lambda r: -r[4])
        for k, shp, a, fd, d, rel in rows:
            if abs(fd) < 1e-8 and d < 1e-8: continue
            flag = "<- PARTIAL TRUNCATION inside" if (d > 1e-4 and rel > 0.02) else ""
            print(f"{k:>34} {str(shp):>16} {a:13.5e} {fd:13.5e} {d:11.3e} {rel:9.2e}  {flag}")
print("\nDONE")
