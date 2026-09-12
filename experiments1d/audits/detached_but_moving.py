"""Which forward outputs MOVE when an atom is displaced but carry NO grad_fn?

Those are truncated position dependences by construction -- no interpretation
needed. Run on both baseline paths so the training-path-specific truncation
that survived LIVE_POS (E_bl 32-92, solute electrostatics 152-154, slab dipole
52-63 meV/A, all only on the frozen-baseline branch) is named by the scan
rather than guessed.

Method per frame and path: one forward with positions requiring grad (records
requires_grad / grad_fn for every tensor output), then two no-grad forwards
with one atom displaced +-h along z (the axis that carries the gap), FD of
every output. An output with |FD| above a floor and no grad_fn is flagged.
Large per-atom or grid outputs are summarised by the max |FD| over their
elements. LIVE_POS=1, DFORCE unset, GRAD_PASSES default.
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
RUN = "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/gate_le"
H = 0.01
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

def fwd(at, keep_sid, grad):
    a = at.copy()
    if not keep_sid: a.info.pop("sample_id", None)
    cfg = mace_data.config_from_atoms(a, key_specification=args.key_specification)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    if grad:
        b["positions"].requires_grad_(True)
        pred = model(b.to_dict(), compute_force=False, training=False)
    else:
        with torch.no_grad(): pred = model(b.to_dict(), compute_force=False, training=False)
    out = {}
    for k, v in pred.items():
        if torch.is_tensor(v) and v.numel() > 0 and v.dtype.is_floating_point:
            out[k] = (v.detach().cpu().numpy().astype(float), bool(v.requires_grad), v.grad_fn is not None)
    _evict(); return out

print(f"model {os.path.basename(mp)} @ {os.path.basename(cp)}  LIVE_POS=1  h={H} A along z")
for sid in pair:
    at = atoms_by_sid[sid]
    f_dft = np.linalg.norm(np.asarray(at.get_array("forces")), axis=1)
    ia = int(np.argmax(f_dft))         # the largest-|F| atom, as in the term split
    for keep_sid, tag in ((True, "training-cache"), (False, "deployment")):
        base = fwd(at, keep_sid, True)
        pp = at.copy(); pp.positions[ia, 2] += H
        pm = at.copy(); pm.positions[ia, 2] -= H
        rp, rm = fwd(pp, keep_sid, False), fwd(pm, keep_sid, False)
        print("\n" + "=" * 78)
        print(f"sid {sid} [{tag}], atom {ia} displaced along z")
        print(f"{'output':>36} {'shape':>14} {'grad_fn':>8} {'max|FD| per A':>14}  flag")
        rows = []
        for k, (v0, rq, gf) in base.items():
            if k not in rp or k not in rm: continue
            vp, vm = rp[k][0], rm[k][0]
            if vp.shape != v0.shape or vm.shape != v0.shape: continue
            fd = np.abs(vp - vm).max() / (2 * H)
            rows.append((k, v0.shape, gf, fd))
        rows.sort(key=lambda r: -r[3])
        for k, shp, gf, fd in rows:
            if fd < 1e-9: continue
            flag = "<- MOVES, NO grad_fn: truncated" if not gf else ""
            print(f"{k:>36} {str(shp):>14} {str(gf):>8} {fd:14.4e}  {flag}")
        still = [k for k, shp, gf, fd in rows if fd < 1e-9]
        print(f"   ({len(still)} outputs did not move under this displacement)")
print("\nDONE")
