"""Does MACE_PB1D_LIVE_POS change ANY forward value? It must not.

The switch is meant to reconnect a gradient path only: positions live into the
solve inputs, E_bl kept in the graph, the compensation profile and the slab
dipole kept live, the cavity's pf_e following. None of those is supposed to
alter a number. This runs the same frames with the switch off and on, in a
single process (the switch is read from the environment at call time, so it can
be toggled between forwards), and compares the total energy and every term
that enters it, on both baseline paths, with and without compute_force.

Anything above float64 noise on the same iterative solve (~1e-9 eV, the level
the analytic-adjoint docstring quotes for value shift) is a defect in the
switch, not a feature.
"""
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
assert not os.environ.get("MACE_PB1D_DFORCE")
os.environ.pop("MACE_PB1D_LIVE_POS", None)

from ase.io import read
from mace import data as mace_data, tools
from mace.data.utils import KeySpecification, update_keyspec_from_kwargs
from mace.tools import torch_geometric, torch_tools
import mace.modules.pb1d_backend as PB

RUN = os.environ.get("KIT_RUN",
                     "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/gate_le")
os.chdir(RUN)
device = torch_tools.init_device(os.environ.get("KIT_DEVICE", "cuda"))
torch_tools.set_default_dtype("float64")
argv = ["--config", "config_pb1d.yaml", "--name", "s3d_gate_le", "--seed",
        "123", "--work_dir", ".", "--device", "cuda"]
args = tools.build_default_arg_parser().parse_args(argv)
args.key_specification = KeySpecification()
update_keyspec_from_kwargs(args.key_specification, vars(args))
kspec = args.key_specification
mp = [p for p in sorted(glob.glob(os.path.join(RUN, "models", "*.model")))
      if "compiled" not in os.path.basename(p)][-1]
cpath = sorted(glob.glob(os.path.join(RUN, "checkpoints", "*_epoch-*.pt")),
               key=lambda p: int(p.rsplit("epoch-", 1)[1].split(".")[0]))[-1]
model = torch.load(f=mp, map_location=device).to(device)
ck = torch.load(cpath, map_location=device)
model.load_state_dict(ck["model"] if "model" in ck else ck.model, strict=False)
model.eval()
for p in model.parameters():
    p.requires_grad = False
model.solvent3d_energy = model.solvent_cavity_energy = True
model.solvent_baseline_coupling = True
z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
_bd = {}
_bk = PB.PB1DBackend.solve_graph
def _wrap(self, *a, **k):
    _bd["b"] = self; return _bk(self, *a, **k)
PB.PB1DBackend.solve_graph = _wrap
def _evict():
    b = _bd.get("b")
    if b is not None:
        for at in ("_bl_ram", "_grids", "_solvers", "_c_units"):
            d = getattr(b, at, None)
            if isinstance(d, dict): d.clear()

TERMS = ["interaction_energy", "electron_energy", "electrostatic_energy",
         "compensation_periodic_1d_energy",
         "compensation_slab_correction_energy", "cavity_energy_g",
         "solvent3d_energy_g", "baseline_coupling_energy_g"]

atoms_by_sid, split_of = {}, {}
for sp in ("train", "val", "test"):
    f = os.path.join(RUN, "data", f"{sp}.xyz")
    if os.path.exists(f):
        for a in read(f, ":"):
            s = a.info.get("sample_id")
            if s is not None:
                atoms_by_sid[int(s)] = a; split_of[int(s)] = sp
VALP = [(k, k + 600) for k in range(1, 401)
        if k in atoms_by_sid and k + 600 in atoms_by_sid
        and split_of.get(k) == "val" and len(atoms_by_sid[k]) == 207][:2]


def forward(at, keep_sid, force):
    a = at.copy()
    if not keep_sid:
        a.info.pop("sample_id", None)
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    if force:
        pred = model(b.to_dict(), compute_force=True, training=False)
    else:
        with torch.no_grad():
            pred = model(b.to_dict(), compute_force=False, training=False)
    out = {"energy": float(pred["energy"].detach().sum())}
    for k in TERMS:
        v = pred.get(k)
        out[k] = float(v.detach().sum()) if v is not None else 0.0
    ef, ed = pred.get("external_field"), pred.get("explicit_dipole")
    out["extfield_dipole"] = float((ef * ed).detach().sum()) if ef is not None and ed is not None else 0.0
    if force:
        out["_F"] = pred["forces"].detach().cpu().numpy()
    _evict()
    return out


print("=" * 78)
print(f"model {os.path.basename(mp)} @ {os.path.basename(cpath)}")
worst = 0.0
for sc, sn in VALP:
    for sid in (sc, sn):
        for keep_sid, tag in ((True, "train"), (False, "deploy")):
            for force in (False, True):
                os.environ.pop("MACE_PB1D_LIVE_POS", None)
                off = forward(atoms_by_sid[sid], keep_sid, force)
                os.environ["MACE_PB1D_LIVE_POS"] = "1"
                on = forward(atoms_by_sid[sid], keep_sid, force)
                os.environ.pop("MACE_PB1D_LIVE_POS", None)
                dmax = max(abs(on[k] - off[k]) for k in off if not k.startswith("_"))
                kmax = max((k for k in off if not k.startswith("_")),
                           key=lambda k: abs(on[k] - off[k]))
                worst = max(worst, dmax)
                line = (f"sid {sid:>4} {tag:>6} force={int(force)}: "
                        f"E off {off['energy']:.9f} on {on['energy']:.9f}  "
                        f"worst term |diff| {dmax:.3e} eV ({kmax})")
                if force:
                    # the FORCE is allowed to change -- that is the point --
                    # so it is reported, not gated
                    dF = float(np.abs(on["_F"] - off["_F"]).max())
                    line += f"   |dF| max {1000*dF:.1f} meV/A (expected to change)"
                print(line, flush=True)
print(f"\nworst forward-VALUE difference across all cells and terms: "
      f"{worst:.3e} eV  ->  "
      + ("VALUES UNCHANGED (below the 1e-9 eV solve-tolerance level)"
         if worst < 1e-9 else "VALUES CHANGED -- the switch is not gradient-only"))
