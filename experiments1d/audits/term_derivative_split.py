"""Step 2: which energy TERM carries the derivative gap? (user-specified,
2026-09-11.)

Step 1 measured, on 8 cells, a stable disagreement between the force the model
returns and the true slope of its own energy: |F_auto - F_FD| = 68-348 meV/A
against an 11-25 meV/A step-size drift, concentrated on z (113-602) with x and
y (20-31) at the drift level. That located the gap to a DIRECTION, not to a
term: solute electrostatics, the slab dipole correction and the field-dependent
local_electron_energy all produce z response, so the solvent energy is not
implicated by the direction alone.

THE DECOMPOSITION covers every term that actually enters total_energy, read
from the assembly rather than assumed (extensions.py:2596, 2613, 2970-2984):
  e0                                  per-species constant, zero force by
                                      construction -- a closure check, not a
                                      candidate
  interaction_energy                  the ordinary energy head
  electron_energy                     local_electron_energy, the head under
                                      study (extensions.py:3109 stores le_total
                                      under this key)
  electrostatic_energy                solute electrostatics
  compensation_periodic_1d_energy
  compensation_slab_correction_energy
  external_field . explicit_dipole
  cavity_energy_g
  solvent3d_energy_g
  baseline_coupling_energy_g

SIGN CONVENTION, fixed:  D_k = F_auto,k - F_FD,k.

THREE CLOSURES, gated before anything is interpreted:
  sum_k E_k        == the model's own energy
  sum_k F_auto,k   == the model's own returned force
  sum_k D_k        == D_total, COMPONENT BY COMPONENT
The third is the one the user asked for. RMS values are never summed: terms are
combined as signed components and only then reduced, because cancellation
between terms is exactly what has to remain visible.

THE SHARPEST SINGLE DIAGNOSTIC. A term whose autograd gradient w.r.t. positions
is None (graph-disconnected) while its finite difference is non-zero is a
truncated derivative by definition, with no interpretation needed. That is
reported per term.

WHY THE DISPLACEMENTS ARE REDONE. Step 1 stored only the total energy at each
displaced geometry, so the per-term values do not exist and cannot be recovered
from its output. The step size is fixed at h = 0.01 A this time rather than
swept, because step 1 measured the drift across 0.005/0.01/0.02 at 0.33-0.74%
of the largest component, far below the gaps being attributed.
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
assert not os.environ.get("MACE_PB1D_DFORCE"), "run with DFORCE unset"

from ase.io import read

from mace import data as mace_data, tools
from mace.data.utils import KeySpecification, update_keyspec_from_kwargs
from mace.tools import torch_geometric, torch_tools
import mace.modules.pb1d_backend as PB

RUN = os.environ.get("KIT_RUN",
                     "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/gate_le")
SMOKE = bool(int(os.environ.get("KIT_SMOKE", "0")))
NATOMS = 2 if SMOKE else int(os.environ.get("KIT_NATOMS", "6"))
NPAIRS = 1 if SMOKE else 2
H = float(os.environ.get("KIT_H", "0.01"))

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
model.solvent3d_energy = True
model.solvent_cavity_energy = True
model.solvent_baseline_coupling = True
z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
print("=" * 78)
print(f"model {os.path.basename(mp)} @ {os.path.basename(cpath)}   h={H} A   "
      f"D_k = F_auto,k - F_FD,k")

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


atoms_by_sid, split_of = {}, {}
for sp in ("train", "val", "test"):
    f = os.path.join(RUN, "data", f"{sp}.xyz")
    if not os.path.exists(f):
        continue
    for a in read(f, ":"):
        s = a.info.get("sample_id")
        if s is not None:
            atoms_by_sid[int(s)] = a
            split_of[int(s)] = sp
VALP = [(k, k + 600) for k in range(1, 401)
        if k in atoms_by_sid and k + 600 in atoms_by_sid
        and split_of.get(k) == "val" and len(atoms_by_sid[k]) == 207][:NPAIRS]

TERMS = ["e0", "interaction_energy", "electron_energy", "electrostatic_energy",
         "compensation_periodic_1d_energy",
         "compensation_slab_correction_energy", "extfield_dipole",
         "cavity_energy_g", "solvent3d_energy_g", "baseline_coupling_energy_g"]
SHORT = {"e0": "e0", "interaction_energy": "energy head",
         "electron_energy": "local_electron", "electrostatic_energy": "solute ES",
         "compensation_periodic_1d_energy": "comp 1D",
         "compensation_slab_correction_energy": "slab dipole",
         "extfield_dipole": "extfield.dip", "cavity_energy_g": "cavity",
         "solvent3d_energy_g": "solvent3D",
         "baseline_coupling_energy_g": "baseline E_bl"}


def build(at, keep_sid):
    a = at.copy()
    if not keep_sid:
        a.info.pop("sample_id", None)
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1)))
    return b.to(device)


def term_tensors(pred, b):
    """every term as it enters total_energy, in the live graph"""
    out = {}
    nat = int(b["ptr"][1] - b["ptr"][0])
    heads = (b["head"][b["batch"]] if "head" in b.to_dict()
             else torch.zeros_like(b["batch"]))
    ar = torch.arange(nat, device=device)
    e0v = model.atomic_energies_fn(b["node_attrs"])
    out["e0"] = (e0v[ar, heads] if e0v.dim() > 1 else e0v).sum()
    for k in TERMS:
        if k in ("e0", "extfield_dipole"):
            continue
        v = pred.get(k)
        out[k] = v.sum() if v is not None else torch.zeros((), device=device)
    # extensions.py:3071 returns "external_field" as external_potential[:, 1:],
    # so the slice in the assembly expression is ALREADY applied here. Slicing
    # again gave 2 columns against explicit_dipole's 3 (job 3431871). The
    # verified form is the one charging_reconcile.py:267-271 uses, and copying
    # the assembly expression instead of the already-checked script is what
    # introduced the error.
    ef, ed = pred.get("external_field"), pred.get("explicit_dipole")
    out["extfield_dipole"] = ((ef * ed).sum() if ef is not None
                              and ed is not None else
                              torch.zeros((), device=device))
    return out


def forward_terms(at, keep_sid, want_grad):
    b = build(at, keep_sid)
    if want_grad:
        b["positions"].requires_grad_(True)
        pred = model(b.to_dict(), compute_force=True, training=False)
        tt = term_tensors(pred, b)
        res = dict(E=float(pred["energy"].detach()),
                   F=pred["forces"].detach().cpu().numpy(),
                   terms={k: float(v.detach()) for k, v in tt.items()})
        # THREE cases, distinguished rather than merged, because they mean
        # different things. allow_unused covers an unused INPUT; it does not
        # cover an OUTPUT with no grad_fn, which raises instead (job 3431950).
        #   no grad_fn        the term is detached from the whole graph -- true
        #                     by construction for e0, a finding for anything else
        #   grad returns None the term is in the graph but does not reach
        #                     positions
        #   otherwise         a real per-term force
        res["Fk"], res["detached"], res["no_pos"] = {}, [], []
        for k, v in tt.items():
            if not v.requires_grad:
                res["Fk"][k] = np.zeros_like(res["F"])
                res["detached"].append(k)
                continue
            g = torch.autograd.grad(v, b["positions"], retain_graph=True,
                                    allow_unused=True)[0]
            if g is None:
                res["Fk"][k] = np.zeros_like(res["F"])
                res["no_pos"].append(k)
            else:
                res["Fk"][k] = (-g).detach().cpu().numpy()
        _evict()
        return res
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
        tt = term_tensors(pred, b)
        res = dict(E=float(pred["energy"]),
                   terms={k: float(v) for k, v in tt.items()})
    _evict()
    return res


for sc, sn in VALP:
    for sid in (sc, sn):
        at = atoms_by_sid[sid]
        syms = np.array(at.get_chemical_symbols())
        f_dft = np.array(at.arrays["forces"], dtype=float)
        mag = np.linalg.norm(f_dft, axis=1)
        pick = []
        for s in sorted(set(syms)):
            idx = np.where(syms == s)[0]
            pick.append(int(idx[np.argmax(mag[idx])]))
        for i in np.argsort(-mag):
            if len(pick) >= NATOMS:
                break
            if int(i) not in pick:
                pick.append(int(i))
        pick = sorted(pick[:NATOMS])
        q = float(at.info.get("total_charge", 0.0))
        for keep_sid, tag in ((True, "training-cache"), (False, "deployment")):
            print("\n" + "=" * 78)
            print(f"sid {sid} ({'charged' if q else 'neutral'}) [{tag}] atoms "
                  + ",".join(str(i) for i in pick))
            base = forward_terms(at, keep_sid, True)
            ssum = sum(base["terms"].values())
            print(f"  CLOSURE 1, sum of terms vs the model's energy: "
                  f"{ssum:+.9f} vs {base['E']:+.9f}, |diff| "
                  f"{abs(ssum - base['E']):.3e} eV")
            fsum = sum(base["Fk"].values())
            d1 = float(np.abs(fsum - base["F"]).max())
            print(f"  CLOSURE 2, sum of per-term autograd forces vs the "
                  f"returned force: max |diff| {d1:.3e} eV/A")
            det = [k for k in base["detached"] if k != "e0"]
            print(f"  terms with NO grad_fn at all (detached from the graph): "
                  + (", ".join(SHORT[k] for k in base["detached"]) or "none")
                  + ("   <- e0 is a per-species constant, expected"
                     if base["detached"] == ["e0"] else
                     "   <- anything beyond e0 here is a finding" if det else ""))
            print(f"  terms in the graph but NOT reaching positions: "
                  + (", ".join(SHORT[k] for k in base["no_pos"]) or "none"))
            if abs(ssum - base["E"]) > 1e-6 or d1 > 1e-6:
                print(f"  CLOSURE FAILED -- the decomposition is not the "
                      f"energy the model uses; stopping this cell")
                continue
            fd = {k: np.zeros((len(pick), 3)) for k in TERMS}
            fd_tot = np.zeros((len(pick), 3))
            for ai, ia in enumerate(pick):
                for c in range(3):
                    pp = at.copy(); pp.positions[ia, c] += H
                    pm = at.copy(); pm.positions[ia, c] -= H
                    rp = forward_terms(pp, keep_sid, False)
                    rm = forward_terms(pm, keep_sid, False)
                    fd_tot[ai, c] = -(rp["E"] - rm["E"]) / (2 * H)
                    for k in TERMS:
                        fd[k][ai, c] = -(rp["terms"][k] - rm["terms"][k]) / (2 * H)
            fdsum = sum(fd.values())
            d3 = float(np.abs(fdsum - fd_tot).max())
            print(f"  CLOSURE 3, sum of per-term FD vs total FD: max |diff| "
                  f"{1000 * d3:.3e} meV/A")
            D, Dtot = {}, base["F"][pick] - fd_tot
            for k in TERMS:
                D[k] = base["Fk"][k][pick] - fd[k]
            dsum = sum(D.values())
            d4 = float(np.abs(dsum - Dtot).max())
            print(f"  CLOSURE 4 (the one that matters), sum_k D_k vs D_total, "
                  f"COMPONENT BY COMPONENT: max |diff| {1000 * d4:.3e} meV/A")
            if d4 > 1e-6:
                print(f"  CLOSURE 4 FAILED -- stopping this cell")
                continue
            print(f"\n  {'term':>16} {'D_k z: signed sum':>18} {'rms':>9} "
                  f"{'|D_k z| max':>12} {'share of |sum|':>15}")
            zs = {k: D[k][:, 2] for k in TERMS}
            tot_z = Dtot[:, 2]
            order = sorted(TERMS, key=lambda k: -np.abs(zs[k]).max())
            for k in order:
                v = zs[k]
                print(f"  {SHORT[k]:>16} {1000 * v.sum():+18.2f} "
                      f"{1000 * float(np.sqrt((v ** 2).mean())):9.2f} "
                      f"{1000 * float(np.abs(v).max()):12.2f} "
                      f"{100 * v.sum() / (tot_z.sum() if abs(tot_z.sum()) > 1e-30 else 1):14.1f}%")
            print(f"  {'TOTAL':>16} {1000 * tot_z.sum():+18.2f} "
                  f"{1000 * float(np.sqrt((tot_z ** 2).mean())):9.2f} "
                  f"{1000 * float(np.abs(tot_z).max()):12.2f}")
            pos = sum(v.sum() for v in zs.values() if v.sum() > 0)
            neg = sum(v.sum() for v in zs.values() if v.sum() < 0)
            print(f"  cancellation: positive contributions {1000 * pos:+.2f}, "
                  f"negative {1000 * neg:+.2f}, net {1000 * (pos + neg):+.2f} "
                  f"meV/A -- the shares above can exceed 100% because terms "
                  f"cancel, which is why RMS values are never summed here")
print("\nDONE")
