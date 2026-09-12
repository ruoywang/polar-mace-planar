"""Is the force-loss gradient w.r.t. PARAMETERS correct through the analytic
PB adjoint? (user-specified step 3 of the E_bl fix, 2026-09-12.)

THE CONCERN, from the solver's own construction (pb1d_solver.py:236-300): it
converges under no_grad, then forms ONE differentiable Newton step
phi = phi* + J_c^{-1}(-resid(theta)) with the coupled Jacobian J_c DETACHED.
By the implicit function theorem the FIRST derivative of phi w.r.t. theta is
exact to solve tolerance. The force loss needs d(force)/d(theta) =
d^2 E / (dR d theta), a MIXED SECOND derivative, and the frozen J_c drops the
term J_c^{-1} (dJ_c/dtheta) J_c^{-1} (d resid/dR), whose second factor is the
first-order position response and is NOT small at convergence. A clean
backward() with non-zero gradients cannot detect this. Directional finite
differences in parameter space can.

WHAT IS COMPARED, per parameter group along a fixed random unit direction v:
  v . grad_theta E                      autograd vs central FD of E(theta)
  v . grad_theta L_F                    autograd vs central FD of L_F(theta)
with L_F = mean (F - F_DFT)^2 over all atoms and components, the same form as
the training force term. The autograd arm is run twice: MACE_PB1D_GRAD_PASSES=1
(the analytic adjoint, what trains) and =0 (the fully unrolled graph, exact by
construction and kept in the code as the FD baseline). The FD arm uses two
step sizes so its own drift is visible.

GROUPS. Three that reach the solve -- field_dependent_charges_maps (the
predicted density coefficients), pb1d_head (p_off), products (the trunk) --
and one control that does NOT reach the solve, local_electron_energy, for
which first and second order must both be exact and any discrepancy would
indicate a defect in this script rather than in the solver.

WHY training=False WITH A MANUAL create_graph. The training path differs from
eval only in the warm-up gate (irrelevant at _pb1d_epoch=39 >= 30) and in
create_graph for the force, which is done here explicitly with
autograd.grad(E, positions, create_graph=True). use_cache_rows is False in the
full-PB stage and fresh_stage1 never writes the profile cache, so a perturbed
forward cannot reuse an unperturbed solve; the solve counter confirms it. Run
under MACE_PB1D_LIVE_POS=1, the path the joint training will use.
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
assert os.environ.get("MACE_PB1D_LIVE_POS"), "run with MACE_PB1D_LIVE_POS=1"

from ase.io import read
from mace import data as mace_data, tools
from mace.data.utils import KeySpecification, update_keyspec_from_kwargs
from mace.tools import torch_geometric, torch_tools
import mace.modules.pb1d_backend as PB

RUN = os.environ.get("KIT_RUN",
                     "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/gate_le")
SMOKE = bool(int(os.environ.get("KIT_SMOKE", "0")))
GROUPS_ALL = ["field_dependent_charges_maps", "pb1d_head", "products",
              "local_electron_energy"]
GROUPS = GROUPS_ALL[:1] + GROUPS_ALL[-1:] if SMOKE else GROUPS_ALL
GPS = [1] if SMOKE else [1, 0]
EPS_REL = [1e-4] if SMOKE else [1e-4, 3e-5]
NFRAMES = 1 if SMOKE else 2
SEED = 7

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
model.solvent3d_energy = model.solvent_cavity_energy = True
model.solvent_baseline_coupling = True
model._pb1d_epoch = 39
z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])

SOLVES = {"n": 0}
_bd = {}
_bk = PB.PB1DBackend.solve_graph
def _wrap(self, *a, **k):
    _bd["b"] = self; SOLVES["n"] += 1; return _bk(self, *a, **k)
PB.PB1DBackend.solve_graph = _wrap
def _evict():
    b = _bd.get("b")
    if b is not None:
        for at in ("_bl_ram", "_grids", "_solvers", "_c_units"):
            d = getattr(b, at, None)
            if isinstance(d, dict): d.clear()

PARAMS = {g: [(n, p) for n, p in model.named_parameters() if n.split(".")[0] == g]
          for g in GROUPS}
for g in GROUPS:
    assert PARAMS[g], f"no parameters in group {g}"
gen = torch.Generator(device="cpu").manual_seed(SEED)
DIRS = {}
for g in GROUPS:
    vs = [torch.randn(p.shape, generator=gen, dtype=torch.float64).to(device)
          for _, p in PARAMS[g]]
    nrm = torch.sqrt(sum((v ** 2).sum() for v in vs))
    DIRS[g] = [v / nrm for v in vs]
THETA_NORM = {g: float(torch.sqrt(sum((p.detach() ** 2).sum()
                                      for _, p in PARAMS[g]))) for g in GROUPS}
print("=" * 78)
print(f"model {os.path.basename(mp)} @ {os.path.basename(cpath)}   "
      f"LIVE_POS={os.environ.get('MACE_PB1D_LIVE_POS')}   _pb1d_epoch=39")
for g in GROUPS:
    print(f"  group {g:<32} {sum(p.numel() for _, p in PARAMS[g]):>8} params  "
          f"||theta|| {THETA_NORM[g]:.4e}")

atoms_by_sid, split_of = {}, {}
for sp in ("train", "val", "test"):
    f = os.path.join(RUN, "data", f"{sp}.xyz")
    if os.path.exists(f):
        for a in read(f, ":"):
            s = a.info.get("sample_id")
            if s is not None:
                atoms_by_sid[int(s)] = a; split_of[int(s)] = sp
pair = [(k, k + 600) for k in range(1, 401)
        if k in atoms_by_sid and k + 600 in atoms_by_sid
        and split_of.get(k) == "val" and len(atoms_by_sid[k]) == 207][0]
FRAMES = [pair[0], pair[1]][:NFRAMES]


def batch_for(sid):
    a = atoms_by_sid[sid].copy()   # sample_id kept: the training-cache path
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    return next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)


def set_requires(flag):
    for p in model.parameters():
        p.requires_grad_(flag)


def energy_and_floss(sid, want_param_grad):
    """E and L_F. With want_param_grad the graph to theta is kept and the force
    is built with create_graph=True, exactly what the training force term
    needs; without it the values are returned detached (the FD arm)."""
    b = batch_for(sid)
    b["positions"].requires_grad_(True)
    f_dft = torch.as_tensor(np.asarray(atoms_by_sid[sid].get_array("forces")),
                            dtype=torch.float64, device=device)
    n0 = SOLVES["n"]
    if want_param_grad:
        pred = model(b.to_dict(), compute_force=False, training=False)
        E = pred["energy"].sum()
        F = -torch.autograd.grad(E, b["positions"], create_graph=True)[0]
        LF = ((F - f_dft) ** 2).mean()
        return E, LF, SOLVES["n"] - n0
    with torch.enable_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
        E = pred["energy"].sum()
        F = -torch.autograd.grad(E, b["positions"], create_graph=False)[0]
    LF = ((F.detach() - f_dft) ** 2).mean()
    return float(E.detach()), float(LF), SOLVES["n"] - n0


def directional(grads, g):
    tot = 0.0
    for (_, p), v in zip(PARAMS[g], DIRS[g]):
        gi = grads.get(id(p))
        if gi is not None:
            tot += float((gi * v).sum())
    return tot


for sid in FRAMES:
    q = float(atoms_by_sid[sid].info.get("total_charge", 0.0))
    print("\n" + "=" * 78)
    print(f"sid {sid} ({'charged' if q else 'neutral'}), training-cache path")
    auto = {}
    for gp in GPS:
        os.environ["MACE_PB1D_GRAD_PASSES"] = str(gp)
        set_requires(True)
        try:
            E, LF, ns = energy_and_floss(sid, True)
            plist = [p for g in GROUPS for _, p in PARAMS[g]]
            gE = torch.autograd.grad(E, plist, retain_graph=True, allow_unused=True)
            gF = torch.autograd.grad(LF, plist, allow_unused=True)
            mapE = {id(p): (gi.detach() if gi is not None else None) for p, gi in zip(plist, gE)}
            mapF = {id(p): (gi.detach() if gi is not None else None) for p, gi in zip(plist, gF)}
            auto[gp] = {g: (directional(mapE, g), directional(mapF, g)) for g in GROUPS}
            nE = sum(1 for g in GROUPS for _, p in PARAMS[g] if mapE.get(id(p)) is None)
            print(f"  autograd, GRAD_PASSES={gp}: E {float(E):.9f}  L_F "
                  f"{float(LF):.6e}  ({ns} solves; {nE} tensors with no E-grad)")
        except RuntimeError as exc:
            print(f"  autograd, GRAD_PASSES={gp}: FAILED -- {str(exc)[:140]}")
            auto[gp] = None
        set_requires(False)
        model.zero_grad(set_to_none=True)
        _evict()
        torch.cuda.empty_cache()
    os.environ["MACE_PB1D_GRAD_PASSES"] = "1"

    fd = {}
    for g in GROUPS:
        saved = [p.detach().clone() for _, p in PARAMS[g]]
        for er in EPS_REL:
            eps = er * THETA_NORM[g]
            vals = []
            for sign in (+1.0, -1.0):
                with torch.no_grad():
                    for (_, p), v, s0 in zip(PARAMS[g], DIRS[g], saved):
                        p.copy_(s0 + sign * eps * v)
                Ev, LFv, ns = energy_and_floss(sid, False)
                vals.append((Ev, LFv, ns))
                _evict()
            with torch.no_grad():
                for (_, p), s0 in zip(PARAMS[g], saved):
                    p.copy_(s0)
            dE = (vals[0][0] - vals[1][0]) / (2 * eps)
            dF = (vals[0][1] - vals[1][1]) / (2 * eps)
            fd[(g, er)] = (dE, dF, vals[0][2] + vals[1][2])

    print(f"\n  {'group':<30} {'quantity':>8} " + " ".join(f"{'auto gp=' + str(gp):>14}" for gp in GPS)
          + " " + " ".join(f"{'FD eps=' + str(er):>14}" for er in EPS_REL) + f" {'FD drift':>9}")
    for g in GROUPS:
        for qi, qn in ((0, "v.dE"), (1, "v.dL_F")):
            row = f"  {g:<30} {qn:>8} "
            for gp in GPS:
                a = auto.get(gp)
                row += f"{(a[g][qi] if a else float('nan')):>14.6e} "
            fds = [fd[(g, er)][qi] for er in EPS_REL]
            row += " ".join(f"{x:>14.6e}" for x in fds)
            drift = abs(fds[0] - fds[-1]) / max(abs(fds[0]), 1e-300) if len(fds) > 1 else 0.0
            row += f" {drift:>9.2e}"
            print(row)
    print(f"\n  relative error of each autograd arm against FD (eps={EPS_REL[0]}):")
    for g in GROUPS:
        for qi, qn in ((0, "v.dE"), (1, "v.dL_F")):
            ref = fd[(g, EPS_REL[0])][qi]
            parts = []
            for gp in GPS:
                a = auto.get(gp)
                if a is None:
                    parts.append(f"gp={gp}: n/a"); continue
                rel = abs(a[g][qi] - ref) / max(abs(ref), 1e-300)
                parts.append(f"gp={gp}: {rel:.3e}")
            print(f"    {g:<30} {qn:>8}  " + "   ".join(parts)
                  + ("   <- control: must be exact at both orders"
                     if g == "local_electron_energy" else ""))
    print(f"  solves per FD forward: "
          f"{fd[(GROUPS[0], EPS_REL[0])][2] // 2} (must be > 0: the perturbed "
          f"solve actually re-ran)")
print("\nREADING: v.dE agreeing at both GRAD_PASSES settings confirms the first "
      "order (IFT). v.dL_F agreeing at gp=0 but NOT at gp=1 is the frozen-J_c "
      "second-order gap, and its size is the number to carry into the joint "
      "training decision. The control group must agree everywhere or this "
      "script, not the solver, is wrong.")
print("DONE")
