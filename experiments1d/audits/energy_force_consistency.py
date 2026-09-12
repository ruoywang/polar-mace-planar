"""Energy-force consistency of the current checkpoint. (user-specified, step 1
of the derivative plan, 2026-09-11.)

THREE COMPARISONS, each answering a different question:
  F_auto vs F_FD   how much of the energy's own position response does the
                   current derivative miss?
  F_auto vs F_DFT  how accurate is the force the model actually returns?
  F_FD  vs F_DFT   how accurate is the slope of the current energy surface?
The third is the one that separates a DERIVATIVE gap from an ENERGY-EXPRESSION
truncation: if F_auto ~ F_FD but both sit far from DFT, the derivative is
consistent with the energy and the energy surface itself is what is wrong.

TWO PATHS, checked separately, because they differ in the baseline:
  training-cache path   sample_id present -> bl_row comes from the frozen
                        per-sid baseline in data/baseline_cache (9.1 GB, on
                        disk). pb1d_backend.py:317-324 says outright that this
                        frozen baseline "cannot carry position gradients
                        (measured artifact forces)". It is also keyed by sid
                        alone, so displacing an atom does not change it.
  deployment path       sample_id removed -> bl_row is None -> the runtime
                        baseline tables are used, which do depend on positions.

WHAT IS HELD FIXED. solvent3d_energy, solvent_cavity_energy and
solvent_baseline_coupling stay ON in every arm, and MACE_PB1D_DFORCE is never
set. That switch does not only change the derivative: the same source comment
says it swaps the frozen per-sid baseline for the runtime tables, so a
comparison across it would confound derivative coverage with a changed energy
definition. Each arm's FD is the slope of that arm's OWN energy.

EVERY DISPLACEMENT RE-SOLVES. The per-sid profile cache (prof1d_<sid>.npz,
geometry-independent by construction) is empty in this run directory, so a
displaced forward is a cache miss and solves fresh -- but that is verified
rather than assumed: PB1DBackend.solve_graph is counted, and a displaced
forward that performs zero solves would mean the solvent was reused and the FD
would be meaningless.

STEP SIZES. Central differences at three h, so stability is demonstrated. A
central difference has truncation error O(h^2) and roundoff O(eps*|E|/h); with
|E| ~ 1.3e3 eV and float64 that floor is ~3e-10 eV/A at h = 0.01, far below
anything here, so a spread across h that is NOT small would indicate a
discontinuous or iteratively-converged energy rather than noise.
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
assert not os.environ.get("MACE_PB1D_DFORCE"), \
    "MACE_PB1D_DFORCE changes the baseline as well as the derivative; this " \
    "audit must run with it unset"

from ase.io import read

from mace import data as mace_data, tools
from mace.data.utils import KeySpecification, update_keyspec_from_kwargs
from mace.tools import torch_geometric, torch_tools
import mace.modules.pb1d_backend as PB

RUN = os.environ.get("KIT_RUN",
                     "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/gate_le")
SMOKE = bool(int(os.environ.get("KIT_SMOKE", "0")))
NATOMS = 2 if SMOKE else int(os.environ.get("KIT_NATOMS", "6"))
HS = [0.01] if SMOKE else [float(x) for x in
                           os.environ.get("KIT_HS", "0.005,0.01,0.02").split(",")]
NPAIRS = 1 if SMOKE else 2

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
print(f"model {os.path.basename(mp)} @ {os.path.basename(cpath)}")
print(f"MACE_PB1D_DFORCE unset; solvent3d/cavity/baseline_coupling all ON in "
      f"every arm")
print(f"profile cache ./cache: {len(glob.glob(os.path.join(RUN, 'cache', '*')))}"
      f" files;  baseline cache present: "
      f"{os.path.exists(os.path.join(RUN, 'data', 'baseline_cache'))}")
print(f"step sizes {HS} A, {NATOMS} atoms x 3 components per frame")

# A non-zero solve count proves the solve was ENTERED, not that it converged
# and not that nothing fell back. The backend already records convergence
# provenance -- n_outer, solver_exit, rms_last, against the max_outer cap --
# precisely because "a small residual is consistent with converging on the
# criterion AND with hitting the cap" (pb1d_backend.py:781-784). All of it is
# collected here. The other two escapes are covered separately: a cache-row
# short-circuit would show as ZERO solves, and the planar warm-up fallback
# cannot fire at all because it is gated on training=True and every forward
# here is training=False.
SOLVES = {"n": 0, "prov": []}
_bk = PB.PB1DBackend.solve_graph
_bd = {}


def _wrap(self, *a, **k):
    _bd["b"] = self
    SOLVES["n"] += 1
    r = _bk(self, *a, **k)
    if isinstance(r, dict):
        SOLVES["prov"].append((r.get("n_outer"), r.get("solver_exit"),
                               r.get("rms_last")))
    return r


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
print(f"geometries: {NPAIRS} NiN44 val pairs "
      + ", ".join(f"{a}/{b}" for a, b in VALP))


def energy(at, want_force, keep_sid):
    a = at.copy()
    if not keep_sid:
        a.info.pop("sample_id", None)
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1)))
    b = b.to(device)
    before, pbefore = SOLVES["n"], len(SOLVES["prov"])
    if want_force:
        pred = model(b.to_dict(), compute_force=True, training=False)
        e = float(pred["energy"].detach())
        f = pred["forces"].detach().cpu().numpy()
    else:
        with torch.no_grad():
            pred = model(b.to_dict(), compute_force=False, training=False)
        e, f = float(pred["energy"]), None
    ns = SOLVES["n"] - before
    prov = SOLVES["prov"][pbefore:]
    _evict()
    return e, f, ns, prov


def rmse(a, b):
    return 1000.0 * float(np.sqrt(((a - b) ** 2).mean()))


for sc, sn in VALP:
    for sid in (sc, sn):
        at = atoms_by_sid[sid]
        f_dft = np.array(at.arrays["forces"], dtype=float)
        syms = np.array(at.get_chemical_symbols())
        mag = np.linalg.norm(f_dft, axis=1)
        # selection rule, stated: the largest-|F_DFT| atom of each species,
        # then filled to NATOMS by |F_DFT| overall. Deterministic.
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
        print("\n" + "=" * 78)
        print(f"sid {sid}  ({'charged' if q else 'neutral'}, q={q:+.3f}), atoms "
              + ", ".join(f"{i}:{syms[i]}" for i in pick))
        for keep_sid, tag in ((True, "training-cache baseline"),
                              (False, "deployment runtime baseline")):
            e_ng, _, ns_ng, pv_ng = energy(at, False, keep_sid)
            e_g, f_auto, ns_g, pv_g = energy(at, True, keep_sid)
            print(f"\n  [{tag}]")
            same = "IDENTICAL" if abs(e_g - e_ng) < 1e-9 else (
                "DIFFER -- the forward is not the same with grad on")
            print(f"    E(no_grad) {e_ng:.9f}  E(grad) {e_g:.9f}  "
                  f"|diff| {abs(e_g - e_ng):.3e} eV   {same}")
            print(f"    PB solves per forward: no_grad {ns_ng}, grad {ns_g}")


            def prov_line(pv, tag):
                if not pv:
                    return f"      {tag}: no solve provenance recorded"
                nout = [x[0] for x in pv if x[0] is not None]
                ex = [str(x[1]) for x in pv if x[1] is not None]
                rl = [x[2] for x in pv if x[2] is not None]
                cap = getattr(_bd.get("b"), "max_outer", None)
                hit = sum(1 for n in nout if cap and n >= cap)
                import collections
                return (f"      {tag}: n_outer "
                        + (f"{min(nout)}..{max(nout)}" if nout else "-")
                        + f" (cap {cap}, at cap {hit}/{len(nout) if nout else 0})"
                        + f"  rms_last max "
                        + (f"{max(rl):.3e}" if rl else "-")
                        + f" (tol {getattr(_bd.get('b'), 'tol', None)})"
                        + f"  exits " + str(dict(collections.Counter(ex))))


            print(f"    solver provenance -- a count alone is not convergence:")
            print(prov_line(pv_ng, "no_grad forward"))
            print(prov_line(pv_g, "grad forward"))
            fd = {}
            for h in HS:
                col = np.zeros((len(pick), 3))
                nsolve = 0
                fdprov = []
                for ai, ia in enumerate(pick):
                    for c in range(3):
                        pp = at.copy(); pp.positions[ia, c] += h
                        pm = at.copy(); pm.positions[ia, c] -= h
                        ep, _, n1, q1 = energy(pp, False, keep_sid)
                        em, _, n2, q2 = energy(pm, False, keep_sid)
                        nsolve += n1 + n2
                        fdprov.extend(q1); fdprov.extend(q2)
                        col[ai, c] = -(ep - em) / (2 * h)
                fd[h] = col
                print(prov_line(fdprov, f"h={h} displaced forwards"))
                print(f"    h={h:<6} FD done, {nsolve} PB solves over "
                      f"{2 * 3 * len(pick)} displaced forwards"
                      + ("   <- ZERO SOLVES: the solvent was reused, FD is "
                         "meaningless" if nsolve == 0 else ""), flush=True)
            hs = sorted(fd)
            if len(hs) > 1:
                sp = max(float(np.abs(fd[a_] - fd[b_]).max())
                         for a_ in hs for b_ in hs)
                print(f"    FD stability across h: max |difference| "
                      f"{1000 * sp:.3f} meV/A "
                      f"({100 * sp / max(float(np.abs(fd[hs[len(hs) // 2]]).max()), 1e-30):.2f}% "
                      f"of the largest FD component)")
            fref = fd[hs[len(hs) // 2]]
            fa = f_auto[pick]
            fdft = f_dft[pick]
            print(f"    RMSE on the selected {len(pick)} atoms x 3 components, "
                  f"meV/A:")
            print(f"      F_auto - F_FD  = {rmse(fa, fref):9.2f}   <- derivative "
                  f"vs the energy's own slope")
            print(f"      F_auto - F_DFT = {rmse(fa, fdft):9.2f}")
            print(f"      F_FD   - F_DFT = {rmse(fref, fdft):9.2f}   <- slope of "
                  f"the current energy surface")
            print(f"      |F_DFT| rms on these atoms = "
                  f"{1000 * float(np.sqrt((fdft ** 2).mean())):.2f} meV/A")
            for ai, ia in enumerate(pick):
                print(f"        atom {ia:>3} {syms[ia]:>2}  auto "
                      + " ".join(f"{1000 * v:+9.1f}" for v in fa[ai])
                      + "   FD " + " ".join(f"{1000 * v:+9.1f}" for v in fref[ai])
                      + "   DFT " + " ".join(f"{1000 * v:+9.1f}" for v in fdft[ai]))
print("\nCOVERAGE: the selected atoms are the largest-|F_DFT| ones, chosen to "
      "expose a gap. Agreement on those components does NOT establish that "
      "every atom's derivative is correct, and this round deliberately does "
      "not expand to a full scan.")
print("READING: a stable auto-vs-FD gap beyond the step-size drift means the "
      "derivative disagrees with THAT path's own forward energy. Agreement "
      "with both far from DFT means the derivative is consistent and the "
      "energy surface is inaccurate -- and this test cannot say whether that "
      "inaccuracy comes from the energy expression's approximations or from "
      "the trained parameters. An FD that moves with h, or a solve that hit "
      "its cap, means the question is undecided rather than answered.")
print("\nDONE")
