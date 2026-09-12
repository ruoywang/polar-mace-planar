"""Step 1 of the E_bl fix: separate the TWO responses inside it, numerically,
without touching the forward. (user-specified, 2026-09-11.)

E_bl is a solvent-SOLUTE COUPLING term, not a solvent term and not a pure
baseline term: pb1d_backend.py builds it as

    Phi_b   = fourier_upsample(phi_base.mean(dim=(0,1))), then mean-removed
    E_bl    = sum_z (rho_ion + rho_bound)(z) * (-Phi_b(z)) * dz * A
    e_bl_t  = e_bl_raw if live_resp else e_bl_raw.detach()

so at fixed cell its position derivative splits as

    dE_bl/dR = int( Phi_b d(rho_solv)/dR + rho_solv d(Phi_b)/dR ) dV

with the first factor the solvent charge response and the second the solute
baseline-potential response. The trailing detach() removes BOTH from the
autograd graph whenever MACE_PB1D_DFORCE is unset, which is the default, so
the energy loss cannot push gradients back through this coupling either.

WHAT THIS SCRIPT DOES AND DOES NOT DO. It measures the two pieces by finite
difference at fixed partner -- rho displaced against the undisplaced Phi_b and
vice versa -- using diagnostic-only exports added to the solve result. It
changes no energy and no value. It does NOT remove the detach: the acceptance
criterion for that is E_bl's OWN autograd force matching its OWN finite
difference, and knowing which of the two responses carries the force is what
tells us which upstream dependence has to be reconnected.

EXPECTED, and therefore worth stating before measuring: in the training-cache
path phi_base comes from the frozen per-sid baseline, which is keyed by sid and
does not move with the geometry, so d(Phi_b)/dR should be zero BY THAT PATH'S
DEFINITION. In the deployment path phi_base comes from the runtime tables,
which do depend on positions, so both pieces should be non-zero. A departure
from that is itself a finding.

CLOSURES: E_bl recomputed from the exports must equal the model's own
baseline_coupling_energy_g, and FD_rho + FD_phi must equal the full FD of E_bl
to the O(h^2) cross term that a central difference leaves behind.
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
OUT = os.environ.get("KIT_OUT", ".")

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
print(f"model {os.path.basename(mp)} @ {os.path.basename(cpath)}  h={H} A")

CAP = {"rows": []}
_bk = PB.PB1DBackend.solve_graph
_bd = {}


def _wrap(self, *a, **k):
    _bd["b"] = self
    r = _bk(self, *a, **k)
    if isinstance(r, dict) and r.get("bl_rho_solv_z") is not None:
        CAP["rows"].append({
            "rho": r["bl_rho_solv_z"].detach().cpu().numpy().copy(),
            "phi": r["bl_phi_b_z"].detach().cpu().numpy().copy(),
            "dz_area": r["bl_dz_area"],
            "e_bl": float(r["e_bl"].detach()) if r.get("e_bl") is not None
            else 0.0})
    return r


PB.PB1DBackend.solve_graph = _wrap


def _evict():
    b = _bd.get("b")
    if b is not None:
        for at in ("_bl_ram", "_grids", "_solvers", "_c_units"):
            d = getattr(b, at, None)
            if isinstance(d, dict):
                d.clear()


def ebl_of(rho, phi, dz, area):
    return float((rho * (-phi)).sum() * dz * area)


def run(at, keep_sid):
    a = at.copy()
    if not keep_sid:
        a.info.pop("sample_id", None)
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1)))
    b = b.to(device)
    CAP["rows"] = []
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
    ebl_model = float(pred["baseline_coupling_energy_g"].sum()) \
        if pred.get("baseline_coupling_energy_g") is not None else 0.0
    rows = CAP["rows"]
    _evict()
    return dict(rows=rows, ebl_model=ebl_model,
                E=float(pred["energy"]))


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
            print(f"sid {sid} ({'charged' if q else 'neutral'}) [{tag}]")
            b0 = run(at, keep_sid)
            if not b0["rows"]:
                print("  no E_bl export captured -- bl_energy off on this path?")
                continue
            n_solves = len(b0["rows"])
            rec = sum(ebl_of(r["rho"], r["phi"], *r["dz_area"])
                      for r in b0["rows"])
            print(f"  CLOSURE A, E_bl recomputed from the exports vs the "
                  f"model's baseline_coupling_energy_g: {rec:+.9f} vs "
                  f"{b0['ebl_model']:+.9f}, |diff| "
                  f"{abs(rec - b0['ebl_model']):.3e} eV   "
                  f"({n_solves} solve(s) exported)")
            if abs(rec - b0["ebl_model"]) > 1e-9:
                print("  CLOSURE A FAILED -- the exports are not the term that "
                      "enters the energy; stopping this cell")
                continue
            rho0 = [r["rho"] for r in b0["rows"]]
            phi0 = [r["phi"] for r in b0["rows"]]
            da = [r["dz_area"] for r in b0["rows"]]
            fd_full = np.zeros((len(pick), 3))
            fd_rho = np.zeros((len(pick), 3))
            fd_phi = np.zeros((len(pick), 3))
            for ai, ia in enumerate(pick):
                for c in range(3):
                    pp = at.copy(); pp.positions[ia, c] += H
                    pm = at.copy(); pm.positions[ia, c] -= H
                    rp, rm = run(pp, keep_sid), run(pm, keep_sid)
                    if len(rp["rows"]) != n_solves or len(rm["rows"]) != n_solves:
                        print(f"    solve count changed under displacement "
                              f"({len(rp['rows'])}, {len(rm['rows'])} vs "
                              f"{n_solves}) -- skipping this component")
                        continue
                    ep = sum(ebl_of(r["rho"], r["phi"], *r["dz_area"])
                             for r in rp["rows"])
                    em = sum(ebl_of(r["rho"], r["phi"], *r["dz_area"])
                             for r in rm["rows"])
                    fd_full[ai, c] = -(ep - em) / (2 * H)
                    # rho displaced against the UNDISPLACED Phi_b
                    ep_r = sum(ebl_of(rp["rows"][j]["rho"], phi0[j], *da[j])
                               for j in range(n_solves))
                    em_r = sum(ebl_of(rm["rows"][j]["rho"], phi0[j], *da[j])
                               for j in range(n_solves))
                    fd_rho[ai, c] = -(ep_r - em_r) / (2 * H)
                    # Phi_b displaced against the UNDISPLACED rho
                    ep_p = sum(ebl_of(rho0[j], rp["rows"][j]["phi"], *da[j])
                               for j in range(n_solves))
                    em_p = sum(ebl_of(rho0[j], rm["rows"][j]["phi"], *da[j])
                               for j in range(n_solves))
                    fd_phi[ai, c] = -(ep_p - em_p) / (2 * H)
                print(f"    atom {ia} done", flush=True)
            resid = fd_rho + fd_phi - fd_full
            print(f"\n  CLOSURE B, FD_rho + FD_phi vs the full FD of E_bl "
                  f"(the remainder is the O(h^2) cross term a central "
                  f"difference leaves): max |resid| "
                  f"{1000 * float(np.abs(resid).max()):.3f} meV/A, i.e. "
                  f"{100 * float(np.abs(resid).max()) / max(float(np.abs(fd_full).max()), 1e-30):.2f}% "
                  f"of the largest full component")
            print(f"\n  {'piece':>34} {'z signed sum':>14} {'z rms':>10} "
                  f"{'z max':>10}")
            for nm, v in (("Phi_b * d(rho_solv)/dR", fd_rho),
                          ("rho_solv * d(Phi_b)/dR", fd_phi),
                          ("full dE_bl/dR", fd_full)):
                z = v[:, 2]
                print(f"  {nm:>34} {1000 * z.sum():+14.2f} "
                      f"{1000 * float(np.sqrt((z ** 2).mean())):10.2f} "
                      f"{1000 * float(np.abs(z).max()):10.2f}")
            fr = float(np.abs(fd_phi).max()) / max(float(np.abs(fd_full).max()), 1e-30)
            exp0 = keep_sid
            print(f"  d(Phi_b)/dR is {100 * fr:.2f}% of the full derivative "
                  f"here; this path's baseline is "
                  + ("the FROZEN per-sid table, so it was expected to be zero"
                     if exp0 else "the runtime table, so it was expected "
                     "non-zero")
                  + ("   <- as expected" if (fr < 1e-6) == exp0 else
                     "   <- NOT as expected, this is a finding"))
            np.savez(os.path.join(OUT, f"ebl_split_{sid}_{tag}.npz"),
                     fd_full=fd_full, fd_rho=fd_rho, fd_phi=fd_phi,
                     pick=np.array(pick))
print("\nDONE")
