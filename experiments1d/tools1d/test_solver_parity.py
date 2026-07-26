"""Torch Solver1D vs numpy solve_frozen parity on cal_18 (both P_off choices),
plus a gradient smoke test (d loss/d p_off finite-difference check)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "polar-mace-planar"))
from mace.modules.pb1d_solver import Solver1D  # noqa: E402

pack = np.load(HERE.parent / "exp_prior_profiles" / "solver_parity_pack.npz")
dev = torch.device("cpu")
nz = int(pack["nz"])
cell = torch.tensor(pack["cell"], dtype=torch.float64)
params = {k: float(pack[k]) for k in ("ZBETA", "theta_b", "n_max", "invBETA", "R_B", "A_K")}
params["LNLION"] = bool(int(pack["LNLION"]))
params["LION"] = bool(int(pack["LION"]))

solver = Solver1D(cell, nz, params, device=dev)
t = lambda k: torch.tensor(pack[k], dtype=torch.float64, device=dev)

def run(poff_key, **kw):
    args = dict(
        cvhar_z=t("cvhar_z"), s_ion=t("s_ion_z"), a1=t("A_scr"), p_off=t(poff_key),
        q_sol=float(pack["q_sol"]),
        val_ion_dipole_z=torch.tensor(float(pack["val_ion_dipole_z"]), dtype=torch.float64),
        c_unit=float(pack["c_unit"]), center_z=float(pack["center_z"]),
        indmin=int(pack["indmin"]), fixsol_steps=5, tol=1.0e-3, max_outer=12,
        fixsol_converge=False,  # numpy refs use the legacy fixed 5-step loop
    )
    args.update(kw)
    return solver.solve(**args)

def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))

ok = True
for name, key, rp, rb, ri in (
    ("exact", "p_exact", "ref_phi_exact", "ref_nb_exact", "ref_nion_exact"),
    ("prior", "p_prior", "ref_phi_prior", "ref_nb_prior", "ref_nion_prior"),
):
    out = run(key)
    phi = out["phi"].numpy()
    nb = out["n_b"].numpy() / float(pack["volume"])
    ni = out["n_ion"].numpy() / float(pack["volume"])
    dphi = rms(phi - pack[rp]); dnb = rms(nb - pack[rb]); dni = rms(ni - pack[ri])
    # scale references for relative context
    print(f"{name}: dphi {dphi:.2e} (ref rms {rms(pack[rp]):.2e})  "
          f"dnb {dnb:.2e} ({rms(pack[rb]):.2e})  dni {dni:.2e} ({rms(pack[ri]):.2e})  "
          f"outer {int(out['n_outer'])}")
    # accuracy also vs the DFT reference potential (the end metric)
    print(f"       phi vs DFT: torch {rms(phi - pack['phi_ref_z']):.4e} | numpy {rms(pack[rp] - pack['phi_ref_z']):.4e}")
    # Pass semantics: both solvers stop at preconditioned-residual rms < 1e-3,
    # so their solutions may differ by up to ~tol in phi. Verified separately:
    # numpy's phi has torch-residual rms 1.04e-3 (its own tol), torch's 1.8e-9
    # (dense Newton converges deeper) -> same equations, tolerance-level parity.
    if dphi > 1.5e-3 or dnb > 2e-5 or dni > 1e-6:
        ok = False

# ---- gradient checks vs central FD (converged fixsol loop) -----------------
# unrolled graph (grad_passes=0) and analytic coupled adjoint (grad_passes=1),
# directional probes on every input channel (single tiny components are FD-
# noise-limited; directions give a clean O(eps^2) signal — see a1 study).
slab = (solver.z > 2) & (solver.z < 8)


def loss_of(out):
    return (out["phi"][slab].mean()
            + 0.1 * out["n_b"].pow(2).sum() + 0.1 * out["n_ion"].pow(2).sum())


rng2 = np.random.default_rng(1)
base_keys = {"cvhar_z": "cvhar_z", "s_ion": "s_ion_z", "a1": "A_scr", "p_off": "p_prior"}
for gp, label in ((0, "unrolled"), (1, "analytic")):
    for arg, packkey in base_keys.items():
        u = torch.tensor(rng2.standard_normal(nz))
        u = u / u.norm()
        v = t(packkey).clone().requires_grad_(True)
        loss_of(run("p_prior", fixsol_converge=True, grad_passes=gp,
                    **{arg: v} if arg != "p_off" else {"p_off": v})).backward()
        g_auto = float((v.grad * u).sum())
        eps = 1e-5
        with torch.no_grad():
            lp = float(loss_of(run("p_prior", fixsol_converge=True, grad_passes=0,
                                   **{arg: t(packkey) + eps * u})))
            lm = float(loss_of(run("p_prior", fixsol_converge=True, grad_passes=0,
                                   **{arg: t(packkey) - eps * u})))
        g_fd = (lp - lm) / (2 * eps)
        ratio = g_auto / g_fd
        good = abs(ratio - 1.0) < 1e-3
        ok = ok and good
        print(f"grad[{label}] d/d{arg}: autograd {g_auto:+.6e}  FD {g_fd:+.6e}  "
              f"ratio {ratio:.6f}  {'OK' if good else 'BAD'}")
print("PARITY:", "PASS" if ok else "FAIL")
