"""Standalone smoke test of PBPlanarSolvent on a synthetic slab.

Builds a NiN-like hexagonal cell with a fake slab+water atom stack, a
synthetic net charge density (Gaussian sheet integrating to total_charge),
and runs the full nonlinear PB solve with the cal18 solvation block.

Checks:
 1. q_ion ~= -total_charge (physics sign of the returned profile)
 2. rho_ion(z) is localized in the ion-accessible region above the water
 3. profile resampling helpers conserve charge
Prints stage timings for the training-cost estimate.

Usage:
  PYTHONPATH=<cep-dip-repo> python test_pb_solvent_synthetic.py <solvation.json>
"""

import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])

from mace.modules.pb_solvent import (  # noqa: E402
    PBPlanarSolvent,
    resample_profile_periodic,
)

TOTAL_CHARGE = -1.0
SIGMA_NET = 2.0
Z_NET = 15.0


def main() -> None:
    config_path = sys.argv[1]
    cell = np.array(
        [[14.802, 0.0, 0.0], [-7.401, 12.818908, 0.0], [0.0, 0.0, 45.0]]
    )
    area = float(np.linalg.norm(np.cross(cell[0], cell[1])))

    rng = np.random.default_rng(20260703)
    n_atoms = 60
    frac_xy = rng.random((n_atoms, 2))
    z = rng.uniform(9.0, 19.0, n_atoms)
    positions = frac_xy @ cell[:2, :2]
    positions = np.column_stack([positions, z])
    z_valence = np.full(n_atoms, 6.0)

    def eval_net_density(points: np.ndarray) -> np.ndarray:
        zz = points[:, 2]
        return (
            TOTAL_CHARGE
            / (area * np.sqrt(2.0 * np.pi) * SIGMA_NET)
            * np.exp(-0.5 * ((zz - Z_NET) / SIGMA_NET) ** 2)
        )

    solver = PBPlanarSolvent(
        config_path=config_path,
        grid_spacing=0.25,
        fixsol_steps=2,
        tol=1.0e-3,
    )
    t0 = time.perf_counter()
    result = solver.solve_rho_ion_z(
        positions=positions,
        cell=cell,
        z_valence=z_valence,
        total_charge=TOTAL_CHARGE,
        neutral_sigma=0.5,
        eval_net_density=eval_net_density,
    )
    t1 = time.perf_counter()

    zg = result["z"]
    rho = result["rho_ion_z"]
    print(f"wall time: {t1 - t0:.1f} s")
    print("diagnostics:", json.dumps(solver.last_diagnostics, indent=1))
    print(f"q_ion = {result['q_ion']:+.6f}  (expect ~ {-TOTAL_CHARGE:+.3f})")
    print(f"layer_mean = {result['layer_mean']:.3f} A")
    imax = int(np.argmax(np.abs(rho)))
    print(f"|rho_ion| peak at z = {zg[imax]:.2f} A, value {rho[imax]:+.3e} e/A^3")

    # charge above/below the water edge
    dz = zg[1] - zg[0]
    q_above = rho[zg > 20.0].sum() * dz * area
    print(f"charge at z > 20 A: {q_above:+.4f} e (should carry most of q_ion)")

    # resampling conserves charge
    for n, half in ((1024, False), (512, True)):
        j = np.arange(n) + (0.5 if half else 0.0)
        z_dst = j * 45.0 / n
        r2 = resample_profile_periodic(zg, rho, z_dst, 45.0)
        q2 = r2.sum() * (45.0 / n) * area
        print(f"resampled n={n} offset_half={half}: q = {q2:+.6f}")

    ok = (
        abs(result["q_ion"] + TOTAL_CHARGE) < 0.05
        and result["layer_mean"] > Z_NET
    )
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
