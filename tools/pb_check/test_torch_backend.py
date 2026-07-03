"""Independent tests for the device-resident PB backend (PBTorchBackend).

1. PARITY: torch backend vs numpy backend (PBPlanarSolvent) on identical
   inputs — real cal_1 geometry (positions/zvals from the config JSON),
   analytic Gaussian net density, same solvation block, same protocol
   (coarse-init cold start, fixsol_steps=2, tol 1e-3, rfft f64).
   Expect profile/moment agreement at machine precision.

2. WARM-START INVARIANCE: warm solve (cached phi as the Newton initial
   guess; the dipole fix-point loop runs the full cold protocol) on a
   slightly perturbed density vs a cold solve of the same density.
   Differences must be at the residual-tolerance scale, far below the
   grid-convergence error budget (0.017 V).

   Parity thresholds are tolerance-limited, not machine precision: the two
   backends order FFTs/reductions differently, so their Newton iterates
   stop at slightly different points inside the same residual tolerance.

3. TIMING: per-stage wall times, cold vs warm, on the target device.

Usage:
  PB_RFFT=1 PYTHONPATH=<cep-dip-repo> python test_torch_backend.py \
      --config cal1_train.json [--device cuda]
"""

import argparse
import os
import sys

os.environ.setdefault("PB_RFFT", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mace.modules.pb_solvent import PBPlanarSolvent, PBTorchBackend  # noqa: E402

TOTAL_CHARGE = -1.0
NET_SIGMA = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--spacing", type=float, default=0.15)
    args = ap.parse_args()

    import json
    cfg = json.load(open(args.config))
    cell = np.asarray(cfg["cell_A"], dtype=float)
    pos_frac = np.asarray(cfg["positions_direct"], dtype=float) % 1.0
    positions = pos_frac @ cell
    zv_map = {k: float(v) for k, v in cfg["zval"].items()}
    zvals = np.concatenate([
        [zv_map[el]] * c for el, c in zip(cfg["elements"], cfg["counts"])
    ])
    area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
    net_z0 = 0.42 * float(np.linalg.norm(cell[2]))

    def net_np(pts):
        zz = pts[:, 2]
        return (TOTAL_CHARGE / (area * np.sqrt(2 * np.pi) * NET_SIGMA)
                * np.exp(-0.5 * ((zz - net_z0) / NET_SIGMA) ** 2))

    def net_t(pts):
        zz = pts[:, 2]
        return (TOTAL_CHARGE / (area * np.sqrt(2 * np.pi) * NET_SIGMA)
                * torch.exp(-0.5 * ((zz - net_z0) / NET_SIGMA) ** 2))

    common = dict(config_path=args.config, grid_spacing=args.spacing,
                  fixsol_steps=2, tol=1e-3, coarse_init=True)
    np_backend = PBPlanarSolvent(**common)
    t_backend = PBTorchBackend(warm_start=True, **common)

    dev = torch.device(args.device)
    pos_t = torch.as_tensor(positions, device=dev, dtype=torch.float64)
    cell_t = torch.as_tensor(cell, device=dev, dtype=torch.float64)
    zv_t = torch.as_tensor(zvals, device=dev, dtype=torch.float64)

    print("=== 0. spectral vs pointwise net density (torch backend) ===")
    from mace.modules.loss import _gto_density_at_points_axis2_pbc
    torch.manual_seed(0)
    sigmas = [0.25, 0.5, 1.0]
    natoms = positions.shape[0]
    coeffs = 0.1 * torch.randn(natoms, len(sigmas), 9, device=dev, dtype=torch.float64)
    # build grid + spectral density via backend internals
    shape0 = t_backend._grid_shape(cell)
    grid0 = t_backend._grid_for(np.ascontiguousarray(cell), shape0, dev)
    pos_frac_t = torch.remainder(pos_t @ torch.linalg.inv(cell_t), 1.0)
    net_g = t_backend._gto_net_density_g(grid0, pos_frac_t, coeffs, sigmas)
    net_spec = grid0.ifft_real(net_g) / grid0.volume  # e/A^3
    # pointwise reference on the same grid points
    nx0, ny0, nz0 = shape0
    fx = torch.arange(nx0, device=dev, dtype=torch.float64) / nx0
    fy = torch.arange(ny0, device=dev, dtype=torch.float64) / ny0
    fz = torch.arange(nz0, device=dev, dtype=torch.float64) / nz0
    pts = torch.stack(torch.meshgrid(fx, fy, fz, indexing="ij"), dim=-1).reshape(-1, 3) @ cell_t
    numbers_dummy = torch.zeros(natoms, dtype=torch.long, device=dev)
    with torch.no_grad():
        ref_chunks = [
            _gto_density_at_points_axis2_pbc(c, coeffs, pos_t, numbers_dummy, cell_t, sigmas)
            for c in pts.split(262144)
        ]
    net_ref = torch.cat(ref_chunks).reshape(shape0)
    dnet = float(torch.max(torch.abs(net_spec - net_ref)) / torch.max(torch.abs(net_ref)))
    ok0 = dnet < 1e-8
    print(f"  max rel diff: {dnet:.2e}  {'OK' if ok0 else 'FAIL'}")

    print("=== 1. parity: torch vs numpy backend ===")
    r_np = np_backend.solve_rho_ion_z(
        positions=positions, cell=cell, z_valence=zvals,
        total_charge=TOTAL_CHARGE, neutral_sigma=0.5, eval_net_density=net_np)
    r_t = t_backend.solve_rho_ion_z(
        positions=pos_t, cell=cell_t, z_valence=zv_t,
        total_charge=TOTAL_CHARGE, neutral_sigma=0.5,
        eval_net_density=net_t, sample_id=None)
    t2n = lambda x: x.detach().cpu().numpy()

    def rel(a, b):
        return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))

    checks = {
        "rho_ion_z": rel(t2n(r_t["rho_ion_z"]), r_np["rho_ion_z"]),
        "rho_bound_z": rel(t2n(r_t["rho_bound_z"]), r_np["rho_bound_z"]),
        "q_ion": abs(r_t["q_ion"] - r_np["q_ion"]),
        "layer_mean": abs(r_t["layer_mean"] - r_np["layer_mean"]),
        "mu_bound": abs(r_t["mu_bound"] - r_np["mu_bound"]),
    }
    lim = {"rho_ion_z": 1e-5, "rho_bound_z": 1e-5, "q_ion": 1e-6,
           "layer_mean": 1e-4, "mu_bound": 1e-4}
    ok1 = all(v < lim[k] for k, v in checks.items())
    for k, v in checks.items():
        print(f"  {k:12s} {v:.2e} (limit {lim[k]:.0e})  {'OK' if v < lim[k] else 'FAIL'}")
    print(f"  numpy diag: {np_backend.last_diagnostics}")
    print(f"  torch diag: {t_backend.last_diagnostics}")

    print("\n=== 2. warm-start invariance (perturbed density) ===")
    # emulate epoch-to-epoch density drift: +2% amplitude
    def net_t2(pts):
        return 1.02 * net_t(pts)
    # seed the cache with the unperturbed solve
    _ = t_backend.solve_rho_ion_z(
        positions=pos_t, cell=cell_t, z_valence=zv_t,
        total_charge=TOTAL_CHARGE, neutral_sigma=0.5,
        eval_net_density=net_t, sample_id=7)
    # warm: cached phi + dipole state, 1 fixstep
    r_warm = t_backend.solve_rho_ion_z(
        positions=pos_t, cell=cell_t, z_valence=zv_t,
        total_charge=TOTAL_CHARGE, neutral_sigma=0.5,
        eval_net_density=net_t2, sample_id=7)
    d_warm = dict(t_backend.last_diagnostics)
    # cold reference of the same perturbed density
    t_backend._phi_cache.clear()
    r_cold = t_backend.solve_rho_ion_z(
        positions=pos_t, cell=cell_t, z_valence=zv_t,
        total_charge=TOTAL_CHARGE, neutral_sigma=0.5,
        eval_net_density=net_t2, sample_id=None)
    d_cold = dict(t_backend.last_diagnostics)
    dz = float(r_cold["height"]) / r_cold["rho_ion_z"].shape[0]
    prof_rms = float(torch.sqrt(torch.mean(
        (r_warm["rho_ion_z"] + r_warm["rho_bound_z"]
         - r_cold["rho_ion_z"] - r_cold["rho_bound_z"]) ** 2)))
    dmu = abs((r_warm["q_ion"] * r_warm["layer_mean"] + r_warm["mu_bound"])
              - (r_cold["q_ion"] * r_cold["layer_mean"] + r_cold["mu_bound"]))
    scale = 4.0 * np.pi * 27.211386245988 / 1.8897261258369282
    dpot = scale * dmu / area
    print(f"  combined-profile rms diff: {prof_rms:.3e} e/A^3")
    print(f"  solvent-potential-term diff: {dpot:.4f} V "
          f"({'OK' if dpot < 5e-3 else 'FAIL'} — budget is 0.017 V grid error)")
    print(f"  warm diag: {d_warm}")
    print(f"  cold diag: {d_cold}")
    ok2 = dpot < 5e-3

    print("\n=== 2b. fixsol convergence probe (informational) ===")
    probe = {}
    for nfs in (1, 2, 6):
        b = PBTorchBackend(config_path=args.config, grid_spacing=args.spacing,
                           fixsol_steps=nfs, tol=1e-3, coarse_init=True,
                           warm_start=False)
        r = b.solve_rho_ion_z(
            positions=pos_t, cell=cell_t, z_valence=zv_t,
            total_charge=TOTAL_CHARGE, neutral_sigma=0.5,
            eval_net_density=net_t, sample_id=None)
        probe[nfs] = r["q_ion"] * r["layer_mean"] + r["mu_bound"]
    for nfs in (1, 2):
        d = scale * abs(probe[nfs] - probe[6]) / area
        print(f"  fixsol={nfs} vs 6: solvent-term diff {d:.4f} V")

    print("\n=== 3. timing (device={}) ===".format(args.device))
    import time
    t_backend._phi_cache.clear()
    t0 = time.perf_counter()
    _ = t_backend.solve_rho_ion_z(
        positions=pos_t, cell=cell_t, z_valence=zv_t,
        total_charge=TOTAL_CHARGE, neutral_sigma=0.5,
        eval_net_density=net_t, sample_id=42)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    cold_s = time.perf_counter() - t0
    print(f"  cold (coarse-init + {common['fixsol_steps']} fixsteps): "
          f"{cold_s:.2f} s  stages={t_backend.last_diagnostics}")
    for i in range(3):
        t0 = time.perf_counter()
        _ = t_backend.solve_rho_ion_z(
            positions=pos_t, cell=cell_t, z_valence=zv_t,
            total_charge=TOTAL_CHARGE, neutral_sigma=0.5,
            eval_net_density=net_t, sample_id=42)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        print(f"  warm #{i+1}: {time.perf_counter()-t0:.2f} s  "
              f"stages={t_backend.last_diagnostics}")

    print("\nRESULT:", "ALL PASS" if (ok0 and ok1 and ok2) else "FAIL")
    sys.exit(0 if (ok0 and ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
