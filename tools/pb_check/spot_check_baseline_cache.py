"""Spot-check the PB baseline cache end-to-end.

For selected sample_ids, run the torch backend through the CACHE route with
the model net density replaced by the DATA net grid (restricted to the PB
grid) — i.e. emulate a perfectly trained model. By linearity,
neutral_cache - net_data == restrict(CHGCAR), so the PB inputs equal the
direct-from-CHGCAR reference pipeline (grid_convergence.py) exactly up to
the float32 rounding of the cached fields. For cal_1 (sample_id 1) the
solvent-potential terms must reproduce the known grid-convergence row
(spacing arg 0.15): pot_ion +21.729, pot_bound -5.778, combined +15.951 V.

Usage:
  PB_RFFT=1 PYTHONPATH=<cep-dip> python spot_check_baseline_cache.py \
      --cache <baseline_cache_dir> --net-dir <grid_cache_dir> \
      --config <cal1_train.json> --xyz <train.xyz> [more xyz ...] \
      --sids 1 150 250 --device cuda
"""

import argparse
import json
import os
import sys

os.environ.setdefault("PB_RFFT", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mace.modules.pb_solvent import PBTorchBackend  # noqa: E402


def _fourier_restrict(values, shape_c):
    spec_f = np.fft.fftn(values) / values.size
    spec_c = np.zeros(shape_c, dtype=complex)
    sl_f, sl_c = [], []
    for nf, nc in zip(values.shape, shape_c):
        h = nc // 2
        sl_f.append((slice(0, h), slice(nf - (h - 1), nf)))
        sl_c.append((slice(0, h), slice(nc - (h - 1), nc)))
    for cf, cc in zip(sl_f[0], sl_c[0]):
        for kf, kc in zip(sl_f[1], sl_c[1]):
            for lf, lc in zip(sl_f[2], sl_c[2]):
                spec_c[cc, kc, lc] = spec_f[cf, kf, lf]
    return np.ascontiguousarray(np.fft.ifftn(spec_c * np.prod(shape_c)).real)

SCALE = 4.0 * np.pi * 27.211386245988 / 1.8897261258369282


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--net-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--xyz", nargs="+", required=True)
    ap.add_argument("--sids", type=int, nargs="+", default=[1, 150, 250])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--spacing", type=float, default=0.15)
    args = ap.parse_args()

    from ase.io import read as ase_read
    structures = {}
    for f in args.xyz:
        for at in ase_read(f, index=":"):
            structures[int(at.info["sample_id"])] = at

    zval_map = {"H": 1.0, "C": 4.0, "N": 5.0, "O": 6.0, "Ni": 10.0}
    backend = PBTorchBackend(
        config_path=args.config, grid_spacing=args.spacing,
        fixsol_steps=2, tol=1e-3, coarse_init=True,
        warm_start=False, baseline_cache=args.cache,
    )
    dev = torch.device(args.device)

    print(f"{'sid':>5} {'route':>9} {'q_ion':>9} {'pot_ion':>9} {'pot_b':>9} {'pot_tot':>9} {'rms':>9}")
    ok = True
    for sid in args.sids:
        at = structures[sid]
        cell = at.cell.array
        area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
        pos_t = torch.as_tensor(at.get_positions(), device=dev, dtype=torch.float64)
        cell_t = torch.as_tensor(cell, device=dev, dtype=torch.float64)
        zv_t = torch.as_tensor(
            [zval_map[s] for s in at.get_chemical_symbols()],
            device=dev, dtype=torch.float64,
        )
        tc = float(at.info["total_charge"])

        # data net grid, restricted to the PB grid, as the "model" density
        net = np.load(os.path.join(args.net_dir, f"density3d_net_grid_{sid}.npz"))["rho"]
        net_xyz = np.ascontiguousarray(net.astype(np.float64).transpose(2, 1, 0))
        shape = backend._grid_shape(cell)
        net_pb = _fourier_restrict(net_xyz, shape)  # e/A^3, physics sign
        net_flat = torch.as_tensor(net_pb.reshape(-1), device=dev, dtype=torch.float64)

        def eval_net(pts, _n=net_flat):
            return _n  # grid._cart_pts ordering == meshgrid ij == reshape order

        results = {}
        for route in ("cache", "gaussian"):
            if route == "gaussian":
                # disable cache to compare against the surrogate route
                idx = backend._bl_index
                backend._bl_index = {}
            r = backend.solve_rho_ion_z(
                positions=pos_t, cell=cell_t, z_valence=zv_t,
                total_charge=tc, neutral_sigma=0.5,
                eval_net_density=eval_net, sample_id=sid,
            )
            if route == "gaussian":
                backend._bl_index = idx
            mu_i = r["q_ion"] * r["layer_mean"]
            pot_i = SCALE * mu_i / area
            pot_b = SCALE * r["mu_bound"] / area
            results[route] = (pot_i, pot_b, pot_i + pot_b)
            print(f"{sid:>5} {route:>9} {r['q_ion']:>+9.4f} {pot_i:>+9.3f} "
                  f"{pot_b:>+9.3f} {pot_i+pot_b:>+9.3f} "
                  f"{backend.last_diagnostics['rms_last']:>9.1e}")
        if sid == 1:
            ref = (21.729, -5.778, 15.951)  # grid-convergence 0.15 row (cal_1)
            d = max(abs(results["cache"][i] - ref[i]) for i in range(3))
            print(f"      cal_1 anchor: max |diff| vs grid-convergence row = {d:.4f} V "
                  f"{'OK' if d < 0.02 else 'FAIL'}")
            ok = ok and d < 0.02

    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
