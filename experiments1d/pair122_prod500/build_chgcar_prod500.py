"""Fully-ML CHGCAR for one frame of pair_122.xyz with the prod500_w1000_ref model (recipe of
exp_band/sid122_bands/build_chgcar_122.py, hard clamp):
    ml_electron = (dft_electron + dft_net) - ml_net     # ion baseline restored from the DFT identity
    ml_electron = max(ml_electron, 0); renormalised to NELECT; augmentation occupancies from the model head.
Usage: python build_chgcar_prod500.py <frame_idx 0|1> <template CHGCAR (DFT of that state)> <net grid npy> <out_dir>
Run from exp_pair122_prod500 (./data -> training data, ./config_pb1d.yaml).
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import ase.io

sys.path.insert(0, "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/exp_band")
from chgcar_utils import read_chgcar, parse_aug_tail, write_chgcar_grid_and_aug
from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
from mace.modules.loss import _gto_density_at_points_axis2_pbc, _parse_density_sigmas

os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")
IDX, TEMPLATE, NPY, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
MODEL = os.environ.get("KIT_MODEL", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_w1000_ref/models/prod500_w1000_ref.model")
XYZ = os.environ.get("KIT_XYZ", "pair_122.xyz")
device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
model = torch.load(MODEL, map_location=device).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
cfg = yaml.safe_load(open("config_pb1d.yaml"))
sigmas = _parse_density_sigmas(cfg["density_3d_sigma"])
atoms = ase.io.read(XYZ, IDX)
sid = int(atoms.info["sample_id"])
chg = read_chgcar(TEMPLATE)
dims = chg["dims"]; vol = abs(float(np.linalg.det(chg["cell"]))); ngrid = int(np.prod(dims))
dft_electron = chg["data_zyx"].astype(np.float64) / vol
target = np.load(NPY).astype(np.float64)
assert target.shape == dft_electron.shape, (target.shape, dft_electron.shape)
assert np.allclose(np.asarray(chg["cell"]), atoms.cell[:], atol=1e-4), "template cell != frame cell"

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin",
               "sample_id": "sample_id", "solvated": "solvated", "fermi_level": "Fermi", "potential": "potential_diff"},
    arrays_keys={"forces": "forces"})
cfgm = mace_data.config_from_atoms(atoms, key_specification=kspec)
ds = [mace_data.AtomicData.from_config(cfgm, z_table=z_table, cutoff=float(model.r_max))]
b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
with torch.no_grad():
    pred = model(b.to_dict(), compute_force=False, training=False)
coeff = pred["charge_density_radial_coefficients"].detach()
occ_pred = pred["aug_occupancies"].detach().cpu().numpy()

positions = b.positions.detach(); zn = b.atomic_numbers.detach().to(dtype=torch.long)
cell = torch.as_tensor(chg["cell"], dtype=torch.float64, device=device)
nx, ny, nz = dims; total = nx * ny * nz
res = np.empty(total, dtype=np.float64)
CH = 65536
for s in range(0, total, CH):
    e = min(s + CH, total)
    idx = torch.arange(s, e, device=device, dtype=torch.long)
    frac = torch.stack([(idx % nx).to(torch.float64) / nx, ((idx // nx) % ny).to(torch.float64) / ny,
                        (idx // (nx * ny)).to(torch.float64) / nz], dim=1)
    with torch.no_grad():
        vals = _gto_density_at_points_axis2_pbc(points=frac @ cell, density_coefficients=coeff,
                                                positions=positions, atomic_numbers=zn, cell=cell, sigma=sigmas)
    res[s:e] = vals.detach().cpu().numpy()
ml_net = res.reshape(nz, ny, nx)
ml_electron = (dft_electron + target) - ml_net
n_clamped = int((ml_electron < 0).sum())
ml_electron = np.maximum(ml_electron, 0.0)
scale = dft_electron.sum() / ml_electron.sum()
ml_electron *= scale
ref_vecs = parse_aug_tail(chg["tail_lines"])
ml_vecs = [occ_pred[i, :len(v)].astype(np.float64) for i, v in enumerate(ref_vecs)]
OUT.mkdir(parents=True, exist_ok=True)
write_chgcar_grid_and_aug(TEMPLATE, OUT / "CHGCAR", ml_electron * vol, aug_vecs=ml_vecs)
diff = ml_net - target
meta = np.load(NPY.replace(".npy", "_meta.npz"))
viz = meta["valid_iz"].astype(np.int64)
np.savez(OUT / "build_summary.npz", sid=sid, nelect=dft_electron.sum() * vol / ngrid, scale=scale, n_clamped=n_clamped,
         res_rmse_win=float(np.sqrt((diff[viz] ** 2).mean())), occ_rmse=float(np.sqrt(((np.concatenate(ml_vecs) - np.concatenate(ref_vecs)) ** 2).mean())),
         ml_net_plane=ml_net.mean(axis=(1, 2)), dft_net_plane=target.mean(axis=(1, 2)), dft_electron_plane=dft_electron.mean(axis=(1, 2)),
         ml_electron_plane=ml_electron.mean(axis=(1, 2)))
print(f"sid {sid}: NELECT {dft_electron.sum() * vol / ngrid:.6f}, renormalisation x{scale:.6f}, clamped points {n_clamped} of {ngrid} "
      f"({100.0 * n_clamped / ngrid:.3f}%), net-density rmse on the valid window {np.sqrt((diff[viz] ** 2).mean()):.4e} e/A^3, "
      f"aug-occupancy rmse {np.sqrt(((np.concatenate(ml_vecs) - np.concatenate(ref_vecs)) ** 2).mean()):.4e} -> {OUT / 'CHGCAR'}", flush=True)
print("all_done", flush=True)
