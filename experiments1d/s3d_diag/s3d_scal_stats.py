"""Training-set statistics of the charge-state scalars for the 3-D solvent head (reviewer 2026-10-03 item 2: "新增输入按训练集
统计归一化"): per atom, total charge Q, the production 1-D potential at the atom phi_1D(z_a) (eV), and the model's per-atom
charge q_a; computed with the baseline model prod500_vsolv_fix over every training frame (one forward each; nothing re-solved).
Output: JSON with mean / std over all training atoms (and per-species means), to be put into the training config as
solvent3d_scal_mean / solvent3d_scal_std.
Usage: python s3d_scal_stats.py <out.json>   (cwd with ./data)
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ["MACE_PB1D_CACHE_READONLY"] = "1"
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")

import numpy as np
import torch
from ase.io import read

from mace import data as mace_data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, utils
from mace.modules.solvent3d import _interp1_periodic

OUT = sys.argv[1] if len(sys.argv) > 1 else "s3d_scal_stats.json"
MODEL = os.environ.get("KIT_MODEL", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_vsolv_fix/models/prod500_vsolv_fix.model")
torch.set_default_dtype(torch.float64)
dev = torch.device(os.environ.get("KIT_DEVICE", "cuda:0"))
model = torch.load(MODEL, map_location=dev).to(dev); model.eval()
for p in model.parameters():
    p.requires_grad_(False)
backend = model._get_pb1d_backend()
atoms_all = read("data/train.xyz", ":")
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin", "sample_id": "sample_id",
                                    "solvated": "solvated", "fermi_level": "Fermi", "potential": "potential_diff"}, arrays_keys={"forces": "forces"})
T0 = time.time()
rows = []; species = []; per_frame = []
for i, atoms in enumerate(atoms_all):
    cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(dev)
    hold = {}; orig = backend.solve_graph

    def spy(*a, **kw):
        res = orig(*a, **kw); hold["res"] = res; hold["kw"] = kw; return res
    backend.solve_graph = spy
    try:
        with torch.no_grad():
            out = model(batch.to_dict(), training=False, compute_force=False)
    finally:
        backend.solve_graph = orig
    if "res" not in hold:
        per_frame.append(dict(sid=int(atoms.info["sample_id"]), skipped=True)); continue
    kw, res = hold["kw"], hold["res"]
    cell = kw["cell"].detach().double().reshape(3, 3); pf = torch.remainder(kw["positions"].detach().double() @ torch.linalg.inv(cell), 1.0)
    phi_at = _interp1_periodic(res["phi_z"].detach().double(), pf[:, 2])
    q = float(kw["total_charge"]); qa = out["charges"].detach().double().reshape(-1)
    sc = torch.stack([torch.full_like(phi_at, q), phi_at, qa], dim=1).cpu().numpy()
    rows.append(sc); species.extend(atoms.get_chemical_symbols())
    per_frame.append(dict(sid=int(atoms.info["sample_id"]), q=q, n=len(atoms), phi_mean=float(phi_at.mean()), qa_mean=float(qa.mean()), qa_std=float(qa.std())))
    if i % 40 == 0:
        print(f"[{time.time() - T0:5.0f}s] frame {i}/{len(atoms_all)} sid {atoms.info['sample_id']} q {q:+.2f} phi_at mean {float(phi_at.mean()):+.3f} q_a mean {float(qa.mean()):+.4f}", flush=True)
A = np.concatenate(rows); species = np.asarray(species)
res = dict(n_frames=len(rows), n_atoms=int(A.shape[0]), names=["Q_total", "phi_1D_at_atom_eV", "q_atom"],
           mean=A.mean(0).tolist(), std=A.std(0).tolist(), min=A.min(0).tolist(), max=A.max(0).tolist(),
           per_species={s: dict(n=int((species == s).sum()), mean=A[species == s].mean(0).tolist(), std=A[species == s].std(0).tolist()) for s in sorted(set(species))},
           frames=per_frame, model=MODEL)
json.dump(res, open(OUT, "w"), indent=1)
print(f"[{time.time() - T0:5.0f}s] {res['n_frames']} frames, {res['n_atoms']} atoms: mean {res['mean']} std {res['std']} min {res['min']} max {res['max']}", flush=True)
print("done", flush=True)
