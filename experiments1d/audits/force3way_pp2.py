"""Step 1 of the 1-D response experiment (user plan 2026-09-07): on frames
with DFT force labels, compare THREE forces in deployment (runtime) mode:
  F_auto = the currently deployed autograd force (lagged solvent),
  F_FD   = central finite difference of the model energy (the force the
           model WOULD have if every derivative channel were connected),
  F_DFT  = the DFT labels.
Question answered: does closing the energy-derivative gap also move the
force TOWARD DFT, or has the trained model already absorbed compensation
(so F_auto is closer to DFT than F_FD)? Both stage-2 flag settings, all
atoms, all 3 components. Vectors saved as npz for later analysis.
cwd = gate dir. Usage: force3way_pp2.py <model> <outdir>
"""
# Paths come from env vars with the LS6 values as defaults, so this
# runs unchanged on Lonestar6 and on a machine that holds the same
# payload elsewhere: KIT_PB_REPO, KIT_MACE_REPO, KIT_DFT,
# KIT_PB_CONFIG. Previously hardcoded, which made every audit
# script non-portable (found by the workstation session 2026-09-09).
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ.get("KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get("KIT_MACE_REPO", "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils

MODEL = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/s3d_gate_pp22_run-123.model"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
H = 0.01

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
model = torch.load(f=MODEL, map_location=device).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff",
               "solvated": "solvated"},
    arrays_keys={"forces": "forces"},
)
want = {"NiN44neusol": None, "NiN44": None}
for a in read("data/train.xyz", ":"):
    ct = a.info.get("config_type")
    if ct in want and want[ct] is None:
        want[ct] = a
    if all(v is not None for v in want.values()):
        break


def run(atoms, force=False):
    cfg = mace_data.config_from_atoms(atoms, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    if force:
        pred = model(b.to_dict(), compute_force=True, compute_stress=False,
                     training=False)
        return (float(pred["energy"].detach().cpu()),
                pred["forces"].detach().cpu().numpy())
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, compute_stress=False,
                     training=False)
    return float(pred["energy"].detach().cpu()), None


def rmse(a, b):
    return 1000.0 * float(np.sqrt(((a - b) ** 2).mean()))


for ct in ("NiN44", "NiN44neusol"):
    src = want[ct]
    f_dft = np.array(src.arrays["forces"], dtype=float)
    frame = src.copy()
    if "sample_id" in frame.info:
        del frame.info["sample_id"]  # runtime (MD) mode
    nat = len(frame)
    syms = np.array(frame.get_chemical_symbols())
    for flags in (False, True):
        model.solvent3d_energy = flags
        model.solvent_cavity_energy = flags
        t0 = time.time()
        e0, f_auto = run(frame, force=True)
        f_fd = np.zeros((nat, 3))
        for ia in range(nat):
            for c in range(3):
                pp = frame.copy(); pp.positions[ia, c] += H
                pm = frame.copy(); pm.positions[ia, c] -= H
                epv, _ = run(pp)
                emv, _ = run(pm)
                f_fd[ia, c] = -(epv - emv) / (2 * H)
            if ia % 50 == 0:
                print(f"   ... {ct} flags={int(flags)} atom {ia}/{nat}",
                      flush=True)
        np.savez(os.path.join(OUT, f"force3way_{ct}_flags{int(flags)}.npz"),
                 f_auto=f_auto, f_fd=f_fd, f_dft=f_dft, syms=syms, e0=e0)
        print(f"\n[{ct} flags={int(flags)}]  E={e0:+.4f}  "
              f"({(time.time()-t0)/60:.0f} min)", flush=True)
        print(f"  RMSE(F_auto - F_DFT) = {rmse(f_auto, f_dft):7.1f} meV/A")
        print(f"  RMSE(F_FD   - F_DFT) = {rmse(f_fd, f_dft):7.1f} meV/A")
        print(f"  RMSE(F_auto - F_FD ) = {rmse(f_auto, f_fd):7.1f} meV/A")
        for s in sorted(set(syms)):
            m = syms == s
            print(f"    {s:>2}: auto-DFT {rmse(f_auto[m], f_dft[m]):7.1f}   "
                  f"FD-DFT {rmse(f_fd[m], f_dft[m]):7.1f}   "
                  f"auto-FD {rmse(f_auto[m], f_fd[m]):7.1f}  (n={int(m.sum())})",
                  flush=True)
        z = 2
        print(f"    z-comp only: auto-DFT {rmse(f_auto[:, z], f_dft[:, z]):7.1f}   "
              f"FD-DFT {rmse(f_fd[:, z], f_dft[:, z]):7.1f}   "
              f"auto-FD {rmse(f_auto[:, z], f_fd[:, z]):7.1f}", flush=True)
print("DONE")
