"""Torch closure_from_fields vs numpy references on cal_18 (needs vmem for
one 3-D grid in torch — run where >8GB vmem is available, or CPU float64 on
a compute node; falls back gracefully on login if it fits)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "polar-mace-planar"))
sys.path.insert(0, "/work/08384/tg876840/ls6/repos/cep-dip-python-pb")

from pure_python import torch_pb as tp  # noqa: E402
from pure_python.pb import derived_params  # noqa: E402
from pure_python.config import load_config  # noqa: E402
from mace.modules.pb1d_closure import closure_from_fields, solute_dipole_z  # noqa: E402

pack = np.load(HERE.parent / "exp_prior_profiles" / "closure_parity_pack.npz")
cfg = load_config("/work/08384/tg876840/ls6/repos/cep-dip-python-pb/pure_python/configs/cal18.json")
params = derived_params(cfg["solvation"])

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cell = np.asarray(pack["cell"], dtype=float)
n_e = torch.tensor(pack["n_e_density"], dtype=torch.float64, device=dev)
cvhar = torch.tensor(pack["cvhar3"], dtype=torch.float64, device=dev)
grid = tp.TorchGrid(cell, tuple(n_e.shape), device=dev, dtype=torch.float64, rspec=True)

out = closure_from_fields(torch.clamp(n_e, min=0.0), cvhar, grid, params, tp)

def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))

ok = True
w_ref = np.clip(pack["var_chi"], 0, None); w_ref = w_ref / w_ref.max()
u_ref = np.cumsum(w_ref); u_ref = u_ref / u_ref[-1]
for name, key, ref in (
    ("A_scr", "A_scr", pack["A_scr"]),
    ("S_ion_z", "S_ion_z", pack["S_ion_z"]),
    ("prior", "prior", pack["prior"]),
    ("w_env", "w_env", w_ref),
    ("u", "u", u_ref),
):
    mine = out[key].detach().cpu().numpy()
    d = rms(mine - ref)
    s = max(rms(ref), 1e-300)
    print(f"{name:8s}: rms diff {d:.3e}  (signal {s:.3e}, rel {d/s:.2e})")
    if d / s > 1e-9:
        ok = False

# solute dipole port
zv = torch.tensor(pack["zv_atoms"], dtype=torch.float64, device=dev)
pos = torch.tensor(pack["positions_direct"], dtype=torch.float64, device=dev)
ne_vals_z = (n_e * grid.volume).mean(dim=(0, 1))
dip = solute_dipole_z(ne_vals_z, pos, zv, torch.tensor(cell, dtype=torch.float64, device=dev))
print(f"dipole_z: torch {float(dip):.10e}  numpy {float(pack['val_ion_dipole_z']):.10e}  "
      f"diff {abs(float(dip) - float(pack['val_ion_dipole_z'])):.2e}")
ok = ok and abs(float(dip) - float(pack["val_ion_dipole_z"])) < 1e-9

print("CLOSURE PARITY:", "PASS" if ok else "FAIL")
