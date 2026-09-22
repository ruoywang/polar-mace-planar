"""Model vs DFT Fermi level (and electrode potential) for the 200 NiN44 charged/neutral twin pairs
of the mix800 set (charged sids 1-200, solvated neutral twins 601-800; frames from train+val+test),
with the prod500_w1000_ref model. One forward per frame (direct model call, PB solve included).
Usage: python fermi_pairs_prod500.py <out npz>   (run from exp_pair122_prod500: ./data, ./config_pb1d.yaml)
"""
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
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.pb1d_backend as PB

OUT = sys.argv[1] if len(sys.argv) > 1 else "fermi_pairs_prod500.npz"
MODEL = os.environ.get("KIT_MODEL", "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/prod500_w1000_ref/models/prod500_w1000_ref.model")
device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
model = torch.load(MODEL, map_location=device).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge", "total_spin": "total_spin",
               "sample_id": "sample_id", "solvated": "solvated", "fermi_level": "Fermi", "potential": "potential_diff"},
    arrays_keys={"forces": "forces"})
frames = []
for f in ("train.xyz", "val.xyz", "test.xyz"):
    frames += read(f"data/{f}", ":")
by_sid = {int(a.info["sample_id"]): a for a in frames}
split_of = {}
for f in ("train", "val", "test"):
    for a in read(f"data/{f}.xyz", ":"):
        split_of[int(a.info["sample_id"])] = f

_bd = {}
_bk = PB.PB1DBackend.solve_graph
def _wrap(self, *a, **k): _bd["b"] = self; return _bk(self, *a, **k)
PB.PB1DBackend.solve_graph = _wrap
def _evict():
    b = _bd.get("b")
    if b is not None:
        for at in ("_bl_ram", "_grids", "_solvers", "_c_units"):
            d = getattr(b, at, None)
            if isinstance(d, dict): d.clear()
    torch.cuda.empty_cache()


def predict(a):
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    with torch.no_grad():
        out = model(b.to_dict(), training=False, compute_force=False)
    fv = out.get("fermi_level_pred")
    if fv is None:
        fv = out["fermi_level"]
    r = (float(fv.detach().cpu().reshape(-1)[0]), float(out["potential"].detach().cpu().reshape(-1)[0]))
    del out, b; _evict()
    return r


rows = []; t0 = time.time()
for c in range(1, 201):
    n = c + 600
    if c not in by_sid or n not in by_sid:
        continue
    ac, an = by_sid[c], by_sid[n]
    fc, pc = predict(ac); fn, pn = predict(an)
    rows.append((c, float(ac.info["Fermi"]), fc, float(an.info["Fermi"]), fn, float(ac.info["potential_diff"]), pc, float(an.info["potential_diff"]), pn,
                 {"train": 0, "val": 1, "test": 2}[split_of[c]]))
    if c % 25 == 0:
        print(f"{c}/200  {time.time() - t0:.0f} s", flush=True)
arr = np.array(rows)
np.savez(OUT, sid=arr[:, 0], dft_c=arr[:, 1], ml_c=arr[:, 2], dft_n=arr[:, 3], ml_n=arr[:, 4],
         pot_dft_c=arr[:, 5], pot_ml_c=arr[:, 6], pot_dft_n=arr[:, 7], pot_ml_n=arr[:, 8], split=arr[:, 9], model=os.path.basename(MODEL))
rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
print(f"{len(rows)} pairs. RMSE fermi charged {rms(arr[:, 2] - arr[:, 1]):.4f} eV | neutral {rms(arr[:, 4] - arr[:, 3]):.4f} eV; "
      f"RMSE potential charged {rms(arr[:, 6] - arr[:, 5]):.4f} | neutral {rms(arr[:, 8] - arr[:, 7]):.4f} eV")
print(f"shift charged-neutral (Fermi): DFT mean {np.mean(arr[:, 1] - arr[:, 3]):+.3f} | model mean {np.mean(arr[:, 2] - arr[:, 4]):+.3f} eV; "
      f"rmse of the pair shift {rms((arr[:, 2] - arr[:, 4]) - (arr[:, 1] - arr[:, 3])):.4f} eV")
for s, nm in ((0, "train"), (1, "val"), (2, "test")):
    m = arr[:, 9] == s
    if m.any():
        print(f"  {nm:5s} {int(m.sum()):3d} pairs: fermi rmse charged {rms(arr[m, 2] - arr[m, 1]):.4f} neutral {rms(arr[m, 4] - arr[m, 3]):.4f} eV")
