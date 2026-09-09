"""Deployment-mode (runtime-baseline, MD path) force error DISTRIBUTION on
the pp2 gate model — the follow-up to the 3-component check (job 3420723),
which was not an upper bound. Central FD (h=0.01 A) of the total energy vs
autograd force:
  frame 1 (neusol): ALL atoms x ALL 3 components
  frame 2 (charged): ALL atoms x z component
Reported: per-element and overall |gap| stats (mean / median / p90 / max),
alongside the force scale. The gap measured here IS the omitted lagged-
solvent response in deployment mode. cwd = gate dir. Usage: force_sweep_pp2.py <model>
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/work/08384/tg876840/ls6/repos/cep-dip-python-pb")
sys.path.insert(0, "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy")

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils

MODEL = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/s3d_gate_pp22_run-123.model"
FLAGS = bool(int(sys.argv[2])) if len(sys.argv) > 2 else True
H = 0.01

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
model = torch.load(f=MODEL, map_location=device).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
model.solvent3d_energy = FLAGS
model.solvent_cavity_energy = FLAGS
print(f"stage-2 energy flags: {FLAGS}")

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


for ct, comps in (("NiN44neusol", (0, 1, 2)), ("NiN44", (2,))):
    frame = want[ct].copy()
    if "sample_id" in frame.info:
        del frame.info["sample_id"]  # runtime-baseline (MD) mode
    nat = len(frame)
    syms = frame.get_chemical_symbols()
    t0 = time.time()
    e0, f0 = run(frame, force=True)
    print(f"\n== {ct}: nat={nat} comps={comps}  E={e0:+.4f}  "
          f"|F| rms {1000*np.sqrt((f0**2).mean()):.1f} meV/A  "
          f"(autograd pass {time.time()-t0:.1f}s)", flush=True)
    t0 = time.time()
    ep_probe, _ = run(frame)
    per_eval = time.time() - t0
    total = nat * len(comps) * 2
    print(f"   per-eval {per_eval:.2f}s -> projected sweep "
          f"{total * per_eval / 60:.0f} min", flush=True)
    gaps, labels = [], []
    for ia in range(nat):
        for c in comps:
            pp = frame.copy(); pp.positions[ia, c] += H
            pm = frame.copy(); pm.positions[ia, c] -= H
            epv, _ = run(pp)
            emv, _ = run(pm)
            fd = -(epv - emv) / (2 * H)
            gaps.append(f0[ia, c] - fd)
            labels.append((ia, syms[ia], c))
        if ia % 25 == 0:
            g = np.abs(np.array(gaps)) * 1000.0
            print(f"   ... atom {ia}/{nat}  running max {g.max():.1f} meV/A",
                  flush=True)
    g = np.abs(np.array(gaps)) * 1000.0
    print(f"[SWEEP {ct}] n={g.size}  mean {g.mean():.1f}  median "
          f"{np.median(g):.1f}  p90 {np.percentile(g, 90):.1f}  p99 "
          f"{np.percentile(g, 99):.1f}  max {g.max():.1f} meV/A", flush=True)
    order = np.argsort(g)[::-1][:8]
    for k in order:
        ia, s, c = labels[k]
        print(f"   worst: atom {ia:3d} {s:>2} comp {'xyz'[c]}  "
              f"|gap| {g[k]:6.1f} meV/A  F={1000*f0[ia,c]:+8.1f}", flush=True)
    per_el = {}
    for (ia, s, c), gv in zip(labels, g):
        per_el.setdefault(s, []).append(gv)
    for s, arr in sorted(per_el.items()):
        arr = np.array(arr)
        print(f"   element {s:>2}: n={arr.size:4d}  mean {arr.mean():6.1f}  "
              f"max {arr.max():6.1f} meV/A", flush=True)
print("DONE")
