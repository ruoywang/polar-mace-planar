"""Direct evidence test for the warmup train/eval-branch energy offset.

Same weights, same 60-frame valid set, stage-2 energy flags ON vs OFF, for
two gate_pp2 checkpoints (EMA weights, the logged-metric convention):
  epoch-25 (pre-PB training: trunk calibrated WITHOUT the terms)
      -> prediction if the offset mechanism is real: OFF ~ old-family level,
         ON inflated by ~(E_cav+E_3d)/atom in quadrature
  epoch-33 (post-PB: trunk absorbed the terms)
      -> prediction: SIGN FLIPS - ON is the good one, OFF inflated
Also prints the measured per-frame E_cav+E_3d (the offset itself).
Profile-cache writes disabled (read-only eval). cwd = gate_pp2.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/work/08384/tg876840/ls6/repos/cep-dip-python-pb")
sys.path.insert(0, "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy")

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
MODEL_SKEL = os.environ.get("MACE_EVAL_MODEL",
                            "checkpoints/s3d_gate_pp22_run-123.model")
model = torch.load(f=MODEL_SKEL, map_location=device).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
model._pb1d_store_profile = lambda *a, **k: None  # read-only cache

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff",
               "solvated": "solvated"},
    arrays_keys={"forces": "forces"},
)
frames = read("data/val.xyz", ":")
print(f"valid frames: {len(frames)}")


def eval_rmse(tag):
    rows, terms = [], []
    for a in frames:
        cfg = mace_data.config_from_atoms(a, key_specification=kspec)
        ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                               cutoff=float(model.r_max))]
        b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
        with torch.no_grad():
            pred = model(b.to_dict(), compute_force=False, training=False)
        de = (float(pred["energy"].detach().cpu()) - float(a.info["energy"])) / len(a)
        ec = pred.get("cavity_energy_g")
        es = pred.get("solvent3d_energy_g")
        t = (float(ec.detach().sum()) if ec is not None else 0.0) + \
            (float(es.detach().sum()) if es is not None else 0.0)
        rows.append((a.info.get("config_type"), de, t,
                     float(a.info.get("total_charge", 0.0))))
        terms.append(t)
    d = np.array([r[1] for r in rows])
    rmse = 1000.0 * float(np.sqrt(np.mean(d * d)))
    t = np.array(terms)
    print(f"  {tag}: RMSE_E_per_atom = {rmse:7.2f} meV   "
          f"E_cav+E_3d per frame = {t.mean():+.3f} +- {t.std():.3f} eV",
          flush=True)
    # per-cohort decomposition: the paired cohorts (neusol vs vac, identical
    # geometries, DFT dE ~ +2.94 eV) carry the un-representable-split error;
    # signed bias per cohort is the crispest signature of who eats it
    for ct in sorted(set(r[0] for r in rows)):
        dc = np.array([r[1] for r in rows if r[0] == ct]) * 1000.0
        tc = np.array([r[2] for r in rows if r[0] == ct])
        print(f"      {ct:>12}: rmse {np.sqrt((dc**2).mean()):7.2f}  "
              f"bias {dc.mean():+7.2f} meV/atom  terms {tc.mean():+6.3f}"
              f"+-{tc.std():5.3f} eV  (n={dc.size})", flush=True)
    # charged cohort: is the E_3d charge-dependence the blocker? per-frame
    # (q_tot, terms, per-atom error) for NiN44 — a wrong charge trend in the
    # terms forces the trunk's charge machinery to counter-fit it
    for ctq in ("NiN44", "NiN88"):
        ch = sorted((r[3], r[2], 1000 * r[1]) for r in rows if r[0] == ctq)
        if ch and any(abs(c[1]) > 1e-9 for c in ch):
            print(f"      {ctq} per-frame (q, E_cav+E_3d eV, err meV/atom):",
                  " ".join(f"({q:+.1f},{t:+.2f},{e:+.1f})" for q, t, e in ch),
                  flush=True)
    return rmse


CKPTS = sys.argv[1:] or ["checkpoints/s3d_gate_pp22_run-123_epoch-25.pt",
                         "checkpoints/s3d_gate_pp22_run-123_epoch-33.pt"]
for ck in CKPTS:
    if ck == "SKEL":  # evaluate the loaded model file's own weights
        print(f"\n== {MODEL_SKEL} (as loaded) ==", flush=True)
    else:
        c = torch.load(ck, map_location=device)
        sd = c["model"] if isinstance(c, dict) and "model" in c else c.model
        model.load_state_dict(sd)
        print(f"\n== {ck.split('/')[-1]} ==", flush=True)
    for flags in (False, True):
        model.solvent3d_energy = flags
        model.solvent_cavity_energy = flags
        if os.environ.get("MACE_EVAL_BL"):
            model.solvent_baseline_coupling = flags
        eval_rmse(f"flags={'ON ' if flags else 'OFF'}")
print("DONE")
