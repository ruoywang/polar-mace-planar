"""Per-plane projection verification (concrete post-change evidence):
A. plane sums of the projected residual: max |plane mean of delta| per
   channel and the residual dipole mu_delta (was -6..-14 e*A junk) -> ~1e-14.
B. supervision profile consistency: r_sup captured from the solve equals an
   independent rebuild from the frozen cavity stash (the loss's convention).
C. loss-gradient exactness: central FD of the s3d point loss along a random
   head-weight direction vs autograd, with profile-cache writes frozen ->
   relative error (the fictitious-gradient guard, cf. the dc.detach lesson).
D. energy values E_cav / E_3d (e_xsol + e_self split) for the record.
cwd = a gate dir (gate_fix5): weights fixed, the CODE is under test.
"""
# Paths come from env vars with the LS6 values as defaults, so this
# runs unchanged on Lonestar6 and on a machine that holds the same
# payload elsewhere: KIT_PB_REPO, KIT_MACE_REPO, KIT_DFT,
# KIT_PB_CONFIG. Previously hardcoded, which made every audit
# script non-portable (found by the workstation session 2026-09-09).
import os
import sys

import torch

sys.path.insert(0, os.environ.get("KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get("KIT_MACE_REPO", "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.pb1d_backend as PB
from mace.modules.solvent3d import (
    normalized_gradient_envelope,
    load_solvent3d_targets,
    attach_solvent3d_samples_to_batch,
    mean_squared_error_solvent3d,
)

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
model = torch.load(f="checkpoints/s3d_gate_fix5_run-123.model", map_location=device).to(device)
model.train()
model.solvent3d_energy = True
model.solvent_cavity_energy = True
head_params = [p for n, p in model.named_parameters() if "solvent3d_head" in n]

cap = {}
_orig = PB.PB1DBackend.solve_graph
def wrap(self, *a, **k):
    out = _orig(self, *a, **k)
    if out.get("s3d_obs") is not None:
        cap["obs"] = out["s3d_obs"]
        cap["grid"] = self._grids[list(self._grids)[-1]]
        cap["coeffs"] = k.get("s3d_coeffs")
        cap["sigmas"] = k.get("s3d_sigmas")
        cap["backend"] = self
        cap["e_cav"] = out.get("e_cav")
        cap["e_s3d"] = out.get("e_s3d")
    return out
PB.PB1DBackend.solve_graph = wrap

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff",
               "solvated": "solvated"},
    arrays_keys={"forces": "forces"},
)
want = {"NiN44": 1, "NiN44neusol": 601}
frames = []
for a in read("data/train.xyz", ":"):
    ct = a.info.get("config_type")
    if ct in want and int(a.info["sample_id"]) == want[ct]:
        frames.append(a)
    if len(frames) == 2:
        break


class StubLoss:
    solvent3d_targets = load_solvent3d_targets("data/solvent3d_points_manifest.json")
    solvent3d_samples = 512
    solvent3d_rng = None


SIG = StubLoss.solvent3d_targets.signal_ms
batches = []
for a in frames:
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    attach_solvent3d_samples_to_batch(b, StubLoss)  # points fixed once, reused
    batches.append((int(a.info["sample_id"]), a, b))

# ---- A/B/D on each frame -------------------------------------------------
for sid, a, b in batches:
    cap.clear()
    with torch.no_grad():
        model(b.to_dict(), training=True, compute_force=False)
    obs = cap["obs"]
    grid = cap["grid"]
    cell64 = torch.as_tensor(grid.cell, device=device, dtype=torch.float64)
    d_b = float(obs["delta_b_pl"].abs().max())
    d_i = float(obs["delta_i_pl"].abs().max())
    mu = float(obs["mu_delta"])
    print(f"[A] sid {sid}: max|delta_b_pl| {d_b:.3e}  max|delta_i_pl| {d_i:.3e} "
          f"e/A^3   mu_delta {mu:+.3e} e*A", flush=True)
    # B: independent rebuild of the supervision profiles from the frozen stash
    cavd = grid._solv3d_cavity
    env_bf = normalized_gradient_envelope(cavd[1], cell64)
    env_if = torch.clamp(cavd[0], 0.0, 1.0)
    be = cap["backend"]
    pos_frac = torch.remainder(
        torch.as_tensor(a.get_positions(), device=device, dtype=torch.float64)
        @ torch.linalg.inv(cell64), 1.0)
    volume = grid.volume
    mb = grid.ifft_real(be._gto_net_density_g(
        grid, pos_frac, cap["coeffs"][:, 0].to(torch.float64), cap["sigmas"])) / volume
    mi = grid.ifft_real(be._gto_net_density_g(
        grid, pos_frac, cap["coeffs"][:, 1].to(torch.float64), cap["sigmas"])) / volume
    rb_re = (env_bf * mb).mean(dim=(0, 1)) / torch.clamp(env_bf.mean(dim=(0, 1)), min=1e-12)
    ri_re = (env_if * mi).mean(dim=(0, 1)) / torch.clamp(env_if.mean(dim=(0, 1)), min=1e-12)
    eb = float((rb_re - obs["r_sup_b"].to(torch.float64)).abs().max())
    ei = float((ri_re - obs["r_sup_i"].to(torch.float64)).abs().max())
    rs = float(obs["r_sup_b"].abs().max())
    print(f"[B] sid {sid}: max|r_sup rebuild diff| b {eb:.3e} i {ei:.3e} "
          f"(profile scale {rs:.3e})", flush=True)
    ec = float(cap["e_cav"].detach()) if cap.get("e_cav") is not None else float("nan")
    es = float(cap["e_s3d"].detach()) if cap.get("e_s3d") is not None else float("nan")
    print(f"[D] sid {sid}: E_cav {ec:+.4f} eV  E_3d {es:+.6f} eV "
          f"(xsol {obs['e_xsol']:+.6f} self {obs['e_self']:+.6f})", flush=True)

# ---- C: loss-gradient exactness (FD vs autograd) -------------------------
# freeze the lagged-SCF profile cache so the forward is a pure function of W
model._pb1d_store_profile = lambda *a, **k: None
W = head_params[0]
gen = torch.Generator(device="cpu").manual_seed(7)
d = torch.randn(W.shape, generator=gen).to(device=device, dtype=W.dtype)
d = d / d.norm()
W0 = W.detach().clone()


def s3d_loss(sid_b):
    _, _, b = sid_b
    pred = model(b.to_dict(), training=True, compute_force=False)
    return mean_squared_error_solvent3d(b, pred, SIG, model.solvent3d_sigmas, ddp=False)


for sid_b in batches:
    sid = sid_b[0]
    # one warm forward with the frozen cache so both FD sides see it
    with torch.no_grad():
        L0 = s3d_loss(sid_b)
    if L0 is None:
        print(f"[C] sid {sid}: no scoreable points (skipped)", flush=True)
        continue
    loss = s3d_loss(sid_b)
    (g,) = torch.autograd.grad(loss, W)
    gd = float((g * d).sum())
    for h in (1.0e-3, 1.0e-4):
        with torch.no_grad():
            W.data.copy_(W0 + h * d)
            Lp = float(s3d_loss(sid_b))
            W.data.copy_(W0 - h * d)
            Lm = float(s3d_loss(sid_b))
            W.data.copy_(W0)
        fd = (Lp - Lm) / (2.0 * h)
        rel = abs(fd - gd) / max(abs(gd), 1e-300)
        print(f"[C] sid {sid} h={h:.0e}: autograd {gd:+.10e}  FD {fd:+.10e}  "
              f"rel {rel:.3e}", flush=True)
print("DONE")
