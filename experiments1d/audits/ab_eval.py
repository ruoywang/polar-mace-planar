"""Per-state evaluation of one A/B arm: absolute energies, forces and the paired
charging energy against DFT, on train and val, reported per state
(charged NiN44 / neutral NiN44 / charged NiN88; there are no neutral NiN88
frames). What the joint A/B was agreed to report (user, 2026-09-12).

  KIT_RUN      run directory (models/*.model + checkpoints/*_epoch-*.pt, latest epoch)
  KIT_NAME     the run's --name (for the arg parser)
  KIT_STRIDE   take every n-th train frame (default 3); val is always complete
  KIT_TAG      label printed in every table row
The environment (MACE_PB1D_LIVE_POS / GRAD_PASSES / DFORCE) is whatever the
caller sets and is printed: an arm is evaluated under the switches it trained
with, and arm A can additionally be evaluated under LIVE_POS=1 to see the force
that is the actual slope of its energy (its trained force is not).
Training-cache path (sample_id kept, frozen per-sid baseline) for every frame,
the same path the training used; the deployment path is a different object and
is not reported here. Energies in eV, per-atom energy errors in meV/atom,
forces in meV/A; paired charging energy dE = E(charged) - E(neutral) for the
pairs (k, k+600) present in the same split, rmse / bias / mean-removed against
DFT. Writes a per-frame npz next to the run's log for later tables.
"""
import glob, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.environ.get("KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get("KIT_MACE_REPO", "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))
os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")
from ase.io import read
from mace import data as mace_data, tools
from mace.data.utils import KeySpecification, update_keyspec_from_kwargs
from mace.tools import torch_geometric, torch_tools
import mace.modules.pb1d_backend as PB
RUN = os.environ["KIT_RUN"]; NAME = os.environ["KIT_NAME"]; TAG = os.environ.get("KIT_TAG", NAME)
STRIDE = int(os.environ.get("KIT_STRIDE", "3")); OUT = os.environ.get("KIT_OUT", os.path.join(RUN, "logs"))
os.chdir(RUN)
device = torch_tools.init_device("cuda"); torch_tools.set_default_dtype("float64")
args = tools.build_default_arg_parser().parse_args(["--config", "config_pb1d.yaml", "--name", NAME, "--seed", "123", "--work_dir", ".", "--device", "cuda"])
args.key_specification = KeySpecification(); update_keyspec_from_kwargs(args.key_specification, vars(args))
mp = [p for p in sorted(glob.glob(os.path.join(RUN, "models", "*.model"))) if "compiled" not in p][-1]
cp = sorted(glob.glob(os.path.join(RUN, "checkpoints", "*_epoch-*.pt")), key=lambda p: int(p.rsplit("epoch-",1)[1].split(".")[0]))[-1]
EPOCH = int(cp.rsplit("epoch-", 1)[1].split(".")[0])
model = torch.load(f=mp, map_location=device).to(device)
model.load_state_dict(torch.load(cp, map_location=device)["model"], strict=False)
model.eval(); model._pb1d_epoch = EPOCH
for p in model.parameters(): p.requires_grad_(False)
z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
_bd = {}; _bk = PB.PB1DBackend.solve_graph
def _wrap(self, *a, **k): _bd["b"] = self; return _bk(self, *a, **k)
PB.PB1DBackend.solve_graph = _wrap
def _evict():
    b = _bd.get("b")
    if b is not None:
        for at in ("_bl_ram", "_grids", "_solvers", "_c_units"):
            d = getattr(b, at, None)
            if isinstance(d, dict): d.clear()
env = {k: os.environ.get(k, "unset") for k in ("MACE_PB1D_LIVE_POS", "MACE_PB1D_GRAD_PASSES", "MACE_PB1D_DFORCE")}
print(f"[{TAG}] model {os.path.basename(mp)} @ epoch {EPOCH}   env {env}   train stride {STRIDE}")

def state_of(a):
    q = float(a.info.get("total_charge", 0.0)); n = len(a)
    return ("charged" if abs(q) > 1e-6 else "neutral") + (" NiN44" if n == 207 else " NiN88" if n == 339 else f" n{n}")

rows = []   # (split, sid, state, natoms, E_model, E_dft, F_rmse_meV, F_max_meV)
for split in ("train", "val"):
    frames = read(os.path.join(RUN, "data", f"{split}.xyz"), ":")
    if split == "train": frames = frames[::STRIDE]
    t0 = time.time()
    for i, a in enumerate(frames):
        sid = int(a.info["sample_id"])
        cfg = mace_data.config_from_atoms(a, key_specification=args.key_specification)
        ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
        b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
        b["positions"].requires_grad_(True)
        with torch.enable_grad():
            pred = model(b.to_dict(), compute_force=False, training=False)
            E = pred["energy"].sum()
            F = -torch.autograd.grad(E, b["positions"])[0]
        fd = np.asarray(a.get_array("forces")); df = F.detach().cpu().numpy() - fd
        rows.append((split, sid, state_of(a), len(a), float(E), float(a.info["energy"]),
                     1e3 * float(np.sqrt((df**2).mean())), 1e3 * float(np.abs(df).max())))
        del pred, E, F, b; _evict(); torch.cuda.empty_cache()
        if (i + 1) % 40 == 0: print(f"  {split}: {i+1}/{len(frames)} frames, {time.time()-t0:.0f} s", flush=True)
    print(f"  {split}: {len(frames)} frames in {time.time()-t0:.0f} s")

np.savez(os.path.join(OUT, f"ab_eval_{TAG}.npz"), rows=np.array(rows, dtype=object), env=str(env), epoch=EPOCH)
def rms(x): x = np.asarray(x, float); return float(np.sqrt((x**2).mean())) if x.size else float("nan")
print(f"\n[{TAG}] PER-STATE ABSOLUTE ENERGY AND FORCES (training-cache path, epoch {EPOCH})")
print(f"  {'split':>5} {'state':>14} {'n':>4} {'E err/atom rmse':>15} {'bias':>9} {'F rmse':>8} {'F max':>8}   (meV/atom, meV/atom, meV/A, meV/A)")
for split in ("train", "val"):
    for st in sorted({r[2] for r in rows if r[0] == split}):
        R = [r for r in rows if r[0] == split and r[2] == st]
        e = [1e3 * (r[4] - r[5]) / r[3] for r in R]
        print(f"  {split:>5} {st:>14} {len(R):>4} {rms(e):15.2f} {np.mean(e):9.2f} {rms([r[6] for r in R]):8.2f} {max(r[7] for r in R):8.1f}")
print(f"\n[{TAG}] PAIRED CHARGING ENERGY dE = E(k) - E(k+600), model vs DFT (eV)")
print(f"  {'split':>5} {'pairs':>5} {'rmse':>8} {'bias':>8} {'mean-removed':>12}")
for split in ("train", "val"):
    by = {r[1]: r for r in rows if r[0] == split}
    res = [(by[k][4] - by[k+600][4]) - (by[k][5] - by[k+600][5]) for k in by if k < 600 and k + 600 in by]
    if res:
        res = np.array(res); print(f"  {split:>5} {len(res):>5} {rms(res):8.4f} {res.mean():8.4f} {res.std():12.4f}")
    else: print(f"  {split:>5}     0  (no complete pairs in the evaluated subset)")
print("DONE")
