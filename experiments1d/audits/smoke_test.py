"""Kit smoke test: does this machine reproduce the LS6 numbers, and how fast
is it in double precision?

Run it from the kit's run/ directory after `source ../env.sh`:
    python -u smoke_test.py                 # compares against reference_values.json
    python -u smoke_test.py --write-ref     # regenerates reference_values.json (LS6 only)

Part 1 measures double precision on this card. It matters because the PB
solve is float64 by design (cep-dip-python-pb/pure_python/torch_pb.py line 20:
"float64 by default to clear the cal_18 validation gate; float32 diverges on
this ill-conditioned problem"), and consumer cards run float64 at 1/64 of
their float32 rate, against 1/2 on an A100. The FFT number is the one to look
at: the audits are dominated by grid FFTs and elementwise grid work, not by
dense linear algebra.

Part 2 runs the two frames the audits use (sid 1 charged, sid 601 neutral) and
compares every solvent energy term against the values this same code produced
on the A100. Anything outside 1e-6 relative means the port is not equivalent
and no result from this machine should be trusted.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ["KIT_PB_REPO"])
sys.path.insert(0, os.environ["KIT_MACE_REPO"])

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.extensions as ext
import mace.modules.pb1d_backend as PB

WRITE_REF = "--write-ref" in sys.argv
# layout-agnostic: the kit sets none of these and gets the kit defaults; a
# machine that already holds the payload elsewhere (e.g. the 81 GB
# migration_kit) points them at its own directories instead.
REF = os.environ.get("KIT_REF", "reference_values.json")
CKDIR = os.environ.get("KIT_CKPT_DIR", "checkpoints")
DATADIR = os.environ.get("KIT_DATA_DIR", "data")
CKPT = os.path.join(CKDIR, "s3d_gate_bl2_run-123.model")
STATE = os.path.join(CKDIR, "s3d_gate_bl2_run-123_epoch-33.pt")
for _p in (CKPT, STATE):
    if not os.path.exists(_p):
        raise SystemExit(f"missing {_p} -- set KIT_CKPT_DIR to the directory "
                         f"holding s3d_gate_bl2_run-123.model and its "
                         f"_epoch-33.pt")

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
name = torch.cuda.get_device_name(0)
tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
print(f"device: {name}  {tot:.1f} GiB  torch {torch.__version__} "
      f"cuda {torch.version.cuda}", flush=True)
print(f"interpreter: {sys.executable}", flush=True)
print(f"cwd: {os.getcwd()}  ckpt dir: {CKDIR}  data dir: {DATADIR}",
      flush=True)
# the PB backend stored inside the saved model resolves its own baseline cache
# and solver config RELATIVE to the current directory ("./data/baseline_cache",
# "./cal1_train.json"), so those two must exist here whatever KIT_* say
for _need in ("cal1_train.json", "data/baseline_cache"):
    if not os.path.exists(_need):
        print(f"  WARNING: ./{_need} not present -- the model's own backend "
              f"resolves that path relative to the cwd and will fail",
              flush=True)

# ---------------- part 1: double precision on this card ----------------
bench = {"device": name, "total_GiB": round(tot, 1)}
n = 2048
A = torch.randn(n, n, dtype=torch.float64, device=device)
B = torch.randn(n, n, dtype=torch.float64, device=device)
for _ in range(2):
    A @ B
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    A @ B
torch.cuda.synchronize()
gf = 5 * 2.0 * n ** 3 / (time.time() - t0) / 1e9
bench["dgemm_GFLOPs"] = round(gf, 1)
del A, B

g = torch.randn(100, 100, 300, dtype=torch.float64, device=device)
for _ in range(2):
    torch.fft.irfftn(torch.fft.rfftn(g), s=g.shape)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    torch.fft.irfftn(torch.fft.rfftn(g), s=g.shape)
torch.cuda.synchronize()
ms = (time.time() - t0) / 20 * 1e3
bench["fft64_100x100x300_ms"] = round(ms, 3)
gbig = torch.randn(168, 168, 500, dtype=torch.float64, device=device)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    torch.fft.irfftn(torch.fft.rfftn(gbig), s=gbig.shape)
torch.cuda.synchronize()
bench["fft64_168x168x500_ms"] = round((time.time() - t0) / 10 * 1e3, 3)
del g, gbig
torch.cuda.empty_cache()
print(f"double precision: DGEMM {bench['dgemm_GFLOPs']:.0f} GFLOP/s | "
      f"FFT 100x100x300 {bench['fft64_100x100x300_ms']:.2f} ms | "
      f"FFT 168x168x500 {bench['fft64_168x168x500_ms']:.2f} ms", flush=True)

# ---------------- part 2: do the physics numbers match? ----------------
model = torch.load(f=CKPT, map_location=device).to(device)
ck = torch.load(STATE, map_location=device)
model.load_state_dict(ck["model"] if "model" in ck else ck.model)
model.eval()
for p in model.parameters():
    p.requires_grad = False
model.solvent3d_energy = True
model.solvent_cavity_energy = True
model.solvent_baseline_coupling = True
# the saved model object carries the LS6 path to the PB repo and prepends it to
# sys.path at forward time; a path that does not exist is harmless (the import
# then resolves through KIT_PB_REPO above), but point it at the kit anyway
model.solvent_pb_repo = os.environ["KIT_PB_REPO"]

cap = {}
_comp = ext._slab_compensation_periodic_1d_energy_radial
def wrap_comp(**kw):
    out = _comp(**kw)
    cap["comp"] = out.detach().clone()
    return out
ext._slab_compensation_periodic_1d_energy_radial = (
    lambda *a, **k: wrap_comp(**k) if not a else _comp(*a, **k))

_solve = PB.PB1DBackend.solve_graph
def wrap_solve(self, *a, **k):
    out = _solve(self, *a, **k)
    if out.get("s3d_obs") is not None:
        cap["obs"] = out["s3d_obs"]
    cap["rho_layer_z"] = out["rho_layer_z"].detach().clone()
    cap["rms_last"] = float(out["rms_last"])
    return out
PB.PB1DBackend.solve_graph = wrap_solve

z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "total_spin": "total_spin", "sample_id": "sample_id",
               "fermi_level": "Fermi", "potential": "potential_diff",
               "solvated": "solvated"},
    arrays_keys={"forces": "forces"},
)
want = {1: None, 601: None}
for split in ("train", "val", "test"):
    try:
        frames = read(os.path.join(DATADIR, f"{split}.xyz"), ":")
    except Exception:
        continue
    for a in frames:
        sid = int(a.info.get("sample_id", -1))
        if sid in want and want[sid] is None:
            want[sid] = a
    if all(v is not None for v in want.values()):
        break
missing = [s for s, v in want.items() if v is None]
if missing:
    raise SystemExit(f"frames missing from {DATADIR}/*.xyz: {missing}")

got = {"bench": bench, "frames": {}}
for sid, a in want.items():
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(
        ds, batch_size=1))).to(device)
    cap.clear()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
    torch.cuda.synchronize()
    wall = time.time() - t0
    obs = cap.get("obs", {})
    d = obs.get("delta_grid")
    rec = {
        "q_tot": float(a.info["total_charge"]),
        "energy": float(pred["energy"].sum()),
        "e_cav": float(pred["cavity_energy_g"].sum()),
        "e_s3d": float(pred["solvent3d_energy_g"].sum()),
        "e_xsol": float(obs.get("e_xsol", float("nan"))),
        "e_self": float(obs.get("e_self", float("nan"))),
        "comp_1d": float(cap["comp"].sum()) if "comp" in cap else float("nan"),
        "rho_layer_z_absint": float(cap["rho_layer_z"].abs().sum()),
        "pb_rms_last": cap.get("rms_last", float("nan")),
        "wall_s": round(wall, 3),
        "peak_GiB": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
    }
    if "baseline_coupling_energy_g" in pred:
        rec["e_bl"] = float(pred["baseline_coupling_energy_g"].sum())
    if d is not None:
        rec["delta_absint"] = float(d.abs().sum())
        rec["delta_plane_max"] = float(d.mean(dim=(0, 1)).abs().max())
    else:
        print("  note: delta_grid absent -- set MACE_S3D_EXPORT_DELTA=1")
    got["frames"][str(sid)] = rec
    print(f"\n== sid {sid} (q={rec['q_tot']:+.2f}) ==  {wall:.2f} s, "
          f"peak {rec['peak_GiB']:.2f} GiB", flush=True)
    for k in ("energy", "comp_1d", "e_cav", "e_bl", "e_s3d", "e_xsol",
              "e_self", "delta_absint", "delta_plane_max", "pb_rms_last"):
        if k in rec:
            print(f"  {k:>16} {rec[k]:+.8e}")

json.dump(got, open("smoke_results.json", "w"), indent=1)
if WRITE_REF:
    json.dump(got, open(REF, "w"), indent=1)
    print(f"\nwrote {REF} from this machine")
    raise SystemExit(0)

if not os.path.exists(REF):
    raise SystemExit(f"\nno {REF} to compare against")
ref = json.load(open(REF))
print(f"\ncomparing against {REF} (recorded on "
      f"{ref.get('bench', {}).get('device', '?')})")
rb = ref.get("bench", {})
for k in ("dgemm_GFLOPs",):
    if k in rb and rb[k]:
        print(f"  {k}: this {bench[k]:.0f} vs A100 {rb[k]:.0f} "
              f"({rb[k]/max(bench[k],1e-9):.1f}x slower here)")
for k in ("fft64_100x100x300_ms", "fft64_168x168x500_ms"):
    if k in rb and bench.get(k):
        print(f"  {k}: this {bench[k]:.2f} ms vs A100 {rb[k]:.2f} ms "
              f"({bench[k]/max(rb[k],1e-9):.1f}x slower here)")
TOL = 1.0e-6
worst, nbad = 0.0, 0
print(f"\n  {'frame':>6} {'term':>16} {'this machine':>18} {'A100':>18} "
      f"{'rel diff':>10}")
for sid, rec in got["frames"].items():
    rr = ref["frames"].get(sid, {})
    for k, v in rec.items():
        if k in ("wall_s", "peak_GiB") or k not in rr:
            continue
        if not isinstance(v, float):
            continue
        den = max(abs(rr[k]), 1.0e-8)
        rel = abs(v - rr[k]) / den
        worst = max(worst, rel)
        if rel > TOL:
            nbad += 1
            print(f"  {sid:>6} {k:>16} {v:+18.9e} {rr[k]:+18.9e} "
                  f"{rel:10.2e}  <-- OVER TOLERANCE")
print(f"\nworst relative difference {worst:.2e} over "
      f"{sum(len(v) for v in got['frames'].values())} numbers, "
      f"{nbad} over the {TOL:g} tolerance")
print("SMOKE PASS" if nbad == 0 else "SMOKE FAIL -- do not trust results here")
