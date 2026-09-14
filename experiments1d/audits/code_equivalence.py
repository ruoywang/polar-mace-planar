"""Old-code vs new-code equivalence on real frames (GPU): energy, full-derivative
forces and the PB solver diagnostics for a few val frames, from a model object +
checkpoint (same loading as ab_eval.py). Run once per code tree via PYTHONPATH and
KIT_OUT; then `compare A.npz B.npz` prints max |dE| and max |dF| per frame. The
same tree run twice gives the CUDA non-determinism floor to read the differences
against. Env: KIT_RUN, KIT_NAME, KIT_CKPT, KIT_MODEL_OBJ, KIT_SIDS (comma list),
KIT_OUT (npz path)."""
import glob, os, sys, time
import numpy as np
if len(sys.argv) >= 4 and sys.argv[1] == "compare":
    a = np.load(sys.argv[2], allow_pickle=True); b = np.load(sys.argv[3], allow_pickle=True)
    print(f"compare {os.path.basename(sys.argv[2])} vs {os.path.basename(sys.argv[3])}  (code {a['code']} vs {b['code']})")
    print(f"  {'sid':>5} {'natoms':>6} {'E_a (eV)':>16} {'|dE| eV':>10} {'max|dF| eV/A':>13} {'rms dF':>10} {'n_outer a/b':>11} {'rms_last a/b':>22}")
    for sid in a["sids"]:
        ia = list(a["sids"]).index(sid); ib = list(b["sids"]).index(sid)
        Fa, Fb = a["F"][ia], b["F"][ib]
        print(f"  {int(sid):>5} {Fa.shape[0]:>6} {a['E'][ia]:16.8f} {abs(a['E'][ia]-b['E'][ib]):10.2e} {np.abs(Fa-Fb).max():13.2e} {np.sqrt(((Fa-Fb)**2).mean()):10.2e} {int(a['n_outer'][ia])}/{int(b['n_outer'][ib])} {a['rms_last'][ia]:.3e}/{b['rms_last'][ib]:.3e}")
    sys.exit(0)
import torch
from ase.io import read
from mace import data as mace_data, tools
from mace.data.utils import KeySpecification, update_keyspec_from_kwargs
from mace.tools import torch_geometric, torch_tools
import mace.modules.pb1d_backend as PB
import subprocess
RUN = os.environ["KIT_RUN"]; NAME = os.environ["KIT_NAME"]; CKPT = os.environ["KIT_CKPT"]; MP = os.environ["KIT_MODEL_OBJ"]
SIDS = [int(x) for x in os.environ.get("KIT_SIDS", "28,628,339").split(",")]; OUT = os.environ["KIT_OUT"]
code = subprocess.run(["git", "log", "--oneline", "-1"], cwd=os.path.dirname(os.path.dirname(os.path.dirname(PB.__file__))), capture_output=True, text=True).stdout.strip()[:60]
os.chdir(RUN)
device = torch_tools.init_device("cuda"); torch_tools.set_default_dtype("float64")
args = tools.build_default_arg_parser().parse_args(["--config", "config_pb1d.yaml", "--name", NAME, "--seed", "123", "--work_dir", ".", "--device", "cuda"])
args.key_specification = KeySpecification(); update_keyspec_from_kwargs(args.key_specification, vars(args))
model = torch.load(f=MP, map_location=device).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device)["model"], strict=True)
model.eval(); model._pb1d_epoch = int(CKPT.rsplit("epoch-", 1)[1].split(".")[0])
for p in model.parameters(): p.requires_grad_(False)
z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])
_bd = {}; _bk = PB.PB1DBackend.solve_graph
def _wrap(self, *a, **k): _bd["b"] = self; return _bk(self, *a, **k)
PB.PB1DBackend.solve_graph = _wrap
frames = {int(a.info["sample_id"]): a for sp in ("train", "val") for a in read(os.path.join(RUN, "data", f"{sp}.xyz"), ":") if int(a.info["sample_id"]) in SIDS}
E_l, F_l, no_l, rms_l, t_l = [], [], [], [], []
for sid in SIDS:
    a = frames[sid]
    cfg = mace_data.config_from_atoms(a, key_specification=args.key_specification)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1))).to(device)
    b["positions"].requires_grad_(True)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.enable_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
        E = pred["energy"].sum(); F = -torch.autograd.grad(E, b["positions"])[0]
    torch.cuda.synchronize(); t_l.append(time.time() - t0)
    d = getattr(_bd.get("b"), "last_diagnostics", {}) or {}
    E_l.append(float(E)); F_l.append(F.detach().cpu().numpy()); no_l.append(int(d.get("n_outer", -1))); rms_l.append(float(d.get("rms_last", float("nan"))))
    print(f"  sid {sid}: E {E_l[-1]:.8f} eV, n_outer {no_l[-1]}, rms_last {rms_l[-1]:.3e}, {t_l[-1]:.2f} s", flush=True)
np.savez(OUT, sids=np.array(SIDS), E=np.array(E_l), F=np.array(F_l, dtype=object), n_outer=np.array(no_l), rms_last=np.array(rms_l), t=np.array(t_l), code=code)
print(f"wrote {OUT}  ({code})")
