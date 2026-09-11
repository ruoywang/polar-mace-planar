"""Is the native local_electron_energy channel able to move a PAIRED charging
energy? (gate_le run, 2026-09-11)

WHY THIS IS A DIFFERENT QUESTION FROM THE EARLIER AUDIT. That audit compared
the READOUT INPUT FEATURES between sid1 and sid601 and found them identical to
1.9e-16 relative, hence d(inter_e) = +0.000000 eV exactly. It did NOT compare
the atomic charges, so it says nothing about this channel. Reading
field_blocks.py:723 the channel computes

    q_in = charges_induced + charges_0
    le   = mlp( [ <node_feats, W_q q_in> , <node_feats, W_v field_feats> ] )

so the charge-blind node_feats are contracted against a charge-carrying vector.
Whether that makes le charge-dependent depends entirely on whether q_in itself
differs between the two states, which is measured here rather than argued:
total_charge enters the charge construction only through the solvent
(extensions.py:178, solvent_charge = -total_charge), and reaches the atomic
charges only via the field feedback at extensions.py:2562. charges_0 is cloned
BEFORE that feedback (2444) and charges_induced after it, so the two have
different exposure and are reported separately.

THE ABLATION GATE. A captured tensor is not proof that the number enters the
total energy. So every frame is run twice: once normally, once with the
channel's output forced to zero. The change in the model's own reported energy
must equal the captured sum exactly, or the probe refuses to interpret itself.

WHAT WOULD COUNT. The paired quantity is the only one that matters:
    d(le) = le(charged) - le(neutral)
If d(le) is zero, training this channel cannot move the charging energy and
the 40-epoch run answers the user's question in the negative regardless of its
RMSE table. If d(le) is nonzero, it is compared against the size of the gap it
would have to cover: eps_Delta = Delta E_model - Delta E_DFT, and against the
one-constant correction a*dN with a = -5.2745 eV that already removes 98.2% of
the bias on the test pairs.
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get(
    "KIT_PB_REPO", "/work/08384/tg876840/ls6/repos/cep-dip-python-pb"))
sys.path.insert(0, os.environ.get(
    "KIT_MACE_REPO",
    "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))

os.environ.setdefault("MACE_PB1D_NO_PRELOAD", "1")

from ase.io import read

from mace import data as mace_data
from mace.data import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils
import mace.modules.pb1d_backend as PB   # for _evict; the reconcile
# audit measured 24 MB/sample retained in _bl_ram and needed per-pair
# eviction to bring a forward pass from 160 MB of growth down to 11 MB.
# MACE_PB1D_NO_PRELOAD alone does not release what a forward has cached.

RUN = os.environ.get("KIT_RUN",
                     "/scratch/08384/tg876840/tmp/c-MACEsol/3-residual_3D/gate_le")
# sid 1/601 live in TRAIN, not val (measured: train.xyz has both, val.xyz has
# neither), so every split is read and the split of each frame is printed. A
# charge dependence seen on a TRAINING pair is not held-out evidence, and the
# table says which pairs are which rather than leaving the reader to assume.
SPLITS = ("train", "val", "test")
PAIRS = [tuple(int(x) for x in p.split(":"))
         for p in os.environ.get("KIT_PAIRS", "1:601").split(",")]
A_CONST = float(os.environ.get("KIT_A_CONST", "-5.2745"))   # eV per electron

# ---------------------------------------------------------------- model ------
def find_model(run):
    cands = (sorted(glob.glob(os.path.join(run, "models", "*.model")))
             + sorted(glob.glob(os.path.join(run, "checkpoints", "*.model"))))
    cands = [c for c in cands if "compiled" not in os.path.basename(c)]
    if not cands:
        raise SystemExit(
            f"no .model in {run}/models or {run}/checkpoints -- mace writes it "
            f"at the END of training, so the run is not finished yet. "
            f"Checkpoints present: "
            f"{[os.path.basename(p) for p in sorted(glob.glob(os.path.join(run,'checkpoints','*.pt')))]}")
    return cands[-1]

def find_ckpt(run):
    pts = sorted(glob.glob(os.path.join(run, "checkpoints", "*_epoch-*.pt")),
                 key=lambda p: int(p.rsplit("epoch-", 1)[1].split(".")[0]))
    return pts[-1] if pts else None

# The pickled model carries a PB1DBackend whose __setstate__ re-runs __init__,
# and that __init__ opens RELATIVE paths taken from the training config --
# './cal1_train.json', './cache', './data/baseline_cache'. Unpickling therefore
# fails with FileNotFoundError unless the process cwd is the run directory
# (measured: job 3430198 died this way in 76 s on a GPU node, so it is not a
# device problem). Chdir here rather than relying on the job script, and note
# that the probe itself writes nothing -- only the solver's own cache under the
# run's ./cache is touched, which is the run's own.
os.chdir(RUN)

device = torch_tools.init_device(os.environ.get("KIT_DEVICE", "cuda"))
mpath = find_model(RUN)
model = torch.load(f=mpath, map_location=device).to(device)
cpath = os.environ.get("KIT_CKPT") or find_ckpt(RUN)
if cpath:
    ck = torch.load(cpath, map_location=device)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck.model
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"   state_dict mismatch: {len(missing)} missing, "
              f"{len(unexpected)} unexpected -> refusing to interpret")
        for k in list(missing)[:5] + list(unexpected)[:5]:
            print(f"     {k}")
        raise SystemExit(1)
model.eval()
for p in model.parameters():
    p.requires_grad = False
model.solvent3d_energy = True
model.solvent_cavity_energy = True
model.solvent_baseline_coupling = True

print(f"=================== the channel itself ===================")
print(f"   model      : {os.path.basename(mpath)}")
print(f"   weights    : {os.path.basename(cpath) if cpath else '(as saved)'}")
mod = getattr(model, "local_electron_energy", None)
if mod is None:
    raise SystemExit("this model has local_electron_energy = None, i.e. the run "
                     "was built with add_local_electron_energy False. Nothing "
                     "to probe.")
npar = sum(p.numel() for p in mod.parameters())
ntot = sum(p.numel() for p in model.parameters())
print(f"   parameters : {npar} in the channel, {ntot} in the model "
      f"({100.0 * npar / ntot:.2f}%)")

# ------------------------------------------------------------ capture --------
_backend = {}
_bk = PB.PB1DBackend.solve_graph

def _wrap_bk(self, *a, **k):
    _backend["b"] = self
    return _bk(self, *a, **k)

PB.PB1DBackend.solve_graph = _wrap_bk

def _evict():
    b = _backend.get("b")
    if b is None:
        return
    for attr in ("_bl_ram", "_grids", "_solvers", "_c_units"):
        d = getattr(b, attr, None)
        if isinstance(d, dict):
            d.clear()

cap = {}
ABLATE = {"on": False}

def pre_hook(m, args, kwargs):
    cap["charges_0"] = kwargs["charges_0"].detach().clone()
    cap["charges_induced"] = kwargs["charges_induced"].detach().clone()
    cap["field_feats"] = kwargs["field_feats"].detach().clone()
    cap["node_feats"] = kwargs["node_feats"].detach().clone()
    return None

def post_hook(m, args, kwargs, out):
    cap["le"] = out.detach().clone()
    if ABLATE["on"]:
        return torch.zeros_like(out)
    return None

mod.register_forward_pre_hook(pre_hook, with_kwargs=True)
mod.register_forward_hook(post_hook, with_kwargs=True)

# -------------------------------------------------------------- data ---------
kspec = KeySpecification(
    info_keys={"energy": "energy", "total_charge": "total_charge",
               "sample_id": "sample_id", "Fermi": "Fermi",
               "fermi_level": "Fermi", "solvated": "solvated",
               "potential": "potential_diff", "head": "head"},
    arrays_keys={"forces": "forces"})
by_sid, split_by_sid = {}, {}
for sp in SPLITS:
    f = os.path.join(RUN, "data", f"{sp}.xyz")
    if not os.path.exists(f):
        continue
    for a in read(f, ":"):
        sid = a.info.get("sample_id")
        if sid is None:
            continue
        by_sid[int(sid)] = a
        split_by_sid[int(sid)] = sp
print(f"   frames     : {len(by_sid)} with a sample_id across "
      f"{[sp for sp in SPLITS if os.path.exists(os.path.join(RUN,'data',sp+'.xyz'))]}")
missing = [s for pr in PAIRS for s in pr if s not in by_sid]
if missing:
    raise SystemExit(f"these sids are in none of the splits: {missing}")
z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

def one(sid):
    a = by_sid[sid]
    cfg = mace_data.config_from_atoms(a, key_specification=kspec)
    ds = [mace_data.AtomicData.from_config(cfg, z_table=z_table,
                                           cutoff=float(model.r_max))]
    b = next(iter(torch_geometric.dataloader.DataLoader(
        ds, batch_size=1))).to(device)
    cap.clear(); ABLATE["on"] = False
    with torch.no_grad():
        pred = model(b.to_dict(), compute_force=False, training=False)
    e_on = float(pred["energy"].sum())
    got = {k: v.clone() for k, v in cap.items()}
    cap.clear(); ABLATE["on"] = True
    with torch.no_grad():
        pred0 = model(b.to_dict(), compute_force=False, training=False)
    e_off = float(pred0["energy"].sum())
    ABLATE["on"] = False
    return dict(sid=sid, atoms=a, e_on=e_on, e_off=e_off,
                q=float(a.info.get("total_charge", 0.0)),
                label=float(a.info["energy"]),
                nat=len(a), **got)

for sc, sn in PAIRS:
    print(f"\n=================== pair {sc} (charged) / {sn} (neutral) "
          f"===================")
    rc, rn = one(sc), one(sn)
    _evict()
    dpos = float(np.abs(rc["atoms"].get_positions()
                        - rn["atoms"].get_positions()).max())
    print(f"   same geometry: max |dpos| {dpos:.3e} A;  q = {rc['q']:+.3f} / "
          f"{rn['q']:+.3f};  dN = {rn['q'] - rc['q']:+.3f} electrons")
    print(f"   splits: sid {sc} in {split_by_sid[sc]}, sid {sn} in "
          f"{split_by_sid[sn]}"
          + ("   <- TRAINING pair, not held-out evidence"
             if "train" in (split_by_sid[sc], split_by_sid[sn]) else
             "   <- held out"))

    print(f"\n   ABLATION GATE -- the captured sum must equal what the channel "
          f"puts into the total energy")
    ok = True
    for r, tag in ((rc, "charged"), (rn, "neutral")):
        cs = float(r["le"].sum())
        de = r["e_on"] - r["e_off"]
        rel = abs(cs - de) / max(abs(cs), 1e-30)
        ok &= rel < 1e-10
        print(f"     {tag:>8}: captured sum {cs:+.9f} eV, energy with minus "
              f"without {de:+.9f} eV, relative {rel:.3e}")
    if not ok:
        print(f"     GATE FAILED -- the captured tensor is not the term that "
              f"enters the energy; nothing below is interpretable")
        continue
    print(f"     gate passed: the captured tensor IS the energy term")

    print(f"\n   IS q_in CHARGE-DEPENDENT? (this is what the earlier audit did "
          f"not test)")
    for k in ("charges_0", "charges_induced", "field_feats", "node_feats"):
        x, y = rc[k], rn[k]
        d = float((x - y).abs().max())
        scale = max(float(y.abs().max()), 1e-30)
        print(f"     {k:>16} {str(tuple(x.shape)):>14}: max |diff| {d:.6e}, "
              f"relative {d / scale:.6e}")
    qc = rc["charges_induced"] + rc["charges_0"]
    qn = rn["charges_induced"] + rn["charges_0"]
    print(f"     {'q_in = sum':>16} {str(tuple(qc.shape)):>14}: max |diff| "
          f"{float((qc - qn).abs().max()):.6e}, relative "
          f"{float((qc - qn).abs().max()) / max(float(qn.abs().max()), 1e-30):.6e}")
    print(f"     column sums of q_in: charged {qc.sum(0).detach().cpu().numpy()}")
    print(f"                          neutral {qn.sum(0).detach().cpu().numpy()}")

    print(f"\n   THE PAIRED NUMBER")
    lec, len_ = float(rc["le"].sum()), float(rn["le"].sum())
    dle = lec - len_
    de_model = rc["e_on"] - rn["e_on"]
    de_label = rc["label"] - rn["label"]
    eps = de_model - de_label
    dN = rn["q"] - rc["q"]
    print(f"     le(charged)            {lec:+.6f} eV")
    print(f"     le(neutral)            {len_:+.6f} eV")
    print(f"     d(le)                  {dle:+.6f} eV    <- can this channel "
          f"move the pair at all?")
    print(f"     Delta E model          {de_model:+.6f} eV")
    print(f"     Delta E DFT            {de_label:+.6f} eV")
    print(f"     eps_Delta              {eps:+.6f} eV    <- the gap d(le) "
          f"would have to cover")
    print(f"     a*dN, a = {A_CONST:+.4f}     {A_CONST * dN:+.6f} eV    "
          f"<- the one-constant correction already measured")
    if abs(dle) < 1e-9:
        print(f"     READ: d(le) is zero to 1e-9, so this channel is as "
              f"charge-blind on this pair as inter_e is")
    else:
        print(f"     READ: d(le) is nonzero, so the channel CAN move the pair; "
              f"it covers {100.0 * dle / eps if abs(eps) > 0 else float('nan'):+.1f}% "
              f"of eps_Delta at this checkpoint")
