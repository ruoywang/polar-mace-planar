"""A/B energy re-fit, phase 2: refit ONLY the energy readout, twice.
(user-specified design, 2026-09-10. NiN44 charged/neutral is the primary criterion.)

  arm   lateral bound charge      trainable
  A     current model prediction  existing energy readout
  B     DFT reference             the same energy readout

Both arms start from the SAME checkpoint weights, see the SAME frames, and get
the SAME budget -- same optimiser, learning rate, schedule, step count and
seed, full-batch so there is no batch order to differ either. Nothing outside
the readout moves: the density, P, the ionic channel and every other feature
are frozen in the cache, scale and shift are registered buffers and were never
trainable, and NO per-system compensation constant is added. The model's one
existing per-class parameter, fermi_vacuum_offset, acts only through
(1 - solvated) and so cannot touch the solvated frames this is judged on; it
stays frozen too.

THE ONLY THING THAT DIFFERS between the arms is a per-frame scalar in the
target, dE_f = E_3d^B - E_3d^A, which the cache phase computed by substituting
lat_DFT = rho_b_DFT - plane_mean(rho_b_DFT) for the model's own lateral bound
residual. dE is zero on the 200 unsolvated frames by construction, so those
frames are identical in both arms and neither arm is a hybrid.

  E_pred^X(theta) = e0_sum + sum_atoms scale_shift( sum_j readout_j,theta(h_j) )
                    + remainder + (dE if X == B else 0)

where `remainder` is the frozen solvent physics the readout does not produce.
Loss: mean over frames of ((E_pred - E_DFT)/n_atoms)^2, identical in both arms
and stated rather than inherited, since only energy is being fitted here.

WHAT IS REPORTED, per the user: on the validation set, NiN44 charged energy
RMSE, neutral energy RMSE, and the DIFFERENCE OF THE TWO GROUPS' SIGNED
BIASES -- separately, never as a cross-system average. The 20 same-geometry
charging pairs that sit wholly inside val are checked as well, and they needed
no new DFT. NiN88 is reported as a secondary line and is excluded from every
headline number.

THE READING, set by the user in advance: if B, after re-adapting, is more
accurate on BOTH charged and neutral AND the inter-group bias is smaller, that
supports going on to actually improve the model's lateral bound charge
prediction. If there is no advantage, the limits of the existing readout and
the frozen features are what to check first -- it does NOT license declaring
the physics correction worthless.

This step only tests whether a more accurate lateral charge makes the energy
easier to learn. Passing it would then mean turning the reference substitution
into the model's own predictable correction and validating the full model's
energy, forces and potential.
"""
import copy
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get(
    "KIT_MACE_REPO",
    "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))

from mace.tools import torch_tools

CKDIR = os.environ.get("KIT_CKPT_DIR", "checkpoints")
OUT = os.environ.get("KIT_OUT", ".")
STEPS = int(os.environ.get("KIT_STEPS", "4000"))
LR = float(os.environ.get("KIT_LR", "1e-3"))
SEED = int(os.environ.get("KIT_SEED", "0"))

device = torch_tools.init_device("cuda")
torch_tools.set_default_dtype("float64")
blob = torch.load(os.path.join(OUT, "ab_refit_cache.pt"), map_location="cpu")
recs, n_ro = blob["recs"], blob["n_readouts"]
scale, shift = blob["scale"], blob["shift"]
print(f"  cache: {len(recs)} frames, {n_ro} readouts, embedding "
      f"{blob['has_embedding']}, scale {scale:.6f}, shift {shift:.6f}")
print(f"  cache gates: feat {blob['gate_feat']:.3e}, e3d {blob['gate_e3d']:.3e}")
if not (blob["gate_feat"] < 1e-6 and blob["gate_e3d"] < 1e-6):
    raise SystemExit("  STOP: the cache failed its own gates, so it "
                     "misrepresents the model and must not be fitted.")

model = torch.load(f=os.path.join(CKDIR, "s3d_gate_bl2_run-123.model"),
                   map_location=device).to(device)
ck = torch.load(os.path.join(CKDIR, "s3d_gate_bl2_run-123_epoch-33.pt"),
                map_location=device)
model.load_state_dict(ck["model"] if "model" in ck else ck.model)
model.eval()
ro_ref = [model.readouts[i] for i in range(n_ro)]
if blob["has_embedding"]:
    ro_ref = [model.embedding_readout] + ro_ref
n_par = sum(p.numel() for r in ro_ref for p in r.parameters())
print(f"  trainable readout parameters: {n_par} across {len(ro_ref)} blocks "
      f"({', '.join(type(r).__name__ for r in ro_ref)})", flush=True)

# ---- build per-natoms blocks so a step is a handful of readout calls -------
idx_keys = sorted(recs[0]["feats"].keys())
print(f"  feature blocks per frame: "
      f"{ {k: recs[0]['feats'][k].shape for k in idx_keys} }")
blocks = {}          # (split, nat) -> dict of stacked tensors
for r in recs:
    key = (r["split"], r["nat"])
    b = blocks.setdefault(key, {"rows": [], **{f"f{k}": [] for k in idx_keys}})
    b["rows"].append(r)
    for k in idx_keys:
        b[f"f{k}"].append(torch.from_numpy(r["feats"][k]))
for key, b in blocks.items():
    for k in idx_keys:
        b[f"f{k}"] = torch.cat(b[f"f{k}"], 0).to(device=device,
                                                 dtype=torch.float64)
    b["nat"] = key[1]
    b["nf"] = len(b["rows"])
    b["heads"] = torch.zeros(b[f"f{idx_keys[0]}"].shape[0],
                             dtype=torch.long, device=device)
    for nm in ("e0_sum", "remainder", "e_dft", "dE"):
        b[nm] = torch.tensor([rr[nm] for rr in b["rows"]],
                             dtype=torch.float64, device=device)
    b["natt"] = torch.tensor([rr["nat"] for rr in b["rows"]],
                             dtype=torch.float64, device=device)
    print(f"    {key[0]:>5} nat={key[1]:4d}: {b['nf']:4d} frames, "
          f"features {tuple(b[f'f{idx_keys[0]}'].shape)}", flush=True)


def predict(ros, b):
    """E_pred without the arm's dE: e0 + sum_atoms scale_shift(sum_j RO_j)."""
    n, nat = b["nf"], b["nat"]
    per_atom = None
    for j, k in enumerate(idx_keys):
        o = ros[j](b[f"f{k}"], b["heads"])
        o = o[:, 0] if o.dim() > 1 else o
        per_atom = o if per_atom is None else per_atom + o
    ss = scale * per_atom + shift
    return b["e0_sum"] + ss.view(n, nat).sum(dim=1) + b["remainder"]


def run_arm(arm):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ros = [copy.deepcopy(r) for r in ro_ref]
    for r in ros:
        for p in r.parameters():
            p.requires_grad_(True)
    pars = [p for r in ros for p in r.parameters()]
    opt = torch.optim.Adam(pars, lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    tr = [b for k, b in blocks.items() if k[0] == "train"]
    hist = []
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        num, den = 0.0, 0
        for b in tr:
            e = predict(ros, b) + (b["dE"] if arm == "B" else 0.0)
            r = (e - b["e_dft"]) / b["natt"]
            num = num + (r * r).sum(); den += b["nf"]
        loss = num / den
        loss.backward(); opt.step(); sch.step()
        if step % max(1, STEPS // 10) == 0 or step == STEPS - 1:
            hist.append((step, float(loss)))
    return ros, hist


def evaluate(ros, arm, split):
    out = {}
    with torch.no_grad():
        for k, b in blocks.items():
            if k[0] != split:
                continue
            e = predict(ros, b) + (b["dE"] if arm == "B" else 0.0)
            per = ((e - b["e_dft"]) / b["natt"]) * 1000.0
            for i, rr in enumerate(b["rows"]):
                out[rr["sid"]] = (rr["group"], float(per[i]), float(e[i]),
                                  rr["e_dft"], rr["nat"])
    return out


res = {}
for arm in ("A", "B"):
    ros, hist = run_arm(arm)
    res[arm] = dict(val=evaluate(ros, arm, "val"),
                    train=evaluate(ros, arm, "train"),
                    test=evaluate(ros, arm, "test"), hist=hist)
    print(f"\n  arm {arm} train loss (meV/atom)^2-ish: " +
          ", ".join(f"{s}:{l*1e6:.2f}" for s, l in hist), flush=True)

GRPS = ["nin44_charged", "neutral44", "unsolvated", "nin88_charged"]


def stats(ev, g):
    v = [x[1] for x in ev.values() if x[0] == g]
    if not v:
        return None
    v = np.array(v)
    return dict(n=len(v), rmse=float(np.sqrt((v * v).mean())),
                bias=float(v.mean()), mae=float(np.abs(v).mean()))


print(f"\n[RESULT] validation, meV/atom -- primary criterion is NiN44 charged "
      f"and neutral; NiN88 is a secondary line and enters no headline number")
print(f"  {'group':>15} {'n':>4} | {'A rmse':>8} {'A bias':>8} | "
      f"{'B rmse':>8} {'B bias':>8} | {'d rmse':>8} {'d |bias|':>9}")
for g in GRPS:
    a, b_ = stats(res["A"]["val"], g), stats(res["B"]["val"], g)
    if a is None:
        continue
    tag = "  <- secondary" if g == "nin88_charged" else ""
    tag += "  <- A == B by construction" if g == "unsolvated" else ""
    print(f"  {g:>15} {a['n']:4d} | {a['rmse']:8.3f} {a['bias']:+8.3f} | "
          f"{b_['rmse']:8.3f} {b_['bias']:+8.3f} | "
          f"{b_['rmse']-a['rmse']:+8.3f} {abs(b_['bias'])-abs(a['bias']):+9.3f}"
          f"{tag}")
for arm in ("A", "B"):
    c = stats(res[arm]["val"], "nin44_charged")
    n = stats(res[arm]["val"], "neutral44")
    print(f"  arm {arm}: inter-group SIGNED bias difference "
          f"(NiN44 charged - neutral) = {c['bias'] - n['bias']:+.3f} meV/atom")
ca, na = stats(res["A"]["val"], "nin44_charged"), stats(res["A"]["val"], "neutral44")
cb, nb = stats(res["B"]["val"], "nin44_charged"), stats(res["B"]["val"], "neutral44")
d_a, d_b = ca["bias"] - na["bias"], cb["bias"] - nb["bias"]
print(f"  -> |difference| {abs(d_a):.3f} (A) vs {abs(d_b):.3f} (B): "
      f"{'B SMALLER' if abs(d_b) < abs(d_a) else 'B not smaller'}")

print(f"\n[CHARGING PAIRS] same geometry, sid k <-> 600+k, both members in "
      f"val; no new DFT was added for this")
for arm in ("A", "B"):
    ev = res[arm]["val"]
    err = []
    for k in range(1, 201):
        if k in ev and 600 + k in ev:
            dp = ev[k][2] - ev[600 + k][2]
            dd = ev[k][3] - ev[600 + k][3]
            err.append((dp - dd, ev[k][4]))
    if err:
        e = np.array([x[0] for x in err]); nat = np.array([x[1] for x in err])
        print(f"  arm {arm}: {len(err)} pairs, charging-energy error RMSE "
              f"{np.sqrt((e*e).mean())*1000:.2f} meV "
              f"({np.sqrt(((e/nat)**2).mean())*1000:.4f} meV/atom), bias "
              f"{e.mean()*1000:+.2f} meV")

print(f"\n[BUDGET] identical by construction: same checkpoint init, same "
      f"frames, Adam lr {LR} with cosine decay over {STEPS} full-batch steps, "
      f"seed {SEED}, no batch order to differ. Final train loss A "
      f"{res['A']['hist'][-1][1]*1e6:.3f} vs B "
      f"{res['B']['hist'][-1][1]*1e6:.3f} (x1e6).")
print(f"\n[READING] B better on BOTH NiN44 charged and neutral AND a smaller "
      f"inter-group bias -> go on to make the lateral bound charge a "
      f"predictable model correction. No advantage -> check the readout's and "
      f"the frozen features' limits FIRST;\n  it does not license calling the "
      f"physics correction worthless. NiN88 does not enter this decision.",
      flush=True)
json.dump({a: {s: {str(k): v for k, v in res[a][s].items()}
               for s in ("val", "train", "test")} for a in ("A", "B")},
          open(os.path.join(OUT, "ab_refit_result.json"), "w"), indent=1)
print(f"\n  per-frame results written to ab_refit_result.json")
print("DONE")
