"""Is a resumed segment the same run as a continuous one? (user-specified, 2026-09-12)

Arguments: state_X state_Y state_X2 log_X log_Y log_X2, where X and X2 are two
continuous runs to the same epoch (their difference is the run-to-run floor:
CUDA scatter/index_add kernels are not deterministic) and Y reached that epoch
through a save/exit/resume. For X-Y and X-X2 this prints, over the raw
parameters, the EMA shadow parameters and every Adam moment: max |diff| and
rms diff against the rms of the values; the Adam step counts; the scheduler
dicts; the best-loss bookkeeping; and for every rank whether each RNG stream
(python, numpy, torch CPU, torch CUDA, loader generators, the loss's two
sample-point RNGs) is identical. The reading: Y must sit at the X-X2 floor
with every RNG stream identical; anything systematically above the floor is a
state the segment file does not carry.
"""
import re, sys
import numpy as np, torch

def load(p):
    try: return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(p, map_location="cpu")

def tstats(a, b):
    a = a.double().flatten(); b = b.double().flatten()
    d = (a - b)
    return float(d.abs().max()), float(torch.sqrt((d ** 2).mean())), float(torch.sqrt((a ** 2).mean()))

def group_stats(pairs):
    mx, ss, n, vs = 0.0, 0.0, 0, 0.0
    for a, b in pairs:
        if a.numel() == 0: continue   # empty buffers (e.g. unused heads) have no statistics
        m, r, v = tstats(a, b); mx = max(mx, m); ss += r * r * a.numel(); n += a.numel(); vs += v * v * a.numel()
    return mx, (ss / n) ** 0.5 if n else float("nan"), (vs / n) ** 0.5 if n else float("nan")

def same(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b): return bool(a.shape == b.shape and torch.equal(a, b))
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray): return bool(a.shape == b.shape and np.array_equal(a, b))
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b

def epoch_lines(path):
    out = {}
    for ln in open(path, errors="replace"):
        m = re.search(r"INFO: Epoch (\d+): head: \S+ loss=([0-9.eE+-]+), RMSE_E_per_atom=\s*([0-9.]+) meV, RMSE_F=\s*([0-9.]+)", ln)
        if m: out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out

sX, sY, sX2, lX, lY, lX2 = sys.argv[1:7]
X, Y, X2 = load(sX), load(sY), load(sX2)
print(f"next_epoch  X {X['next_epoch']}  Y {Y['next_epoch']}  X2 {X2['next_epoch']}   world_size {X['world_size']}")
for label, A, Bs in (("X vs Y (resumed)", X, Y), ("X vs X2 (floor)", X, X2)):
    print("\n" + "=" * 78); print(label)
    pm = [(A["model"][k], Bs["model"][k]) for k in A["model"] if torch.is_tensor(A["model"][k]) and A["model"][k].is_floating_point()]
    mx, rd, rv = group_stats(pm); print(f"  raw parameters        max|d| {mx:.3e}   rms d {rd:.3e}   rms value {rv:.3e}   ({len(pm)} tensors)")
    if A.get("ema") is not None:
        em = list(zip(A["ema"]["shadow_params"], Bs["ema"]["shadow_params"]))
        mx, rd, rv = group_stats(em); print(f"  EMA shadow params     max|d| {mx:.3e}   rms d {rd:.3e}   rms value {rv:.3e}   num_updates {A['ema']['num_updates']} / {Bs['ema']['num_updates']}   decay {A['ema']['decay']}")
    oa, ob = A["optimizer"]["state"], Bs["optimizer"]["state"]
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        prs = [(oa[i][key], ob[i][key]) for i in oa if key in oa[i] and i in ob and key in ob[i]]
        if prs:
            mx, rd, rv = group_stats(prs); print(f"  Adam {key:<15} max|d| {mx:.3e}   rms d {rd:.3e}   rms value {rv:.3e}")
    stA = sorted({int(oa[i]["step"]) if not torch.is_tensor(oa[i]["step"]) else int(oa[i]["step"].item()) for i in oa})
    stB = sorted({int(ob[i]["step"]) if not torch.is_tensor(ob[i]["step"]) else int(ob[i]["step"].item()) for i in ob})
    print(f"  Adam step counts      {stA} / {stB}   {'identical' if stA == stB else 'DIFFERENT'}")
    la, lb = A["lr_scheduler"], Bs["lr_scheduler"]
    keys = sorted(set(la) | set(lb))
    diff = [k for k in keys if not same(la.get(k), lb.get(k))]
    print(f"  scheduler             {'identical' if not diff else 'differs in ' + str(diff)}   " + "  ".join(f"{k}={la.get(k)}" for k in ("last_epoch", "num_bad_epochs", "best", "cooldown_counter", "_last_lr") if k in la))
    if diff:
        for k in diff: print(f"      {k}: {la.get(k)}  vs  {lb.get(k)}")
    print(f"  lowest_loss           {A['lowest_loss']:.10f} / {Bs['lowest_loss']:.10f}   valid_loss {A['valid_loss']:.10f} / {Bs['valid_loss']:.10f}   patience {A['patience_counter']} / {Bs['patience_counter']}   keep_last {A['keep_last']} / {Bs['keep_last']}")
    for r, (ra, rb) in enumerate(zip(A["rng"], Bs["rng"])):
        flags = []
        for k in ("python", "numpy", "torch_cpu", "torch_cuda", "loader_generator", "valid_generators", "density_3d_rng", "solvent3d_rng"):
            flags.append(f"{k}={'same' if same(ra.get(k), rb.get(k)) else 'DIFF'}")
        print(f"  rank {r} RNG streams    " + "  ".join(flags))
print("\n" + "=" * 78); print("validation lines (loss, RMSE_E meV/atom, RMSE_F meV/A)")
eX, eY, eX2 = epoch_lines(lX), epoch_lines(lY), epoch_lines(lX2)
for ep in sorted(set(eX) | set(eY) | set(eX2)):
    print(f"  epoch {ep}:  X {eX.get(ep)}   Y {eY.get(ep)}   X2 {eX2.get(ep)}")
print("\nREADING: X-Y must sit at the X-X2 floor for every quantity and every RNG stream must be identical;")
print("a quantity where X-Y stands above X-X2 names a state the segment file does not carry.")
print("DONE")
