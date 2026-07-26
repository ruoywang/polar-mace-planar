"""Local field factor: value parity vs torch_pb + IFT backward vs autograd.

1. values: our forward vs tp._local_field_factor on the pack's real field
   magnitudes (must agree to the loop tolerance, 1e-10 rel)
2. gradient: our IFT backward vs autograd through the unrolled tp loop,
   random cotangent dot products + per-point comparison
3. closure end-to-end: closure gradients (all five outputs, random
   cotangents) with our operator vs with autograd-through-loop
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "polar-mace-planar"))
sys.path.insert(0, "/work/08384/tg876840/ls6/repos/cep-dip-python-pb")

from pure_python import torch_pb as tp  # noqa: E402
from pure_python.pb import derived_params  # noqa: E402
from mace.modules.pb1d_localfield import local_field_factor  # noqa: E402

import json
with open(str(HERE.parent / "exp_pb1d_smoke" / "cal1_train.json")) as f:
    cfg = json.load(f)
params = derived_params(cfg.get("solvation", cfg))

torch.manual_seed(0)
ok = True

# representative field magnitudes: zero, tiny, moderate, near-saturation
emag = torch.cat([
    torch.zeros(50, dtype=torch.float64),
    10.0 ** torch.linspace(-6, 1.2, 4000, dtype=torch.float64),
    torch.rand(4000, dtype=torch.float64) * 10.0,
])

# 1) value parity
f_ref = tp._local_field_factor(emag, params)
f_new = local_field_factor(emag, params)
dv = float((f_new - f_ref).abs().max())
print(f"value: max|diff| {dv:.3e} (loop tol 1e-10)  {'OK' if dv < 1e-9 else 'BAD'}")
ok &= dv < 1e-9

# 2) gradient vs autograd through the unrolled loop
e1 = emag.clone().requires_grad_(True)
e2 = emag.clone().requires_grad_(True)
cot = torch.randn_like(emag)
(tp._local_field_factor(e1, params) * cot).sum().backward()
(local_field_factor(e2, params) * cot).sum().backward()
g_ref, g_new = e1.grad, e2.grad
scale = g_ref.abs().max()
dg = float((g_new - g_ref).abs().max() / scale)
print(f"grad : max rel diff {dg:.3e} (vs unrolled autograd)  {'OK' if dg < 1e-6 else 'BAD'}")
ok &= dg < 1e-6

# 3) end-to-end closure gradients on the real pack fields
import importlib
import mace.modules.pb1d_closure as clo_mod
pack = np.load(str(HERE.parent / "exp_prior_profiles" / "closure_parity_pack.npz"))
grid = tp.TorchGrid(pack["cell"].astype(float), tuple(int(x) for x in pack["shape"]),
                    device="cpu", dtype=torch.float64, rspec=True)
ne = torch.tensor(pack["n_e_density"], dtype=torch.float64)
cv = torch.tensor(pack["cvhar3"], dtype=torch.float64)

def closure_grads(use_new):
    if use_new:
        importlib.reload(clo_mod)
    else:
        importlib.reload(clo_mod)
        clo_mod.local_field_factor = None  # force tp path
        src = clo_mod.response_a3.__code__  # noqa: F841
    ne_v = ne.clone().requires_grad_(True)
    cv_v = cv.clone().requires_grad_(True)
    if not use_new:
        # monkeypatch: restore the original tp call
        def response_a3_ref(emag, s_diel, prm, tpm):
            f_loc = tpm._local_field_factor(emag, prm)
            y = float(prm["PBETA"]) * emag * f_loc
            g = clo_mod.langevin_g_torch(y) if bool(prm["LNLDIEL"]) else torch.ones_like(y)
            poe = (float(prm["alpha0_rot"]) / clo_mod.EDEPS * g
                   + float(prm["alpha_pol"]) / clo_mod.EDEPS)
            return f_loc * float(prm["N_MOL"]) * s_diel * poe
        clo_mod.response_a3 = response_a3_ref
    out = clo_mod.closure_from_fields(ne_v, cv_v, grid, params, tp)
    torch.manual_seed(1)
    s = sum((out[k] * torch.randn_like(out[k])).sum()
            for k in ("A_scr", "S_ion_z", "prior", "w_env", "u"))
    s.backward()
    return ne_v.grad.clone(), cv_v.grad.clone()

g_ne_ref, g_cv_ref = closure_grads(False)
g_ne_new, g_cv_new = closure_grads(True)
for name, a, b in (("d/d n_e", g_ne_ref, g_ne_new), ("d/d cvhar", g_cv_ref, g_cv_new)):
    rel = float((a - b).abs().max() / a.abs().max())
    print(f"closure {name}: max rel diff {rel:.3e}  {'OK' if rel < 1e-6 else 'BAD'}")
    ok &= rel < 1e-6

print("LOCALFIELD:", "PASS" if ok else "FAIL")
