"""Head unit tests + c_max calibration under the PHYSICAL envelope.

1) representation: lstsq-fit the true cal_18 residual (prior - exact, from
   the solver pack) in the head's exact basis (physical w_env, u, K=8,
   sigma_z blur) -> fit rms must beat the 1e-3 training bar; max|c| sets
   c_max (with ~2x headroom).
2) unit tests: zero-init => dP == 0; coefficients bounded; gradients flow
   to node features, positions and MLP weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "polar-mace-planar"))
from mace.modules.pb1d_head import (  # noqa: E402
    PB1DResidualHead, gaussian_blur_periodic,
)

sp = np.load(HERE.parent / "exp_prior_profiles" / "solver_parity_pack.npz")
cp = np.load(HERE.parent / "exp_prior_profiles" / "closure_parity_pack.npz")
dt = torch.float64
lz = float(sp["length_z"])
nz = int(sp["nz"])

y = torch.tensor(sp["p_prior"] - sp["p_exact"], dtype=dt)
w_ref = np.clip(cp["var_chi"], 0, None); w_ref = w_ref / w_ref.max()
w_env = torch.tensor(w_ref, dtype=dt)
u = torch.cumsum(w_env, dim=0); u = u / u[-1]

K = 8
uk = (torch.arange(K, dtype=dt) + 0.5) / K
basis = torch.exp(-0.5 * ((u[:, None] - uk[None, :]) / (1.0 / K)) ** 2)
M = torch.stack([gaussian_blur_periodic(w_env * basis[:, k], 0.2, lz) for k in range(K)], dim=1)
c, *_ = torch.linalg.lstsq(M, y[:, None])
c = c.squeeze(-1)
resid = y - M @ c
print(f"representation (physical envelope): fit rms {float(resid.pow(2).mean().sqrt()):.3e} "
      f"(target rms {float(y.pow(2).mean().sqrt()):.3e}); max|c| {float(c.abs().max()):.4f}")
c_max_needed = float(c.abs().max())
c_max = 0.25  # fitted max ~0.115 with the normalized var_chi envelope
print(f"-> c_max calibration: fitted max |c| = {c_max_needed:.3f}, choose c_max = {c_max}")
ok = float(resid.pow(2).mean().sqrt()) < 1.0e-3

# ---- head unit tests ----
torch.manual_seed(0)
n_atoms, d = 207, 64
head = PB1DResidualHead(feat_dim=d, c_max=c_max).to(dt)
feats = torch.randn(n_atoms, d, dtype=dt, requires_grad=True)
z_at = (torch.rand(n_atoms, dtype=dt) * lz).requires_grad_(True)
z_grid = torch.arange(nz, dtype=dt) * lz / nz
prior_t = torch.tensor(sp["p_prior"], dtype=dt)
q = torch.tensor(-1.0, dtype=dt)

coef = head.coefficients(feats, z_at, w_env, u, prior_t, z_grid, lz, q)
dp = head.delta_p(coef, w_env, u, lz)
assert float(dp.abs().max()) == 0.0, "zero-init must give dP == 0"
print("zero-init: dP == 0  OK")

with torch.no_grad():
    for p in head.mlp.parameters():
        p.normal_(0, 0.5)
coef = head.coefficients(feats, z_at, w_env, u, prior_t, z_grid, lz, q)
assert float(coef.abs().max()) <= c_max + 1e-12, "coefficients must be bounded"
dp = head.delta_p(coef, w_env, u, lz)
loss = dp.pow(2).sum()
loss.backward()
gf = float(feats.grad.abs().sum())
gz = float(z_at.grad.abs().sum())
gw = sum(float(p.grad.abs().sum()) for p in head.mlp.parameters())
print(f"bounded coeffs OK (max {float(coef.abs().max()):.3f} <= {c_max}); "
      f"grads: feats {gf:.2e}, z_atoms {gz:.2e}, mlp {gw:.2e}")
ok = ok and gf > 0 and gz > 0 and gw > 0
# support: dP must vanish where the envelope does (up to the sigma_z blur)
outside = (w_env == 0)
if outside.any():
    leak = float(dp[outside].abs().max()) / max(float(dp.abs().max()), 1e-30)
    print(f"support leak outside envelope (rel): {leak:.2e}")
    ok = ok and leak < 1e-6

print("HEAD:", "PASS" if ok else "FAIL")
