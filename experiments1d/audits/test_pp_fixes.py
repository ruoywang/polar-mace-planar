"""Micro regression tests for the two user-reported issues (2026-09-07):
1. constant cavity -> bound envelope must be ZERO, not all-ones
   (old: sqrt(0+1e-30)/max = 1e-15/1e-15 = 1 everywhere);
2. supervision/energy projection consistency: a pure plane mode is
   annihilated on the energy grid; the loss must see zero at OFF-PLANE
   points too (old direct-GTO-minus-interpolated-r path does not).
CPU, synthetic fields, seconds. cwd anywhere."""
# Paths come from env vars with the LS6 values as defaults, so this
# runs unchanged on Lonestar6 and on a machine that holds the same
# payload elsewhere: KIT_PB_REPO, KIT_MACE_REPO, KIT_DFT,
# KIT_PB_CONFIG. Previously hardcoded, which made every audit
# script non-portable (found by the workstation session 2026-09-09).
import os
import sys

import torch

sys.path.insert(0, os.environ.get("KIT_MACE_REPO", "/scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/pmp-s3denergy"))
from mace.modules.solvent3d import (
    _grad_mag_periodic,
    _interp3_periodic,
    normalized_gradient_envelope,
)

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
cell = torch.diag(torch.tensor([10.0, 10.0, 25.0]))
nx, ny, nz = 24, 24, 60

# ---- 1. envelope guard ----------------------------------------------------
const = torch.full((nx, ny, nz), 0.7)
g_old = _grad_mag_periodic(const, cell)
env_old = g_old / torch.clamp(g_old.max(), min=1.0e-30)
env_new = normalized_gradient_envelope(const, cell)
print(f"[1] constant cavity: old env min/max {float(env_old.min()):.3f}/"
      f"{float(env_old.max()):.3f} (bug: all-ones)  new env max "
      f"{float(env_new.abs().max()):.3e} (want 0)")
assert float(env_old.min()) > 0.99, "old-path repro failed"
assert float(env_new.abs().max()) == 0.0, "guard failed"
# real interface: guard must be a no-op
zc = torch.arange(nz) / nz * cell[2, 2]
step = torch.tanh((zc - 8.0) / 1.0)[None, None, :].expand(nx, ny, nz).contiguous()
g_r = _grad_mag_periodic(step, cell)
env_r_old = g_r / torch.clamp(g_r.max(), min=1.0e-30)
env_r_new = normalized_gradient_envelope(step, cell)
d = float((env_r_old - env_r_new).abs().max())
print(f"[1] real interface: |old-new| max {d:.3e} (want 0), env max "
      f"{float(env_r_new.max()):.3f}")
assert d == 0.0 and abs(float(env_r_new.max()) - 1.0) < 1e-12

# ---- 2. plane-mode projection consistency ---------------------------------
# positive interface-like envelope, z-dependent with xy texture
xg = torch.arange(nx) / nx
yg = torch.arange(ny) / ny
zg = torch.arange(nz) / nz
env = (torch.exp(-((zg[None, None, :] - 0.35) / 0.06) ** 2)
       * (1.0 + 0.3 * torch.sin(2 * torch.pi * xg)[:, None, None]
          * torch.cos(2 * torch.pi * yg)[None, :, None]) + 1e-6)


def m_of_z(zfrac):
    return (torch.sin(2 * torch.pi * zfrac)
            + 0.3 * torch.cos(4 * torch.pi * zfrac))


raw = env * m_of_z(zg)[None, None, :]
r = raw.mean(dim=(0, 1)) / torch.clamp(env.mean(dim=(0, 1)), min=1e-12)
d_grid = raw - r[None, None, :] * env
print(f"[2] plane mode on grid: max|projected| {float(d_grid.abs().max()):.3e} "
      f"(annihilated on the energy grid)")
assert float(d_grid.abs().max()) < 1e-13

# off-plane sample points (deliberately mid-plane in z)
P = 500
frac = torch.rand(P, 3)
frac[:, 2] = (torch.randint(0, nz, (P,)) + 0.5) / nz
env_p = _interp3_periodic(env, frac)
# OLD path: continuum density at the point minus interpolated r(z) * env
t = frac[:, 2] * nz
k0 = torch.floor(t).long() % nz
w = t - torch.floor(t)
r_interp = r[k0] * (1 - w) + r[(k0 + 1) % nz] * w
old_resid = env_p * m_of_z(frac[:, 2]) - r_interp * env_p
# NEW path: interpolate the projected grid field itself
new_resid = _interp3_periodic(d_grid, frac)
print(f"[2] off-plane points: OLD residual max {float(old_resid.abs().max()):.3e} "
      f"(the inconsistency)  NEW residual max {float(new_resid.abs().max()):.3e} "
      f"(want ~0); envelope scale {float(env_p.max()):.3f}, "
      f"m scale {float(m_of_z(frac[:, 2]).abs().max()):.3f}")
assert float(old_resid.abs().max()) > 1e-4, "old-path repro failed"
assert float(new_resid.abs().max()) < 1e-13, "unified path not exact"

# general (non-plane) mode: interpolation is exact at grid nodes
m3 = env * torch.randn(nx, ny, nz).mul(0.01).add(
    m_of_z(zg)[None, None, :] * 0.5)
r3 = m3.mean(dim=(0, 1)) / torch.clamp(env.mean(dim=(0, 1)), min=1e-12)
d3 = m3 - r3[None, None, :] * env
ii = torch.randint(0, nx, (50,)); jj = torch.randint(0, ny, (50,))
kk = torch.randint(0, nz, (50,))
node_frac = torch.stack([ii / nx, jj / ny, kk / nz], dim=1)
diff = float((_interp3_periodic(d3, node_frac) - d3[ii, jj, kk]).abs().max())
print(f"[2] general mode at grid nodes: |interp - grid| max {diff:.3e} (want 0)")
assert diff < 1e-14
# gradient flows through the interpolation (loss-path liveness)
d3l = d3.clone().requires_grad_(True)
_interp3_periodic(d3l, node_frac).sum().backward()
print(f"[2] grad through interp: nonzero entries "
      f"{int((d3l.grad != 0).sum())} (want >0)")
assert int((d3l.grad != 0).sum()) > 0
print("ALL PASS")
