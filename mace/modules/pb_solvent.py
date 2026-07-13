"""Profile-resampling utility shared by the solvent modes.

The 3-D nonlinear PB solver that used to live here (PBTorchBackend /
PBPlanarSolvent, validated against cep-dip-python-pb in the 3-D era) was
retired in favour of the 1-D closure route: see pb1d_backend / pb1d_solver /
pb1d_closure / pb1d_head, which reuse its spectral density assembly and
baseline-cache conventions.
"""

from __future__ import annotations

import torch


def resample_profile_periodic_torch(
    profile: torch.Tensor,
    height: float,
    num_grid: int,
    offset_half: bool,
) -> torch.Tensor:
    """Periodic linear resampling of a uniform-grid profile (torch, device)."""
    nz = profile.shape[0]
    j = torch.arange(num_grid, device=profile.device, dtype=profile.dtype)
    u = (j + (0.5 if offset_half else 0.0)) * (nz / float(num_grid))
    i0 = torch.floor(u).to(torch.long) % nz
    i1 = (i0 + 1) % nz
    w = (u - torch.floor(u)).to(profile.dtype)
    return profile[i0] * (1.0 - w) + profile[i1] * w
