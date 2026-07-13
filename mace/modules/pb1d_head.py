"""The P_z residual head (v1, free-trained).

Design (published note: polar-mace-planar docs/v1-head-design.html):

  dP(z) = G_{sigma_z} * [ w(z) * sum_k c_k B_k(u(z)) ],   K = 8
  c_k   = c_max * tanh( MLP([F_k || F_glob || s_k]) )
  F_k   = sum_i omega_ik f_i / sum_i omega_ik,
  omega_ik = w(z_i) * exp(-(u(z_i) - u_k)^2 / (2 sigma_f^2)), sigma_f = 1.5/K
  s_k   = [q_tot/10, u_k, prior(u_k)/prior_scale]

One shared MLP for all zones; final layer zero-initialized so dP == 0 at
step 0 (the model starts exactly at the prior). Everything differentiable:
w/u/prior enter through interpolation, atom weights through smooth omegas.

Operates per structure (profiles are per-graph); the wiring loops graphs.
"""
from __future__ import annotations

import math
from typing import Dict

import torch

TPI = 2.0 * math.pi


def interp_profile(profile: torch.Tensor, z_grid: torch.Tensor, z_query: torch.Tensor,
                   lz: float) -> torch.Tensor:
    """Periodic linear interpolation of a profile at query heights (differentiable
    in both the profile values and the query positions)."""
    nz = profile.shape[0]
    dz = lz / nz
    x = torch.remainder(z_query, lz) / dz
    i0 = torch.floor(x).long() % nz
    i1 = (i0 + 1) % nz
    frac = x - torch.floor(x)
    return profile[i0] * (1.0 - frac) + profile[i1] * frac


def value_at_u(profile: torch.Tensor, u: torch.Tensor, u_query: torch.Tensor) -> torch.Tensor:
    """profile value where the (monotone) progress coordinate crosses u_query."""
    idx = torch.searchsorted(u.detach().contiguous(), u_query.detach().contiguous())
    idx = torch.clamp(idx, 1, u.shape[0] - 1)
    u0, u1 = u[idx - 1], u[idx]
    f = (u_query - u0) / torch.clamp(u1 - u0, min=1.0e-30)
    f = torch.clamp(f, 0.0, 1.0)
    return profile[idx - 1] * (1.0 - f) + profile[idx] * f


def gaussian_blur_periodic(profile: torch.Tensor, sigma: float, lz: float) -> torch.Tensor:
    """G_sigma * profile via the spectral transfer function (periodic, exact)."""
    nz = profile.shape[0]
    g = torch.arange(nz // 2 + 1, dtype=profile.dtype, device=profile.device) / lz
    transfer = torch.exp(-0.5 * (sigma * TPI * g) ** 2)
    return torch.fft.irfft(torch.fft.rfft(profile) * transfer, n=nz)


class PB1DResidualHead(torch.nn.Module):
    def __init__(
        self,
        feat_dim: int,
        n_zones: int = 8,
        hidden_dim: int = 128,
        c_max: float = 0.25,
        sigma_z: float = 0.2,
        prior_scale: float = 0.3,
        sigma_f_factor: float = 1.5,
    ) -> None:
        super().__init__()
        self.n_zones = int(n_zones)
        self.c_max = float(c_max)
        self.sigma_z = float(sigma_z)
        self.prior_scale = float(prior_scale)
        self.sigma_f = sigma_f_factor / float(n_zones)
        uk = (torch.arange(n_zones, dtype=torch.float64) + 0.5) / n_zones
        self.register_buffer("u_k", uk)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * feat_dim + 3, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def coefficients(
        self,
        node_feats: torch.Tensor,   # [n_atoms, feat_dim]
        z_atoms: torch.Tensor,      # [n_atoms] cartesian z
        w_env: torch.Tensor,        # [nz] envelope on the closure grid
        u: torch.Tensor,            # [nz] progress coordinate
        prior: torch.Tensor,        # [nz]
        z_grid: torch.Tensor,       # [nz]
        lz: float,
        q_tot: torch.Tensor,        # scalar
    ) -> torch.Tensor:
        dt = w_env.dtype
        feats = node_feats.to(dt)
        w_at = interp_profile(w_env, z_grid, z_atoms.to(dt), lz)
        u_at = interp_profile(u, z_grid, z_atoms.to(dt), lz)
        u_k = self.u_k.to(device=feats.device, dtype=dt)
        omega = w_at[:, None] * torch.exp(
            -0.5 * ((u_at[:, None] - u_k[None, :]) / self.sigma_f) ** 2
        )  # [n_atoms, K]
        denom = torch.clamp(omega.sum(dim=0), min=1.0e-30)
        f_k = (omega.transpose(0, 1) @ feats) / denom[:, None]      # [K, d]
        f_glob = feats.mean(dim=0, keepdim=True).expand(self.n_zones, -1)
        prior_at_k = value_at_u(prior, u, u_k) / self.prior_scale
        scalars = torch.stack(
            [ (q_tot.to(dt) / 10.0).expand(self.n_zones), u_k, prior_at_k ], dim=-1
        )
        x = torch.cat([f_k, f_glob, scalars], dim=-1).to(self.mlp[0].weight.dtype)
        raw = self.mlp(x).squeeze(-1).to(dt)
        return self.c_max * torch.tanh(raw)

    def delta_p(
        self,
        c: torch.Tensor,        # [K]
        w_env: torch.Tensor,    # [nz_solve] envelope on the SOLVE grid
        u: torch.Tensor,        # [nz_solve]
        lz: float,
    ) -> torch.Tensor:
        u_k = self.u_k.to(device=c.device, dtype=w_env.dtype)
        sig = 1.0 / self.n_zones
        basis = torch.exp(-0.5 * ((u[:, None] - u_k[None, :]) / sig) ** 2)
        shape = basis @ c.to(w_env.dtype)
        return gaussian_blur_periodic(w_env * shape, self.sigma_z, lz)

    def forward(self, **kw) -> torch.Tensor:  # pragma: no cover - thin wrapper
        c = self.coefficients(
            kw["node_feats"], kw["z_atoms"], kw["w_env_closure"], kw["u_closure"],
            kw["prior_closure"], kw["z_grid_closure"], kw["lz"], kw["q_tot"],
        )
        return self.delta_p(c, kw["w_env_solve"], kw["u_solve"], kw["lz"])
