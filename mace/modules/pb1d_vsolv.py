"""Parent (VASPsol++) solvent effective potential at the stage-1 field, as a charge-recursion
input (user design 2026-09-21; audits/vsolv_eval.py is the standalone check of these formulas).

    v_new(r) = d(A_cav + A_diel + A_ion)/dn_e(r) |_{phi = phi_stage1}

  A_cav  = TAU * int |grad S_cav[n_e]| dV          (pb1d_backend's regularised expression, doubled Stern mask)
  A_diel = N_MOL * int S_diel[n_e] lambda_diel(E_loc) dV,  lambda_diel = lambda_rot + lambda_pol + lambda_sic
  A_ion  = n_max * int S_ion[n_e] lambda_ion(phi) dV,      lambda_ion = -(1/beta) ln(1 - theta + theta cosh(z beta phi))

The partial is taken with phi an INDEPENDENT input (g(n, phi) = d_n A(n, phi)); the caller passes the
live stage-1 phi so that forces and loss gradients keep the solve dependence (create_graph=True),
and the whole evaluation is checkpointed (recomputed in the backward) because the double-backward
graph of the 3-D cavity construction peaks at ~21 GiB per frame if stored.

Projection: v_new is a POTENTIAL (energy per electron, electron-PE convention). The electrostatic
channel feeds the physics-sign potential of the solvent charge (potential energy of a positive unit
charge) smoothed with the receiver Gaussians and its gradient at the atoms; the equivalent
physics-sign potential of v_new is -v_new (an added electron density dn_e is a physics charge
-dn_e). VSOLV_SIGN carries that; the sampler test (tools1d/test_vsolv_wiring.py) fixes it.
No Poisson solve, no de-meaning: v_new vanishes away from the cavity shell.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from .pb1d_solver import EDEPS

VSOLV_SIGN = -1.0   # physics-sign potential equivalent of an energy-per-electron functional derivative


def lambda_diel(emag_loc: torch.Tensor, params: dict, tp) -> torch.Tensor:
    """Parent molecular dielectric free energy at the LOCAL field magnitude (eV)."""
    invb = float(params["invBETA"]); pbeta = float(params["PBETA"])
    a0, ap, isic = float(params["alpha0_rot"]), float(params["alpha_pol"]), float(params["invalpha_sic"])
    y = pbeta * emag_loc
    if bool(params["LNLDIEL"]):
        small = y < 1.0e-3
        ys = torch.where(small, torch.ones_like(y), y)
        ln_direct = torch.log(torch.sinh(torch.clamp(ys, max=700.0)) / ys)
        ln_series = y * y / 6.0 - y ** 4 / 180.0
        lam_rot = -invb * torch.where(small, ln_series, ln_direct)
        g = tp._dielectric_g(y, params)
    else:
        lam_rot = -invb * (y * y / 6.0)
        g = torch.ones_like(y)
    lam_pol = -0.5 * ap * emag_loc ** 2 / EDEPS
    lam_sic = 0.5 * isic * ((a0 * g + ap) * emag_loc) ** 2 / EDEPS
    return lam_rot + lam_pol + lam_sic


def lambda_ion(phi: torch.Tensor, params: dict) -> torch.Tensor:
    invb = float(params["invBETA"]); zb = float(params["ZBETA"]); th = float(params["theta_b"])
    if not (bool(params["LION"]) and bool(params["LNLION"]) and th > 0.0):
        raise NotImplementedError("v_ion is written for the lattice-gas ionic model (LNLION, theta_b > 0)")
    x = torch.clamp(zb * phi, -100.0, 100.0)
    return -invb * torch.log(1.0 - th + th * torch.cosh(x))


def stern_mask(grid, params: dict, tp) -> Optional[torch.Tensor]:
    if bool(params["LVAC"]) and float(params["SOL_Z1"]) > float(params["SOL_Z0"]):
        return tp._smooth_box(grid, float(params["SOL_Z0"]) + float(params["D_STERN"]),
                              float(params["SOL_Z1"]) - float(params["D_STERN"]), float(params["SOL_SIGMA"]))
    return None


def free_energies(n_e: torch.Tensor, phi3: torch.Tensor, grid, params: dict, tp, w_b, eps_area: float
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A_cav, A_diel, A_ion (eV) at fixed phi3; n_e (e/A^3 on the PB grid) is the differentiated input."""
    s_ion, s_diel, s_cav = tp.create_cavity_torch(n_e, grid, params)
    dV = grid.volume / float(grid.ngrid)
    m = stern_mask(grid, params, tp)
    if m is not None:
        s_cav = s_cav * m
    gx, gy, gz, _ = grid.grad_from_recip(grid.fft(s_cav))
    a_cav = float(params["TAU"]) * torch.sqrt(gx * gx + gy * gy + gz * gz + eps_area).sum() * dV
    fq = tp._field_quantities(phi3, s_ion, s_diel, grid, params, w_b)   # emag = |E_loc| (f_loc included)
    a_diel = float(params["N_MOL"]) * (s_diel * lambda_diel(fq["emag"], params, tp)).sum() * dV
    a_ion = float(params["n_max"]) * (s_ion * lambda_ion(phi3, params)).sum() * dV
    return a_cav, a_diel, a_ion


def fourier_resample_1d(f: torch.Tensor, n_new: int) -> torch.Tensor:
    n = f.shape[0]
    F = torch.fft.rfft(f) / n
    G = torch.zeros(n_new // 2 + 1, dtype=F.dtype, device=f.device)
    m = min(F.shape[0], G.shape[0])
    G[:m] = F[:m]
    return torch.fft.irfft(G * n_new, n=n_new)


def periodic_trilinear(field: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
    """Periodic trilinear interpolation of field [nx, ny, nz] at fractional positions frac [n, 3];
    differentiable in frac (piecewise-linear weights)."""
    nx, ny, nz = field.shape
    dims = torch.tensor([nx, ny, nz], dtype=frac.dtype, device=frac.device)
    u = torch.remainder(frac, 1.0) * dims
    i0 = torch.floor(u)
    w1 = u - i0
    w0 = 1.0 - w1
    i0 = i0.to(torch.long)
    out = torch.zeros(frac.shape[0], dtype=field.dtype, device=field.device)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix = torch.remainder(i0[:, 0] + dx, nx)
                iy = torch.remainder(i0[:, 1] + dy, ny)
                iz = torch.remainder(i0[:, 2] + dz, nz)
                w = ((w1[:, 0] if dx else w0[:, 0]) * (w1[:, 1] if dy else w0[:, 1]) * (w1[:, 2] if dz else w0[:, 2]))
                out = out + w * field[ix, iy, iz]
    return out


def smoothed_value_and_gradient(v: torch.Tensor, grid, frac: torch.Tensor, sigma: float
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Receiver-Gaussian-smoothed potential and its gradient at the atoms (spectral smoothing and
    derivative, periodic trilinear sampling)."""
    v_g = grid.fft(v) * torch.exp(-0.5 * (2.0 * math.pi) ** 2 * grid.gsq * float(sigma) ** 2)
    v_s = grid.ifft_real(v_g)
    gx, gy, gz, _ = grid.grad_from_recip(v_g)
    val = periodic_trilinear(v_s, frac)
    grad = torch.stack([periodic_trilinear(gx, frac), periodic_trilinear(gy, frac), periodic_trilinear(gz, frac)], dim=1)
    return val, grad


def _vsolv_core(n_e: torch.Tensor, phi_z_solver: torch.Tensor, pos_frac: torch.Tensor, grid, params: dict,
                tp, sigma_b: float, eps_area: float, sigmas: List[float]) -> torch.Tensor:
    """node fields [n_atoms, n_sigma, 4] = [V, dV/dx, dV/dy, dV/dz] with V = VSOLV_SIGN * smoothed v_new."""
    nz = int(grid.shape[2])
    phi_z = fourier_resample_1d(phi_z_solver, nz)
    phi3 = phi_z[None, None, :].expand(*grid.shape)
    w_b = tp._normalized_gaussian_kernel_g(grid, sigma_b)
    if not n_e.requires_grad:
        n_e = n_e.detach().requires_grad_(True)
    a_cav, a_diel, a_ion = free_energies(n_e, phi3, grid, params, tp, w_b, eps_area)
    (gr,) = torch.autograd.grad(a_cav + a_diel + a_ion, n_e, create_graph=True)
    v_new = gr / (grid.volume / float(grid.ngrid))
    v_phys = VSOLV_SIGN * v_new
    rows = []
    for s in sigmas:
        val, grad = smoothed_value_and_gradient(v_phys, grid, pos_frac, float(s))
        rows.append(torch.cat([val[:, None], grad], dim=1))
    return torch.stack(rows, dim=1)


def vsolv_node_fields(n_e: torch.Tensor, phi_z_solver: torch.Tensor, pos_frac: torch.Tensor, grid, params: dict,
                      tp, sigma_b: float, eps_area: float, sigmas: List[float], checkpoint: bool = True
                      ) -> torch.Tensor:
    """Checkpointed wrapper: the training backward recomputes the double-backward graph instead of
    storing it (measured 21-22 GiB peak per frame when stored)."""
    if checkpoint and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint as _ckpt

        def _fn(n_e_, phi_, pf_):
            return _vsolv_core(n_e_, phi_, pf_, grid, params, tp, sigma_b, eps_area, sigmas)
        return _ckpt(_fn, n_e, phi_z_solver, pos_frac, use_reentrant=False)
    return _vsolv_core(n_e, phi_z_solver, pos_frac, grid, params, tp, sigma_b, eps_area, sigmas)
