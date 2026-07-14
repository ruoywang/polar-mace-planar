"""1-D closure quantities from live 3-D fields (torch, differentiable).

Given the current electron density and the solute potential on the PB grid,
produce everything the pb1d route needs per forward:

  A_scr(z)   — mean response, saturation evaluated at the screened vacuum
               field (the published consistent pairing)
  S_ion(z)   — plane-averaged ion cavity
  prior(z)   — screened-vacuum P_off prior (its own covariance)
  w(z)       — envelope var_xy[chi], normalized to max 1 per structure
               (the residual's permitted region). MEASURED on cal_18
               (2026-07-13): plain var_chi fits the true residual to
               4.6e-4 while the Cauchy-Schwarz product
               sqrt(var_chi)*sqrt(var_E) degrades 4x to 2.0e-3 — the
               field-fluctuation factor distorts the shape. Amplitude
               information flows through the q_tot scalar instead.
  u(z)       — interface progress coordinate cumsum(w)/sum(w)

The heavy 3-D machinery (TorchGrid, cavity, kernels) is reused from
cep-dip-python-pb's `pure_python.torch_pb` (GPU-validated against numpy to
~1e-12 in the 3-D era). This module contains no 3-D solve.
"""
from __future__ import annotations

import math
from typing import Dict

import torch

TPI = 2.0 * math.pi
RYTOEV = 13.605826
AUTOA = 0.529177249
EDEPS = 4.0 * math.pi * 2.0 * RYTOEV * AUTOA


def langevin_g_torch(y: torch.Tensor) -> torch.Tensor:
    """3(coth y - 1/y)/y with the numpy port's small/large guards."""
    yc = torch.clamp(y, min=1.0e-12)
    mid = 3.0 * (1.0 / torch.tanh(yc) - 1.0 / yc) / yc
    large = 3.0 * (1.0 - 1.0 / yc) / yc
    out = torch.where(y > 100.0, large, mid)
    return torch.where(y < 2.0e-4, torch.ones_like(y), out)


def response_a3(emag: torch.Tensor, s_diel: torch.Tensor, params: dict, tp) -> torch.Tensor:
    """A(r) = f_loc * N_MOL * s_diel * (alpha0_rot*g(y) + alpha_pol)/EDEPS."""
    from .pb1d_localfield import local_field_factor
    f_loc = local_field_factor(emag, params)
    y = float(params["PBETA"]) * emag * f_loc
    g = langevin_g_torch(y) if bool(params["LNLDIEL"]) else torch.ones_like(y)
    poe = float(params["alpha0_rot"]) / EDEPS * g + float(params["alpha_pol"]) / EDEPS
    return f_loc * float(params["N_MOL"]) * s_diel * poe


def plane_mean(a: torch.Tensor) -> torch.Tensor:
    return a.mean(dim=(0, 1))


def plane_var(a: torch.Tensor) -> torch.Tensor:
    m = a.mean(dim=(0, 1))
    return (a * a).mean(dim=(0, 1)) - m * m


def closure_from_fields(
    n_e_density: torch.Tensor,
    phi_sol: torch.Tensor,
    grid,
    params: dict,
    tp,
) -> Dict[str, torch.Tensor]:
    """All 1-D closure quantities from live 3-D fields. Differentiable.

    n_e_density: electron density (e/A^3, >=0) on the PB grid [nx,ny,nz]
    phi_sol:     solute potential (eV) on the same grid
    grid:        torch_pb.TorchGrid
    tp:          the torch_pb module
    """
    s_ion3, s_diel3, _ = tp.create_cavity_torch(n_e_density, grid, params)

    sigma_b = float(params["R_B"]) if float(params["R_B"]) > 0.0 else float(params["A_K"])
    cached = getattr(grid, "_pb1d_wb_kernel", None)
    if cached is not None and cached[0] == sigma_b:
        w_b = cached[1]
    else:
        w_b = tp._normalized_gaussian_kernel_g(grid, sigma_b)
        grid._pb1d_wb_kernel = (sigma_b, w_b)

    # zero-field response -> local dielectric for the screening heuristic.
    # At E=0 the response is a CONSTANT times s_diel (Langevin g(0)=1, local
    # field factor constant): evaluate the constant on one element instead of
    # running grid-wide transcendentals (verified identical to 1e-16).
    zero1 = s_diel3.new_zeros(1)
    resp_unit = response_a3(zero1, s_diel3.new_ones(1), params, tp)[0]
    a3_zero = resp_unit * s_diel3
    eps3 = 1.0 + EDEPS * a3_zero

    # screened vacuum field: E = -(w_b * grad phi_sol) / eps(r)
    phi_g = grid.fft(phi_sol)
    ex, ey, ez, emag = grid.grad_from_recip(-torch.conj(w_b) * phi_g)
    emag_scr = emag / eps3
    ez_scr = ez / eps3

    # consistent pairing: saturation of the response at the screened field
    a3_scr = response_a3(emag_scr, s_diel3, params, tp)

    A_scr = plane_mean(a3_scr)
    S_ion_z = plane_mean(s_ion3)
    ez_scr_mean = plane_mean(ez_scr)
    prior = plane_mean(a3_scr * ez_scr) - A_scr * ez_scr_mean

    var_chi = torch.clamp(plane_var(a3_scr), min=0.0)
    w_env = var_chi / torch.clamp(var_chi.max(), min=1.0e-30)
    u = torch.cumsum(w_env, dim=0)
    u = u / torch.clamp(u[-1], min=1.0e-30)

    return {
        "A_scr": A_scr,
        "S_ion_z": S_ion_z,
        "prior": prior,
        "w_env": w_env,
        "u": u,
        "s_diel_mean": plane_mean(s_diel3),
        "n_e_mean": plane_mean(n_e_density),
    }


def solute_dipole_z(
    n_e_values_z: torch.Tensor,
    pos_frac: torch.Tensor,
    z_valence: torch.Tensor,
    cell: torch.Tensor,
    width: float = 4.0,
) -> torch.Tensor:
    """z-dipole of (valence electrons + ion cores), torch/differentiable.

    Faithful port of cep-dip's valence_ion_dipole_cart (z channel, poscen
    0.5): electrons via the sawtooth-windowed grid moment of the plane
    profile in DIRECT coordinates, ions via min-image fractional
    displacements (with the boundary snap), then direct -> cartesian.
    """
    nz = n_e_values_z.shape[0]
    dt, dev = n_e_values_z.dtype, n_e_values_z.device
    nouth = nz // 2
    indmin = int((nouth + int(0.5 * nz) + 10 * nz) % nz + 1)
    indices = torch.arange(1, nz + 1, dtype=dt, device=dev)
    ii = torch.remainder(indices - indmin + nz, nz) - nouth
    xx = torch.abs(torch.abs(ii) - nouth)
    cutoff = torch.where(xx > width, torch.ones_like(xx),
                         torch.abs(torch.sin(math.pi * xx / width / 2.0)))
    denlin = n_e_values_z / nz
    direct_z = torch.sum(denlin * ii * (1.0 / nz) * cutoff)

    anorm_z = torch.linalg.norm(cell[2])
    disp = torch.remainder(pos_frac[:, 2] - 0.5 + 10.5, 1.0) - 0.5
    tiny = 1.0e-2
    snap = torch.abs(torch.abs(disp) - 0.5) < tiny / anorm_z
    disp = torch.where(snap, torch.zeros_like(disp), disp)
    direct_z = direct_z - torch.sum(z_valence * disp)
    return direct_z * cell[2, 2]
