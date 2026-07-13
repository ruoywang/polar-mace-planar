"""Differentiable 1-D nonlinear PB solver (frozen affine dielectric closure).

Torch port of cep-dip-python-pb's `solve_frozen` (tools/pb_check/
solve_1d_frozen_response.py) specialised to a (1, 1, nz) grid, with the
fixsol dipole-correction loop.  Numerics:

- All spectral operators (derivative, w_b convolution, Poisson/L0) are
  precomputed as dense nz x nz real matrices (circulants built by applying
  the exact rfft-convention ops to an identity batch), so one solve is a
  handful of matmuls plus `torch.linalg.solve` per Newton step.  This
  deliberately avoids backprop through CG iterations (measured unstable in
  the 3-D era); `linalg.solve` has an exact, stable adjoint.
- Conventions follow the numpy source exactly: fft = rfft/n, ifft = irfft*n,
  derivative g-mesh has the Nyquist zeroed, the Poisson gsq keeps it,
  densities are in the VASP "values" convention (sum/n = charge).
- The residual norm used for tolerances reproduces numpy's preconditioned
  norm sqrt(g0^2 + |l0_inv resid|^2), so iteration counts are comparable.

Everything is float64 and differentiable end-to-end (Newton unrolled; the
line-search branch decisions are treated as constants by autograd, which is
exact almost everywhere).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

TPI = 2.0 * math.pi
RYTOEV = 13.605826
AUTOA = 0.529177249
EDEPS = 4.0 * math.pi * 2.0 * RYTOEV * AUTOA
DTYPE = torch.float64


def _rfft_matrix_ops(nz: int, lz: float, volume: float, device) -> Dict[str, torch.Tensor]:
    """Dense matrices for the exact spectral ops on a (1,1,nz) grid."""
    eye = torch.eye(nz, dtype=DTYPE, device=device)
    spec = torch.fft.rfft(eye, dim=-1) / nz  # rows: fft of basis vectors
    nzh = nz // 2 + 1
    hz = torch.arange(nzh, dtype=DTYPE, device=device)
    gz_deriv = hz.clone()
    if nz % 2 == 0:
        gz_deriv[-1] = 0.0  # Nyquist zeroed for derivatives
    gz_deriv = gz_deriv / lz
    gsq = (hz / lz) ** 2  # Poisson mesh keeps the Nyquist

    def to_real(mat_spec: torch.Tensor) -> torch.Tensor:
        # rows are spectra of columns of the operator applied to identity
        return torch.fft.irfft(mat_spec * nz, n=nz, dim=-1).transpose(0, 1).contiguous()

    # derivative operator: ifft( i*2pi*g * fft(x) )
    D = to_real(spec * (1j * TPI * gz_deriv))
    # L0 (Poisson) operator: ifft( (2pi)^2 gsq * V/EDEPS * fft(x) )
    L0 = to_real(spec * (TPI**2 * gsq * volume / EDEPS))
    # preconditioned-norm weight for the residual rms (numpy parity)
    w_spec = torch.full((nzh,), 2.0, dtype=DTYPE, device=device)
    w_spec[0] = 1.0
    if nz % 2 == 0:
        w_spec[-1] = 1.0
    l0_inv = torch.zeros(nzh, dtype=DTYPE, device=device)
    l0_inv[1:] = EDEPS / (TPI**2 * gsq[1:]) / volume
    return {"D": D, "L0": L0, "w_spec": w_spec, "l0_inv": l0_inv}


def _gaussian_wb_matrix(nz: int, lz: float, sigma: float, device) -> torch.Tensor:
    """w_b convolution as a dense circulant (numpy normalized_gaussian_kernel_g)."""
    z = torch.arange(nz, dtype=DTYPE, device=device) * lz / nz
    r = torch.minimum(z, lz - z)  # periodic distance from origin (1-D grid line)
    real = torch.exp(-0.5 * (r / sigma) ** 2) / (sigma * math.sqrt(TPI)) ** 3
    real = real * (nz / real.sum())
    ker_g = torch.fft.rfft(real) / nz  # real & symmetric -> spectrum is real
    eye = torch.eye(nz, dtype=DTYPE, device=device)
    spec = torch.fft.rfft(eye, dim=-1) / nz
    return torch.fft.irfft(spec * ker_g * nz, n=nz, dim=-1).transpose(0, 1).contiguous()


def ion_density_values(phi: torch.Tensor, s_ion: torch.Tensor, params: Dict[str, float],
                       volume: float) -> torch.Tensor:
    """rho_ion * volume (values convention). Port of ion_density_values_from_phi."""
    if not params["LION"]:
        return torch.zeros_like(phi)
    x = params["ZBETA"] * phi
    theta = params["theta_b"]
    if params["LNLION"] and theta > 0.0:
        xc = torch.clamp(x, -100.0, 100.0)
        den = 1.0 + theta * (torch.cosh(xc) - 1.0)
        mid = theta * torch.sinh(xc) / den
        large = torch.sign(x)
        small = theta * x
        n_work = torch.where(torch.abs(x) > 100.0, large,
                             torch.where(torch.abs(x) < math.sqrt(theta) * 2.0e-4, small, mid))
    elif params["LNLION"]:
        n_work = torch.sinh(torch.clamp(x, -100.0, 100.0))
    else:
        n_work = x
    rho = -params["n_max"] * params["invBETA"] * params["ZBETA"] * s_ion * n_work
    return rho * volume


def ion_density_derivative(phi: torch.Tensor, s_ion: torch.Tensor, params: Dict[str, float],
                           volume: float) -> torch.Tensor:
    """d(rho_ion*volume)/d(phi), elementwise (analytic)."""
    if not params["LION"]:
        return torch.zeros_like(phi)
    zb = params["ZBETA"]
    x = zb * phi
    theta = params["theta_b"]
    if params["LNLION"] and theta > 0.0:
        xc = torch.clamp(x, -100.0, 100.0)
        ch = torch.cosh(xc)
        sh = torch.sinh(xc)
        den = 1.0 + theta * (ch - 1.0)
        dmid = theta * ch / den - (theta * sh) * (theta * sh) / (den * den)
        dsmall = torch.full_like(x, theta)
        dlarge = torch.zeros_like(x)
        dnw = torch.where(torch.abs(x) > 100.0, dlarge,
                          torch.where(torch.abs(x) < math.sqrt(theta) * 2.0e-4, dsmall, dmid))
    elif params["LNLION"]:
        xc = torch.clamp(x, -100.0, 100.0)
        dnw = torch.cosh(xc)
        dnw = torch.where(torch.abs(x) > 100.0, torch.zeros_like(x), dnw)
    else:
        dnw = torch.ones_like(x)
    return -params["n_max"] * params["invBETA"] * zb * s_ion * dnw * zb * volume


def cdipol_potential_1d(nz: int, length: float, ef_direct_z: torch.Tensor, indmin: int,
                        device, width: float = 4.0) -> torch.Tensor:
    indices = torch.arange(1, nz + 1, dtype=DTYPE, device=device)
    nouth = nz // 2
    ii = torch.remainder(indices - indmin + nz, nz) - nouth
    xx = torch.abs(torch.abs(ii) - nouth)
    cutoff = torch.where(xx > width, torch.ones_like(xx),
                         torch.abs(torch.sin(math.pi * xx / width / 2.0)))
    e_comp = ef_direct_z * length
    return (-e_comp * length / nz) * ii * cutoff


class Solver1D:
    """Per-cell 1-D PB solver with precomputed operators (reusable across calls)."""

    def __init__(self, cell: torch.Tensor, nz: int, params: Dict[str, float],
                 device: Optional[torch.device] = None):
        device = device or cell.device
        self.device = device
        self.nz = nz
        cell64 = cell.to(DTYPE)
        self.lz = float(cell64[2, 2])
        self.volume = float(torch.abs(torch.linalg.det(cell64)))
        self.params = params
        ops = _rfft_matrix_ops(nz, self.lz, self.volume, device)
        self.D = ops["D"]
        self.L0 = ops["L0"]
        self.w_spec = ops["w_spec"]
        self.l0_inv = ops["l0_inv"]
        sigma_b = params["R_B"] if params["R_B"] > 0.0 else params["A_K"]
        self.WB = _gaussian_wb_matrix(nz, self.lz, sigma_b, device)
        self.z = torch.arange(nz, dtype=DTYPE, device=device) * self.lz / nz

    # -- pieces -----------------------------------------------------------
    def bound_matrix(self, a1: torch.Tensor) -> torch.Tensor:
        """n_b(phi part) = B @ phi.

        E_z = -(w_b * grad phi) = -(D @ WB) phi;  P = a1 * E_z (+ p_off);
        n_b = -V * WB @ D @ P  =>  B = +V * WB @ D @ diag(a1) @ D @ WB.
        """
        core = self.D @ self.WB
        return self.volume * (self.WB @ (self.D @ (a1[:, None] * core)))

    def bound_offset(self, p_off: torch.Tensor) -> torch.Tensor:
        """n_b contribution of the constant offset P_off: -V * WB @ D @ p_off."""
        return -self.volume * (self.WB @ (self.D @ p_off))

    def resid_rms(self, resid: torch.Tensor) -> torch.Tensor:
        rg = torch.fft.rfft(resid) / self.nz
        g0 = rg[0].real
        cw = rg * self.l0_inv.to(torch.complex128)
        nrm2 = torch.sum(self.w_spec * (cw.real**2 + cw.imag**2))
        return torch.sqrt(g0 * g0 + nrm2)

    def residual(self, phi: torch.Tensor, phi_sol: torch.Tensor, s_ion: torch.Tensor,
                 B: torch.Tensor, nb_off: torch.Tensor, q_sol: float
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_b = B @ phi + nb_off
        n_ion = ion_density_values(phi, s_ion, self.params, self.volume)
        l0term = self.L0 @ (phi - phi_sol)
        resid = (n_b + n_ion) - l0term + q_sol  # G0 of l0 set to -q_sol (numpy parity)
        return resid, n_b, n_ion

    # -- one nonlinear solve (fixed phi_sol) --------------------------------
    def newton(self, phi0: torch.Tensor, phi_sol: torch.Tensor, s_ion: torch.Tensor,
               a1: torch.Tensor, p_off: torch.Tensor, q_sol: float,
               tol: float, max_outer: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        B = self.bound_matrix(a1)
        nb_off = self.bound_offset(p_off)
        phi = phi0
        resid, n_b, n_ion = self.residual(phi, phi_sol, s_ion, B, nb_off, q_sol)
        rms = self.resid_rms(resid)
        n_outer = 0
        for outer in range(max_outer + 1):
            if float(rms) < tol and outer >= 1:
                break
            n_outer = outer + 1
            dion = ion_density_derivative(phi, s_ion, self.params, self.volume)
            J = B + torch.diag(dion) - self.L0
            dphi = torch.linalg.solve(J, -resid)
            alpha = 1.0
            for _ in range(7):
                trial = phi + alpha * dphi
                t_resid, t_nb, t_nion = self.residual(trial, phi_sol, s_ion, B, nb_off, q_sol)
                t_rms = self.resid_rms(t_resid)
                if float(t_rms) <= float(rms) or alpha <= 1.0 / 64.0:
                    phi, resid, rms = trial, t_resid, t_rms
                    n_b, n_ion = t_nb, t_nion
                    break
                alpha *= 0.5
        return phi, n_b, n_ion, n_outer

    # -- full solve with the fixsol dipole loop -----------------------------
    def solve(self, cvhar_z: torch.Tensor, s_ion: torch.Tensor, a1: torch.Tensor,
              p_off: torch.Tensor, q_sol: float, val_ion_dipole_z: torch.Tensor,
              c_unit: float, center_z: float, indmin: int,
              fixsol_steps: int = 5, tol: float = 1.0e-3, max_outer: int = 12,
              phi_init: Optional[torch.Tensor] = None
              ) -> Dict[str, torch.Tensor]:
        nz, dev = self.nz, self.device
        phi = phi_init if phi_init is not None else torch.zeros(nz, dtype=DTYPE, device=dev)
        qsol_cache = torch.zeros((), dtype=DTYPE, device=dev)
        dsol_z = torch.zeros((), dtype=DTYPE, device=dev)
        # EwaldDipoleMixer state (z channel only; ef = c_unit * d_mix, exact linearity)
        dip_tmp = torch.zeros((), dtype=DTYPE, device=dev)
        res_old = torch.zeros((), dtype=DTYPE, device=dev)
        total_outer = 0
        for _step in range(fixsol_steps):
            dip_z = val_ion_dipole_z + dsol_z - qsol_cache * center_z
            dip_in = torch.clamp(dip_z, -20.0, 20.0)
            res = dip_in - dip_tmp
            alpha = torch.where(torch.abs(res) > 1.0,
                                0.6 / (torch.abs(res) * torch.abs(res)),
                                torch.full_like(res, 0.6))
            alpha = torch.where((res * res_old < 0.0) & (torch.abs(res) > 0.7 * torch.abs(res_old)),
                                alpha * 0.5, alpha)
            d_mix = (1.0 - alpha) * dip_tmp + alpha * dip_in
            res_old = res
            dip_tmp = d_mix
            ef_z = c_unit * d_mix
            cvdip = cdipol_potential_1d(nz, self.lz, ef_z, indmin, dev)
            phi_sol = cvhar_z + cvdip
            phi, n_b, n_ion, n_outer = self.newton(
                phi, phi_sol, s_ion, a1, p_off, q_sol, tol, max_outer)
            total_outer += n_outer
            charge = n_b + n_ion
            qsol_cache = charge.mean()
            dsol_z = (charge * self.z).mean()
        return {
            "phi": phi,
            "n_b": n_b,
            "n_ion": n_ion,
            "phi_sol": phi_sol,
            "n_outer": torch.tensor(total_outer),
        }
