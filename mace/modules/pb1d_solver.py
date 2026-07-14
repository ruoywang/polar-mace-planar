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

Everything is float64 and differentiable end-to-end.  Gradients come from
the analytic coupled adjoint (grad_passes=1, FD-verified on all five input
channels incl. the rank-1 dipole-feedback elimination — see Solver1D.solve)
or the fully-unrolled legacy graph (grad_passes=0, kept as the FD baseline).
The fixsol dipole loop runs to convergence by default (fixsol_converge).
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
        self._core = self.D @ self.WB  # reused by every bound_matrix call
        self.z = torch.arange(nz, dtype=DTYPE, device=device) * self.lz / nz

    # -- pieces -----------------------------------------------------------
    def bound_matrix(self, a1: torch.Tensor) -> torch.Tensor:
        """n_b(phi part) = B @ phi.

        E_z = -(w_b * grad phi) = -(D @ WB) phi;  P = a1 * E_z (+ p_off);
        n_b = -V * WB @ D @ P  =>  B = +V * WB @ D @ diag(a1) @ D @ WB.
        """
        return self.volume * (self.WB @ (self.D @ (a1[:, None] * self._core)))

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
               tol: float, max_outer: int,
               B: Optional[torch.Tensor] = None,
               nb_off: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if B is None:
            B = self.bound_matrix(a1)
        if nb_off is None:
            nb_off = self.bound_offset(p_off)
        phi = phi0
        resid, n_b, n_ion = self.residual(phi, phi_sol, s_ion, B, nb_off, q_sol)
        rms = self.resid_rms(resid)
        n_outer = 0
        self._last_rms = float(rms)
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
            self._last_rms = float(rms)
        return phi, n_b, n_ion, n_outer

    # -- full solve with the fixsol dipole loop -----------------------------
    def solve(self, cvhar_z: torch.Tensor, s_ion: torch.Tensor, a1: torch.Tensor,
              p_off: torch.Tensor, q_sol: float, val_ion_dipole_z: torch.Tensor,
              c_unit: float, center_z: float, indmin: int,
              fixsol_steps: int = 5, tol: float = 1.0e-3, max_outer: int = 12,
              phi_init: Optional[torch.Tensor] = None, grad_passes: int = 0,
              fixsol_converge: bool = True, fixsol_max_steps: int = 60
              ) -> Dict[str, torch.Tensor]:
        """Solve; gradients via the analytic coupled adjoint (grad_passes>=1)
        or the fully-unrolled graph (grad_passes=0).

        Analytic mode: converge under no_grad, then form ONE differentiable
        residual at the converged point and take a single Newton step with
        the COUPLED Jacobian J_c = J + g1 w^T (detached):

          J  = B + diag(dion) - L0          (Newton Jacobian at phi*)
          g1 = c_unit * L0 @ k              (dipole-field column: dR/dd)
          w  = C^T (z - center_z)/nz        (charge-moment row:  dd/dphi,
                                             C = B + diag(dion))

        The rank-1 term solves the dipole-feedback fix point EXACTLY (direct
        elimination; immune to the loop gain > 1 that makes naive iteration
        of the differentiable tail diverge).  By the implicit function
        theorem the step's gradient equals the converged algorithm's, up to
        the solve tolerance (rms ~1e-9); values shift by the same
        negligible amount.  Graph: ~30 ops vs thousands for the unroll.
        """
        want_grad = torch.is_grad_enabled() and any(
            isinstance(x, torch.Tensor) and x.requires_grad
            for x in (cvhar_z, s_ion, a1, p_off, val_ion_dipole_z)
        )
        if not want_grad or grad_passes <= 0:
            return self._solve_unrolled(
                cvhar_z, s_ion, a1, p_off, q_sol, val_ion_dipole_z,
                c_unit, center_z, indmin, fixsol_steps, tol, max_outer, phi_init,
                fixsol_converge, fixsol_max_steps)
        with torch.no_grad():
            pre = self._solve_unrolled(
                cvhar_z.detach(), s_ion.detach(), a1.detach(), p_off.detach(),
                q_sol, val_ion_dipole_z.detach(), c_unit, center_z, indmin,
                fixsol_steps, tol, max_outer, phi_init,
                fixsol_converge, fixsol_max_steps)
        phi_star = pre["phi"]  # detached (no_grad)
        nz, dev = self.nz, self.device

        # theta-differentiable pieces at the converged point
        B = self.bound_matrix(a1)
        nb_off = self.bound_offset(p_off)
        charge_star = (B @ phi_star + nb_off
                       + ion_density_values(phi_star, s_ion, self.params, self.volume))
        dip_z = (val_ion_dipole_z + (charge_star * self.z).mean()
                 - charge_star.mean() * center_z)
        d_new = torch.clamp(dip_z, -20.0, 20.0)  # mixer fix point: d = dip_in
        cvdip = cdipol_potential_1d(nz, self.lz, c_unit * d_new, indmin, dev)
        phi_sol = cvhar_z + cvdip
        resid, _, _ = self.residual(phi_star, phi_sol, s_ion, B, nb_off, q_sol)

        # detached coupled Jacobian (rank-1 dipole-feedback correction)
        with torch.no_grad():
            dion = ion_density_derivative(phi_star, s_ion.detach(), self.params, self.volume)
            C = B.detach() + torch.diag(dion)
            J = C - self.L0
            unit = torch.ones((), dtype=DTYPE, device=dev)
            k_vec = cdipol_potential_1d(nz, self.lz, unit, indmin, dev)
            g1 = c_unit * (self.L0 @ k_vec)
            mask = (torch.abs(dip_z.detach()) < 20.0).to(DTYPE)
            w_vec = mask * (C.transpose(0, 1) @ ((self.z - center_z) / nz))
            J_c = J + torch.outer(g1, w_vec)

        dphi = torch.linalg.solve(J_c, -resid)
        phi = phi_star + dphi
        n_b = B @ phi + nb_off
        n_ion = ion_density_values(phi, s_ion, self.params, self.volume)
        with torch.no_grad():
            fres, _, _ = self.residual(phi, phi_sol, s_ion, B, nb_off, q_sol)
            self._last_rms = float(self.resid_rms(fres))
        return {
            "phi": phi,
            "n_b": n_b,
            "n_ion": n_ion,
            "phi_sol": phi_sol,
            "n_outer": pre["n_outer"],
            "rms_last": self._last_rms,
        }

    def _solve_unrolled(self, cvhar_z: torch.Tensor, s_ion: torch.Tensor, a1: torch.Tensor,
                        p_off: torch.Tensor, q_sol: float, val_ion_dipole_z: torch.Tensor,
                        c_unit: float, center_z: float, indmin: int,
                        fixsol_steps: int = 5, tol: float = 1.0e-3, max_outer: int = 12,
                        phi_init: Optional[torch.Tensor] = None,
                        fixsol_converge: bool = True, fixsol_max_steps: int = 60
                        ) -> Dict[str, torch.Tensor]:
        """fixsol_converge (default): treat fixsol_steps as a MINIMUM and keep
        mixing until the dipole residual is at the fix point (|res| <= 1e-9,
        cap fixsol_max_steps).  Measured on the parity pack: 5 fixed steps
        leave d_mix 6% off (res 0.36); ~20 steps reach 1e-11.  The converged
        d is the physical answer and what the analytic adjoint differentiates.
        False reproduces the legacy fixed-count loop (numpy parity)."""
        nz, dev = self.nz, self.device
        phi = phi_init if phi_init is not None else torch.zeros(nz, dtype=DTYPE, device=dev)
        qsol_cache = torch.zeros((), dtype=DTYPE, device=dev)
        dsol_z = torch.zeros((), dtype=DTYPE, device=dev)
        # EwaldDipoleMixer state (z channel only; ef = c_unit * d_mix, exact linearity)
        dip_tmp = torch.zeros((), dtype=DTYPE, device=dev)
        res_old = torch.zeros((), dtype=DTYPE, device=dev)
        B_fix = self.bound_matrix(a1)
        nb_off_fix = self.bound_offset(p_off)
        total_outer = 0
        _step = 0
        while True:
            _step += 1
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
                phi, phi_sol, s_ion, a1, p_off, q_sol, tol, max_outer,
                B=B_fix, nb_off=nb_off_fix)
            total_outer += n_outer
            charge = n_b + n_ion
            qsol_cache = charge.mean()
            dsol_z = (charge * self.z).mean()
            if _step >= fixsol_steps:
                if not fixsol_converge or _step >= fixsol_max_steps:
                    break
                if float(torch.abs(res)) <= 1.0e-7 * max(1.0, float(torch.abs(dip_in))):
                    break
        return {
            "phi": phi,
            "n_b": n_b,
            "n_ion": n_ion,
            "phi_sol": phi_sol,
            "n_outer": torch.tensor(total_outer),
            "rms_last": float(getattr(self, "_last_rms", float("nan"))),
            # mixer/moment state for the IFT tail
            "dip_tmp": dip_tmp,
            "res_old": res_old,
            "qsol_cache": qsol_cache,
            "dsol_z": dsol_z,
        }
