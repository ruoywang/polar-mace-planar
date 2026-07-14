"""Local field factor with an analytic implicit-function backward.

torch_pb._local_field_factor solves, pointwise, the fixed point

    f = clamp( 1 / (1 - (alpha_rot * g(beta*f*e) + alpha_pol) * nu), lo, hi )

with g the rotational Langevin factor g(u) = 3(u - tanh u)/(u^2 tanh u).
Autograd through its (up to) 80 unrolled iterations dominates the closure
backward graph and each iteration carries a CPU-GPU sync. Here:

- forward: the same damped-free iteration, convergence checked every 8
  iterations (only MORE converged than the original's every-step check;
  same 1e-10 tolerance).
- backward: implicit differentiation at the fixed point,
      df/de = A g'(u) beta f / (1 - A g'(u) beta e),   A = f^2 alpha_rot nu,
  zero where the clamp is active or e == 0. Exact up to the solve tolerance.

Verified against autograd-through-the-loop (see tools1d/test_localfield_grad.py).
"""
from __future__ import annotations

import torch


def _g_rot(u: torch.Tensor) -> torch.Tensor:
    out = torch.ones_like(u)
    small = u < 2.0e-4
    us = u[~small]
    out[~small] = 3.0 * (us - torch.tanh(us)) / (us * us * torch.tanh(us))
    return out


def _g_rot_prime(u: torch.Tensor) -> torch.Tensor:
    """d g / d u, analytic; series -2u/15 below the small-u switch."""
    out = -2.0 * u / 15.0
    small = u < 1.0e-2
    us = u[~small]
    t = torch.tanh(us)
    tp = 1.0 - t * t
    num = 3.0 * ((1.0 - tp) * us * us * t - (us - t) * (2.0 * us * t + us * us * tp))
    out[~small] = num / (us * us * t) ** 2
    return out


class _LocalFieldFactorFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e_mag, params):
        alpha_pol = float(params["alpha_pol"])
        alpha0_rot = float(params["alpha0_rot"])
        nu = float(params["invalpha_sic"])
        beta = float(params["PBETA"])
        if not bool(params["LNLDIEL"]):
            f = torch.full_like(e_mag, 1.0 / (1.0 - (alpha_pol + alpha0_rot) * nu))
            ctx.save_for_backward(torch.zeros_like(e_mag))
            ctx.linear = True
            return f
        lo = 1.0 / (1.0 - alpha_pol * nu)
        hi = (1.0 / (1.0 - (alpha_pol + alpha0_rot) * nu)) * (1.0 + 1.0e-8)
        x0 = beta * e_mag
        f = torch.full_like(e_mag, hi)
        zero = x0 == 0.0
        with torch.no_grad():
            for it in range(80):
                gx = _g_rot(f * x0)
                new = 1.0 / (1.0 - (gx * alpha0_rot + alpha_pol) * nu)
                new = torch.clamp(new, lo, hi)
                if it % 8 == 7:
                    diff = torch.max(torch.abs(new - f))
                    f = new
                    if float(diff) <= 1.0e-10 * max(1.0, hi):
                        break
                else:
                    f = new
            f[zero] = hi
        ctx.linear = False
        ctx.scalars = (alpha_pol, alpha0_rot, nu, beta, lo, hi)
        ctx.save_for_backward(e_mag, f)
        return f

    @staticmethod
    def backward(ctx, grad_f):
        if ctx.linear:
            return torch.zeros_like(grad_f), None
        e_mag, f = ctx.saved_tensors
        alpha_pol, alpha0_rot, nu, beta, lo, hi = ctx.scalars
        u = beta * f * e_mag
        gp = _g_rot_prime(u)
        A = f * f * alpha0_rot * nu
        denom = 1.0 - A * gp * beta * e_mag
        dfde = A * gp * beta * f / denom
        # clamp-active or zero-field points carry no gradient
        eps = 1.0e-12 * hi
        active = (f > lo + eps) & (f < hi * (1.0 - 1.0e-12)) & (e_mag != 0.0)
        dfde = torch.where(active, dfde, torch.zeros_like(dfde))
        return grad_f * dfde, None


def local_field_factor(e_mag: torch.Tensor, params: dict) -> torch.Tensor:
    return _LocalFieldFactorFn.apply(e_mag, params)
