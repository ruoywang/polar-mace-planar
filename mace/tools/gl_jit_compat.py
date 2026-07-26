"""TorchScript-compatibility shims for the pinned graph_longrange package.

graph_longrange 0.4.0 is pinned and unmodified on disk; a few of its forward
methods use constructs TorchScript cannot script (star-unpacked shapes,
out=-into-slice writes). This module replaces those methods at import time
with mathematically identical, scriptable rewrites (verified bit-identical
by the training identity probe). Only torch.stack/mul reorderings — no
numerical change.
"""
from __future__ import annotations

import torch


def _patch_gto_form_factors() -> None:
    from graph_longrange.gto_utils import RadialIntegralDirect as cls

    def forward(self, k_mods: torch.Tensor) -> torch.Tensor:
        k2 = k_mods * k_mods
        exp_term = torch.exp(-0.5 * k2.unsqueeze(-1) * self.sigma2)
        if self.max_l == 0:
            out = self.pref0 * exp_term
            return out.unsqueeze(-1)
        a = self.pref0 * exp_term
        b = self.pref1 * (k_mods.unsqueeze(-1) * exp_term)
        return torch.stack((a, b), dim=-1)

    cls.forward = forward


def _patch_gto_basis() -> None:
    from graph_longrange import gto_utils

    cls = getattr(gto_utils, "GTOBasis", None)
    if cls is None or not hasattr(cls, "_evaluate_fourier_basis"):
        return

    def _evaluate_fourier_basis(
        self, k_moduli: torch.Tensor, yklm: torch.Tensor
    ) -> torch.Tensor:
        fnlk = self.radial_spline(k_moduli) * self.cl_scale
        expanded_fnlk = torch.index_select(fnlk, -1, self.expanded_l_indices)
        xnlk = expanded_fnlk * yklm.unsqueeze(-2)
        re = xnlk * self.real_phase_factors
        im = xnlk * self.imag_phase_factors
        return torch.stack((re, im), dim=-1)

    cls._evaluate_fourier_basis = _evaluate_fourier_basis


def _patch_gto_selfinteraction() -> None:
    from graph_longrange.gto_utils import GTOSelfInteractionBlock as cls

    @torch.jit.ignore
    def _features_dim(self) -> int:
        return int(self.features_irreps.dim)

    def forward(self, charge_density: torch.Tensor) -> torch.Tensor:
        qs_expanded = torch.index_select(
            charge_density, dim=-1, index=self.select_indices
        )
        features = torch.zeros(
            (charge_density.shape[0], self._features_dim()),
            device=charge_density.device,
            dtype=charge_density.dtype,
        )
        features[..., : self.non_zero_terms] = torch.einsum(
            "d,nd->nd", self.overlap_constants, qs_expanded
        )
        return features

    cls._features_dim = _features_dim
    cls.forward = forward


def apply() -> None:
    _patch_gto_form_factors()
    _patch_gto_basis()
    _patch_gto_selfinteraction()


apply()
