"""On-the-fly PB baseline fields for structures outside the training cache.

The cached baseline fields are superpositions of per-element radial
functions; fit_form_factors.py (experiments1d/exp_runtime_baseline)
extracts the per-element tables once from the training cache. At runtime a
new geometry only needs its structure factors:

    F(G) = sum_el f_el(|G|) * S_el(G),   S_el(G) = sum_i exp(-2pi i h . u_i)

followed by an inverse FFT. Blind validation against held-out cached
structures: neutral rel-rmse ~1e-4, phi_base ~1.5e-4 (float32 cache noise
floor). v1 scope: the fixed training cell/grid (tables store both and are
checked before use).
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
from ase.data import atomic_numbers as _ase_atomic_numbers


class RuntimeBaselineTables:
    def __init__(self, path: str):
        d = np.load(path)
        self.g2_shells = d["g2_shells"]
        self.f_neutral = d["f_neutral"]
        self.f_phi = d["f_phi"]
        self.elements = [str(x) for x in d["elements"]]
        self.z_list = [int(_ase_atomic_numbers[el]) for el in self.elements]
        self.cell = np.asarray(d["cell"], dtype=float)
        self.shape = tuple(int(x) for x in d["pb_shape"])
        self.f1d_neutral = None
        self.f1d_phi = None
        if "f1d_neutral_re" in d:
            self.f1d_neutral = d["f1d_neutral_re"] + 1j * d["f1d_neutral_im"]
            self.f1d_phi = d["f1d_phi_re"] + 1j * d["f1d_phi_im"]
        self._per_device: Dict[str, Dict[str, torch.Tensor]] = {}

    def matches(self, cell_np: np.ndarray, shape) -> bool:
        return tuple(shape) == self.shape and np.allclose(
            np.asarray(cell_np, dtype=float), self.cell, atol=1e-4
        )

    def _tables_on(self, device: torch.device) -> Dict[str, torch.Tensor]:
        key = str(device)
        if key not in self._per_device:
            nx, ny, nz = self.shape
            hx = np.fft.fftfreq(nx) * nx
            hy = np.fft.fftfreq(ny) * ny
            hz = np.arange(nz // 2 + 1)
            H = np.stack(
                np.meshgrid(hx, hy, hz, indexing="ij"), axis=-1
            ).reshape(-1, 3)
            inv_cell = np.linalg.inv(self.cell)
            gmat = inv_cell @ inv_cell.T
            g2r = np.round(np.einsum("ki,ij,kj->k", H, gmat, H), 9)
            sidx = np.clip(
                np.searchsorted(self.g2_shells, g2r), 0, len(self.g2_shells) - 1
            )
            # searchsorted returns the right insertion point for exact hits
            # computed the same way as at fit time; guard against drift
            off = np.abs(self.g2_shells[sidx] - g2r)
            sidx_lo = np.clip(sidx - 1, 0, len(self.g2_shells) - 1)
            use_lo = np.abs(self.g2_shells[sidx_lo] - g2r) < off
            sidx = np.where(use_lo, sidx_lo, sidx)
            if np.abs(self.g2_shells[sidx] - g2r).max() > 1e-7:
                raise RuntimeError(
                    "runtime baseline shell mapping mismatch; tables were "
                    "built for a different grid"
                )
            self._per_device[key] = {
                "H": torch.tensor(H, dtype=torch.float64, device=device),
                "fn": torch.tensor(
                    self.f_neutral[sidx, :].T, dtype=torch.float64, device=device
                ),
                "fp": torch.tensor(
                    self.f_phi[sidx, :].T, dtype=torch.float64, device=device
                ),
            }
        return self._per_device[key]



    def _line_profiles(
        self, frac_z: torch.Tensor, node_z: torch.Tensor, device: torch.device
    ):
        """(neutral, phi) plane profiles [nz] from the high-fidelity 1d fit."""
        if self.f1d_neutral is None:
            return None
        nz = self.shape[2]
        key = "f1d:" + str(device)
        if key not in self._per_device:
            self._per_device[key] = {
                "fn": torch.tensor(self.f1d_neutral.T, dtype=torch.complex128, device=device),
                "fp": torch.tensor(self.f1d_phi.T, dtype=torch.complex128, device=device),
                "hz": torch.arange(nz // 2 + 1, dtype=torch.float64, device=device),
            }
        t = self._per_device[key]
        pn = torch.zeros(nz // 2 + 1, dtype=torch.complex128, device=device)
        pp = torch.zeros_like(pn)
        z = node_z.view(-1).to(device)
        fz = frac_z.to(device=device, dtype=torch.float64)
        for e_i, z_el in enumerate(self.z_list):
            uz = fz[z == z_el]
            if uz.shape[0] == 0:
                continue
            s_line = torch.exp(
                -2j * math.pi * t["hz"][:, None] * uz[None, :]
            ).sum(dim=1)
            pn += t["fn"][e_i] * s_line
            pp += t["fp"][e_i] * s_line
        return torch.fft.irfft(pn, n=nz), torch.fft.irfft(pp, n=nz)


    def profile_z(
        self, pos_frac: torch.Tensor, node_z: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Plane-averaged neutral profile: only the (0,0,gz) reciprocal line.

        Returns (z_grid [nz], profile [nz]) float64 on device; same values
        convention as plane-averaging the 3-D neutral field (the crossing
        detector normalizes, so the overall scale is irrelevant anyway)."""
        nx, ny, nz = self.shape
        if self.f1d_neutral is not None:
            key0 = "f1dz:" + str(device)
            if key0 not in self._per_device:
                self._per_device[key0] = {
                    "z": torch.arange(nz, dtype=torch.float64, device=device)
                    * float(np.linalg.norm(self.cell[2])) / nz
                }
            prof_n, _ = self._line_profiles(pos_frac[:, 2], node_z, device)
            return self._per_device[key0]["z"], prof_n
        key = "profile:" + str(device)
        if key not in self._per_device:
            inv_cell = np.linalg.inv(self.cell)
            gzz = float((inv_cell @ inv_cell.T)[2, 2])
            g2r = np.round((np.arange(nz // 2 + 1) ** 2) * gzz, 9)
            sidx = np.clip(
                np.searchsorted(self.g2_shells, g2r), 0, len(self.g2_shells) - 1
            )
            sidx_lo = np.clip(sidx - 1, 0, len(self.g2_shells) - 1)
            use_lo = (np.abs(self.g2_shells[sidx_lo] - g2r)
                      < np.abs(self.g2_shells[sidx] - g2r))
            sidx = np.where(use_lo, sidx_lo, sidx)
            self._per_device[key] = {
                "fn_line": torch.tensor(
                    self.f_neutral[sidx, :].T, dtype=torch.float64, device=device
                ),
                "hz": torch.arange(
                    nz // 2 + 1, dtype=torch.float64, device=device
                ),
                "z": torch.arange(nz, dtype=torch.float64, device=device)
                * float(np.linalg.norm(self.cell[2])) / nz,
            }
        t = self._per_device[key]
        fk = torch.zeros(nz // 2 + 1, dtype=torch.complex128, device=device)
        frac_z = pos_frac[:, 2].to(device=device, dtype=torch.float64)
        z = node_z.view(-1).to(device)
        for e_i, z_el in enumerate(self.z_list):
            uz = frac_z[z == z_el]
            if uz.shape[0] == 0:
                continue
            phase = t["hz"][:, None] * uz[None, :]
            fk += t["fn_line"][e_i] * torch.exp(
                -2j * math.pi * phase
            ).sum(dim=1)
        profile = torch.fft.irfft(fk, n=nz)
        return t["z"], profile

    def fields(
        self, pos_frac: torch.Tensor, node_z: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """neutral and phi_base on the PB grid (float64, on device)."""
        t = self._tables_on(device)
        H = t["H"]
        nk = H.shape[0]
        f_neutral_g = torch.zeros(nk, dtype=torch.complex128, device=device)
        f_phi_g = torch.zeros(nk, dtype=torch.complex128, device=device)
        frac = pos_frac.to(device=device, dtype=torch.float64)
        z = node_z.view(-1).to(device)
        for e_i, z_el in enumerate(self.z_list):
            sel = frac[z == z_el]
            if sel.shape[0] == 0:
                continue
            s_el = torch.zeros(nk, dtype=torch.complex128, device=device)
            # atom chunk: 16 fits the login node's 8 GB vmem; GPUs take 64
            # (transient phase block 1.5e6 x 64 complex128 ~ 1.5 GiB)
            chunk = 64 if device.type == "cuda" else 16
            for c0 in range(0, sel.shape[0], chunk):
                phase = H @ sel[c0:c0 + chunk].T
                s_el += torch.exp(-2j * math.pi * phase).sum(dim=1)
            f_neutral_g += t["fn"][e_i] * s_el
            f_phi_g += t["fp"][e_i] * s_el
        nx, ny, nz = self.shape
        neutral = torch.fft.irfftn(
            f_neutral_g.reshape(nx, ny, nz // 2 + 1), s=self.shape
        )
        phi_base = torch.fft.irfftn(
            f_phi_g.reshape(nx, ny, nz // 2 + 1), s=self.shape
        )
        # pin the plane averages to the high-fidelity 1d fit: the 3-D shell
        # fit is core-amplitude normalized and its plateau noise (~0.05 eV)
        # is visible to the PB closure
        line = self._line_profiles(frac[:, 2], node_z, device)
        if line is not None:
            prof_n, prof_p = line
            neutral = neutral + (prof_n - neutral.mean(dim=(0, 1)))[None, None, :]
            phi_base = phi_base + (prof_p - phi_base.mean(dim=(0, 1)))[None, None, :]
        return neutral, phi_base
