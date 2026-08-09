"""Probe residual head for the 3-D net density (probe3d, v1).

The model's net density becomes GTO + this residual; the residual enters the
PB electrostatics (added to net_g on the PB grid) and the density/profile
losses. Design constraints (see claude/3-charge_probe/DESIGN.md):

  - invariants from the backbone's equivariant features contracted with Ylm
    of the atom->point direction (same e3nn convention MACE uses on raw
    Cartesian vectors, component normalization)
  - per-l channel projection first (channel mixing within each angular block
    keeps rotational behavior intact) so grid-scale evaluation moves small
    vectors, not the raw 1152-dim features
  - species-conditioned radial gating (DeepH-R conditions probe messages on
    the neighboring atom species), cosine cutoff at rc
  - displacement convention identical to loss._gto_density_at_points_axis2_pbc:
    minimum image along the two in-plane axes only
  - smooth z-window: zero outside [z_lo, z_hi] (the supervised solvent
    window); cosine roll of width edge_width inside the boundaries
  - zero net charge is enforced by the caller (mean subtraction on the grid)
  - last layer zero-initialized: at step 0 the residual is exactly zero and
    the model reproduces the production forward bit-for-bit
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import torch
from e3nn import o3


def _parse_layer_blocks(irreps_per_layer: List[str]) -> List[Tuple[int, int, int]]:
    """Concatenated node_feats layout -> [(offset, l, mul)] in storage order."""
    blocks: List[Tuple[int, int, int]] = []
    off = 0
    for s in irreps_per_layer:
        for part in str(s).split("+"):
            m = re.match(r"\s*(\d+)x(\d+)[eo]\s*", part)
            if m is None:
                raise ValueError(f"cannot parse irreps chunk: {part!r}")
            mul, l = int(m.group(1)), int(m.group(2))
            blocks.append((off, l, mul))
            off += mul * (2 * l + 1)
    return blocks


class Probe3DResidualHead(torch.nn.Module):
    def __init__(
        self,
        irreps_per_layer: List[str],
        zs: List[int],
        rc: float = 6.0,
        n_rbf: int = 16,
        proj: int = 32,
        hidden: int = 256,
        gate_hidden: int = 64,
        max_l: int = 2,
        y_scale: float = 0.03,
        edge_width: float = 1.0,
    ) -> None:
        super().__init__()
        blocks = _parse_layer_blocks(irreps_per_layer)
        max_l = int(min(max_l, max(l for (_, l, _) in blocks)))
        self.max_l = max_l
        # group storage blocks by l (channels concatenated across layers)
        self.blocks_by_l: List[List[Tuple[int, int]]] = []
        for l in range(max_l + 1):
            group = [(off, mul) for (off, l_b, mul) in blocks if l_b == l]
            if not group:
                raise ValueError(f"no irreps blocks with l={l} in {irreps_per_layer}")
            self.blocks_by_l.append(group)
        self.rc = float(rc)
        self.edge_width = float(edge_width)
        self.proj = int(proj)
        self.zs = [int(z) for z in zs]
        self.register_buffer("y_scale", torch.tensor(float(y_scale)))

        self.proj_maps = torch.nn.ModuleList()
        for l in range(max_l + 1):
            c_in = sum(mul for (_, mul) in self.blocks_by_l[l])
            self.proj_maps.append(torch.nn.Linear(c_in, proj, bias=False))
        self.n_inv = proj * (max_l + 1)

        centers = torch.linspace(0.0, rc, n_rbf)
        self.register_buffer("rbf_centers", centers)
        self.rbf_gamma = (n_rbf / rc) ** 2

        self.gate_mlp = torch.nn.Sequential(
            torch.nn.Linear(n_rbf + len(self.zs), gate_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(gate_hidden, self.n_inv),
        )
        self.out_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.n_inv + 1, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )
        torch.nn.init.zeros_(self.out_mlp[-1].weight)
        torch.nn.init.zeros_(self.out_mlp[-1].bias)

    # -- per-graph precomputations -------------------------------------
    def project_feats(self, node_feats: torch.Tensor) -> List[torch.Tensor]:
        """[N, D] -> per-l projected blocks [N, proj, 2l+1] (equivariant)."""
        out: List[torch.Tensor] = []
        for l, group in enumerate(self.blocks_by_l):
            chunks = [
                node_feats[:, off : off + mul * (2 * l + 1)].reshape(-1, mul, 2 * l + 1)
                for (off, mul) in group
            ]
            stacked = torch.cat(chunks, dim=1)  # [N, C_l, 2l+1]
            w = self.proj_maps[l].weight.to(stacked.dtype)  # [proj, C_l]
            out.append(torch.einsum("pc,nck->npk", w, stacked))
        return out

    def species_onehot(self, node_z: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        oh = torch.zeros((node_z.shape[0], len(self.zs)), dtype=dtype, device=node_z.device)
        for i, z in enumerate(self.zs):
            oh[:, i] = (node_z == z).to(dtype)
        return oh

    # -- core chunk evaluation ------------------------------------------
    def _eval_chunk(
        self,
        pts: torch.Tensor,          # [P, 3] cartesian
        proj_flat: torch.Tensor,    # [N, n_inv * ...] flattened projected feats
        species_oh: torch.Tensor,   # [N, S]
        frac_atoms: torch.Tensor,   # [N, 3]
        cell: torch.Tensor,         # [3, 3]
        inv_cell: torch.Tensor,     # [3, 3]
        z_lo: torch.Tensor,
        z_hi: torch.Tensor,
    ) -> torch.Tensor:
        dt = pts.dtype
        frac_p = pts @ inv_cell
        d = frac_p[:, None, :] - frac_atoms[None, :, :]
        d = torch.cat([d[..., 0:2] - torch.round(d[..., 0:2]), d[..., 2:3]], dim=-1)
        dc = torch.einsum("pni,ij->pnj", d, cell)
        dist = torch.linalg.norm(dc, dim=-1)
        mask = dist < self.rc
        idx = mask.nonzero(as_tuple=False)
        ep, ea = idx[:, 0], idx[:, 1]
        ev = dc[mask]
        r = dist[mask].clamp(min=1.0e-9)

        env = 0.5 * (torch.cos(torch.pi * torch.clamp(r / self.rc, max=1.0)) + 1.0)
        sh = o3.spherical_harmonics(
            list(range(len(self.blocks_by_l))), ev, normalize=True, normalization="component"
        )
        # unflatten projected feats per l and contract with Ylm
        invs = []
        col = 0
        for l in range(len(self.blocks_by_l)):
            width = self.proj * (2 * l + 1)
            fb = proj_flat[:, col : col + width].reshape(-1, self.proj, 2 * l + 1)[ea]
            y = sh[:, l * l : (l + 1) * (l + 1)]
            invs.append(torch.einsum("emk,ek->em", fb, y))
            col += width
        inv_all = torch.cat(invs, dim=-1)  # [E, n_inv]

        rbf = torch.exp(-self.rbf_gamma * (r[:, None] - self.rbf_centers.to(dt)[None, :]) ** 2)
        gate = self.gate_mlp(torch.cat([rbf, species_oh[ea]], dim=-1).to(self.gate_mlp[0].weight.dtype)).to(dt)
        msg = inv_all * gate * env[:, None]

        probe = pts.new_zeros((pts.shape[0], self.n_inv))
        probe.index_add_(0, ep, msg)
        coord = pts.new_zeros((pts.shape[0],))
        coord.index_add_(0, ep, env)

        raw = self.out_mlp(
            torch.cat([probe, coord[:, None]], dim=-1).to(self.out_mlp[0].weight.dtype)
        ).squeeze(-1).to(dt)

        # smooth z-window
        z = pts[:, 2]
        w_lo = torch.clamp((z - z_lo) / self.edge_width, 0.0, 1.0)
        w_hi = torch.clamp((z_hi - z) / self.edge_width, 0.0, 1.0)
        wz = 0.5 * (1.0 - torch.cos(torch.pi * w_lo)) * 0.5 * (1.0 - torch.cos(torch.pi * w_hi))
        return raw * self.y_scale.to(dt) * wz

    # -- public evaluators ------------------------------------------------
    def points_residual(
        self,
        node_feats: torch.Tensor,   # [N, D]
        node_z: torch.Tensor,       # [N] atomic numbers
        positions: torch.Tensor,    # [N, 3] cartesian (detached by caller)
        cell: torch.Tensor,         # [3, 3]
        points: torch.Tensor,       # [P, 3] cartesian
        z_window: Tuple[float, float],
        chunk: int = 16384,
        use_ckpt: bool = False,
    ) -> torch.Tensor:
        dt = node_feats.dtype
        cell = cell.to(dt)
        inv_cell = torch.linalg.inv(cell)
        frac_atoms = positions.to(dt) @ inv_cell
        proj_blocks = self.project_feats(node_feats)
        proj_flat = torch.cat([b.reshape(b.shape[0], -1) for b in proj_blocks], dim=-1)
        species_oh = self.species_onehot(node_z, dt)
        z_lo = cell.new_tensor(float(z_window[0]))
        z_hi = cell.new_tensor(float(z_window[1]))

        outs = []
        for c0 in range(0, points.shape[0], chunk):
            pts = points[c0 : c0 + chunk].to(dt)
            if use_ckpt and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint as _ckpt

                val = _ckpt(
                    self._eval_chunk, pts, proj_flat, species_oh, frac_atoms,
                    cell, inv_cell, z_lo, z_hi, use_reentrant=False,
                )
            else:
                val = self._eval_chunk(
                    pts, proj_flat, species_oh, frac_atoms, cell, inv_cell, z_lo, z_hi
                )
            outs.append(val)
        return torch.cat(outs, dim=0)

    def profile_z(
        self,
        node_feats: torch.Tensor,
        node_z: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        z_window: Tuple[float, float],
        nz: int = 512,
        nxy: int = 8,
        chunk: int = 16384,
        use_ckpt: bool = False,
    ) -> torch.Tensor:
        """Plane-averaged residual (physical density units) on a uniform z grid."""
        dt = node_feats.dtype
        dev = node_feats.device
        fx = (torch.arange(nxy, device=dev, dtype=dt) + 0.5) / nxy
        fz = torch.arange(nz, device=dev, dtype=dt) / nz
        gx, gy, gz = torch.meshgrid(fx, fx, fz, indexing="ij")
        frac = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
        pts = frac @ cell.to(dt)
        vals = self.points_residual(
            node_feats, node_z, positions, cell, pts, z_window, chunk, use_ckpt
        )
        return vals.view(nxy, nxy, nz).mean(dim=(0, 1))
