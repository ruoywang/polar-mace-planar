"""TorchScript-compatibility shims for the pinned graph_longrange package.

graph_longrange 0.4.0 stays unmodified on disk; a few of its methods use
constructs TorchScript cannot script (star-unpacked shapes, out=-into-slice
writes, decorated wrappers, tensor tuple-unpacking, get_default_dtype).
This module installs mathematically identical, scriptable rewrites at import
time. All replacement functions live at module level so TorchScript resolves
their cross-references as plain module globals; numeric constants are inlined
as literals (TorchScript rejects closed-over/global python floats here).
Numerical equivalence is guarded by the training identity probe.
"""
from typing import Dict, List, Optional, Tuple

import torch

from mace.tools.scatter import scatter_sum

# FIELD_CONSTANT == graph_longrange.utils.FIELD_CONSTANT == 180.95128022746667
# 4*pi == 12.566370614359172  (kept as literals: see module docstring)


def _rid_forward(self, k_mods: torch.Tensor) -> torch.Tensor:
    k2 = k_mods * k_mods
    exp_term = torch.exp(-0.5 * k2.unsqueeze(-1) * self.sigma2)
    if self.max_l == 0:
        out = self.pref0 * exp_term
        return out.unsqueeze(-1)
    a = self.pref0 * exp_term
    b = self.pref1 * (k_mods.unsqueeze(-1) * exp_term)
    return torch.stack((a, b), dim=-1)


def _gto_evaluate_fourier_basis(
    self, k_moduli: torch.Tensor, yklm: torch.Tensor
) -> torch.Tensor:
    fnlk = self.radial_spline(k_moduli) * self.cl_scale
    expanded_fnlk = torch.index_select(fnlk, -1, self.expanded_l_indices)
    xnlk = expanded_fnlk * yklm.unsqueeze(-2)
    re = xnlk * self.real_phase_factors
    im = xnlk * self.imag_phase_factors
    return torch.stack((re, im), dim=-1)


def _gto_selfinteraction_forward(self, charge_density: torch.Tensor) -> torch.Tensor:
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


@torch.jit.ignore
def _gto_selfinteraction_features_dim(self) -> int:
    return int(self.features_irreps.dim)


def batch_complete_graph_excluding_self_duplicates_vector(
    batch: torch.Tensor, N: int
) -> torch.Tensor:
    with torch.no_grad():
        batch = batch.long()
        orig = torch.arange(batch.size(0), device=batch.device)
        batch2 = batch.repeat_interleave(N)
        orig2 = orig.repeat_interleave(N)
        G = int(batch2.max().item()) + 1
        edges: List[torch.Tensor] = []
        for g in range(G):
            mask = batch2 == g
            nodes = mask.nonzero().view(-1)
            if nodes.numel() <= 1:
                continue
            D = nodes.size(0)
            row = nodes.view(-1, 1).expand(-1, D).reshape(-1)
            col = nodes.view(1, -1).expand(D, -1).reshape(-1)
            orig_row = orig2[mask].view(-1, 1).expand(-1, D).reshape(-1)
            orig_col = orig2[mask].view(1, -1).expand(D, -1).reshape(-1)
            keep = orig_row != orig_col
            edges.append(torch.stack([row[keep], col[keep]], dim=0))
        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=batch.device)
        return torch.cat(edges, dim=1)


def charges_features_from_graph(
    charges: torch.Tensor,
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    total_width_factors: torch.Tensor,
) -> torch.Tensor:
    num_nodes = positions.shape[0]
    sender = edge_index[0]
    receiver = edge_index[1]
    R_ij = positions[sender] - positions[receiver]
    d_ij = torch.norm(R_ij, dim=-1, keepdim=True)
    smooth_reciprocal = torch.erf(0.5 * d_ij / total_width_factors) / (d_ij + 1e-6)
    features = scatter_sum(
        charges[sender].unsqueeze(-1) * smooth_reciprocal,
        receiver,
        dim=0,
        dim_size=num_nodes,
    )
    return 180.95128022746667 * features / 12.566370614359172


def charges_energy_from_graph(
    charges: torch.Tensor,
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    density_smearing_width: float,
) -> torch.Tensor:
    sender = edge_index[0]
    receiver = edge_index[1]
    R_ij = positions[receiver] - positions[sender]
    d_ij = torch.linalg.norm(R_ij, dim=-1)
    smooth_reciprocal = torch.erf(d_ij * 0.5 / density_smearing_width) / (
        torch.abs(d_ij) + 1e-6
    )
    edge_energy = (
        0.5
        * 180.95128022746667
        * smooth_reciprocal
        * charges[sender]
        * charges[receiver]
        / 12.566370614359172
    )
    if edge_energy.numel() == 0:
        return torch.zeros(
            (int(batch.max().item()) + 1,), dtype=charges.dtype, device=charges.device
        )
    node_energies = scatter_sum(src=edge_energy.squeeze(-1), index=receiver, dim=-1)
    return scatter_sum(src=node_energies, index=batch, dim=-1)


def _rsfd_call_density_1_feats_1(
    self,
    source_feats: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    extended_positions = positions.repeat_interleave(4, dim=0)
    extended_positions[1::4] += self.x
    extended_positions[2::4] += self.y
    extended_positions[3::4] += self.z
    extended_batch = batch.repeat_interleave(4)
    charges = torch.zeros_like(extended_positions[:, 0])
    charges[1::4] = source_feats[:, 3] / self.offset
    charges[2::4] = source_feats[:, 1] / self.offset
    charges[3::4] = source_feats[:, 2] / self.offset
    charges[0::4] = source_feats[:, 0] - (
        charges[1::4] + charges[2::4] + charges[3::4]
    )
    edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 4)
    scalar_features = charges_features_from_graph(
        charges=charges,
        positions=extended_positions,
        edge_index=edge_index,
        batch=extended_batch,
        total_width_factors=self.total_width_factors.unsqueeze(0),
    )
    all_features = torch.zeros(
        batch.size(0),
        4 * self.num_radial,
        dtype=positions.dtype,
        device=batch.device,
    )
    all_features[:, : self.num_radial] = self.l0_factors * scalar_features[0::4]
    all_features[:, self.num_radial :: 3] = self.l1_factors * (
        scalar_features[2::4] - scalar_features[0::4]
    )
    all_features[:, self.num_radial + 1 :: 3] = self.l1_factors * (
        scalar_features[3::4] - scalar_features[0::4]
    )
    all_features[:, self.num_radial + 2 :: 3] = self.l1_factors * (
        scalar_features[1::4] - scalar_features[0::4]
    )
    return all_features


def _rsfd_forward(
    self,
    source_feats: torch.Tensor,
    node_positions: torch.Tensor,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if self.density_max_l == 0 and self.projection_max_l == 0:
        features = self.call_density_0_feats_0(
            source_feats.squeeze(-2), node_positions, batch
        )
    elif self.density_max_l == 1 and self.projection_max_l == 0:
        all_feats = self.call_density_1_feats_1(
            source_feats.squeeze(-2), node_positions, batch
        )
        features = all_feats[:, : self.num_radial]
    elif self.density_max_l == 0 and self.projection_max_l == 1:
        padded_source_feats = torch.zeros(
            source_feats.shape[0],
            4,
            dtype=source_feats.dtype,
            device=source_feats.device,
        )
        padded_source_feats[:, 0] = source_feats[:, 0, 0]
        features = self.call_density_1_feats_1(
            padded_source_feats, node_positions, batch
        )
    else:
        features = self.call_density_1_feats_1(
            source_feats.squeeze(-2), node_positions, batch
        )
    self_interaction_terms = self.self_interaction(source_feats.squeeze(-2))
    if self.include_self_interaction:
        features += self_interaction_terms
    return features, self_interaction_terms, None


def _cpb_forward(
    self,
    charge_coefficients: torch.Tensor,
    positions: torch.Tensor,
    volumes: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    # CUBIC_MADELUNG = 2.837297; pi literals as in the module docstring
    total_charge = scatter_sum(src=charge_coefficients[:, 0], index=batch, dim=-1)
    q_r = positions * charge_coefficients[:, 0].unsqueeze(-1)
    total_dipole = scatter_sum(src=q_r, index=batch, dim=0)
    r_squared = torch.sum(torch.square(positions), dim=-1)
    q_rr = r_squared * charge_coefficients[:, 0]
    quadrupole = scatter_sum(src=q_rr, index=batch, dim=0)

    if self.density_max_l > 0:
        sel = torch.tensor([3, 1, 2], dtype=torch.long, device=charge_coefficients.device)
        local_dipoles_cartesian = torch.index_select(charge_coefficients, -1, sel)
        total_dipole = total_dipole + scatter_sum(
            src=local_dipoles_cartesian, index=batch, dim=-2
        )
        p_dot_r = torch.einsum("bi,bi->b", positions, local_dipoles_cartesian)
        quadrupole = quadrupole + 2 * scatter_sum(src=p_dot_r, index=batch, dim=0)

    spread_dipoles = torch.index_select(total_dipole, 0, batch)
    spread_total_charge = torch.index_select(total_charge, 0, batch)
    spread_volumes = torch.index_select(volumes, 0, batch)
    spread_total_quadrupole = torch.index_select(quadrupole, 0, batch)

    node_fields = torch.zeros(
        (positions.shape[0], 4),
        dtype=positions.dtype,
        device=positions.device,
    )
    Ls = torch.pow(volumes, 0.333333)
    delta_V_0 = 2.837297 * self.const * total_charge / Ls
    node_delta_V = torch.index_select(delta_V_0, 0, batch)
    node_delta_V += (
        -self.const
        * 2
        * 3.141592653589793
        * spread_total_charge
        * r_squared
        / (3 * spread_volumes)
    )
    node_delta_V += (
        self.const
        * 4
        * 3.141592653589793
        * torch.einsum("bi,bi->b", spread_dipoles, positions)
        / (3 * spread_volumes)
    )
    node_delta_V += (
        -self.const * 2 * 3.141592653589793 * spread_total_quadrupole / (3 * spread_volumes)
    )
    node_fields[:, 0] = node_delta_V
    quantity_a = spread_dipoles - spread_total_charge.unsqueeze(-1) * positions
    node_fields[:, 1:] = (
        4 * 3.141592653589793 * self.const * quantity_a / (3 * spread_volumes.unsqueeze(-1))
    )
    return node_fields


def _is_batch1(batch: torch.Tensor) -> bool:
    if batch.numel() == 0:
        return True
    return int(batch.max()) == 0


def _get_total_dipole_z(
    source_feats: torch.Tensor, node_positions: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    charges = source_feats[:, 0]
    if _is_batch1(batch):
        total_dipole_z = (node_positions[:, 2] * charges).sum().unsqueeze(0)
        if source_feats.shape[-1] > 1:
            local_dipoles = source_feats[:, 1:4]
            total_dipole_z = total_dipole_z + local_dipoles.sum(dim=0)[1].unsqueeze(0)
        return total_dipole_z
    total_dipole_z = scatter_sum(
        src=node_positions[:, 2] * charges, index=batch, dim=0
    )
    if source_feats.shape[-1] > 1:
        local_dipoles = source_feats[:, 1:4]
        total_dipole_p = scatter_sum(src=local_dipoles, index=batch, dim=0)
        total_dipole_z = total_dipole_z + total_dipole_p[:, 1]
    return total_dipole_z


def slab_dipole_correction_energy(
    source_feats: torch.Tensor,
    node_positions: torch.Tensor,
    volumes: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    total_dipole_z = _get_total_dipole_z(source_feats, node_positions, batch)
    A = 180.95128022746667 / 12.566370614359172
    dipole_norms_squared = total_dipole_z ** 2
    return A * 2 * 3.141592653589793 * dipole_norms_squared / volumes


def slab_dipole_correction_total_field(
    total_dipole: torch.Tensor, volumes: torch.Tensor
) -> torch.Tensor:
    A = 180.95128022746667 / 12.566370614359172
    total_field_z = A * 4 * 3.141592653589793 * total_dipole[:, 2] / volumes
    total_field = torch.zeros_like(total_dipole)
    total_field[:, 2] = total_field_z
    return total_field


def slab_dipole_correction_node_fields(
    source_feats: torch.Tensor,
    node_positions: torch.Tensor,
    volumes: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    total_dipole_z = _get_total_dipole_z(source_feats, node_positions, batch)
    A = 180.95128022746667 / 12.566370614359172
    total_field_z = A * 4 * 3.141592653589793 * total_dipole_z / volumes
    spread_total_field_z = torch.index_select(total_field_z, 0, batch)
    delta_V_nodes = spread_total_field_z * node_positions[:, 2]
    node_fields = torch.zeros(
        (node_positions.shape[0], 4),
        dtype=node_positions.dtype,
        device=node_positions.device,
    )
    node_fields[:, 0] = delta_V_nodes
    node_fields[:, 3] = spread_total_field_z
    return node_fields


def _npfc_forward(
    self,
    source_feats: torch.Tensor,
    node_positions: torch.Tensor,
    batch: torch.Tensor,
    volumes: torch.Tensor,
    pbc: torch.Tensor,
    correction_mode: Optional[int] = None,
    correction_node_masks: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    # CORRECTION_MODE_{PBC,MOLECULE,SLAB,MIXED} = 0,1,2,3 (module literals)
    source_feats_lm = source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
    if correction_mode is None:
        pbc_bool = pbc.to(dtype=torch.bool)
        is_pbc_graph = pbc_bool.all(dim=1)
        is_molecule_graph = (~pbc_bool).all(dim=1)
        is_slab_graph = pbc_bool[:, 0] & pbc_bool[:, 1] & (~pbc_bool[:, 2])
        if bool(is_pbc_graph.all()):
            correction_mode = 0
        elif bool(is_molecule_graph.all()):
            correction_mode = 1
        elif bool(is_slab_graph.all()):
            correction_mode = 2
        else:
            correction_mode = 3
            correction_node_masks = {
                "is_molecule_node": torch.index_select(is_molecule_graph, 0, batch),
                "is_slab_node": torch.index_select(is_slab_graph, 0, batch),
            }
    if correction_mode == 0:
        n_nodes = node_positions.size(0)
        return node_positions.new_zeros(
            (n_nodes, self.displaced_interactions.projections_dim)
        )
    if correction_mode == 1:
        node_fields = self.self_field(
            charge_coefficients=source_feats_lm,
            positions=node_positions,
            volumes=volumes,
            batch=batch,
        )
        return self.displaced_interactions(
            batch=batch, positions=node_positions, node_fields=node_fields
        )
    if correction_mode == 2:
        node_fields = slab_dipole_correction_node_fields(
            source_feats=source_feats_lm,
            node_positions=node_positions,
            volumes=volumes,
            batch=batch,
        )
        return self.displaced_interactions(
            batch=batch, positions=node_positions, node_fields=node_fields
        )
    if correction_node_masks is None:
        pbc_bool = pbc.to(dtype=torch.bool)
        is_molecule_graph = (~pbc_bool).all(dim=1)
        is_slab_graph = pbc_bool[:, 0] & pbc_bool[:, 1] & (~pbc_bool[:, 2])
        correction_node_masks = {
            "is_molecule_node": torch.index_select(is_molecule_graph, 0, batch),
            "is_slab_node": torch.index_select(is_slab_graph, 0, batch),
        }
    node_fields_molecule = self.self_field(
        charge_coefficients=source_feats_lm,
        positions=node_positions,
        volumes=volumes,
        batch=batch,
    )
    node_fields_slab = slab_dipole_correction_node_fields(
        source_feats=source_feats_lm,
        node_positions=node_positions,
        volumes=volumes,
        batch=batch,
    )
    is_molecule = correction_node_masks["is_molecule_node"]
    is_slab = correction_node_masks["is_slab_node"]
    node_fields = torch.zeros_like(node_fields_molecule)
    node_fields[is_molecule] = node_fields_molecule[is_molecule]
    node_fields[is_slab] = node_fields_slab[is_slab]
    return self.displaced_interactions(
        batch=batch, positions=node_positions, node_fields=node_fields
    )


def _mdcb_forward(
    self,
    charge_coefficients: torch.Tensor,
    positions: torch.Tensor,
    volumes: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    total_charge = scatter_sum(src=charge_coefficients[:, 0], index=batch, dim=-1)
    charge_norms_squared = torch.square(total_charge)
    q_r = positions * charge_coefficients[:, 0].unsqueeze(-1)
    total_dipole = scatter_sum(src=q_r, index=batch, dim=0)
    r_squared = torch.sum(torch.square(positions), dim=-1)
    q_rr = r_squared * charge_coefficients[:, 0]
    quadrupole = scatter_sum(src=q_rr, index=batch, dim=0)
    if self.density_max_l > 0:
        sel = torch.tensor([3, 1, 2], dtype=torch.long, device=charge_coefficients.device)
        local_dipoles_cartesian = torch.index_select(charge_coefficients, -1, sel)
        total_dipole = total_dipole + scatter_sum(
            src=local_dipoles_cartesian, index=batch, dim=-2
        )
        p_dot_r = torch.einsum("bi,bi->b", positions, local_dipoles_cartesian)
        quadrupole = quadrupole + 2 * scatter_sum(src=p_dot_r, index=batch, dim=0)
    Ls = torch.pow(volumes, 0.3333)
    delta_E = 0.5 * 2.837297 * self.const * charge_norms_squared / Ls
    delta_E += (
        2
        * self.const
        * 3.141592653589793
        * torch.sum(torch.square(total_dipole), dim=-1)
        / (3 * volumes)
    )
    delta_E += -2 * self.const * 3.141592653589793 * total_charge * quadrupole / (3 * volumes)
    return delta_E


def _as_method(fn, name: str):
    """TorchScript registers methods by __name__; align it with the slot."""
    fn.__name__ = name
    fn.__qualname__ = name
    return fn


def apply() -> None:
    from graph_longrange import gto_utils
    from graph_longrange import realspace_electrostatics as rse

    gto_utils.RadialIntegralDirect.forward = _as_method(_rid_forward, "forward")
    gto_utils.GTOBasis._evaluate_fourier_basis = _as_method(
        _gto_evaluate_fourier_basis, "_evaluate_fourier_basis"
    )
    gto_utils.GTOSelfInteractionBlock._features_dim = _as_method(
        _gto_selfinteraction_features_dim, "_features_dim"
    )
    gto_utils.GTOSelfInteractionBlock.forward = _as_method(
        _gto_selfinteraction_forward, "forward"
    )
    rse.batch_complete_graph_excluding_self_duplicates_vector = (
        batch_complete_graph_excluding_self_duplicates_vector
    )
    rse.charges_features_from_graph = charges_features_from_graph
    rse.charges_energy_from_graph = charges_energy_from_graph
    rse.RealSpaceFiniteDifferenceElectrostaticFeatures.call_density_1_feats_1 = (
        _as_method(_rsfd_call_density_1_feats_1, "call_density_1_feats_1")
    )
    rse.RealSpaceFiniteDifferenceElectrostaticFeatures.forward = _as_method(
        _rsfd_forward, "forward"
    )
    from graph_longrange import slabs as gl_slabs
    from graph_longrange import features as gl_features
    gl_slabs.CorrectivePotentialBlock.forward = _as_method(_cpb_forward, "forward")
    gl_slabs.MonopoleDipoleCorrectionBlock.forward = _as_method(_mdcb_forward, "forward")
    gl_slabs._is_batch1 = _is_batch1
    gl_slabs._get_total_dipole_z = _get_total_dipole_z
    gl_slabs.slab_dipole_correction_energy = slab_dipole_correction_energy
    gl_slabs.slab_dipole_correction_total_field = slab_dipole_correction_total_field
    gl_slabs.slab_dipole_correction_node_fields = slab_dipole_correction_node_fields
    gl_features.slab_dipole_correction_node_fields = slab_dipole_correction_node_fields
    gl_features.NonPeriodicFeatureCorrections.forward = _as_method(_npfc_forward, "forward")
    # GTOElectrostaticFeatures' dict-cache API is not scriptable; keep the
    # whole module as a python fallback (recursive scripting skips ignored
    # forwards). mace calls it via jit-ignored boundary methods anyway.
    gl_features.GTOElectrostaticFeatures.forward = torch.jit.ignore(
        gl_features.GTOElectrostaticFeatures.forward
    )
    # the jit-ignored boundary methods in extensions.py call these two on the
    # SCRIPTED module at runtime; ignore keeps a python binding on the
    # RecursiveScriptModule (plain undecorated methods are dropped).
    gl_features.GTOElectrostaticFeatures.precompute_geometry = torch.jit.ignore(
        gl_features.GTOElectrostaticFeatures.precompute_geometry
    )
    gl_features.GTOElectrostaticFeatures.forward_dynamic = torch.jit.ignore(
        gl_features.GTOElectrostaticFeatures.forward_dynamic
    )
    from graph_longrange import energy as gl_energy
    gl_energy.GTOElectrostaticEnergy.forward = torch.jit.ignore(
        gl_energy.GTOElectrostaticEnergy.forward
    )


apply()
