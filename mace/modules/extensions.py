import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

try:
    from graph_longrange.energy import GTOElectrostaticEnergy
    from graph_longrange.features import GTOElectrostaticFeatures
    from graph_longrange.gto_utils import (
        DisplacedGTOExternalFieldBlock,
        gto_basis_kspace_cutoff,
    )
    from graph_longrange.kspace import compute_k_vectors_flat
    from graph_longrange.utils import FIELD_CONSTANT

    GRAPH_LONGRANGE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    GRAPH_LONGRANGE_AVAILABLE = False

from mace.modules.blocks import (
    LinearReadoutBlock,
    NonLinearBiasReadoutBlock,
    NonLinearReadoutBlock,
)
from mace.modules.models import ScaleShiftMACE
from mace.modules.utils import get_atomic_virials_stresses, get_outputs, prepare_graph
from mace.modules.wrapper_ops import (
    CuEquivarianceConfig,
    OEQConfig,
    TransposeIrrepsLayoutWrapper,
)
from mace.tools.scatter import scatter_mean, scatter_sum

from .field_blocks import (
    EnvironmentDependentSpinSourceBlock,
    MultiLayerFeatureMixer,
    field_readout_blocks,
    field_update_blocks,
)
from .solvent_charge_layer import (
    _axis_box_length,
    _truncated_gaussian_mean,
    compute_interface_qm_summaries,
    compute_density_threshold_crossing_from_baseline_profile,
    compute_density_threshold_crossing_from_mace_multipoles,
    compute_oh_structure_crossings,
    compute_ze_crossing_from_mace_multipoles,
    periodic_gaussian_layer_potential_field_nodes,
    periodic_profile_layer_potential_field_nodes,
    require_explicit_valence_electrons,
    predict_potential_from_dipole_and_solvent_layer,
    predict_potential_from_dipole_and_solvent_mu,
)
from .utils import compute_total_charge_dipole_permuted


def _copy_mace_readout(
    mace_readout: torch.nn.Module, cueq_config: Optional[CuEquivarianceConfig] = None
) -> torch.nn.Module:
    """
    Helper function to copy a MACE readout block.
    """
    if isinstance(mace_readout, LinearReadoutBlock):
        return LinearReadoutBlock(
            irreps_in=mace_readout.linear.irreps_in,  # type: ignore
            irrep_out=mace_readout.linear.irreps_out,  # type: ignore
            cueq_config=cueq_config,
        )
    if isinstance(mace_readout, NonLinearReadoutBlock):  # type: ignore
        return NonLinearReadoutBlock(
            irreps_in=mace_readout.linear_1.irreps_in,  # type: ignore
            MLP_irreps=mace_readout.hidden_irreps,
            gate=mace_readout.non_linearity._modules["acts"][  # pylint: disable=W0212
                0
            ].f,
            irrep_out=mace_readout.linear_2.irreps_out,  # type: ignore
            num_heads=mace_readout.num_heads,
            cueq_config=cueq_config,
        )
    raise TypeError("Unsupported readout type.")


def _get_readout_input_dim(block: torch.nn.Module) -> int:
    if isinstance(block, LinearReadoutBlock):
        return block.linear.irreps_in.dim  # type: ignore
    if isinstance(block, NonLinearReadoutBlock):  # type: ignore
        return block.linear_1.irreps_in.dim  # type: ignore
    raise TypeError("Unsupported readout type for input dimension retrieval.")


def _load_center_density_baselines_npz(path: str) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
    arrays = np.load(Path(path))
    required = {"sample_ids", "z_A", "raw_neutral_e"}
    missing = required.difference(arrays.files)
    if missing:
        raise ValueError(
            f"Missing arrays in baseline density file {path}: {sorted(missing)}"
        )
    sample_ids = np.asarray(arrays["sample_ids"], dtype=np.int64)
    z_values = np.asarray(arrays["z_A"], dtype=np.float64)
    neutral_values = np.asarray(arrays["raw_neutral_e"], dtype=np.float64)
    if z_values.shape != neutral_values.shape:
        raise ValueError(f"z_A and raw_neutral_e shape mismatch in {path}")
    if z_values.shape[0] != sample_ids.shape[0]:
        raise ValueError(f"sample_ids length does not match z_A in {path}")
    return {
        int(sample_id): (z_values[idx], neutral_values[idx])
        for idx, sample_id in enumerate(sample_ids)
    }


def _cell_area_for_axis_batch(cells: torch.Tensor, axis: int) -> torch.Tensor:
    if axis == 0:
        v1, v2 = cells[:, 1, :], cells[:, 2, :]
    elif axis == 1:
        v1, v2 = cells[:, 0, :], cells[:, 2, :]
    else:
        v1, v2 = cells[:, 0, :], cells[:, 1, :]
    return torch.clamp(
        torch.linalg.norm(torch.cross(v1, v2, dim=-1), dim=-1), min=1.0e-12
    )


def _slab_dipole_correction_delta(
    explicit_dipole: torch.Tensor,
    total_dipole: torch.Tensor,
    volume: torch.Tensor,
    pbc: torch.Tensor,
    axis: int,
) -> torch.Tensor:
    """Replace explicit-only slab correction with compensated-total correction."""
    if int(axis) != 2:
        raise NotImplementedError("MACEPolar slab correction is defined for z-axis slabs only")
    pbc_g = pbc.view(-1, 3).to(torch.bool)
    slab = torch.ones(3, dtype=torch.bool, device=pbc_g.device)
    slab[axis] = False
    slab_mask = torch.all(pbc_g == slab, dim=1).to(total_dipole.dtype)
    volume_g = torch.clamp(volume.view(-1).to(total_dipole.dtype), min=1.0e-12)
    const = total_dipole.new_tensor(0.5 * float(FIELD_CONSTANT))
    explicit_mu = explicit_dipole[:, axis]
    total_mu = total_dipole[:, axis]
    delta = const * (torch.square(total_mu) - torch.square(explicit_mu)) / volume_g
    return delta * slab_mask


def _slab_compensation_gaussian_potential_field_nodes(
    total_charge: torch.Tensor,
    center: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch: torch.Tensor,
    positions: torch.Tensor,
    sigma_g: float,
    axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cells = cell
    if cells.dim() == 2 and cells.shape[1] == 3:
        cells = cells.view(-1, 3, 3)
    total_charge_g = total_charge.view(-1).to(positions.dtype)
    center_g = center.view(-1).to(positions.dtype)
    pbc_g = pbc.view(-1, 3).to(torch.bool)
    slab = torch.ones(3, dtype=torch.bool, device=pbc_g.device)
    slab[axis] = False
    slab_mask = torch.all(pbc_g == slab, dim=1).to(positions.dtype)

    area = _cell_area_for_axis_batch(cells.to(positions.dtype), axis)
    solvent_charge = -total_charge_g
    prefactor = positions.new_tensor(0.5 * float(FIELD_CONSTANT)) * solvent_charge / area
    sigma = positions.new_tensor(float(max(sigma_g, 1.0e-12)))
    inv_sqrt2 = positions.new_tensor(1.0 / math.sqrt(2.0))
    sqrt_2_over_pi = positions.new_tensor(math.sqrt(2.0 / math.pi))

    dz = positions[:, axis] - center_g[batch]
    u = dz * inv_sqrt2 / sigma
    shape_int = dz * torch.erf(u) + sqrt_2_over_pi * sigma * torch.exp(-(u * u))
    phi = -prefactor[batch] * shape_int
    field = positions.new_zeros(positions.shape)
    field[:, axis] = -prefactor[batch] * torch.erf(u)
    mask = slab_mask[batch]
    return phi * mask, field * mask.unsqueeze(-1)


def _slab_compensation_gaussian_features(
    external_field_block: torch.nn.Module,
    total_charge: torch.Tensor,
    center: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch: torch.Tensor,
    positions: torch.Tensor,
    sigma_g: float,
    feature_sigmas: List[float],
    axis: int,
    convention: str = "periodic",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matrix = external_field_block.matrix.to(positions.dtype)
    features = positions.new_zeros((positions.shape[0], matrix.shape[0]))
    phi_ref = positions.new_zeros(positions.shape[0])
    field_ref = positions.new_zeros(positions.shape)
    n_sigmas = len(feature_sigmas)
    for i_s, receiver_sigma in enumerate(feature_sigmas):
        if convention == "periodic":
            # Same G!=0 convention as the periodic evaluator used for the
            # explicit density; together with the slab dipole-correction
            # features this reproduces the neutral-total potential without a
            # spurious jellium parabola for charged explicit subsystems.
            phi, field = periodic_gaussian_layer_potential_field_nodes(
                total_charge=total_charge,
                center=center,
                cell=cell,
                pbc=pbc,
                batch=batch,
                positions=positions,
                sigma_g=sigma_g,
                receiver_sigma=float(receiver_sigma),
                axis=axis,
            )
        elif convention == "isolated":
            sigma_eff = math.sqrt(float(sigma_g) ** 2 + float(receiver_sigma) ** 2)
            phi, field = _slab_compensation_gaussian_potential_field_nodes(
                total_charge=total_charge,
                center=center,
                cell=cell,
                pbc=pbc,
                batch=batch,
                positions=positions,
                sigma_g=sigma_eff,
                axis=axis,
            )
        else:
            raise ValueError(
                f"Unsupported solvent_plane_feature_convention: {convention!r}"
            )
        if i_s == 0:
            phi_ref = phi
            field_ref = field
        node_fields = positions.new_zeros((positions.shape[0], 4))
        node_fields[:, 0] = phi
        node_fields[:, 1:] = field
        node_fields = node_fields[:, [0, 3, 1, 2]]
        rows = [i_s]
        if matrix.shape[0] >= n_sigmas * 4:
            rows.extend(range(n_sigmas + i_s * 3, n_sigmas + (i_s + 1) * 3))
        row_index = torch.tensor(rows, dtype=torch.long, device=positions.device)
        features[:, row_index] = torch.einsum(
            "rf,nf->nr", torch.index_select(matrix, 0, row_index), node_fields
        )
    return features, phi_ref, field_ref


def _slab_compensation_profile_features(
    external_field_block: torch.nn.Module,
    profile: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch: torch.Tensor,
    positions: torch.Tensor,
    feature_sigmas: List[float],
    axis: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PB analogue of _slab_compensation_gaussian_features: SCF field
    features of an arbitrary laterally uniform rho_ion(z) profile
    [n_graphs, num_grid] in the periodic G!=0 convention."""
    matrix = external_field_block.matrix.to(positions.dtype)
    features = positions.new_zeros((positions.shape[0], matrix.shape[0]))
    phi_ref = positions.new_zeros(positions.shape[0])
    field_ref = positions.new_zeros(positions.shape)
    n_sigmas = len(feature_sigmas)
    for i_s, receiver_sigma in enumerate(feature_sigmas):
        phi, field = periodic_profile_layer_potential_field_nodes(
            profile=profile,
            cell=cell,
            pbc=pbc,
            batch=batch,
            positions=positions,
            receiver_sigma=float(receiver_sigma),
            axis=axis,
        )
        if i_s == 0:
            phi_ref = phi
            field_ref = field
        node_fields = positions.new_zeros((positions.shape[0], 4))
        node_fields[:, 0] = phi
        node_fields[:, 1:] = field
        node_fields = node_fields[:, [0, 3, 1, 2]]
        rows = [i_s]
        if matrix.shape[0] >= n_sigmas * 4:
            rows.extend(range(n_sigmas + i_s * 3, n_sigmas + (i_s + 1) * 3))
        row_index = torch.tensor(rows, dtype=torch.long, device=positions.device)
        features[:, row_index] = torch.einsum(
            "rf,nf->nr", torch.index_select(matrix, 0, row_index), node_fields
        )
    return features, phi_ref, field_ref


def _slab_compensation_slab_correction_features(
    external_field_block: torch.nn.Module,
    total_charge: torch.Tensor,
    center: torch.Tensor,
    cell: torch.Tensor,
    volume: torch.Tensor,
    pbc: torch.Tensor,
    batch: torch.Tensor,
    positions: torch.Tensor,
    sigma_g: float,
    axis: int,
    solvent_mu_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if int(axis) != 2:
        raise NotImplementedError("MACEPolar slab correction is defined for z-axis slabs only")
    cells = cell
    if cells.dim() == 2 and cells.shape[1] == 3:
        cells = cells.view(-1, 3, 3)
    pbc_g = pbc.view(-1, 3).to(torch.bool)
    slab = torch.ones(3, dtype=torch.bool, device=pbc_g.device)
    slab[axis] = False
    slab_mask = torch.all(pbc_g == slab, dim=1).to(positions.dtype)
    if solvent_mu_override is not None:
        solvent_mu = solvent_mu_override.view(-1).to(positions.dtype)
    else:
        l_axis = torch.stack(
            [_axis_box_length(cells[i], axis) for i in range(cells.shape[0])]
        ).to(positions.dtype)
        layer_mean = _truncated_gaussian_mean(
            center=center.view(-1).to(positions.dtype),
            sigma=sigma_g,
            lower=torch.zeros_like(l_axis),
            upper=l_axis,
        )
        solvent_mu = -total_charge.view(-1).to(positions.dtype) * layer_mean
    volume_g = torch.clamp(volume.view(-1).to(positions.dtype), min=1.0e-12)
    field_z = positions.new_tensor(float(FIELD_CONSTANT)) * solvent_mu / volume_g
    field_z = field_z * slab_mask
    spread_field_z = torch.index_select(field_z, 0, batch)

    node_fields = positions.new_zeros((positions.shape[0], 4))
    node_fields[:, 0] = spread_field_z * positions[:, axis]
    node_fields[:, 1 + axis] = spread_field_z
    node_fields = node_fields[:, [0, 3, 1, 2]]
    return torch.einsum(
        "pf,nf->np", external_field_block.matrix.to(positions.dtype), node_fields
    )


def _slab_compensation_periodic_1d_energy_radial(
    radial_coefficients: torch.Tensor,
    atomic_density_sigmas: List[float],
    total_charge: torch.Tensor,
    center: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch: torch.Tensor,
    positions: torch.Tensor,
    num_graphs: int,
    sigma_g: float,
    axis: int,
    num_grid: int = 512,
    comp_profile: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compensation contribution in the same 1D Poisson convention as profile loss.

    When ``comp_profile`` [num_graphs, num_grid] is given (e/A^3 on the
    z_j = (j + 0.5) * H / num_grid grid, physics sign), it replaces the
    truncated-Gaussian compensation profile; the exact 1D periodic cross +
    self energy is valid for any laterally uniform rho_ion(z)."""
    if comp_profile is not None and int(comp_profile.shape[1]) != int(num_grid):
        raise ValueError(
            "comp_profile second dimension must equal num_grid "
            f"({int(comp_profile.shape[1])} != {int(num_grid)})"
        )
    if int(axis) != 2:
        raise NotImplementedError("1D compensation energy is currently defined for z-axis slabs")
    if radial_coefficients.dim() != 3:
        raise ValueError("radial_coefficients must have shape [n_atoms, n_sigmas, n_coeffs]")

    cells = cell
    if cells.dim() == 2 and cells.shape[1] == 3:
        cells = cells.view(-1, 3, 3)
    pbc_g = pbc.view(-1, 3).to(torch.bool)
    slab = torch.ones(3, dtype=torch.bool, device=pbc_g.device)
    slab[axis] = False
    slab_mask = torch.all(pbc_g == slab, dim=1).to(positions.dtype)

    energies = positions.new_zeros(num_graphs)
    sigma_comp = positions.new_tensor(float(max(sigma_g, 1.0e-12)))
    inv_sqrt2 = positions.new_tensor(1.0 / math.sqrt(2.0))
    sqrt_2pi = positions.new_tensor(math.sqrt(2.0 * math.pi))
    mode = torch.fft.fftfreq(num_grid, d=1.0, device=positions.device).to(positions.dtype)
    mode = mode * float(num_grid)

    for graph_idx in range(num_graphs):
        if graph_idx >= cells.shape[0] or slab_mask[graph_idx] == 0:
            continue
        atom_mask = batch == graph_idx
        if not torch.any(atom_mask):
            continue
        cell_g = cells[graph_idx].to(positions.dtype)
        area = _cell_area_for_axis_batch(cell_g.view(1, 3, 3), axis)[0]
        height = _axis_box_length(cell_g, axis).to(positions.dtype)
        volume = torch.clamp(area * height, min=1.0e-12)
        z_grid = (torch.arange(num_grid, dtype=positions.dtype, device=positions.device) + 0.5)
        z_grid = z_grid * height / float(num_grid)

        z_atom = positions[atom_mask, axis].to(positions.dtype)
        coeff = radial_coefficients[atom_mask].to(positions.dtype)
        rho_exp = z_grid.new_zeros(num_grid)
        for i_s, source_sigma in enumerate(atomic_density_sigmas):
            sigma = z_grid.new_tensor(float(max(source_sigma, 1.0e-12)))
            dz = z_grid[:, None] - z_atom[None, :]
            gaussian = torch.exp(-0.5 * torch.square(dz / sigma)) / (sqrt_2pi * sigma)
            c = coeff[:, i_s, :]
            rho_exp = rho_exp + (gaussian * (c[:, 0] / area)[None, :]).sum(dim=1)
            if c.shape[1] > 1:
                gaussian_d1 = -(dz / (sigma * sigma)) * gaussian
                rho_exp = rho_exp + (
                    gaussian_d1 * ((-c[:, 2] / area)[None, :])
                ).sum(dim=1)
            if c.shape[1] >= 9:
                sigma2 = sigma * sigma
                gaussian_d2 = ((dz * dz) / (sigma2 * sigma2) - 1.0 / sigma2) * gaussian
                rho_exp = rho_exp + (
                    gaussian_d2 * (((c[:, 6] / 3.0) / area)[None, :])
                ).sum(dim=1)

        if comp_profile is not None:
            rho_comp = comp_profile[graph_idx].to(positions.dtype)
        else:
            lower = z_grid.new_zeros(())
            upper = height
            center_g = center.view(-1).to(positions.dtype)[graph_idx]
            a = (lower - center_g) / sigma_comp
            b = (upper - center_g) / sigma_comp
            norm = 0.5 * (
                torch.erf(b * inv_sqrt2) - torch.erf(a * inv_sqrt2)
            )
            norm = torch.clamp(norm, min=1.0e-12)
            dz_comp = z_grid - center_g
            gauss_profile = torch.exp(-0.5 * torch.square(dz_comp / sigma_comp))
            gauss_profile = gauss_profile / (sqrt_2pi * sigma_comp * norm)
            solvent_charge = -total_charge.view(-1).to(positions.dtype)[graph_idx]
            rho_comp = (solvent_charge / area) * gauss_profile

        raw_exp = rho_exp * volume
        raw_comp = rho_comp * volume
        g_abs = torch.abs(2.0 * math.pi * mode / height)
        phi_rec = torch.zeros(
            num_grid, dtype=torch.complex128 if positions.dtype == torch.float64 else torch.complex64,
            device=positions.device,
        )
        rec_comp = torch.fft.fft(raw_comp.to(positions.dtype)) / float(num_grid)
        mask = g_abs > 1.0e-14
        phi_rec[mask] = rec_comp[mask] * float(FIELD_CONSTANT) / (
            torch.square(g_abs[mask]) * volume
        )
        phi_comp = torch.fft.ifft(phi_rec * float(num_grid)).real
        phi_comp = phi_comp - torch.mean(phi_comp)
        energies[graph_idx] = (
            torch.mean(raw_exp * phi_comp)
            + 0.5 * torch.mean(raw_comp * phi_comp)
        )
    return energies


@compile_mode("script")
class MACELES(ScaleShiftMACE):
    def __init__(self, les_arguments: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        try:
            from les import Les
        except ImportError as exc:
            raise ImportError(
                "Cannot import 'les'. Please install the 'les' library from https://github.com/ChengUCB/les."
            ) from exc
        if les_arguments is None:
            les_arguments = {"use_atomwise": False}
        self.compute_bec = les_arguments.get("compute_bec", False)
        self.bec_output_index = les_arguments.get("bec_output_index", None)
        self.les = Les(les_arguments=les_arguments)
        self.les_readouts = torch.nn.ModuleList()
        self.readout_input_dims = [
            _get_readout_input_dim(readout) for readout in self.readouts  # type: ignore
        ]
        cueq_config = kwargs.get("cueq_config", None)
        for readout in self.readouts:  # type: ignore
            self.les_readouts.append(
                _copy_mace_readout(readout, cueq_config=cueq_config)
            )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
        compute_bec: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )
        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        if cell.dim() == 2 and cell.shape[1] == 3 and cell.shape[0] == num_graphs * 3:
            cell = cell.view(num_graphs, 3, 3)
        elif cell.dim() == 2 and cell.shape[0] == num_graphs and cell.shape[1] == 9:
            cell = cell.view(num_graphs, 3, 3)
        elif cell.dim() == 1 and cell.numel() == num_graphs * 9:
            cell = cell.view(num_graphs, 3, 3)
        node_heads = ctx.node_heads
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        # Setting LES cell input to zero when boundary conditions are not periodic
        cell_les = cell.clone()
        pbc_tensor = data["pbc"].to(device=data["cell"].device)
        no_pbc_mask_cfg = ~pbc_tensor.any(dim=-1)
        no_pbc_mask_rows = no_pbc_mask_cfg.repeat_interleave(3)
        cell_les[no_pbc_mask_rows] = torch.zeros(
            (no_pbc_mask_rows.sum(), 3), dtype=cell_les.dtype, device=cell_les.device
        )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(
            vectors.dtype
        )  # [n_graphs, num_heads]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        # Embeddings of additional features
        if hasattr(self, "joint_embedding"):
            embedding_features: Dict[str, torch.Tensor] = {}
            for name, _ in self.embedding_specs.items():
                embedding_features[name] = data[name]
            node_feats += self.joint_embedding(
                data["batch"],
                embedding_features,
            )
            if hasattr(self, "embedding_readout"):
                embedding_node_energy = self.embedding_readout(
                    node_feats, node_heads
                ).squeeze(-1)
                embedding_energy = scatter_sum(
                    src=embedding_node_energy,
                    index=data["batch"],
                    dim=0,
                    dim_size=num_graphs,
                )
                e0 += embedding_energy

        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list: List[torch.Tensor] = []
        node_qs_list: List[torch.Tensor] = []

        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            node_attrs_slice = data["node_attrs"]
            if is_lammps and i > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(i == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and i == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_list.append(node_feats)

        for i, (readout, les_readout) in enumerate(
            zip(self.readouts, self.les_readouts)
        ):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es = readout(node_feats_list[feat_idx], node_heads)[
                num_atoms_arange, node_heads
            ]
            node_qs = les_readout(node_feats_list[feat_idx], node_heads)[
                num_atoms_arange, node_heads
            ]  # type: ignore
            node_qs_list.append(node_qs)
            node_es_list.append(node_es)

        node_feats_out = torch.cat(node_feats_list, dim=-1)
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)

        total_energy = e0 + inter_e
        node_energy = node_e0.clone().double() + node_inter_es.clone().double()

        les_q = torch.sum(torch.stack(node_qs_list, dim=1), dim=1)
        les_result = self.les(
            latent_charges=les_q,
            positions=positions,
            cell=cell_les.view(-1, 3, 3),
            batch=data["batch"],
            compute_energy=True,
            compute_bec=(compute_bec or self.compute_bec),
            bec_output_index=self.bec_output_index,
        )
        les_energy_opt = les_result["E_lr"]
        if les_energy_opt is None:
            les_energy = torch.zeros_like(total_energy)
        else:
            les_energy = les_energy_opt
        total_energy += les_energy

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=inter_e + les_energy,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
        )

        atomic_virials: Optional[torch.Tensor] = None
        atomic_stresses: Optional[torch.Tensor] = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )
        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "displacement": displacement,
            "hessian": hessian,
            "node_feats": node_feats_out,
            "les_energy": les_energy,
            "latent_charges": les_q,
            "BEC": les_result["BEC"],
        }


def _permute_to_e3nn_convention(x: torch.Tensor) -> torch.Tensor:
    return x[..., torch.LongTensor([1, 2, 0]).to(x.device)]


@compile_mode("script")
class PolarMACE(ScaleShiftMACE):
    def __init__(
        self,
        kspace_cutoff_factor: float = 1.5,
        atomic_multipoles_max_l: int = 0,
        atomic_multipoles_smearing_width: float = 1.0,
        field_feature_max_l: int = 0,
        field_feature_widths: List[float] = (1.0,),
        num_recursion_steps: int = 1,
        field_si: bool = False,
        include_electrostatic_self_interaction: bool = False,
        add_local_electron_energy: bool = False,
        quadrupole_feature_corrections: bool = False,
        return_electrostatic_potentials: bool = False,
        field_feature_norms: Optional[List[float]] = None,
        field_norm_factor: Optional[float] = 0.02,
        fixedpoint_update_config: Optional[Dict[str, Any]] = None,
        field_readout_config: Optional[Dict[str, Any]] = None,
        atomic_density_sigmas: Optional[List[float]] = None,
        element_charge_residual_scale: float = 1.0,
        solvent_sigma_g: float = 0.85,
        solvent_density_threshold: Optional[float] = None,
        solvent_ze_level: float = 0.5,
        solvent_window_inward: float = 5.0,
        solvent_window_outward: float = 5.0,
        solvent_potential_axis: int = 2,
        solvent_potential_sign: float = 1.0,
        solvent_center_mean_shift: float = 0.0,
        solvent_plane_feature_convention: str = "periodic",
        solvent_model: str = "planar",
        solvent_pb_config: Optional[str] = None,
        solvent_pb_repo: Optional[str] = None,
        solvent_pb_grid_spacing: float = 0.15,
        solvent_pb_fixsol_steps: int = 2,
        solvent_pb_tol: float = 1.0e-3,
        solvent_pb_nuclear_sigma: float = 0.4,
        solvent_pb_coarse_init: bool = True,
        solvent_pb_include_bound: bool = True,
        solvent_pb_backend: str = "torch",
        solvent_pb_warm_start: bool = True,
        solvent_pb_warm_fixsol_steps: int = 0,
        fermi_level_baseline: float = 0.0,
        atomic_valence_electrons: Optional[List[float]] = None,
        potential_1d_profile_file: Optional[str] = None,
        learn_solvent_center_residual: bool = True,
        **kwargs,
    ):
        if not GRAPH_LONGRANGE_AVAILABLE:
            raise ImportError(
                "Cannot import 'graph_longrange'. Please install graph_electrostatics "
                "from https://github.com/WillBaldwin0/graph_electrostatics."
            )
        try:
            hidden_irreps: o3.Irreps = kwargs["hidden_irreps"]
            MLP_irreps_raw = kwargs["MLP_irreps"]
            if isinstance(MLP_irreps_raw, int):
                MLP_irreps = o3.Irreps(f"{int(MLP_irreps_raw)}x0e")
            else:
                MLP_irreps = o3.Irreps(str(MLP_irreps_raw))
            gate = kwargs["gate"]
            avg_num_neighbors: float = kwargs["avg_num_neighbors"]
            num_interactions: int = kwargs["num_interactions"]
            num_elements: int = kwargs["num_elements"]
        except KeyError as exc:
            missing = str(exc).strip("'")
            raise KeyError(
                f"Missing required argument '{missing}' in kwargs for PolarMACE. "
                "Pass all ScaleShiftMACE/MACE constructor args as keyword arguments."
            ) from exc

        cueq_config: Optional[CuEquivarianceConfig] = kwargs.get("cueq_config", None)
        oeq_config: Optional[OEQConfig] = kwargs.get("oeq_config", None)
        # Keep a reference for config extraction tools
        self.cueq_config = cueq_config
        self.oeq_config = oeq_config

        # Defaults to mirror previous behavior
        heads = kwargs.get("heads", ["Default"]) or ["Default"]
        kwargs.setdefault("heads", heads)
        kwargs.setdefault("readout_cls", NonLinearReadoutBlock)
        kwargs.setdefault("use_agnostic_product", True)
        kwargs.setdefault("apply_cutoff", True)
        # Provide default atomic_inter_scale/shift if not passed
        kwargs.setdefault("atomic_inter_scale", [1.0] * len(heads))
        kwargs.setdefault("atomic_inter_shift", [0.0] * len(heads))

        # Swallow optional convenience args that shouldn't flow to super().__init__
        kwargs.pop("field_dependence_type", None)
        kwargs.pop("final_field_readout_type", None)
        kwargs.pop("keep_last_layer_irreps", None)

        # Initialize the MACE backbone (interactions, products, readouts, embeddings)
        super().__init__(**kwargs, keep_last_layer_irreps=True)

        self.kspace_cutoff_factor = float(kspace_cutoff_factor)
        self.num_recursion_steps = int(num_recursion_steps)
        self.atomic_multipoles_max_l = int(atomic_multipoles_max_l)
        self.atomic_multipoles_smearing_width = float(atomic_multipoles_smearing_width)
        self.atomic_density_sigmas = (
            [self.atomic_multipoles_smearing_width]
            if atomic_density_sigmas is None
            else [float(value) for value in atomic_density_sigmas]
        )
        self.atomic_density_num_sigmas = len(self.atomic_density_sigmas)
        self.atomic_multipoles_dim = (self.atomic_multipoles_max_l + 1) ** 2
        self.electrostatic_multipoles_max_l = min(self.atomic_multipoles_max_l, 1)
        self.electrostatic_multipoles_dim = (
            self.electrostatic_multipoles_max_l + 1
        ) ** 2
        self.field_feature_max_l = int(field_feature_max_l)
        self.field_feature_widths = list(field_feature_widths)
        self.field_norm_factor = float(field_norm_factor)
        self._field_feature_norms = field_feature_norms
        self.include_electrostatic_self_interaction = (
            include_electrostatic_self_interaction
        )
        self.atomic_multipoles_smearing_width = float(atomic_multipoles_smearing_width)
        self.add_local_electron_energy = add_local_electron_energy
        self.quadrupole_feature_corrections = quadrupole_feature_corrections
        self.field_si = field_si
        self.keep_last_layer_irreps = True
        self.element_charge_residual_scale = float(element_charge_residual_scale)
        self.register_buffer(
            "element_charge_baseline",
            torch.zeros((len(heads), num_elements), dtype=torch.get_default_dtype()),
        )
        self.solvent_sigma_g = float(solvent_sigma_g)
        self.solvent_density_threshold = (
            None
            if solvent_density_threshold is None
            else float(solvent_density_threshold)
        )
        self.solvent_ze_level = float(solvent_ze_level)
        self.solvent_window_inward = float(solvent_window_inward)
        self.solvent_window_outward = float(solvent_window_outward)
        self.solvent_potential_axis = int(solvent_potential_axis)
        self.solvent_potential_sign = float(solvent_potential_sign)
        self.solvent_center_mean_shift = float(solvent_center_mean_shift)
        if solvent_plane_feature_convention not in ("periodic", "isolated"):
            raise ValueError(
                "solvent_plane_feature_convention must be 'periodic' or 'isolated', "
                f"got {solvent_plane_feature_convention!r}"
            )
        self.solvent_plane_feature_convention = str(solvent_plane_feature_convention)
        if solvent_model not in ("planar", "pb"):
            raise ValueError(
                f"solvent_model must be 'planar' or 'pb', got {solvent_model!r}"
            )
        if solvent_model == "pb" and not solvent_pb_config:
            raise ValueError("solvent_model='pb' requires solvent_pb_config")
        self.solvent_model = str(solvent_model)
        self.solvent_pb_config = solvent_pb_config
        self.solvent_pb_repo = solvent_pb_repo
        self.solvent_pb_grid_spacing = float(solvent_pb_grid_spacing)
        self.solvent_pb_fixsol_steps = int(solvent_pb_fixsol_steps)
        self.solvent_pb_tol = float(solvent_pb_tol)
        self.solvent_pb_nuclear_sigma = float(solvent_pb_nuclear_sigma)
        self.solvent_pb_coarse_init = bool(solvent_pb_coarse_init)
        self.solvent_pb_include_bound = bool(solvent_pb_include_bound)
        if solvent_pb_backend not in ("numpy", "torch"):
            raise ValueError(
                f"solvent_pb_backend must be 'numpy' or 'torch', got {solvent_pb_backend!r}"
            )
        self.solvent_pb_backend = str(solvent_pb_backend)
        self.solvent_pb_warm_start = bool(solvent_pb_warm_start)
        self.solvent_pb_warm_fixsol_steps = int(solvent_pb_warm_fixsol_steps)
        self._pb_solver = None
        self.register_buffer(
            "fermi_level_baseline",
            torch.tensor(float(fermi_level_baseline), dtype=torch.get_default_dtype()),
        )
        self.center_density_baselines = (
            _load_center_density_baselines_npz(potential_1d_profile_file)
            if potential_1d_profile_file is not None
            else {}
        )
        self.solvent_residual_hidden_dim = 16
        self.fermi_residual_hidden_dim = 16
        self.solvent_residual_max_abs_shift = 1.5
        self.solvent_residual_init_shift = 0.0
        self.learn_solvent_center_residual = bool(learn_solvent_center_residual)
        explicit_valence = None
        if atomic_valence_electrons is not None:
            explicit_valence = torch.tensor(
                atomic_valence_electrons,
                dtype=torch.get_default_dtype(),
                device=self.atomic_numbers.device,
            )
        valence_electrons = require_explicit_valence_electrons(
            self.atomic_numbers, explicit_valence
        )
        self.register_buffer("atomic_valence_electrons", valence_electrons)
        # k-space cutoff heuristic
        kspace_cutoff = kspace_cutoff_factor * gto_basis_kspace_cutoff(
            [atomic_multipoles_smearing_width] + list(field_feature_widths),
            max(atomic_multipoles_max_l, field_feature_max_l),
        )
        self.register_buffer(
            "kspace_cutoff",
            torch.tensor(kspace_cutoff, dtype=torch.get_default_dtype()),
        )

        # Normalization for field features
        if field_feature_norms is not None:
            assert len(field_feature_norms) == len(field_feature_widths) * (
                field_feature_max_l + 1
            ), f"{len(field_feature_widths) * (field_feature_max_l+1)}, {len(field_feature_norms)}"
        else:
            field_feature_norms = (
                [1.0] * len(field_feature_widths) * (field_feature_max_l + 1)
            )
        expanded: List[float] = []
        for l in range(field_feature_max_l + 1):
            for j in range(len(field_feature_widths)):
                expanded += [field_feature_norms[l * len(field_feature_widths) + j]] * (
                    2 * l + 1
                )
        self.register_buffer(
            "field_feature_norms",
            torch.tensor(expanded, dtype=torch.get_default_dtype()),
        )

        self.lr_source_maps = torch.nn.ModuleList(
            EnvironmentDependentSpinSourceBlock(
                irreps_in=hidden_irreps,
                max_l=atomic_multipoles_max_l,
                num_radial=self.atomic_density_num_sigmas,
                cueq_config=cueq_config,
            )
            for _ in range(num_interactions)
        )

        # Field-dependent components
        radial_source_irreps = (
            self.atomic_density_num_sigmas
            * o3.Irreps.spherical_harmonics(atomic_multipoles_max_l)
        ).sort()[0].simplify()
        self.charges_irreps = radial_source_irreps + radial_source_irreps
        charges_layout = (
            cueq_config.layout_str
            if (cueq_config is not None and cueq_config.enabled)
            else "mul_ir"
        )
        self._charges_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.charges_irreps,
            source=charges_layout,
            target="mul_ir",
            cueq_config=cueq_config,
        )
        self._charges_from_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.charges_irreps,
            source="mul_ir",
            target=charges_layout,
            cueq_config=cueq_config,
        )
        lr_sh_irreps = o3.Irreps.spherical_harmonics(field_feature_max_l)
        self.field_irreps = (
            (lr_sh_irreps * len(field_feature_widths)).sort()[0].simplify()
        )
        self.potential_irreps = (
            self.field_irreps * 2
        )  # 2 spin channels for the potential irreps

        self.electric_potential_descriptor = GTOElectrostaticFeatures(
            density_max_l=self.electrostatic_multipoles_max_l,
            density_smearing_width=atomic_multipoles_smearing_width,
            feature_max_l=field_feature_max_l,
            feature_smearing_widths=list(field_feature_widths),
            kspace_cutoff=kspace_cutoff,
            include_self_interaction=field_si,
            quadrupole_feature_corrections=quadrupole_feature_corrections,
            integral_normalization="receiver",
        )
        field_layout_target = (
            cueq_config.layout_str
            if (cueq_config is not None and cueq_config.enabled)
            else "mul_ir"
        )
        self._field_from_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.field_irreps,
            source="mul_ir",
            target=field_layout_target,
            cueq_config=cueq_config,
        )

        self.fukui_source_map = NonLinearBiasReadoutBlock(
            hidden_irreps,
            MLP_irreps.simplify(),
            gate,
            o3.Irreps("2x0e"),
            cueq_config=None,
            oeq_config=oeq_config,
        )
        fukui_layout = (
            cueq_config.layout_str
            if (cueq_config is not None and cueq_config.enabled)
            else "mul_ir"
        )
        self._fukui_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=hidden_irreps,
            source=fukui_layout,
            target="mul_ir",
            cueq_config=cueq_config,
        )
        if fixedpoint_update_config is None:
            fixedpoint_update_config = {}
        fixedpoint_update_config = fixedpoint_update_config.copy()
        self._fixedpoint_update_config = fixedpoint_update_config.copy()
        lr_source_cls = fixedpoint_update_config.pop(
            "type", "AgnosticEmbeddedOneBodyVariableUpdate"
        )
        if isinstance(lr_source_cls, str):
            lr_source_cls = field_update_blocks[lr_source_cls]
        # Map optional class names to implementations
        pe_cls = fixedpoint_update_config.get("potential_embedding_cls", None)
        if isinstance(pe_cls, str):
            # currently only AgnosticChargeBiasedLinearPotentialEmbedding is required
            from .field_blocks import (
                AgnosticChargeBiasedLinearPotentialEmbedding as _PE,
            )

            fixedpoint_update_config["potential_embedding_cls"] = _PE
        nl_cls = fixedpoint_update_config.get("nonlinearity_cls", None)
        if isinstance(nl_cls, str):
            from .field_blocks import MLPNonLinearity as _NL

            fixedpoint_update_config["nonlinearity_cls"] = _NL
        # Reconstruct irreps needed for field update maps
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        max_ell_field_update = 2
        field_update_sh_irreps = o3.Irreps.spherical_harmonics(max_ell_field_update)
        self.from_ell_max_field_update = (max_ell_field_update + 1) ** 2
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        field_interaction_irreps = (
            (field_update_sh_irreps * num_features).sort()[0].simplify()
        )
        self.field_dependent_charges_maps = torch.nn.ModuleList()
        for _ in range(num_recursion_steps):
            self.field_dependent_charges_maps.append(
                lr_source_cls(
                    node_attrs_irreps=node_attr_irreps,
                    node_feats_irreps=hidden_irreps,
                    edge_attrs_irreps=field_update_sh_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    target_irreps=field_interaction_irreps,
                    hidden_irreps=hidden_irreps,
                    avg_num_neighbors=avg_num_neighbors,
                    potential_irreps=self.potential_irreps,
                    charges_irreps=self.charges_irreps,
                    num_elements=num_elements,
                    field_norm_factor=float(field_norm_factor or 1.0),
                    cueq_config=cueq_config,
                    oeq_config=oeq_config,
                    **fixedpoint_update_config,
                )
            )

        # Post-SCF readout
        self.add_local_electron_energy = add_local_electron_energy
        if field_readout_config is None:
            field_readout_config = {}
        field_readout_config = field_readout_config.copy()
        self._field_readout_config = field_readout_config.copy()
        field_readout_cls = field_readout_config.pop("type", "OneBodyMLPFieldReadout")
        if isinstance(field_readout_cls, str):
            field_readout_cls = field_readout_blocks[field_readout_cls]
        if self.add_local_electron_energy:
            self.local_electron_energy = field_readout_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=field_update_sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=field_interaction_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                potential_irreps=self.potential_irreps,
                charges_irreps=self.charges_irreps,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
                **field_readout_config,
            )
        else:
            self.local_electron_energy = None

        self.external_field_contribution = DisplacedGTOExternalFieldBlock(
            field_feature_max_l, list(field_feature_widths), "receiver"
        )
        self.coulomb_energy = GTOElectrostaticEnergy(
            density_max_l=self.electrostatic_multipoles_max_l,
            density_smearing_width=atomic_multipoles_smearing_width,
            kspace_cutoff=float(kspace_cutoff),
            include_self_interaction=include_electrostatic_self_interaction,
        )
        self.return_electrostatic_potentials = return_electrostatic_potentials
        self.layer_feature_mixer = MultiLayerFeatureMixer(
            node_feats_irreps=hidden_irreps,
            num_interactions=num_interactions,
            cueq_config=cueq_config,
        )
        self.solvent_scalar_feature_dim = 3
        self.solvent_interface_pool_sigma = 1.0
        self.solvent_center_residual = torch.nn.Sequential(
            torch.nn.Linear(
                hidden_irreps.dim * num_interactions * 2
                + self.solvent_scalar_feature_dim,
                self.solvent_residual_hidden_dim,
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(self.solvent_residual_hidden_dim, 1),
        )
        final_linear = self.solvent_center_residual[-1]
        torch.nn.init.zeros_(final_linear.weight)
        init_frac = max(
            min(
                self.solvent_residual_init_shift
                / max(self.solvent_residual_max_abs_shift, 1.0e-12),
                0.999,
            ),
            -0.999,
        )
        torch.nn.init.constant_(
            final_linear.bias,
            float(torch.atanh(torch.tensor(init_frac, dtype=final_linear.bias.dtype))),
        )
        self.fermi_level_residual = torch.nn.Sequential(
            torch.nn.Linear(
                hidden_irreps.dim * num_interactions,
                self.fermi_residual_hidden_dim,
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(self.fermi_residual_hidden_dim, 1),
        )
        fermi_final_linear = self.fermi_level_residual[-1]
        torch.nn.init.zeros_(fermi_final_linear.weight)
        torch.nn.init.zeros_(fermi_final_linear.bias)
    def set_element_charge_baseline(self, baseline: torch.Tensor) -> None:
        if baseline.shape != self.element_charge_baseline.shape:
            raise ValueError(
                f"Expected element_charge_baseline shape {tuple(self.element_charge_baseline.shape)}, got {tuple(baseline.shape)}"
            )
        self.element_charge_baseline.copy_(
            baseline.to(
                device=self.element_charge_baseline.device,
                dtype=self.element_charge_baseline.dtype,
            )
        )

    def _radial_flat_to_blocks(self, radial_flat: torch.Tensor) -> torch.Tensor:
        out = radial_flat.new_zeros(
            (
                radial_flat.shape[0],
                self.atomic_density_num_sigmas,
                self.atomic_multipoles_dim,
            )
        )
        offset = 0
        for ell in range(self.atomic_multipoles_max_l + 1):
            width = 2 * ell + 1
            block = radial_flat[
                :, offset : offset + self.atomic_density_num_sigmas * width
            ].view(radial_flat.shape[0], self.atomic_density_num_sigmas, width)
            out[:, :, ell * ell : (ell + 1) * (ell + 1)] = block
            offset += self.atomic_density_num_sigmas * width
        return out

    def _radial_flat_to_effective(self, radial_flat: torch.Tensor) -> torch.Tensor:
        return self._radial_flat_to_blocks(radial_flat).sum(dim=1)

    def _center_profile_inputs(
        self,
        data: Dict[str, torch.Tensor],
        num_graphs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[tuple[List[torch.Tensor], List[torch.Tensor]]]:
        if not self.center_density_baselines:
            return None
        sample_ids = data.get("sample_id")
        if sample_ids is None:
            raise KeyError(
                "A baseline density profile file was provided, but batch data has no sample_id"
            )
        sample_ids = sample_ids.view(-1)
        if sample_ids.numel() != num_graphs:
            raise ValueError(
                f"sample_id count {sample_ids.numel()} does not match num_graphs {num_graphs}"
            )
        z_values: List[torch.Tensor] = []
        raw_neutral_values: List[torch.Tensor] = []
        for value in sample_ids:
            sample_id = int(value.detach().cpu().item())
            if sample_id not in self.center_density_baselines:
                raise KeyError(
                    f"sample_id {sample_id} is missing from baseline density profile file"
                )
            z_np, neutral_np = self.center_density_baselines[sample_id]
            z_values.append(torch.as_tensor(z_np, device=device, dtype=dtype))
            raw_neutral_values.append(
                torch.as_tensor(neutral_np, device=device, dtype=dtype)
            )
        return z_values, raw_neutral_values

    def _get_pb_solver(self):
        if self._pb_solver is None:
            common = dict(
                config_path=self.solvent_pb_config,
                repo_path=self.solvent_pb_repo,
                grid_spacing=self.solvent_pb_grid_spacing,
                fixsol_steps=self.solvent_pb_fixsol_steps,
                tol=self.solvent_pb_tol,
                coarse_init=self.solvent_pb_coarse_init,
                nuclear_sigma=self.solvent_pb_nuclear_sigma,
                axis=self.solvent_potential_axis,
            )
            if self.solvent_pb_backend == "torch":
                from .pb_solvent import PBTorchBackend

                self._pb_solver = PBTorchBackend(
                    warm_start=self.solvent_pb_warm_start,
                    warm_fixsol_steps=self.solvent_pb_warm_fixsol_steps,
                    **common,
                )
            else:
                from .pb_solvent import PBPlanarSolvent

                self._pb_solver = PBPlanarSolvent(**common)
        return self._pb_solver

    def _solve_pb_profiles(
        self,
        data: Dict[str, torch.Tensor],
        positions: torch.Tensor,
        cell: torch.Tensor,
        radial_blocks: torch.Tensor,
        node_valence_electrons: torch.Tensor,
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:
        """Detached per-graph nonlinear PB solve on the current (pre-SCF)
        density; returns rho_ion(z) resampled onto the feature grid (1024,
        offset 0) and the 1D-energy grid (512, offset 1/2), plus the layer
        charge, mean and dipole moment used by the observables.

        The same profile (from the pre-SCF density) is reused post-SCF for
        the energy and observables — the PB analogue of the planar model's
        pre/post-SCF center inconsistency, kept deliberately so a single
        solve per forward suffices.
        """
        from .loss import _gto_density_at_points_axis2_pbc
        from .pb_solvent import (
            profiles_to_tensors,
            resample_profile_periodic_torch,
        )

        solver = self._get_pb_solver()
        use_torch = self.solvent_pb_backend == "torch"
        eval_chunk = 262144 if positions.is_cuda else 4096
        axis = self.solvent_potential_axis
        cells = cell.detach()
        if cells.dim() == 2 and cells.shape[1] == 3:
            cells = cells.view(-1, 3, 3)
        pbc_g = data["pbc"].view(-1, 3).to(torch.bool)
        slab = torch.ones(3, dtype=torch.bool, device=pbc_g.device)
        slab[axis] = False
        slab_mask = torch.all(pbc_g == slab, dim=1)
        element_index = torch.argmax(data["node_attrs"], dim=-1)
        atomic_numbers_nodes = self.atomic_numbers[element_index]
        total_charge_g = data["total_charge"].view(-1)
        sample_ids = data.get("sample_id")
        if sample_ids is not None:
            sample_ids = sample_ids.view(-1)

        profiles: List[Optional[Dict[str, Any]]] = []
        heights: List[float] = []
        q_ion = positions.new_zeros(num_graphs)
        solvent_mu = positions.new_zeros(num_graphs)
        prof_feat = positions.new_zeros(num_graphs, 1024)
        prof_energy = positions.new_zeros(num_graphs, 512)
        for g in range(num_graphs):
            cell_g = cells[g]
            heights.append(float(_axis_box_length(cell_g, axis).item()))
            atom_mask = data["batch"] == g
            if not bool(slab_mask[g].item()) or not bool(torch.any(atom_mask).item()):
                profiles.append(None)
                continue
            pos_g = positions[atom_mask].detach()
            coeffs_g = radial_blocks[atom_mask].detach()
            numbers_g = atomic_numbers_nodes[atom_mask]
            zval_g = node_valence_electrons[atom_mask].detach()
            sid = (
                int(sample_ids[g].detach().cpu().item())
                if sample_ids is not None
                else None
            )

            def eval_net_density_t(pts):
                with torch.no_grad():
                    chunks = []
                    for chunk in pts.to(positions.dtype).split(eval_chunk):
                        chunks.append(
                            _gto_density_at_points_axis2_pbc(
                                chunk,
                                coeffs_g,
                                pos_g,
                                numbers_g,
                                cell_g,
                                self.atomic_density_sigmas,
                            )
                        )
                    return torch.cat(chunks).detach()

            if use_torch:
                result = solver.solve_rho_ion_z(
                    positions=pos_g,
                    cell=cell_g,
                    z_valence=zval_g,
                    total_charge=float(total_charge_g[g].item()),
                    neutral_sigma=float(self.atomic_multipoles_smearing_width),
                    sample_id=sid,
                    radial_coeffs=coeffs_g,
                    sigmas=self.atomic_density_sigmas,
                )
            else:
                def eval_net_density(points_np):
                    pts = torch.as_tensor(
                        points_np, device=positions.device, dtype=positions.dtype
                    )
                    return eval_net_density_t(pts).cpu().numpy()

                result = solver.solve_rho_ion_z(
                    positions=pos_g.cpu().numpy(),
                    cell=cell_g.cpu().numpy(),
                    z_valence=zval_g.cpu().numpy(),
                    total_charge=float(total_charge_g[g].item()),
                    neutral_sigma=float(self.atomic_multipoles_smearing_width),
                    eval_net_density=eval_net_density,
                )
            # The solvent layer the model sees: ionic charge alone, or the
            # full implicit-region solvent charge (ionic + bound). The bound
            # (polarization) charge integrates to ~0 but carries the
            # dielectric screening dipole of the implicit region.
            result = dict(result)
            if self.solvent_pb_include_bound:
                result["rho_layer_z"] = (
                    result["rho_ion_z"] + result["rho_bound_z"]
                )
                mu_g = (
                    result["q_ion"] * result["layer_mean"]
                    + result["mu_bound"]
                )
            else:
                result["rho_layer_z"] = result["rho_ion_z"]
                mu_g = result["q_ion"] * result["layer_mean"]
            profiles.append(result)
            q_ion[g] = float(result["q_ion"])
            solvent_mu[g] = float(mu_g)
            if use_torch:
                layer = result["rho_layer_z"].to(positions.dtype)
                height_g = float(result["height"])
                prof_feat[g] = resample_profile_periodic_torch(
                    layer, height_g, 1024, False
                )
                prof_energy[g] = resample_profile_periodic_torch(
                    layer, height_g, 512, True
                )

        if not use_torch:
            prof_feat = profiles_to_tensors(
                profiles, heights, 1024, False, positions.device, positions.dtype,
                key="rho_layer_z",
            )
            prof_energy = profiles_to_tensors(
                profiles, heights, 512, True, positions.device, positions.dtype,
                key="rho_layer_z",
            )
        # Effective layer mean of the combined profile (dipole / charge);
        # the charge is the ionic one (bound integrates to ~0).
        layer_mean = solvent_mu / torch.where(
            torch.abs(q_ion) > 1.0e-12, q_ion, torch.full_like(q_ion, 1.0e-12)
        )
        return {
            "profile_features": prof_feat,
            "profile_energy": prof_energy,
            "q_ion": q_ion,
            "layer_mean": layer_mean,
            "solvent_mu": solvent_mu,
        }

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
        use_pbc_evaluator: bool = False,
        fermi_level: Optional[torch.Tensor] = None,
        external_field: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if not GRAPH_LONGRANGE_AVAILABLE:
            raise ImportError(
                "Cannot import 'graph_longrange'. Please install graph_electrostatics "
                "from https://github.com/WillBaldwin0/graph_electrostatics."
            )
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )

        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        if fermi_level is None:
            fermi_level = data["fermi_level"]
        if external_field is None:
            external_field = data["external_field"]
        external_potential = torch.hstack(
            (torch.zeros_like(fermi_level).unsqueeze(-1), external_field)
        )
        charges_to_mul_ir = getattr(self, "_charges_to_mul_ir", None)

        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(vectors.dtype)

        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(_permute_to_e3nn_convention(vectors))
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        node_es_list: List[torch.Tensor] = []
        node_feats_list: List[torch.Tensor] = []
        spin_charge_density = torch.zeros(
            (data["batch"].size(-1), self.charges_irreps.dim),
            device=data["batch"].device,
            dtype=vectors.dtype,
        )
        node_charge_baseline = torch.sum(
            self.element_charge_baseline.index_select(0, node_heads.to(torch.long))
            * data["node_attrs"].to(self.element_charge_baseline.dtype),
            dim=-1,
        ).to(vectors.dtype)
        node_charge_residual_scale = torch.full_like(
            node_charge_baseline, self.element_charge_residual_scale
        )

        for i, (interaction, product, lr_src) in enumerate(
            zip(self.interactions, self.products, self.lr_source_maps)
        ):
            node_attrs_slice = data["node_attrs"]
            if is_lammps and i > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(i == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and i == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_list.append(node_feats)

            feat_idx = -1 if len(self.readouts) == 1 else min(i, len(self.readouts) - 1)
            node_es = self.readouts[feat_idx](node_feats, node_heads)[
                num_atoms_arange, node_heads
            ]
            node_es_list.append(node_es)

            spin_charge_sources = lr_src(node_feats).squeeze(-2)
            if self.element_charge_residual_scale != 1.0:
                spin_charge_sources = spin_charge_sources.view(
                    spin_charge_sources.shape[0], 2, -1
                )
                spin_charge_sources = spin_charge_sources.clone()
                spin_charge_sources[:, :, : self.atomic_density_num_sigmas] = (
                    node_charge_residual_scale[:, None, None]
                    * spin_charge_sources[:, :, : self.atomic_density_num_sigmas]
                )
                spin_charge_sources = spin_charge_sources.view(
                    spin_charge_sources.shape[0], -1
                )
            spin_charge_density = spin_charge_density + spin_charge_sources

        node_feats_out = torch.cat(node_feats_list, dim=-1)
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)

        # Build k-grid
        (
            k_vectors,
            kv_norms_squared,
            k_vectors_batch,
            k_vectors_0mask,
        ) = compute_k_vectors_flat(
            self.kspace_cutoff, cell.view(-1, 3, 3), data["rcell"].view(-1, 3, 3)
        )

        field_feature_cache = self.electric_potential_descriptor.precompute_geometry(
            k_vectors=k_vectors,
            k_norm2=kv_norms_squared,
            k_vector_batch=k_vectors_batch,
            k0_mask=k_vectors_0mask,
            node_positions=positions,
            batch=data["batch"],
            volume=data["volume"],
            pbc=data["pbc"].view(-1, 3),
            force_pbc_evaluator=use_pbc_evaluator,
        )

        # SCF fixed point
        features_mixed = self.layer_feature_mixer(torch.stack(node_feats_list, dim=0))
        spin_charge_density = spin_charge_density.view(
            spin_charge_density.shape[0], 2, -1
        )
        spin_charge_density = spin_charge_density.clone()
        half_node_charge_baseline = 0.5 * node_charge_baseline
        spin_charge_density[:, 0, : self.atomic_density_num_sigmas] = (
            spin_charge_density[:, 0, : self.atomic_density_num_sigmas]
            + half_node_charge_baseline[:, None] / float(self.atomic_density_num_sigmas)
        )
        spin_charge_density[:, 1, : self.atomic_density_num_sigmas] = (
            spin_charge_density[:, 1, : self.atomic_density_num_sigmas]
            + half_node_charge_baseline[:, None] / float(self.atomic_density_num_sigmas)
        )
        fukui_input = node_feats
        fukui_to_mul_ir = getattr(self, "_fukui_to_mul_ir", None)
        if fukui_to_mul_ir is not None:
            fukui_input = fukui_to_mul_ir(fukui_input)
        fukui_sources = self.fukui_source_map(fukui_input)
        fukui_norm = scatter_sum(
            src=fukui_sources.double(),
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )[data["batch"]].to(vectors.dtype)
        fukui_norm = torch.where(
            fukui_norm == 0, torch.ones_like(fukui_norm), fukui_norm
        )
        fukui_sources = fukui_sources / fukui_norm
        Q_p_S = (data["total_charge"] + (data["total_spin"] - 1))[data["batch"]]
        Q_m_S = (data["total_charge"] - (data["total_spin"] - 1))[data["batch"]]
        pred_total_charges_0 = scatter_sum(
            src=spin_charge_density[:, :, : self.atomic_density_num_sigmas]
            .sum(dim=-1)
            .double(),
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )[data["batch"]].to(vectors.dtype)
        spin_charge_density = spin_charge_density.clone()
        spin_charge_density[:, 0, : self.atomic_density_num_sigmas] = (
            spin_charge_density[:, 0, : self.atomic_density_num_sigmas]
            + fukui_sources[:, 0, None]
            * ((Q_p_S / 2) - pred_total_charges_0[:, 0])[:, None]
            / float(self.atomic_density_num_sigmas)
        )
        spin_charge_density[:, 1, : self.atomic_density_num_sigmas] = (
            spin_charge_density[:, 1, : self.atomic_density_num_sigmas]
            + fukui_sources[:, 1, None]
            * ((Q_m_S / 2) - pred_total_charges_0[:, 1])[:, None]
            / float(self.atomic_density_num_sigmas)
        )
        # print("spin_charge_density", spin_charge_density)

        potential_features = torch.zeros(
            (data["batch"].size(-1), self.potential_irreps.dim),
            device=data["batch"].device,
            dtype=vectors.dtype,
        )
        field_independent_spin_charge_density = spin_charge_density.clone()
        element_index = torch.argmax(data["node_attrs"], dim=-1)
        node_valence_electrons = self.atomic_valence_electrons[element_index]
        center_profile_inputs = self._center_profile_inputs(
            data=data,
            num_graphs=num_graphs,
            device=positions.device,
            dtype=positions.dtype,
        )
        comp_charge_density = field_independent_spin_charge_density.sum(dim=1)
        comp_density_coefficients = self._radial_flat_to_effective(
            comp_charge_density
        )
        if (
            self.solvent_density_threshold is not None
            and center_profile_inputs is not None
        ):
            comp_density_width_threshold = max(
                float(self.solvent_density_threshold) - 0.05,
                1.0e-6,
            )
            comp_z_base, _ = compute_density_threshold_crossing_from_baseline_profile(
                z_values=center_profile_inputs[0],
                raw_neutral_values=center_profile_inputs[1],
                radial_coefficients=self._radial_flat_to_blocks(comp_charge_density).detach(),
                positions=positions.detach(),
                batch=data["batch"],
                cell=cell.detach(),
                atomic_density_sigmas=self.atomic_density_sigmas,
                density_threshold=float(self.solvent_density_threshold),
                width_density_threshold=comp_density_width_threshold,
                inward=self.solvent_window_inward,
                outward=self.solvent_window_outward,
                axis=self.solvent_potential_axis,
            )
        elif self.solvent_density_threshold is not None:
            comp_density_width_threshold = max(
                float(self.solvent_density_threshold) - 0.05,
                1.0e-6,
            )
            comp_z_base, _ = compute_density_threshold_crossing_from_mace_multipoles(
                density_coefficients=comp_density_coefficients.detach(),
                positions=positions.detach(),
                node_valence_electrons=node_valence_electrons.detach(),
                batch=data["batch"],
                cell=cell.detach(),
                density_smearing_width=self.atomic_multipoles_smearing_width,
                density_threshold=float(self.solvent_density_threshold),
                width_density_threshold=comp_density_width_threshold,
                inward=self.solvent_window_inward,
                outward=self.solvent_window_outward,
                axis=self.solvent_potential_axis,
            )
        else:
            comp_z_base, _ = compute_ze_crossing_from_mace_multipoles(
                density_coefficients=comp_density_coefficients.detach(),
                positions=positions.detach(),
                node_valence_electrons=node_valence_electrons.detach(),
                batch=data["batch"],
                cell=cell.detach(),
                density_smearing_width=self.atomic_multipoles_smearing_width,
                level=self.solvent_ze_level,
                inward=self.solvent_window_inward,
                outward=self.solvent_window_outward,
                axis=self.solvent_potential_axis,
            )
        comp_center_init = (
            comp_z_base
            + comp_z_base.new_full(
                comp_z_base.shape, float(self.solvent_center_mean_shift)
            )
        ).detach()
        pb_solvent_data: Optional[Dict[str, torch.Tensor]] = None
        if self.solvent_model == "pb":
            pb_solvent_data = self._solve_pb_profiles(
                data=data,
                positions=positions,
                cell=cell,
                radial_blocks=self._radial_flat_to_blocks(comp_charge_density),
                node_valence_electrons=node_valence_electrons,
                num_graphs=num_graphs,
            )
            (
                compensation_external_features,
                _compensation_phi_nodes,
                _compensation_external_field_nodes,
            ) = _slab_compensation_profile_features(
                external_field_block=self.external_field_contribution,
                profile=pb_solvent_data["profile_features"],
                cell=cell.detach(),
                pbc=data["pbc"].view(-1, 3),
                batch=data["batch"],
                positions=positions,
                feature_sigmas=self.field_feature_widths,
                axis=self.solvent_potential_axis,
            )
            compensation_slab_features = _slab_compensation_slab_correction_features(
                external_field_block=self.external_field_contribution,
                total_charge=data["total_charge"],
                center=comp_center_init,
                cell=cell.detach(),
                volume=data["volume"],
                pbc=data["pbc"].view(-1, 3),
                batch=data["batch"],
                positions=positions,
                sigma_g=self.solvent_sigma_g,
                axis=self.solvent_potential_axis,
                solvent_mu_override=pb_solvent_data["solvent_mu"],
            )
        else:
            (
                compensation_external_features,
                _compensation_phi_nodes,
                _compensation_external_field_nodes,
            ) = _slab_compensation_gaussian_features(
                external_field_block=self.external_field_contribution,
                total_charge=data["total_charge"],
                center=comp_center_init,
                cell=cell.detach(),
                pbc=data["pbc"].view(-1, 3),
                batch=data["batch"],
                positions=positions,
                sigma_g=self.solvent_sigma_g,
                feature_sigmas=self.field_feature_widths,
                axis=self.solvent_potential_axis,
                convention=self.solvent_plane_feature_convention,
            )
            compensation_slab_features = _slab_compensation_slab_correction_features(
                external_field_block=self.external_field_contribution,
                total_charge=data["total_charge"],
                center=comp_center_init,
                cell=cell.detach(),
                volume=data["volume"],
                pbc=data["pbc"].view(-1, 3),
                batch=data["batch"],
                positions=positions,
                sigma_g=self.solvent_sigma_g,
                axis=self.solvent_potential_axis,
            )
        compensation_external_features = (
            compensation_external_features + compensation_slab_features
        )
        compensation_external_features = compensation_external_features.to(vectors.dtype)
        scf_external_potential = external_potential
        esps: Optional[torch.Tensor] = None

        for i in range(self.num_recursion_steps):
            source_feats_alpha = self._radial_flat_to_effective(
                spin_charge_density[:, 0, :]
            )
            source_feats_beta = self._radial_flat_to_effective(
                spin_charge_density[:, 1, :]
            )
            if charges_to_mul_ir is not None:
                source_feats_alpha = charges_to_mul_ir(source_feats_alpha)
                source_feats_beta = charges_to_mul_ir(source_feats_beta)
            source_feats_alpha = source_feats_alpha[
                :, : self.electrostatic_multipoles_dim
            ]
            source_feats_beta = source_feats_beta[
                :, : self.electrostatic_multipoles_dim
            ]
            field_feats_alpha = self.electric_potential_descriptor.forward_dynamic(
                cache=field_feature_cache,
                source_feats=source_feats_alpha.unsqueeze(-2),
                pbc=data["pbc"].view(-1, 3),
            )
            field_feats_beta = self.electric_potential_descriptor.forward_dynamic(
                cache=field_feature_cache,
                source_feats=source_feats_beta.unsqueeze(-2),
                pbc=data["pbc"].view(-1, 3),
            )
            field_from_mul_ir = getattr(self, "_field_from_mul_ir", None)
            if field_from_mul_ir is not None:
                field_feats_alpha = field_from_mul_ir(field_feats_alpha)
                field_feats_beta = field_from_mul_ir(field_feats_beta)
            esps = None

            # Add external field contribution and subtract barycenter for gauge invariance
            barycenter = scatter_mean(
                src=positions.double(),
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            ).to(positions.dtype)
            half_external_field = 0.5 * (
                self.external_field_contribution(
                    data["batch"],
                    positions - barycenter[data["batch"], :],
                    scf_external_potential,
                )
                + compensation_external_features
            )
            field_feats_alpha = (
                field_feats_alpha + half_external_field
            ) / self.field_feature_norms
            field_feats_beta = (
                field_feats_beta + half_external_field
            ) / self.field_feature_norms

            potential_features = torch.cat(
                (field_feats_alpha, field_feats_beta), dim=-1
            )
            charge_sources_out = self.field_dependent_charges_maps[i](
                node_attrs=data["node_attrs"],
                node_feats=features_mixed,
                edge_attrs=edge_attrs[:, : self.from_ell_max_field_update],
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                potential_features=potential_features,
                local_charges=spin_charge_density.view(
                    spin_charge_density.shape[0], -1
                ),
            )

            current_fukui_sources = charge_sources_out[:, -2:]
            charge_sources = charge_sources_out[:, :-2]
            # print("charge_sources", charge_sources)
            # print("current_fukui_sources", current_fukui_sources)
            spin_charge_density_sources = charge_sources.view(
                spin_charge_density.shape[0], 2, -1
            )
            if self.element_charge_residual_scale != 1.0:
                spin_charge_density_sources = spin_charge_density_sources.clone()
                spin_charge_density_sources[:, :, : self.atomic_density_num_sigmas] = (
                    node_charge_residual_scale[:, None, None]
                    * spin_charge_density_sources[
                        :, :, : self.atomic_density_num_sigmas
                    ]
                )
            spin_charge_density = spin_charge_density + spin_charge_density_sources

            fukui_norm2 = scatter_sum(
                src=current_fukui_sources.double(),
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )[data["batch"]].to(vectors.dtype)
            fukui_norm2 = torch.where(
                fukui_norm2 == 0, torch.ones_like(fukui_norm2), fukui_norm2
            )
            current_fukui_sources = current_fukui_sources / fukui_norm2
            pred_total_charges = scatter_sum(
                src=spin_charge_density[:, :, : self.atomic_density_num_sigmas]
                .sum(dim=-1)
                .double(),
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )[data["batch"]].to(vectors.dtype)
            spin_charge_density = spin_charge_density.clone()
            spin_charge_density[:, 0, : self.atomic_density_num_sigmas] = (
                spin_charge_density[:, 0, : self.atomic_density_num_sigmas]
                + current_fukui_sources[:, 0, None]
                * ((Q_p_S / 2) - pred_total_charges[:, 0])[:, None]
                / float(self.atomic_density_num_sigmas)
            )
            spin_charge_density[:, 1, : self.atomic_density_num_sigmas] = (
                spin_charge_density[:, 1, : self.atomic_density_num_sigmas]
                + current_fukui_sources[:, 1, None]
                * ((Q_m_S / 2) - pred_total_charges[:, 1])[:, None]
                / float(self.atomic_density_num_sigmas)
            )

        total_energy = e0 + inter_e
        if self.local_electron_energy is not None:
            local_q_e = self.local_electron_energy(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs[:, : self.from_ell_max_field_update],
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                field_feats=potential_features,
                charges_0=field_independent_spin_charge_density.view(
                    field_independent_spin_charge_density.shape[0], -1
                ),
                charges_induced=spin_charge_density.view(spin_charge_density.shape[0], -1),
            )
            le_total = scatter_sum(
                src=local_q_e, index=data["batch"], dim=-1, dim_size=num_graphs
            )
            total_energy = total_energy + le_total
        else:
            le_total = torch.zeros_like(total_energy)

        charge_density = spin_charge_density.sum(dim=1)
        spin_density = spin_charge_density[:, 0, :] - spin_charge_density[:, 1, :]
        charge_density_radial_mul_ir = charge_density
        spin_density_radial_mul_ir = spin_density
        spin_charge_density_radial_mul_ir = spin_charge_density
        charge_density_mul_ir = self._radial_flat_to_effective(
            charge_density_radial_mul_ir
        )
        spin_density_mul_ir = self._radial_flat_to_effective(spin_density_radial_mul_ir)
        spin_charge_density_mul_ir = torch.stack(
            [
                self._radial_flat_to_effective(spin_charge_density_radial_mul_ir[:, 0, :]),
                self._radial_flat_to_effective(spin_charge_density_radial_mul_ir[:, 1, :]),
            ],
            dim=1,
        )
        total_charge, explicit_dipole = compute_total_charge_dipole_permuted(
            charge_density_mul_ir, positions, data["batch"], num_graphs
        )
        if charge_density_mul_ir.shape[1] > 1:
            atomic_dipole = charge_density_mul_ir[:, 1:4][:, [2, 0, 1]]
        else:
            atomic_dipole = torch.zeros_like(positions)
        element_index = torch.argmax(data["node_attrs"], dim=-1)
        node_valence_electrons = self.atomic_valence_electrons[element_index]
        charge_density_radial_coefficients = self._radial_flat_to_blocks(
            charge_density_radial_mul_ir
        )
        density_width_threshold = None
        if (
            self.solvent_density_threshold is not None
            and center_profile_inputs is not None
        ):
            density_width_threshold = max(
                float(self.solvent_density_threshold) - 0.05,
                1.0e-6,
            )
            z_e50, w_e50 = compute_density_threshold_crossing_from_baseline_profile(
                z_values=center_profile_inputs[0],
                raw_neutral_values=center_profile_inputs[1],
                radial_coefficients=charge_density_radial_coefficients.detach(),
                positions=positions.detach(),
                batch=data["batch"],
                cell=cell.detach(),
                atomic_density_sigmas=self.atomic_density_sigmas,
                density_threshold=float(self.solvent_density_threshold),
                width_density_threshold=density_width_threshold,
                inward=self.solvent_window_inward,
                outward=self.solvent_window_outward,
                axis=self.solvent_potential_axis,
            )
        elif self.solvent_density_threshold is not None:
            density_width_threshold = max(
                float(self.solvent_density_threshold) - 0.05,
                1.0e-6,
            )
            z_e50, w_e50 = compute_density_threshold_crossing_from_mace_multipoles(
                density_coefficients=charge_density_mul_ir.detach(),
                positions=positions.detach(),
                node_valence_electrons=node_valence_electrons.detach(),
                batch=data["batch"],
                cell=cell.detach(),
                density_smearing_width=self.atomic_multipoles_smearing_width,
                density_threshold=float(self.solvent_density_threshold),
                width_density_threshold=density_width_threshold,
                inward=self.solvent_window_inward,
                outward=self.solvent_window_outward,
                axis=self.solvent_potential_axis,
            )
        else:
            z_e50, w_e50 = compute_ze_crossing_from_mace_multipoles(
                density_coefficients=charge_density_mul_ir.detach(),
                positions=positions.detach(),
                node_valence_electrons=node_valence_electrons.detach(),
                batch=data["batch"],
                cell=cell.detach(),
                density_smearing_width=self.atomic_multipoles_smearing_width,
                level=self.solvent_ze_level,
                inward=self.solvent_window_inward,
                outward=self.solvent_window_outward,
                axis=self.solvent_potential_axis,
            )
        z50_o, w_o, z50_h, w_h = compute_oh_structure_crossings(
            positions=positions.detach(),
            node_attrs=data["node_attrs"],
            batch=data["batch"],
            cell=cell.detach(),
            atomic_numbers=self.atomic_numbers,
            level=0.5,
            axis=self.solvent_potential_axis,
        )
        qsum_o, qsum_h, msum_o, msum_h = compute_interface_qm_summaries(
            charges=charge_density_mul_ir[:, 0].detach(),
            atomic_dipole=atomic_dipole.detach(),
            positions=positions.detach(),
            node_attrs=data["node_attrs"],
            batch=data["batch"],
            atomic_numbers=self.atomic_numbers,
            z_o_anchor=z50_o.detach(),
            z_h_anchor=z50_h.detach(),
            axis=self.solvent_potential_axis,
        )
        graph_feats_global = scatter_mean(
            src=node_feats_out, index=data["batch"], dim=0, dim_size=num_graphs
        )
        z_axis = positions[:, self.solvent_potential_axis].detach()
        z_center_nodes = z_e50[data["batch"]].detach()
        sigma_if = max(float(self.solvent_interface_pool_sigma), 1.0e-12)
        interface_w = torch.exp(
            -0.5
            * torch.square(
                (z_axis - z_center_nodes)
                / z_axis.new_tensor(float(sigma_if))
            )
        )
        interface_w_sum = torch.clamp(
            scatter_sum(
                src=interface_w,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            ),
            min=1.0e-12,
        )
        graph_feats_if = scatter_sum(
            src=node_feats_out * interface_w.unsqueeze(-1),
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        ) / interface_w_sum.unsqueeze(-1)
        input_total_charge = data["total_charge"].view(-1).to(total_charge.dtype)
        explicit_potential_base, _, _ = predict_potential_from_dipole_and_solvent_layer(
            dipole=explicit_dipole,
            total_charge=torch.zeros_like(total_charge),
            center=z_e50,
            cell=cell.detach(),
            sigma_g=self.solvent_sigma_g,
            axis=self.solvent_potential_axis,
            potential_sign=self.solvent_potential_sign,
        )
        solvent_scalar_inputs = torch.stack(
            [
                z_e50.to(graph_feats_global.dtype) / 10.0,
                input_total_charge.to(graph_feats_global.dtype) / 10.0,
                explicit_potential_base.to(graph_feats_global.dtype) / 10.0,
            ],
            dim=-1,
        )
        solvent_head_inputs = torch.cat(
            [graph_feats_global, graph_feats_if, solvent_scalar_inputs], dim=-1
        )
        solvent_raw_shift = self.solvent_center_residual(solvent_head_inputs).view(-1)
        if self.learn_solvent_center_residual:
            solvent_center_residual = (
                self.solvent_residual_max_abs_shift
                * torch.tanh(solvent_raw_shift)
            )
        else:
            solvent_center_residual = solvent_raw_shift * 0.0
        solv_center = (
            z_e50
            + z_e50.new_full(z_e50.shape, float(self.solvent_center_mean_shift))
            + solvent_center_residual
        )
        cell_for_axis = cell.detach()
        n_graphs_local = int(solv_center.shape[0])
        if cell_for_axis.dim() == 1 and cell_for_axis.numel() == n_graphs_local * 9:
            cell_for_axis = cell_for_axis.view(n_graphs_local, 3, 3)
        elif cell_for_axis.dim() == 2:
            if cell_for_axis.shape == (3, 3) and n_graphs_local == 1:
                cell_for_axis = cell_for_axis.unsqueeze(0)
            elif (
                cell_for_axis.shape[1] == 9
                and cell_for_axis.shape[0] == n_graphs_local
            ):
                cell_for_axis = cell_for_axis.view(n_graphs_local, 3, 3)
            elif (
                cell_for_axis.shape[1] == 3
                and cell_for_axis.shape[0] == n_graphs_local * 3
            ):
                cell_for_axis = cell_for_axis.view(n_graphs_local, 3, 3)
        if cell_for_axis.dim() == 3:
            l_axis = torch.stack(
                [
                    _axis_box_length(cell_for_axis[i], self.solvent_potential_axis)
                    for i in range(cell_for_axis.shape[0])
                ]
            ).to(solv_center.dtype)
        elif cell_for_axis.dim() == 2 and cell_for_axis.shape[1] == 3:
            l_axis = torch.clamp(
                torch.abs(cell_for_axis[:, self.solvent_potential_axis]),
                min=1.0e-12,
            ).to(solv_center.dtype)
        else:
            raise ValueError(
                f"Unsupported cell shape for solvent axis length: {tuple(cell_for_axis.shape)}"
            )
        if pb_solvent_data is not None:
            layer_mean = pb_solvent_data["layer_mean"].to(solv_center.dtype)
            solvent_mu = pb_solvent_data["solvent_mu"].to(solv_center.dtype)
            # The physical center is the PB profile mean; the learned
            # residual head is kept in the graph with zero weight so
            # distributed training sees all parameters.
            solv_center = layer_mean + solvent_raw_shift * 0.0
        else:
            layer_mean = _truncated_gaussian_mean(
                center=solv_center,
                sigma=self.solvent_sigma_g,
                lower=torch.zeros_like(solv_center),
                upper=l_axis,
            )
            solvent_mu = (-input_total_charge.view(-1)) * layer_mean
        solvent_dipole = torch.zeros_like(explicit_dipole)
        solvent_dipole[:, self.solvent_potential_axis] = solvent_mu.to(
            solvent_dipole.dtype
        )
        total_dipole = explicit_dipole + solvent_dipole
        if pb_solvent_data is not None:
            _, explicit_potential_base, solvent_potential = (
                predict_potential_from_dipole_and_solvent_mu(
                    dipole=explicit_dipole,
                    solvent_mu=solvent_mu,
                    cell=cell.detach(),
                    axis=self.solvent_potential_axis,
                    potential_sign=self.solvent_potential_sign,
                )
            )
        else:
            _, explicit_potential_base, solvent_potential = (
                predict_potential_from_dipole_and_solvent_layer(
                    dipole=explicit_dipole,
                    total_charge=input_total_charge,
                    center=solv_center,
                    cell=cell.detach(),
                    sigma_g=self.solvent_sigma_g,
                    axis=self.solvent_potential_axis,
                    potential_sign=self.solvent_potential_sign,
                )
            )
        explicit_potential = explicit_potential_base
        total_potential = explicit_potential + solvent_potential
        fermi_level_residual = self.fermi_level_residual(graph_feats_global).view(-1)
        fermi_level_pred = (
            self.fermi_level_baseline.to(dtype=total_potential.dtype)
            - total_potential
            + fermi_level_residual.to(dtype=total_potential.dtype)
        )
        electro_energy = self.coulomb_energy(
            k_vectors=k_vectors,
            k_norm2=kv_norms_squared,
            k_vector_batch=k_vectors_batch,
            k0_mask=k_vectors_0mask,
            source_feats=charge_density_mul_ir[
                :, : self.electrostatic_multipoles_dim
            ],
            node_positions=positions,
            batch=data["batch"],
            volume=data["volume"],
            pbc=data["pbc"].view(-1, 3),
            force_pbc_evaluator=use_pbc_evaluator,
        )
        compensation_periodic_1d_energy = _slab_compensation_periodic_1d_energy_radial(
            radial_coefficients=charge_density_radial_coefficients,
            atomic_density_sigmas=self.atomic_density_sigmas,
            total_charge=input_total_charge,
            center=solv_center,
            cell=cell.detach(),
            pbc=data["pbc"].view(-1, 3),
            batch=data["batch"],
            positions=positions,
            num_graphs=num_graphs,
            sigma_g=self.solvent_sigma_g,
            axis=self.solvent_potential_axis,
            comp_profile=(
                pb_solvent_data["profile_energy"]
                if pb_solvent_data is not None
                else None
            ),
        )
        compensation_slab_correction_energy = _slab_dipole_correction_delta(
            explicit_dipole=explicit_dipole,
            total_dipole=total_dipole,
            volume=data["volume"],
            pbc=data["pbc"].view(-1, 3),
            axis=self.solvent_potential_axis,
        )
        total_energy = (
            total_energy
            + electro_energy
            + compensation_periodic_1d_energy
            + compensation_slab_correction_energy
            + torch.sum(external_potential[:, 1:] * explicit_dipole, dim=-1)
        )

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=total_energy,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces or compute_atomic_stresses,
        )

        atomic_virials: Optional[torch.Tensor] = None
        atomic_stresses: Optional[torch.Tensor] = None
        if compute_atomic_stresses and edge_forces is not None:
            from .utils import get_atomic_virials_stresses as _gav

            atomic_virials, atomic_stresses = _gav(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )

        return {
            "energy": total_energy,
            "node_energy": node_e0.clone().double() + node_inter_es.clone().double(),
            "interaction_energy": inter_e,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "hessian": hessian,
            "displacement": displacement,
            "node_feats": node_feats_out,
            "density_coefficients": charge_density_mul_ir,
            "charge_density_radial_coefficients": charge_density_radial_coefficients,
            "spin_density": spin_density_mul_ir,
            "charges_history": torch.stack(
                [spin_charge_density_mul_ir.clone().detach()], dim=-1
            ),
            "fermi_level": external_potential[:, 0],
            "external_field": external_potential[:, 1:],
            "charges": charge_density_mul_ir[:, 0],
            "spins": spin_density_mul_ir[:, 0],
            "dipole": total_dipole,
            "explicit_dipole": explicit_dipole,
            "solvent_dipole": solvent_dipole,
            "atomic_dipole": atomic_dipole,
            "total_charge": total_charge,
            "potential": total_potential,
            "fermi_level_pred": fermi_level_pred,
            "fermi_level_residual": fermi_level_residual,
            "fermi_level_baseline": self.fermi_level_baseline.expand_as(total_potential),
            "explicit_potential": explicit_potential,
            "explicit_potential_base": explicit_potential_base,
            "solvent_potential": solvent_potential,
            "z_e50": z_e50,
            "w_e50": w_e50,
            "z_e0p15": z_e50,
            "w_e0p15": w_e50,
            "z50_o": z50_o,
            "w_o": w_o,
            "z50_h": z50_h,
            "w_h": w_h,
            "qsum_o": qsum_o,
            "qsum_h": qsum_h,
            "msum_o": msum_o,
            "msum_h": msum_h,
            "solv_center_physical_shift": torch.zeros_like(solv_center),
            "solv_center": solv_center,
            "solv_center_residual": solvent_center_residual,
            "compensation_scf_center": comp_center_init,
            "solvent_layer_mean": layer_mean,
            "electrostatic_energy": electro_energy,
            "compensation_periodic_1d_energy": compensation_periodic_1d_energy,
            "compensation_slab_correction_energy": compensation_slab_correction_energy,
            "electron_energy": le_total,
            "electrostatic_potentials": esps,
            "spin_charge_density": spin_charge_density_mul_ir,
        }
