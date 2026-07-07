###########################################################################################
# Implementation of different loss functions
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import ast
import csv
import json
import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.distributed as dist

from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch
from .solvent_charge_layer import (
    _axis_box_length,
    _truncated_gaussian_mean,
    predict_potential_from_dipole_and_solvent_layer,
)

try:
    from graph_longrange.realspace_grid_integrals import (
        tensor_realspace_GTO_evaluation,
    )
except ImportError:  # pragma: no cover - graph_longrange is required by PolarMACE
    tensor_realspace_GTO_evaluation = None


# ------------------------------------------------------------------------------
# Helper function for loss reduction that handles DDP correction
# ------------------------------------------------------------------------------
def is_ddp_enabled():
    return dist.is_initialized() and dist.get_world_size() > 1


def reduce_loss(raw_loss: torch.Tensor, ddp: Optional[bool] = None) -> torch.Tensor:
    """
    Reduces an element-wise loss tensor.

    If ddp is True and distributed is initialized, the function computes:

        loss = (local_sum * world_size) / global_num_elements

    Otherwise, it returns the regular mean.
    """
    ddp = is_ddp_enabled() if ddp is None else ddp
    if ddp and dist.is_initialized():
        world_size = dist.get_world_size()
        n_local = raw_loss.numel()
        loss_sum = raw_loss.sum()
        total_samples = torch.tensor(
            n_local, device=raw_loss.device, dtype=raw_loss.dtype
        )
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        return loss_sum * world_size / total_samples
    return raw_loss.mean()


# ------------------------------------------------------------------------------
# Energy Loss Functions
# ------------------------------------------------------------------------------


def mean_squared_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    raw_loss = torch.square(ref["energy"] - pred["energy"])
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Calculate per-graph number of atoms.
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # shape: [n_graphs]
    raw_loss = (
        ref.weight
        * ref.energy_weight
        * torch.square((ref["energy"] - pred["energy"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


def weighted_mean_absolute_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    raw_loss = (
        ref.weight
        * ref.energy_weight
        * torch.abs((ref["energy"] - pred["energy"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Stress and Virials Loss Functions
# ------------------------------------------------------------------------------


def weighted_mean_squared_stress(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
    raw_loss = (
        configs_weight
        * configs_stress_weight
        * torch.square(ref["stress"] - pred["stress"])
    )
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_virials(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_virials_weight = ref.virials_weight.view(-1, 1, 1)
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)
    raw_loss = (
        configs_weight
        * configs_virials_weight
        * torch.square((ref["virials"] - pred["virials"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Forces Loss Functions
# ------------------------------------------------------------------------------


def mean_squared_error_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Repeat per-graph weights to per-atom level.
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    raw_loss = (
        configs_weight
        * configs_forces_weight
        * torch.square(ref["forces"] - pred["forces"])
    )
    return reduce_loss(raw_loss, ddp)


def mean_normed_error_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    raw_loss = torch.linalg.vector_norm(ref["forces"] - pred["forces"], ord=2, dim=-1)
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Dipole Loss Function
# ------------------------------------------------------------------------------


def weighted_mean_squared_error_dipole(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
    ref_dipole = reference_total_dipole(ref)
    if ref_dipole is None:
        ref_dipole = ref["dipole"]
    raw_loss = (
        ref.weight.view(-1, 1)
        * ref.dipole_weight
        * torch.square((ref_dipole - pred["dipole"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Polarizability Loss Function
# ------------------------------------------------------------------------------


def weighted_mean_squared_error_polarizability(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[
        bool
    ] = None,  # ,mean: Optional[torch.Tensor] = None , std: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # polarizability: [n_graphs, ]
    # ref_polar = ref["polarizability"].view(-1, 3, 3) * std.view(1, 3, 3) + mean.view(1, 3, 3) if mean is not None and std is not None else ref["polarizability"]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)  # [n_graphs,1]
    raw_loss = (
        ref.weight.view(-1, 1, 1)
        * ref.polarizability_weight
        * torch.square(
            (ref["polarizability"].view(-1, 3, 3) - pred["polarizability"]) / num_atoms
        )
    )
    return reduce_loss(raw_loss, ddp)


POTENTIAL_FROM_DIPOLE_SCALE_EV_PER_EANG_A2 = (
    4.0 * math.pi * 27.211386245988 / 1.8897261258369282
)


def _batch_get(batch: Batch, key: str):
    if isinstance(batch, dict):
        return batch.get(key, None)
    return getattr(batch, key, None)


def attach_density_3d_samples_to_batch(batch: Batch, loss_fn) -> None:
    density_targets = getattr(loss_fn, "density_3d_targets", None)
    samples_per_graph = int(getattr(loss_fn, "density_3d_samples", 0))
    if not density_targets or samples_per_graph <= 0:
        return
    sample_ids = _batch_get(batch, "sample_id")
    if sample_ids is None:
        return
    rng = getattr(loss_fn, "density_3d_rng", None)
    positions = batch["positions"] if isinstance(batch, dict) else batch.positions
    points_all = []
    rho_all = []
    graph_all = []
    for graph_idx, sample_id_value in enumerate(sample_ids.view(-1)):
        sample_id = int(sample_id_value.detach().cpu().item())
        if hasattr(density_targets, "sample_points"):
            points, rho_ref = density_targets.sample_points(
                sample_id=sample_id,
                n_points=samples_per_graph,
                rng=rng,
                dtype=positions.dtype,
                device=positions.device,
            )
        else:
            point_list, rho_list = density_targets[sample_id]
            if samples_per_graph > 0 and samples_per_graph < len(point_list):
                rand = rng if rng is not None else random
                idx = rand.sample(range(len(point_list)), samples_per_graph)
                point_list = [point_list[i] for i in idx]
                rho_list = [rho_list[i] for i in idx]
            points = positions.new_tensor(point_list)
            rho_ref = positions.new_tensor(rho_list)
        points_all.append(points)
        rho_all.append(rho_ref)
        graph_all.append(
            torch.full(
                (points.shape[0],),
                graph_idx,
                device=positions.device,
                dtype=torch.long,
            )
        )
    if not points_all:
        return
    batch.density_probe_points = torch.cat(points_all, dim=0)
    batch.density_probe_rho = torch.cat(rho_all, dim=0)
    batch.density_probe_graph_index = torch.cat(graph_all, dim=0)


def reference_total_dipole(
    ref: Batch,
    axis: int = 2,
    sigma_g: float = 0.85,
) -> Optional[torch.Tensor]:
    ref_explicit_dipole = _batch_get(ref, "explicit_dipole")
    if ref_explicit_dipole is None:
        ref_explicit_dipole = _batch_get(ref, "dipole")
    if ref_explicit_dipole is None:
        return None
    return ref_explicit_dipole.view(-1, 3)


def weighted_mean_squared_error_charges(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> Optional[torch.Tensor]:
    if pred.get("charges") is None or _batch_get(ref, "charges") is None:
        return None
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    config_weight = torch.repeat_interleave(ref.weight * ref.charges_weight, num_atoms)
    raw_loss = config_weight * torch.square(ref["charges"] - pred["charges"])
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_error_atomic_dipole(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> Optional[torch.Tensor]:
    if pred.get("atomic_dipole") is None or _batch_get(ref, "atomic_dipole") is None:
        return None
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    config_weight = torch.repeat_interleave(
        ref.weight * ref.atomic_dipole_weight, num_atoms
    ).unsqueeze(-1)
    raw_loss = config_weight * torch.square(
        ref["atomic_dipole"] - pred["atomic_dipole"]
    )
    return reduce_loss(raw_loss, ddp)


def predict_potential_from_dipole(
    ref: Batch,
    pred: TensorDict,
    axis: int = 2,
    potential_sign: float = 1.0,
) -> Optional[torch.Tensor]:
    dipole = pred.get("dipole")
    cell = _batch_get(ref, "cell")
    if dipole is None or cell is None:
        return None
    if axis not in (0, 1, 2):
        raise ValueError("potential_axis must be 0/1/2")

    cells = cell
    if cells.dim() == 2 and cells.shape[1] == 3:
        if cells.shape[0] % 3 != 0:
            return None
        cells = cells.view(-1, 3, 3)
    elif cells.dim() != 3:
        return None

    dipoles = dipole.view(-1, 3)
    if dipoles.shape[0] != cells.shape[0]:
        return None

    if axis == 0:
        v1, v2 = cells[:, 1, :], cells[:, 2, :]
    elif axis == 1:
        v1, v2 = cells[:, 0, :], cells[:, 2, :]
    else:
        v1, v2 = cells[:, 0, :], cells[:, 1, :]

    area = torch.linalg.norm(torch.cross(v1, v2, dim=-1), dim=-1)
    area = torch.clamp(area.to(dipoles.dtype), min=1e-12)
    scale = dipoles.new_tensor(POTENTIAL_FROM_DIPOLE_SCALE_EV_PER_EANG_A2)
    sign = dipoles.new_tensor(float(potential_sign))
    return sign * scale * dipoles[:, axis] / area


def mean_squared_error_potential_from_dipole(
    ref: Batch,
    pred: TensorDict,
    axis: int = 2,
    potential_sign: float = 1.0,
    ddp: Optional[bool] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    pred_potential = pred.get("potential")
    if pred_potential is None:
        pred_potential = predict_potential_from_dipole(
            ref=ref, pred=pred, axis=axis, potential_sign=potential_sign
        )
    if pred_potential is None or _batch_get(ref, "potential") is None:
        return None, None
    raw_loss = (
        ref.weight
        * ref.potential_weight
        * torch.square(ref["potential"].view(-1) - pred_potential.view(-1))
    )
    return reduce_loss(raw_loss, ddp), pred_potential


def mean_squared_error_fermi_level(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[bool] = None,
) -> Optional[torch.Tensor]:
    pred_fermi = pred.get("fermi_level_pred")
    ref_fermi = getattr(ref, "fermi_level", None)
    if pred_fermi is None or ref_fermi is None:
        return None
    raw_loss = torch.square(ref_fermi.view(-1) - pred_fermi.view(-1))
    return reduce_loss(raw_loss, ddp)


def mean_squared_error_fermi_residual(
    pred: TensorDict,
    ddp: Optional[bool] = None,
) -> Optional[torch.Tensor]:
    residual = pred.get("fermi_level_residual")
    if residual is None:
        return None
    raw_loss = torch.square(residual.view(-1))
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_error_explicit_potential(
    ref: Batch,
    pred: TensorDict,
    axis: int = 2,
    potential_sign: float = 1.0,
    ddp: Optional[bool] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    pred_explicit_potential = pred.get("explicit_potential")
    ref_dipole = _batch_get(ref, "explicit_dipole")
    if ref_dipole is None:
        ref_dipole = _batch_get(ref, "dipole")
    ref_cell = _batch_get(ref, "cell")
    if pred_explicit_potential is None or ref_dipole is None or ref_cell is None:
        return None, None
    zeros = torch.zeros(
        (ref_dipole.view(-1, 3).shape[0],),
        dtype=ref_dipole.dtype,
        device=ref_dipole.device,
    )
    _, ref_explicit_potential, _ = predict_potential_from_dipole_and_solvent_layer(
        dipole=ref_dipole.view(-1, 3),
        total_charge=zeros,
        center=zeros,
        cell=ref_cell,
        sigma_g=0.85,
        axis=axis,
        potential_sign=potential_sign,
    )
    raw_loss = ref.weight * torch.square(
        ref_explicit_potential.view(-1) - pred_explicit_potential.view(-1)
    )
    return reduce_loss(raw_loss, ddp), pred_explicit_potential


def _load_density_3d_tsv(
    path: Union[str, Path],
) -> dict[int, tuple[list[list[float]], list[float]]]:
    grouped: dict[int, list[tuple[int, list[float], float]]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"sample_id", "point_index", "x_A", "y_A", "z_A", "rho_e_per_A3"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in density_3d_file {path}: {sorted(missing)}")
        for row in reader:
            sid = int(row["sample_id"])
            grouped.setdefault(sid, []).append(
                (
                    int(row["point_index"]),
                    [float(row["x_A"]), float(row["y_A"]), float(row["z_A"])],
                    float(row["rho_e_per_A3"]),
                )
            )
    out: dict[int, tuple[list[list[float]], list[float]]] = {}
    for sid, values in grouped.items():
        values.sort(key=lambda x: x[0])
        out[sid] = ([v[1] for v in values], [v[2] for v in values])
    return out


class Density3DGridTargets:
    def __init__(self, manifest_path: Union[str, Path], max_cache_size: int = 32):
        self.manifest_path = Path(manifest_path)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != "density3d_grid_npz_v1":
            raise ValueError(
                f"Unsupported density grid manifest format in {self.manifest_path}"
            )
        self.entries = {int(k): v for k, v in payload["entries"].items()}
        self.max_cache_size = int(max_cache_size)
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = (
            OrderedDict()
        )

    def __contains__(self, sample_id: int) -> bool:
        return int(sample_id) in self.entries

    def _load(self, sample_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample_id = int(sample_id)
        if sample_id in self._cache:
            self._cache.move_to_end(sample_id)
            return self._cache[sample_id]
        entry = self.entries[sample_id]
        path = Path(entry["path"])
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        with np.load(path) as data:
            rho = np.asarray(data["rho"], dtype=np.float32)
            lattice = np.asarray(data["lattice"], dtype=np.float64)
            valid_iz = np.asarray(data["valid_iz"], dtype=np.int64)
        self._cache[sample_id] = (rho, lattice, valid_iz)
        self._cache.move_to_end(sample_id)
        while len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)
        return rho, lattice, valid_iz

    def sample_points(
        self,
        sample_id: int,
        n_points: int,
        rng: Optional[random.Random],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n_points <= 0:
            raise ValueError(
                "density_3d_samples must be > 0 when density_3d_file is a full-grid manifest"
            )
        rho, lattice, valid_iz = self._load(sample_id)
        nz, ny, nx = rho.shape
        n_valid = int(valid_iz.size) * ny * nx
        if n_points > n_valid:
            raise ValueError(
                f"Requested {n_points} density points but only {n_valid} grid points are available"
            )
        rand = rng if rng is not None else random
        linear = np.asarray(rand.sample(range(n_valid), int(n_points)), dtype=np.int64)
        plane = linear // (ny * nx)
        rem = linear - plane * ny * nx
        iy = rem // nx
        ix = rem - iy * nx
        iz = valid_iz[plane]
        frac = np.column_stack(
            [
                ix.astype(np.float64) / float(nx),
                iy.astype(np.float64) / float(ny),
                iz.astype(np.float64) / float(nz),
            ]
        )
        points = frac @ lattice
        rho_ref = rho[iz, iy, ix]
        return (
            torch.as_tensor(points, dtype=dtype, device=device),
            torch.as_tensor(rho_ref, dtype=dtype, device=device),
        )

    def plane_average(
        self,
        sample_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        rho, _, _ = self._load(sample_id)
        rho_1d = rho.mean(axis=(1, 2))
        return torch.as_tensor(rho_1d, dtype=dtype, device=device)


def _load_density_3d_targets(path: Union[str, Path]):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return Density3DGridTargets(path)
    return _load_density_3d_tsv(path)


def _cell_area_for_axis(cell_g: torch.Tensor, axis: int) -> torch.Tensor:
    if cell_g.dim() == 1:
        cell_g = cell_g.view(3, 3)
    if axis == 0:
        v1, v2 = cell_g[1, :], cell_g[2, :]
    elif axis == 1:
        v1, v2 = cell_g[0, :], cell_g[2, :]
    else:
        v1, v2 = cell_g[0, :], cell_g[1, :]
    return torch.clamp(torch.linalg.norm(torch.cross(v1, v2, dim=-1), dim=-1), min=1.0e-12)


def _cells_from_batch(ref: Batch, num_graphs: int) -> torch.Tensor:
    cells = ref["cell"]
    if cells.dim() == 3:
        return cells
    if cells.dim() == 2 and cells.shape == (3, 3) and num_graphs == 1:
        return cells.unsqueeze(0)
    if cells.dim() == 2 and cells.shape[0] == num_graphs * 3 and cells.shape[1] == 3:
        return cells.view(num_graphs, 3, 3)
    if cells.dim() == 2 and cells.shape[0] == num_graphs and cells.shape[1] == 9:
        return cells.view(num_graphs, 3, 3)
    if cells.dim() == 1 and cells.numel() == num_graphs * 9:
        return cells.view(num_graphs, 3, 3)
    raise ValueError(f"Unsupported batched cell shape: {tuple(cells.shape)}")


def _gaussian_1d(z_grid: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma_t = z_grid.new_tensor(float(max(sigma, 1.0e-12)))
    dz = z_grid[:, None] - centers[None, :]
    return torch.exp(-0.5 * torch.square(dz / sigma_t)) / (
        math.sqrt(2.0 * math.pi) * sigma_t
    )


def _gaussian_d1_1d(z_grid: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    g = _gaussian_1d(z_grid, centers, sigma)
    sigma_t = z_grid.new_tensor(float(max(sigma, 1.0e-12)))
    dz = z_grid[:, None] - centers[None, :]
    return -(dz / (sigma_t * sigma_t)) * g


def _gaussian_d2_1d(z_grid: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    g = _gaussian_1d(z_grid, centers, sigma)
    sigma_t = z_grid.new_tensor(float(max(sigma, 1.0e-12)))
    dz = z_grid[:, None] - centers[None, :]
    sigma2 = sigma_t * sigma_t
    return ((dz * dz) / (sigma2 * sigma2) - 1.0 / sigma2) * g


def _infer_multipole_max_l(num_coefficients: int) -> int:
    max_l = int(round(math.sqrt(float(num_coefficients)) - 1.0))
    if (max_l + 1) ** 2 != int(num_coefficients):
        raise ValueError(
            "density_coefficients dimension must be a complete spherical-harmonic "
            f"block, got {num_coefficients}"
        )
    return max_l


def _axis2_area_height(cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    area = _cell_area_for_axis(cell, 2)
    volume = torch.abs(torch.linalg.det(cell))
    return area, torch.clamp(volume / area, min=1.0e-12)


def _height_from_uniform_z_grid(z_ref: torch.Tensor) -> torch.Tensor:
    if int(z_ref.numel()) < 2:
        raise ValueError("potential_1d_profile target must contain at least two z points")
    dz = z_ref[1] - z_ref[0]
    return torch.abs(dz) * float(z_ref.numel())


def _electron_density_coefficients_from_mace_multipoles(
    density_coefficients: torch.Tensor,
    atomic_numbers: torch.Tensor,
) -> torch.Tensor:
    raise ValueError(
        "Converting non-radial charge coefficients to electron density requires "
        "explicit atomic_valence_electrons. Current density/potential losses should "
        "use PolarMACE charge_density_radial_coefficients instead."
    )


def _parse_density_sigmas(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [float(item) for item in parsed]
        return [float(parsed)]
    return [float(value)]


def _electron_density_radial_coefficients(
    density_coefficients: torch.Tensor,
    atomic_numbers: torch.Tensor,
) -> torch.Tensor:
    if density_coefficients.dim() == 3:
        return density_coefficients
    return _electron_density_coefficients_from_mace_multipoles(
        density_coefficients, atomic_numbers
    )[:, None, :]


def _gto_density_at_points_axis2_pbc(
    points: torch.Tensor,
    density_coefficients: torch.Tensor,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell: torch.Tensor,
    sigma,
) -> torch.Tensor:
    if tensor_realspace_GTO_evaluation is None:
        raise ImportError("graph_longrange is required for GTO density loss")

    point_frac = torch.linalg.solve(cell.T.to(dtype=points.dtype), points.T).T
    atom_frac = torch.linalg.solve(
        cell.T.to(dtype=positions.dtype), positions.T
    ).T.to(dtype=points.dtype)
    disp_frac = point_frac[:, None, :] - atom_frac[None, :, :]
    disp_frac[..., 0:2] = disp_frac[..., 0:2] - torch.round(disp_frac[..., 0:2])
    disp_cart = torch.einsum("...i,ij->...j", disp_frac, cell.to(dtype=points.dtype))

    coeff_e = _electron_density_radial_coefficients(
        density_coefficients, atomic_numbers
    ).to(dtype=points.dtype)
    sigmas = _parse_density_sigmas(sigma)
    if coeff_e.shape[1] != len(sigmas):
        raise ValueError(
            f"density radial coefficient count {coeff_e.shape[1]} does not match "
            f"number of sigmas {len(sigmas)}"
        )
    max_l = _infer_multipole_max_l(coeff_e.shape[2])
    rho = points.new_zeros(points.shape[0])
    zero = points.new_zeros(3)
    for radial_idx, sigma_value in enumerate(sigmas):
        for ell in range(max_l + 1):
            start = ell * ell
            stop = (ell + 1) * (ell + 1)
            basis = tensor_realspace_GTO_evaluation(
                disp_cart,
                ell,
                float(sigma_value),
                zero,
                normalize="multipoles",
            )
            rho = rho + torch.einsum(
                "pnl,nl->p", basis, coeff_e[:, radial_idx, start:stop]
            )
    return rho


def density_3d_residuals(
    ref: Batch,
    pred: TensorDict,
    density_targets: dict[int, tuple[list[list[float]], list[float]]],
    density_smearing_width: float,
    samples_per_graph: int = 0,
    rng: Optional[random.Random] = None,
) -> Optional[torch.Tensor]:
    density_coefficients = pred.get("charge_density_radial_coefficients")
    if density_coefficients is None:
        raise KeyError(
            "density_3d loss requires PolarMACE charge_density_radial_coefficients"
        )
    if (
        _batch_get(ref, "sample_id") is None
        or _batch_get(ref, "atomic_numbers") is None
    ):
        return None

    num_graphs = int(ref.ptr.numel() - 1)
    cells = _cells_from_batch(ref, num_graphs)
    atomic_numbers = _batch_get(ref, "atomic_numbers").to(dtype=torch.long)
    losses = []
    for graph_idx in range(num_graphs):
        sample_id = int(_batch_get(ref, "sample_id").view(-1)[graph_idx].detach().cpu().item())
        if sample_id not in density_targets:
            raise KeyError(f"sample_id {sample_id} is missing from density_3d_file")
        attached_points = _batch_get(ref, "density_probe_points")
        attached_rho = _batch_get(ref, "density_probe_rho")
        attached_graph_index = _batch_get(ref, "density_probe_graph_index")
        if (
            attached_points is not None
            and attached_rho is not None
            and attached_graph_index is not None
        ):
            graph_mask = attached_graph_index.to(device=attached_points.device) == graph_idx
            points = attached_points[graph_mask]
            rho_ref = attached_rho[graph_mask]
        elif hasattr(density_targets, "sample_points"):
            points, rho_ref = density_targets.sample_points(
                sample_id=sample_id,
                n_points=int(samples_per_graph),
                rng=rng,
                dtype=ref["positions"].dtype,
                device=ref["positions"].device,
            )
        else:
            point_list, rho_list = density_targets[sample_id]
            if samples_per_graph > 0 and samples_per_graph < len(point_list):
                rand = rng if rng is not None else random
                idx = rand.sample(range(len(point_list)), int(samples_per_graph))
                point_list = [point_list[i] for i in idx]
                rho_list = [rho_list[i] for i in idx]
            points = ref["positions"].new_tensor(point_list)
            rho_ref = ref["positions"].new_tensor(rho_list)
        start = int(ref.ptr[graph_idx].item())
        stop = int(ref.ptr[graph_idx + 1].item())
        rho_pred = _gto_density_at_points_axis2_pbc(
            points=points,
            density_coefficients=density_coefficients[start:stop],
            positions=ref["positions"][start:stop],
            atomic_numbers=atomic_numbers[start:stop],
            cell=cells[graph_idx].to(ref["positions"].dtype),
            sigma=density_smearing_width,
        )
        losses.append(rho_pred - rho_ref)
    if not losses:
        return None
    return torch.cat(losses, dim=0)


def mean_squared_error_density_3d(
    ref: Batch,
    pred: TensorDict,
    density_targets: dict[int, tuple[list[list[float]], list[float]]],
    density_smearing_width: float,
    samples_per_graph: int = 0,
    rng: Optional[random.Random] = None,
    ddp: Optional[bool] = None,
) -> Optional[torch.Tensor]:
    residuals = density_3d_residuals(
        ref=ref,
        pred=pred,
        density_targets=density_targets,
        density_smearing_width=density_smearing_width,
        samples_per_graph=samples_per_graph,
        rng=rng,
    )
    if residuals is None:
        return None
    return reduce_loss(torch.square(residuals), ddp)


FELECT_EV_ANG = 14.39964546866782


def _load_potential_1d_profiles_npz(path: Union[str, Path]) -> dict[int, dict[str, np.ndarray]]:
    arrays = np.load(path)
    required = {
        "sample_ids",
        "z_A",
        "phi_eV",
        "raw_neutral_e",
        "raw_ion_potcar_e",
        "cdipol_indmin",
    }
    missing = required.difference(arrays.files)
    if missing:
        raise ValueError(f"Missing arrays in potential_1d_profile_file {path}: {sorted(missing)}")
    sample_ids = np.asarray(arrays["sample_ids"], dtype=np.int64)
    z_values = np.asarray(arrays["z_A"], dtype=np.float64)
    phi_values = np.asarray(arrays["phi_eV"], dtype=np.float64)
    neutral_values = np.asarray(arrays["raw_neutral_e"], dtype=np.float64)
    ion_values = np.asarray(arrays["raw_ion_potcar_e"], dtype=np.float64)
    if not (z_values.shape == phi_values.shape == neutral_values.shape == ion_values.shape):
        raise ValueError(
            "z_A, phi_eV, raw_neutral_e and raw_ion_potcar_e must have identical shapes "
            f"in {path}"
        )
    indmin_values = np.asarray(arrays["cdipol_indmin"], dtype=np.int64)
    if indmin_values.shape[0] != sample_ids.shape[0]:
        raise ValueError(f"cdipol_indmin length does not match sample_ids in {path}")
    if "align_mask" in arrays.files:
        align_mask_values = np.asarray(arrays["align_mask"], dtype=np.float64)
        if align_mask_values.shape != z_values.shape:
            raise ValueError(f"align_mask shape does not match z_A in {path}")
    else:
        align_mask_values = np.ones_like(z_values, dtype=np.float64)
    return {
        int(sample_id): {
            "z_A": z_values[idx],
            "phi_eV": phi_values[idx],
            "raw_neutral_e": neutral_values[idx],
            "raw_ion_potcar_e": ion_values[idx],
            "align_mask": align_mask_values[idx],
            "cdipol_indmin": np.asarray(indmin_values[idx], dtype=np.int64),
        }
        for idx, sample_id in enumerate(sample_ids)
    }


def _solvent_profile_1d_torch(
    z_ref: torch.Tensor,
    area: torch.Tensor,
    height: torch.Tensor,
    total_charge: torch.Tensor,
    solvent_center: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    sigma_t = z_ref.new_tensor(float(max(sigma, 1.0e-12)))
    lower = z_ref.new_tensor(0.0)
    upper = height.to(dtype=z_ref.dtype)
    center = solvent_center.to(dtype=z_ref.dtype)
    inv_sqrt2 = z_ref.new_tensor(1.0 / math.sqrt(2.0))
    a = (lower - center) / sigma_t
    b = (upper - center) / sigma_t
    norm = 0.5 * (torch.erf(b * inv_sqrt2) - torch.erf(a * inv_sqrt2))
    norm = torch.clamp(norm, min=1.0e-12)
    dz = z_ref - center
    gaussian = torch.exp(-0.5 * torch.square(dz / sigma_t)) / (
        z_ref.new_tensor(math.sqrt(2.0 * math.pi)) * sigma_t
    )
    layer_charge = -total_charge.to(dtype=z_ref.dtype)
    return (layer_charge / area.to(dtype=z_ref.dtype)) * gaussian / norm


def _residual_plane_density_from_radial_coefficients(
    z_ref: torch.Tensor,
    density_coefficients: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    sigma,
    axis: int,
) -> torch.Tensor:
    if axis != 2:
        raise NotImplementedError("potential_1d_profile loss currently supports axis=2")
    area = _cell_area_for_axis(cell.to(dtype=z_ref.dtype), axis)
    z_atom = positions[:, axis].to(dtype=z_ref.dtype)
    coeff = density_coefficients.to(dtype=z_ref.dtype)
    if coeff.dim() != 3:
        coeff = coeff[:, None, :]
    sigmas = _parse_density_sigmas(sigma)
    if coeff.shape[1] != len(sigmas):
        raise ValueError(
            f"density radial coefficient count {coeff.shape[1]} does not match "
            f"number of sigmas {len(sigmas)}"
        )
    rho = z_ref.new_zeros(z_ref.shape)
    for radial_idx, sigma_value in enumerate(sigmas):
        c = coeff[:, radial_idx, :]
        rho = rho + (_gaussian_1d(z_ref, z_atom, float(sigma_value)) * (c[:, 0] / area)[None, :]).sum(dim=1)
        if c.shape[1] > 1:
            # Match graph_longrange.tensor_realspace_GTO_evaluation, which
            # evaluates e3nn spherical harmonics on coordinates permuted as
            # (y, z, x).  For l=1, Cartesian z is local component 1,
            # i.e. full coefficient index 2 after the l=0 monopole.
            if axis == 2:
                dipole_component = c[:, 2]
            else:
                raise NotImplementedError("potential_1d_profile loss currently supports axis=2")
            rho = rho + (
                _gaussian_d1_1d(z_ref, z_atom, float(sigma_value))
                * ((-dipole_component / area)[None, :])
            ).sum(dim=1)
        if c.shape[1] >= 9:
            rho = rho + (
                _gaussian_d2_1d(z_ref, z_atom, float(sigma_value))
                * (((c[:, 6] / 3.0) / area)[None, :])
            ).sum(dim=1)
    return rho


def _solve_potential_1d_from_raw_profile(
    raw_profile_e: torch.Tensor,
    height: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    nz = raw_profile_e.numel()
    freq = torch.fft.fftfreq(nz, d=1.0 / float(nz), device=raw_profile_e.device).to(raw_profile_e.dtype)
    gabs = torch.abs(2.0 * math.pi * freq / height.to(dtype=raw_profile_e.dtype))
    coeff = torch.fft.fft(raw_profile_e) / float(nz)
    phi_coeff = torch.zeros_like(coeff)
    mask = gabs > raw_profile_e.new_tensor(1.0e-14)
    phi_coeff[mask] = coeff[mask] * raw_profile_e.new_tensor(4.0 * math.pi * FELECT_EV_ANG) / (
        torch.square(gabs[mask]) * volume.to(dtype=raw_profile_e.dtype)
    )
    phi = torch.real(torch.fft.ifft(phi_coeff * float(nz)))
    return phi - torch.mean(phi)


def _apply_cdipol_correction_torch(
    phi: torch.Tensor,
    height: torch.Tensor,
    volume: torch.Tensor,
    mu_field_eang: torch.Tensor,
    total_charge_net_phys_e: torch.Tensor,
    indmin: int,
) -> torch.Tensor:
    nz = int(phi.numel())
    nouth = nz // 2
    e_compensate = -4.0 * math.pi * FELECT_EV_ANG * mu_field_eang.to(dtype=phi.dtype) / volume.to(dtype=phi.dtype)
    deltaq_vasp = -total_charge_net_phys_e.to(dtype=phi.dtype)
    quad_field = -(2.0 * math.pi) * deltaq_vasp * phi.new_tensor(FELECT_EV_ANG) / volume.to(dtype=phi.dtype) / 3.0
    dipfac = -e_compensate * height.to(dtype=phi.dtype) / float(nz)
    quadfac = quad_field * torch.square(height.to(dtype=phi.dtype) / float(nz))
    indices = torch.arange(1, nz + 1, device=phi.device, dtype=torch.long)
    ii = torch.remainder(indices - int(indmin) + nz, nz) - nouth
    ii_f = ii.to(dtype=phi.dtype)
    xx = torch.abs(torch.abs(ii_f) - float(nouth))
    cutoff = torch.where(
        xx > phi.new_tensor(4.0),
        torch.ones_like(xx),
        torch.abs(torch.sin(phi.new_tensor(math.pi / 8.0) * xx)),
    )
    return phi + dipfac * ii_f * cutoff + quadfac * torch.square(ii_f) * torch.square(cutoff)


def _resample_profile_to_zref(
    profile_src: torch.Tensor, height: torch.Tensor, z_ref: torch.Tensor
) -> torch.Tensor:
    """Periodic linear resampling of a uniform-grid profile [nsrc] (sampled at
    z_j = j*H/nsrc) onto arbitrary z_ref points (mod H)."""
    nsrc = int(profile_src.shape[0])
    u = torch.remainder(z_ref.to(profile_src.dtype) / height.to(profile_src.dtype), 1.0) * nsrc
    i0 = torch.floor(u).to(torch.long) % nsrc
    i1 = (i0 + 1) % nsrc
    w = (u - torch.floor(u)).to(profile_src.dtype)
    return profile_src[i0] * (1.0 - w) + profile_src[i1] * w


def potential_1d_profile_residuals(
    ref: Batch,
    pred: TensorDict,
    potential_targets: dict[int, dict[str, np.ndarray]],
    density_smearing_width,
    axis: int = 2,
    solvent_sigma_g: float = 0.85,
    align: str = "mean",
    use_solvent_profile: bool = False,
) -> Optional[torch.Tensor]:
    density_coefficients = pred.get("charge_density_radial_coefficients")
    if density_coefficients is None:
        raise KeyError(
            "potential_1d_profile loss requires PolarMACE charge_density_radial_coefficients"
        )
    if (
        _batch_get(ref, "sample_id") is None
        or pred.get("solv_center") is None
    ):
        return None

    num_graphs = int(ref.ptr.numel() - 1)
    cells = _cells_from_batch(ref, num_graphs)
    total_charge_ref = _batch_get(ref, "total_charge")
    if total_charge_ref is None:
        return None
    residuals = []
    align = str(align).lower()
    for graph_idx in range(num_graphs):
        sample_id = int(_batch_get(ref, "sample_id").view(-1)[graph_idx].detach().cpu().item())
        if sample_id not in potential_targets:
            raise KeyError(f"sample_id {sample_id} is missing from potential_1d_profile_file")
        target = potential_targets[sample_id]
        z_ref = ref["positions"].new_tensor(target["z_A"])
        phi_ref = ref["positions"].new_tensor(target["phi_eV"])
        raw_neutral = ref["positions"].new_tensor(target["raw_neutral_e"])
        raw_ion = ref["positions"].new_tensor(target["raw_ion_potcar_e"])
        align_mask = ref["positions"].new_tensor(target["align_mask"])
        indmin = int(np.asarray(target["cdipol_indmin"]).item())
        start = int(ref.ptr[graph_idx].item())
        stop = int(ref.ptr[graph_idx + 1].item())
        cell = cells[graph_idx].to(ref["positions"].dtype)
        area = _cell_area_for_axis(cell, 2).to(dtype=z_ref.dtype)
        height = _height_from_uniform_z_grid(z_ref).to(dtype=z_ref.dtype)
        volume = area * height
        rho_residual = _residual_plane_density_from_radial_coefficients(
            z_ref=z_ref,
            density_coefficients=density_coefficients[start:stop],
            positions=ref["positions"][start:stop],
            cell=cell,
            sigma=density_smearing_width,
            axis=axis,
        )
        total_charge = total_charge_ref.view(-1)[graph_idx]
        solvent_center = pred["solv_center"].view(-1)[graph_idx]
        rho_solvent = _solvent_profile_1d_torch(
            z_ref=z_ref,
            area=area,
            height=height,
            total_charge=total_charge,
            solvent_center=solvent_center,
            sigma=solvent_sigma_g,
        )
        raw_residual = rho_residual * volume.to(dtype=z_ref.dtype)
        raw_solvent = -rho_solvent * volume.to(dtype=z_ref.dtype)
        # In PB mode the actual solvent charge is the solved profile
        # (ion + bound polarization), not a gaussian at solv_center. When
        # enabled and available, score the real profile: resample it onto
        # z_ref and rescale to the same net charge as the gaussian layer
        # (robust to sign/unit conventions; the gaussian carries the correct
        # q_sol). Falls back to the gaussian for planar graphs / when off.
        if use_solvent_profile:
            sp = pred.get("solvent_profile_features")
            if sp is not None:
                prof_g = sp.view(sp.shape[0], -1)[graph_idx].to(dtype=z_ref.dtype)
                if float(torch.sum(torch.abs(prof_g))) > 1.0e-8:
                    raw_prof = _resample_profile_to_zref(prof_g, height, z_ref) * volume.to(dtype=z_ref.dtype)
                    s_prof = torch.sum(raw_prof)
                    s_gauss = torch.sum(raw_solvent)
                    if float(torch.abs(s_prof)) > 1.0e-8:
                        raw_solvent = raw_prof * (s_gauss / s_prof)
        raw_total = raw_neutral - raw_residual + raw_ion + raw_solvent
        phi_pred = _solve_potential_1d_from_raw_profile(
            raw_profile_e=raw_total,
            height=height,
            volume=volume,
        )
        dipole = pred.get("dipole")
        if dipole is None:
            raw_mu = torch.sum(raw_total * (z_ref - 0.5 * height.to(dtype=z_ref.dtype))) / float(raw_total.numel())
        else:
            raw_mu = -dipole.view(-1, 3)[graph_idx, axis].to(dtype=z_ref.dtype)
        phi_pred = _apply_cdipol_correction_torch(
            phi=phi_pred,
            height=height,
            volume=volume,
            mu_field_eang=raw_mu,
            total_charge_net_phys_e=-torch.sum(raw_total) / float(raw_total.numel()),
            indmin=indmin,
        )
        if align == "mean":
            phi_cmp = phi_pred - torch.mean(phi_pred)
            phi_ref_cmp = phi_ref - torch.mean(phi_ref)
        elif align == "upper":
            denom = torch.clamp(torch.sum(align_mask), min=align_mask.new_tensor(1.0))
            phi_cmp = phi_pred - torch.sum(phi_pred * align_mask) / denom
            phi_ref_cmp = phi_ref - torch.sum(phi_ref * align_mask) / denom
        elif align == "none":
            phi_cmp = phi_pred
            phi_ref_cmp = phi_ref
        else:
            raise ValueError(f"Unsupported potential_1d_profile_align={align!r}")
        residuals.append(phi_cmp - phi_ref_cmp)
    if not residuals:
        return None
    return torch.cat(residuals, dim=0)


def reference_solvent_layer_mean_from_partition(
    ref: Batch,
    axis: int = 2,
    potential_sign: float = 1.0,
) -> Optional[torch.Tensor]:
    ref_potential = _batch_get(ref, "potential")
    ref_total_charge = _batch_get(ref, "total_charge")
    ref_cell = _batch_get(ref, "cell")
    ref_charges = _batch_get(ref, "charges")
    ref_atomic_dipole = _batch_get(ref, "atomic_dipole")
    ref_positions = _batch_get(ref, "positions")
    if (
        ref_potential is None
        or ref_total_charge is None
        or ref_cell is None
        or ref_charges is None
        or ref_atomic_dipole is None
        or ref_positions is None
    ):
        return None
    if axis not in (0, 1, 2):
        raise ValueError("potential_axis must be 0/1/2")

    num_graphs = int(ref.ptr.numel() - 1)
    cells = _cells_from_batch(ref, num_graphs)
    ref_potential = ref_potential.view(-1)
    ref_total_charge = ref_total_charge.view(-1)
    if ref_potential.shape[0] != num_graphs or ref_total_charge.shape[0] != num_graphs:
        return None

    charges = ref_charges.view(-1).to(dtype=ref_positions.dtype)
    atomic_dipole = ref_atomic_dipole.view(-1, 3).to(dtype=ref_positions.dtype)
    positions = ref_positions.to(dtype=ref_positions.dtype)
    explicit_mu = []
    for graph_idx in range(num_graphs):
        start = int(ref.ptr[graph_idx].item())
        stop = int(ref.ptr[graph_idx + 1].item())
        mu_g = torch.sum(charges[start:stop] * positions[start:stop, axis])
        mu_g = mu_g + torch.sum(atomic_dipole[start:stop, axis])
        explicit_mu.append(mu_g)
    explicit_mu_axis = torch.stack(explicit_mu, dim=0).to(dtype=ref_potential.dtype)
    areas = torch.stack(
        [_cell_area_for_axis(cells[i], axis) for i in range(num_graphs)], dim=0
    ).to(dtype=ref_potential.dtype)
    scale = ref_potential.new_tensor(POTENTIAL_FROM_DIPOLE_SCALE_EV_PER_EANG_A2)
    sign = ref_potential.new_tensor(float(potential_sign))
    total_mu_ref = ref_potential * areas / (sign * scale)
    solvent_charge = -ref_total_charge.to(dtype=ref_potential.dtype)
    valid = torch.abs(solvent_charge) > ref_potential.new_tensor(1.0e-12)
    if not bool(torch.all(valid).detach().cpu().item()):
        return None
    return (total_mu_ref - explicit_mu_axis) / solvent_charge


def reference_solvent_layer_mean_from_density(
    ref: Batch,
    potential_targets: dict[int, dict[str, np.ndarray]],
    density_targets,
    axis: int = 2,
    potential_sign: float = 1.0,
) -> Optional[torch.Tensor]:
    ref_potential = _batch_get(ref, "potential")
    ref_total_charge = _batch_get(ref, "total_charge")
    ref_cell = _batch_get(ref, "cell")
    sample_ids = _batch_get(ref, "sample_id")
    if (
        ref_potential is None
        or ref_total_charge is None
        or ref_cell is None
        or sample_ids is None
        or not potential_targets
        or density_targets is None
    ):
        return None
    if axis != 2:
        raise NotImplementedError("density-derived solvent center currently supports axis=2")
    if not hasattr(density_targets, "plane_average"):
        raise ValueError(
            "density-derived solvent center requires a full-grid density_3d_file manifest; "
            "random-point TSV density targets do not define a total residual dipole"
        )

    num_graphs = int(ref.ptr.numel() - 1)
    cells = _cells_from_batch(ref, num_graphs)
    ref_potential = ref_potential.view(-1)
    ref_total_charge = ref_total_charge.view(-1)
    sample_ids = sample_ids.view(-1)
    if ref_potential.shape[0] != num_graphs or ref_total_charge.shape[0] != num_graphs:
        return None

    targets = []
    scale = ref_potential.new_tensor(POTENTIAL_FROM_DIPOLE_SCALE_EV_PER_EANG_A2)
    sign = ref_potential.new_tensor(float(potential_sign))
    for graph_idx in range(num_graphs):
        sample_id = int(sample_ids[graph_idx].detach().cpu().item())
        if sample_id not in potential_targets:
            raise KeyError(f"sample_id {sample_id} is missing from potential_1d_profile_file")
        if sample_id not in density_targets:
            raise KeyError(f"sample_id {sample_id} is missing from density_3d_file")
        target = potential_targets[sample_id]
        z_ref = ref["positions"].new_tensor(target["z_A"])
        raw_neutral = ref["positions"].new_tensor(target["raw_neutral_e"])
        raw_ion = ref["positions"].new_tensor(target["raw_ion_potcar_e"])
        rho_residual = density_targets.plane_average(
            sample_id=sample_id,
            dtype=z_ref.dtype,
            device=z_ref.device,
        )
        if rho_residual.shape != z_ref.shape:
            raise ValueError(
                f"sample_id {sample_id}: density_3d nz={rho_residual.numel()} "
                f"does not match potential_1d nz={z_ref.numel()}"
            )
        cell = cells[graph_idx].to(ref["positions"].dtype)
        area = _cell_area_for_axis(cell, axis).to(dtype=z_ref.dtype)
        height = _height_from_uniform_z_grid(z_ref).to(dtype=z_ref.dtype)
        volume = area * height
        raw_residual = rho_residual * volume.to(dtype=z_ref.dtype)
        raw_explicit = raw_neutral - raw_residual + raw_ion
        explicit_mu = -torch.mean(raw_explicit * z_ref)
        total_mu_ref = ref_potential[graph_idx].to(dtype=z_ref.dtype) * area / (
            sign.to(dtype=z_ref.dtype) * scale.to(dtype=z_ref.dtype)
        )
        solvent_charge = -ref_total_charge[graph_idx].to(dtype=z_ref.dtype)
        if bool((torch.abs(solvent_charge) <= z_ref.new_tensor(1.0e-12)).detach().cpu().item()):
            return None
        targets.append((total_mu_ref - explicit_mu) / solvent_charge)
    if not targets:
        return None
    return torch.stack(targets, dim=0).to(dtype=ref_potential.dtype)


def _predicted_solvent_layer_mean(
    ref: Batch,
    pred: TensorDict,
    axis: int,
    sigma_g: float,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    pred_center = pred.get("solv_center")
    ref_cell = _batch_get(ref, "cell")
    if pred_center is None or ref_cell is None:
        return None, None
    if axis not in (0, 1, 2):
        raise ValueError("potential_axis must be 0/1/2")

    centers = pred_center.view(-1)
    n_graphs = int(centers.shape[0])
    cells = _cells_from_batch(ref, n_graphs)
    if int(cells.shape[0]) != n_graphs:
        return None, None
    l_axis = torch.stack([_axis_box_length(cells[i], axis) for i in range(n_graphs)]).to(
        centers.dtype
    )
    pred_layer_mean = _truncated_gaussian_mean(
        center=centers,
        sigma=sigma_g,
        lower=torch.zeros_like(centers),
        upper=l_axis,
    )
    return pred_layer_mean, centers


def solvent_layer_mean_residuals_from_partition(
    ref: Batch,
    pred: TensorDict,
    axis: int = 2,
    potential_sign: float = 1.0,
    sigma_g: float = 0.85,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    pred_layer_mean, centers = _predicted_solvent_layer_mean(
        ref=ref,
        pred=pred,
        axis=axis,
        sigma_g=sigma_g,
    )
    if pred_layer_mean is None or centers is None:
        return None, None, None

    target_layer_mean = reference_solvent_layer_mean_from_partition(
        ref=ref,
        axis=axis,
        potential_sign=potential_sign,
    )
    if target_layer_mean is None:
        return None, pred_layer_mean, None
    target_layer_mean = target_layer_mean.to(device=centers.device, dtype=centers.dtype)
    return target_layer_mean - pred_layer_mean, pred_layer_mean, target_layer_mean


def solvent_layer_mean_residuals_from_density(
    ref: Batch,
    pred: TensorDict,
    potential_targets: dict[int, dict[str, np.ndarray]],
    density_targets,
    axis: int = 2,
    potential_sign: float = 1.0,
    sigma_g: float = 0.85,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    pred_layer_mean, centers = _predicted_solvent_layer_mean(
        ref=ref,
        pred=pred,
        axis=axis,
        sigma_g=sigma_g,
    )
    if pred_layer_mean is None or centers is None:
        return None, None, None

    target_layer_mean = reference_solvent_layer_mean_from_density(
        ref=ref,
        potential_targets=potential_targets,
        density_targets=density_targets,
        axis=axis,
        potential_sign=potential_sign,
    )
    if target_layer_mean is None:
        return None, pred_layer_mean, None
    target_layer_mean = target_layer_mean.to(device=centers.device, dtype=centers.dtype)
    return target_layer_mean - pred_layer_mean, pred_layer_mean, target_layer_mean


def solvent_layer_mean_residuals(
    ref: Batch,
    pred: TensorDict,
    potential_targets: dict[int, dict[str, np.ndarray]],
    density_targets,
    use_density_center: bool,
    use_partition_center: bool,
    axis: int = 2,
    potential_sign: float = 1.0,
    sigma_g: float = 0.85,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if use_density_center and use_partition_center:
        raise ValueError(
            "solvent_center_weight is ambiguous: density/profile and partition center targets "
            "are both enabled. Use one method at a time."
        )
    if use_density_center:
        return solvent_layer_mean_residuals_from_density(
            ref=ref,
            pred=pred,
            potential_targets=potential_targets,
            density_targets=density_targets,
            axis=axis,
            potential_sign=potential_sign,
            sigma_g=sigma_g,
        )
    if use_partition_center:
        return solvent_layer_mean_residuals_from_partition(
            ref=ref,
            pred=pred,
            axis=axis,
            potential_sign=potential_sign,
            sigma_g=sigma_g,
        )
    return None, None, None


def weighted_mean_squared_error_solvent_layer_mean_from_partition(
    ref: Batch,
    pred: TensorDict,
    axis: int = 2,
    potential_sign: float = 1.0,
    sigma_g: float = 0.85,
    ddp: Optional[bool] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    residuals, pred_layer_mean, _ = solvent_layer_mean_residuals_from_partition(
        ref=ref,
        pred=pred,
        axis=axis,
        potential_sign=potential_sign,
        sigma_g=sigma_g,
    )
    if residuals is None:
        return None, pred_layer_mean
    raw_loss = ref.weight * torch.square(residuals)
    return reduce_loss(raw_loss, ddp), pred_layer_mean


def weighted_mean_squared_error_solvent_layer_mean(
    ref: Batch,
    pred: TensorDict,
    potential_targets: dict[int, dict[str, np.ndarray]],
    density_targets,
    use_density_center: bool,
    use_partition_center: bool,
    axis: int = 2,
    potential_sign: float = 1.0,
    sigma_g: float = 0.85,
    ddp: Optional[bool] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    residuals, pred_layer_mean, _ = solvent_layer_mean_residuals(
        ref=ref,
        pred=pred,
        potential_targets=potential_targets,
        density_targets=density_targets,
        use_density_center=use_density_center,
        use_partition_center=use_partition_center,
        axis=axis,
        potential_sign=potential_sign,
        sigma_g=sigma_g,
    )
    if residuals is None:
        return None, pred_layer_mean
    raw_loss = ref.weight * torch.square(residuals)
    return reduce_loss(raw_loss, ddp), pred_layer_mean


def mean_squared_error_potential_1d_profile(
    ref: Batch,
    pred: TensorDict,
    potential_targets: dict[int, dict[str, np.ndarray]],
    density_smearing_width,
    axis: int = 2,
    solvent_sigma_g: float = 0.85,
    align: str = "mean",
    ddp: Optional[bool] = None,
    use_solvent_profile: bool = False,
) -> Optional[torch.Tensor]:
    residuals = potential_1d_profile_residuals(
        ref=ref,
        pred=pred,
        potential_targets=potential_targets,
        density_smearing_width=density_smearing_width,
        axis=axis,
        solvent_sigma_g=solvent_sigma_g,
        align=align,
        use_solvent_profile=use_solvent_profile,
    )
    if residuals is None:
        return None
    return reduce_loss(torch.square(residuals), ddp)


# ------------------------------------------------------------------------------
# Conditional Losses for Forces
# ------------------------------------------------------------------------------


def conditional_mse_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    # Define multiplication factors for different regimes.
    factors = torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=ref["forces"].device, dtype=ref["forces"].dtype
    )
    err = ref["forces"] - pred["forces"]
    se = torch.zeros_like(err)
    norm_forces = torch.norm(ref["forces"], dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    se[c1] = torch.square(err[c1]) * factors[0]
    se[c2] = torch.square(err[c2]) * factors[1]
    se[c3] = torch.square(err[c3]) * factors[2]
    se[~(c1 | c2 | c3)] = torch.square(err[~(c1 | c2 | c3)]) * factors[3]
    raw_loss = configs_weight * configs_forces_weight * se
    return reduce_loss(raw_loss, ddp)


def conditional_huber_forces(
    ref_forces: torch.Tensor,
    pred_forces: torch.Tensor,
    huber_delta: float,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    factors = huber_delta * torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=ref_forces.device, dtype=ref_forces.dtype
    )
    norm_forces = torch.norm(ref_forces, dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    c4 = ~(c1 | c2 | c3)
    se = torch.zeros_like(pred_forces)
    se[c1] = torch.nn.functional.huber_loss(
        ref_forces[c1], pred_forces[c1], reduction="none", delta=factors[0]
    )
    se[c2] = torch.nn.functional.huber_loss(
        ref_forces[c2], pred_forces[c2], reduction="none", delta=factors[1]
    )
    se[c3] = torch.nn.functional.huber_loss(
        ref_forces[c3], pred_forces[c3], reduction="none", delta=factors[2]
    )
    se[c4] = torch.nn.functional.huber_loss(
        ref_forces[c4], pred_forces[c4], reduction="none", delta=factors[3]
    )
    return reduce_loss(se, ddp)


# ------------------------------------------------------------------------------
# Loss Modules Combining Multiple Quantities
# ------------------------------------------------------------------------------


class WeightedEnergyForcesLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        charges_weight=0.0,
        atomic_dipole_weight=0.0,
        potential_weight=0.0,
        potential_axis=2,
        potential_sign=1.0,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "charges_weight",
            torch.tensor(charges_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "atomic_dipole_weight",
            torch.tensor(atomic_dipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "potential_weight",
            torch.tensor(potential_weight, dtype=torch.get_default_dtype()),
        )
        self.potential_axis = int(potential_axis)
        self.potential_sign = float(potential_sign)

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, dtype=torch.get_default_dtype(), device=ref.weight.device)
        if self.energy_weight > 1e-12 and pred.get("energy") is not None:
            loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
            loss = loss + self.energy_weight * loss_energy
        if self.forces_weight > 1e-12 and pred.get("forces") is not None:
            loss_forces = mean_squared_error_forces(ref, pred, ddp)
            loss = loss + self.forces_weight * loss_forces
        if self.charges_weight > 1e-12:
            loss_charges = weighted_mean_squared_error_charges(ref, pred, ddp)
            if loss_charges is not None:
                loss = loss + self.charges_weight * loss_charges
        if self.atomic_dipole_weight > 1e-12:
            loss_atomic_dipole = weighted_mean_squared_error_atomic_dipole(ref, pred, ddp)
            if loss_atomic_dipole is not None:
                loss = loss + self.atomic_dipole_weight * loss_atomic_dipole
        if self.potential_weight > 1e-12:
            loss_potential, pred_potential = mean_squared_error_potential_from_dipole(
                ref,
                pred,
                axis=self.potential_axis,
                potential_sign=self.potential_sign,
                ddp=ddp,
            )
            if loss_potential is not None:
                pred["potential"] = pred_potential
                loss = loss + self.potential_weight * loss_potential
        return loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, charges_weight={self.charges_weight:.3f}, "
            f"atomic_dipole_weight={self.atomic_dipole_weight:.3f}, "
            f"potential_weight={self.potential_weight:.3f}, "
            f"potential_axis={self.potential_axis}, "
            f"potential_sign={self.potential_sign:.3f})"
        )


class WeightedForcesLoss(torch.nn.Module):
    def __init__(self, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        return self.forces_weight * loss_forces

    def __repr__(self):
        return f"{self.__class__.__name__}(forces_weight={self.forces_weight:.3f})"


class WeightedEnergyForcesStressLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_stress = weighted_mean_squared_stress(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedHuberEnergyForcesStressLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    ) -> None:
        super().__init__()
        # We store the huber_delta rather than a loss with fixed reduction.
        self.huber_delta = huber_delta
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        if ddp:
            loss_energy = torch.nn.functional.huber_loss(
                ref["energy"] / num_atoms,
                pred["energy"] / num_atoms,
                reduction="none",
                delta=self.huber_delta,
            )
            loss_energy = reduce_loss(loss_energy, ddp)
            loss_forces = torch.nn.functional.huber_loss(
                ref["forces"], pred["forces"], reduction="none", delta=self.huber_delta
            )
            loss_forces = reduce_loss(loss_forces, ddp)
            loss_stress = torch.nn.functional.huber_loss(
                ref["stress"], pred["stress"], reduction="none", delta=self.huber_delta
            )
            loss_stress = reduce_loss(loss_stress, ddp)
        else:
            loss_energy = torch.nn.functional.huber_loss(
                ref["energy"] / num_atoms,
                pred["energy"] / num_atoms,
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_forces = torch.nn.functional.huber_loss(
                ref["forces"], pred["forces"], reduction="mean", delta=self.huber_delta
            )
            loss_stress = torch.nn.functional.huber_loss(
                ref["stress"], pred["stress"], reduction="mean", delta=self.huber_delta
            )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class UniversalLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    ) -> None:
        super().__init__()
        self.huber_delta = huber_delta
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
        configs_energy_weight = ref.energy_weight
        configs_forces_weight = torch.repeat_interleave(
            ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
        ).unsqueeze(-1)
        if ddp:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="none",
                delta=self.huber_delta,
            )
            loss_energy = reduce_loss(loss_energy, ddp)
            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )
            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="none",
                delta=self.huber_delta,
            )
            loss_stress = reduce_loss(loss_stress, ddp)
        else:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )
            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="mean",
                delta=self.huber_delta,
            )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedEnergyForcesVirialsLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, virials_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "virials_weight",
            torch.tensor(virials_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_virials = weighted_mean_squared_virials(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.virials_weight * loss_virials
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, virials_weight={self.virials_weight:.3f})"
        )


class DipoleSingleLoss(torch.nn.Module):
    def __init__(
        self,
        dipole_weight=1.0,
        charges_weight=0.0,
        atomic_dipole_weight=0.0,
        potential_weight=0.0,
        potential_axis=2,
        potential_sign=1.0,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "charges_weight",
            torch.tensor(charges_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "atomic_dipole_weight",
            torch.tensor(atomic_dipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "potential_weight",
            torch.tensor(potential_weight, dtype=torch.get_default_dtype()),
        )
        self.potential_axis = int(potential_axis)
        self.potential_sign = float(potential_sign)

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss = self.dipole_weight * (
            weighted_mean_squared_error_dipole(ref, pred, ddp) * 100.0
        )
        if self.charges_weight > 1e-12:
            loss_charges = weighted_mean_squared_error_charges(ref, pred, ddp)
            if loss_charges is not None:
                loss = loss + self.charges_weight * loss_charges
        if self.atomic_dipole_weight > 1e-12:
            loss_atomic_dipole = weighted_mean_squared_error_atomic_dipole(ref, pred, ddp)
            if loss_atomic_dipole is not None:
                loss = loss + self.atomic_dipole_weight * loss_atomic_dipole
        if self.potential_weight > 1e-12:
            loss_potential, pred_potential = mean_squared_error_potential_from_dipole(
                ref,
                pred,
                axis=self.potential_axis,
                potential_sign=self.potential_sign,
                ddp=ddp,
            )
            if loss_potential is not None:
                pred["potential"] = pred_potential
                loss = loss + self.potential_weight * loss_potential
        return loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(dipole_weight={self.dipole_weight:.3f}, "
            f"charges_weight={self.charges_weight:.3f}, atomic_dipole_weight={self.atomic_dipole_weight:.3f}, "
            f"potential_weight={self.potential_weight:.3f}, "
            f"potential_axis={self.potential_axis}, potential_sign={self.potential_sign:.3f})"
        )


class DipolePolarLoss(torch.nn.Module):
    def __init__(
        self, dipole_weight=1.0, polarizability_weight=1.0
    ) -> (
        None
    ):  # dipole_mean=None,dipole_std=None,polarizability_mean=None,polarizability_std=None
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarizability_weight",
            torch.tensor(polarizability_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_dipole = weighted_mean_squared_error_dipole(
            ref, pred, ddp
        )  # ,self.dipole_mean,self.dipole_std) #* 100.0  # scale adjustment

        loss_polarizability = weighted_mean_squared_error_polarizability(
            ref, pred, ddp
        )  # ,self.polarizability_mean,self.polarizability_std) #* 100.0  # scale adjustment
        return (
            self.dipole_weight * loss_dipole
            + self.polarizability_weight * loss_polarizability
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dipole_weight={self.dipole_weight:.3f}, polarizability_weight={self.polarizability_weight:.3f})"
        )


class WeightedEnergyForcesDipoleLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        dipole_weight=1.0,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_dipole = weighted_mean_squared_error_dipole(ref, pred, ddp) * 100.0
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.dipole_weight * loss_dipole
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"dipole_weight={self.dipole_weight:.3f})"
        )


class WeightedEnergyForcesElectrostaticsLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        dipole_weight=1.0,
        charges_weight=0.0,
        atomic_dipole_weight=0.0,
        potential_weight=0.0,
        fermi_level_weight=0.0,
        fermi_residual_reg_weight=0.1,
        density_3d_weight=0.0,
        density_3d_file=None,
        density_3d_sigma=0.5,
        density_3d_samples=0,
        density_3d_seed=12345,
        potential_1d_profile_weight=0.0,
        potential_1d_profile_file=None,
        potential_1d_profile_align="mean",
        potential_1d_profile_use_solvent_profile=False,
        solvent_center_weight=0.0,
        potential_axis=2,
        potential_sign=1.0,
        solvent_sigma_g=0.85,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        # Keep the constructor argument for backward compatibility, but the
        # total-dipole loss is intentionally disabled in this training path.
        self.register_buffer(
            "dipole_weight",
            torch.tensor(0.0, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "charges_weight",
            torch.tensor(charges_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "atomic_dipole_weight",
            torch.tensor(atomic_dipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "potential_weight",
            torch.tensor(potential_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "fermi_level_weight",
            torch.tensor(fermi_level_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "fermi_residual_reg_weight",
            torch.tensor(fermi_residual_reg_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "density_3d_weight",
            torch.tensor(density_3d_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "potential_1d_profile_weight",
            torch.tensor(potential_1d_profile_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "solvent_center_weight",
            torch.tensor(solvent_center_weight, dtype=torch.get_default_dtype()),
        )
        self.density_3d_file = density_3d_file
        self.density_3d_targets = (
            _load_density_3d_targets(density_3d_file)
            if density_3d_file is not None
            and (
                float(density_3d_weight) > 1.0e-12
                or (
                    float(solvent_center_weight) > 1.0e-12
                    and potential_1d_profile_file is not None
                )
            )
            else {}
        )
        self.density_3d_sigma = (
            _parse_density_sigmas(density_3d_sigma)
            if isinstance(density_3d_sigma, (list, tuple, str))
            else float(density_3d_sigma)
        )
        self.density_3d_samples = int(density_3d_samples)
        density_seed = int(density_3d_seed)
        if dist.is_available() and dist.is_initialized():
            density_seed += int(dist.get_rank())
        self.density_3d_seed = density_seed
        self.density_3d_rng = random.Random(density_seed)
        self.potential_1d_profile_file = potential_1d_profile_file
        self.potential_1d_profile_targets = (
            _load_potential_1d_profiles_npz(potential_1d_profile_file)
            if potential_1d_profile_file is not None
            and (
                float(potential_1d_profile_weight) > 1.0e-12
                or (
                    float(solvent_center_weight) > 1.0e-12
                    and density_3d_file is not None
                )
            )
            else {}
        )
        self.potential_1d_profile_align = str(potential_1d_profile_align)
        self.potential_1d_profile_use_solvent_profile = bool(
            potential_1d_profile_use_solvent_profile
        )
        self.potential_axis = int(potential_axis)
        self.potential_sign = float(potential_sign)
        self.solvent_sigma_g = float(solvent_sigma_g)
        self.use_density_center_target = bool(
            float(solvent_center_weight) > 1.0e-12
            and density_3d_file is not None
            and potential_1d_profile_file is not None
        )
        self.use_partition_center_target = bool(
            float(solvent_center_weight) > 1.0e-12
            and not self.use_density_center_target
        )
        if self.use_density_center_target and (
            float(charges_weight) > 1.0e-12 or float(atomic_dipole_weight) > 1.0e-12
        ):
            raise ValueError(
                "Ambiguous electrostatic training setup: density-derived solvent center "
                "target is enabled by density_3d_file + potential_1d_profile_file, but "
                "partition charge/atomic-dipole losses are also enabled."
            )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, dtype=torch.get_default_dtype(), device=ref.weight.device)
        if self.energy_weight > 1e-12 and pred.get("energy") is not None:
            loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
            loss = loss + self.energy_weight * loss_energy
        if self.forces_weight > 1e-12 and pred.get("forces") is not None:
            loss_forces = mean_squared_error_forces(ref, pred, ddp)
            loss = loss + self.forces_weight * loss_forces
        if self.charges_weight > 1e-12:
            loss_charges = weighted_mean_squared_error_charges(ref, pred, ddp)
            if loss_charges is not None:
                loss = loss + self.charges_weight * loss_charges
        if self.atomic_dipole_weight > 1e-12:
            loss_atomic_dipole = weighted_mean_squared_error_atomic_dipole(ref, pred, ddp)
            if loss_atomic_dipole is not None:
                loss = loss + self.atomic_dipole_weight * loss_atomic_dipole
        if self.potential_weight > 1e-12:
            loss_potential, pred_potential = mean_squared_error_potential_from_dipole(
                ref,
                pred,
                axis=self.potential_axis,
                potential_sign=self.potential_sign,
                ddp=ddp,
            )
            if loss_potential is not None:
                pred["potential"] = pred_potential
                loss = loss + self.potential_weight * loss_potential
        if self.fermi_level_weight > 1e-12:
            loss_fermi = mean_squared_error_fermi_level(ref, pred, ddp)
            if loss_fermi is not None:
                loss = loss + self.fermi_level_weight * loss_fermi
        if self.fermi_level_weight > 1e-12 and self.fermi_residual_reg_weight > 1e-12:
            loss_fermi_residual = mean_squared_error_fermi_residual(pred, ddp)
            if loss_fermi_residual is not None:
                loss = loss + self.fermi_residual_reg_weight * loss_fermi_residual
        if self.density_3d_weight > 1e-12:
            loss_density_3d = mean_squared_error_density_3d(
                ref=ref,
                pred=pred,
                density_targets=self.density_3d_targets,
                density_smearing_width=self.density_3d_sigma,
                samples_per_graph=self.density_3d_samples,
                rng=self.density_3d_rng,
                ddp=ddp,
            )
            if loss_density_3d is not None:
                loss = loss + self.density_3d_weight * loss_density_3d
        if self.potential_1d_profile_weight > 1e-12:
            loss_potential_1d_profile = mean_squared_error_potential_1d_profile(
                ref=ref,
                pred=pred,
                potential_targets=self.potential_1d_profile_targets,
                density_smearing_width=self.density_3d_sigma,
                axis=self.potential_axis,
                solvent_sigma_g=self.solvent_sigma_g,
                align=self.potential_1d_profile_align,
                ddp=ddp,
                use_solvent_profile=self.potential_1d_profile_use_solvent_profile,
            )
            if loss_potential_1d_profile is not None:
                loss = loss + self.potential_1d_profile_weight * loss_potential_1d_profile
        if self.solvent_center_weight > 1e-12:
            loss_solvent_layer_mean, pred_solvent_layer_mean = (
                weighted_mean_squared_error_solvent_layer_mean(
                    ref=ref,
                    pred=pred,
                    potential_targets=self.potential_1d_profile_targets,
                    density_targets=self.density_3d_targets,
                    use_density_center=self.use_density_center_target,
                    use_partition_center=self.use_partition_center_target,
                    axis=self.potential_axis,
                    potential_sign=self.potential_sign,
                    sigma_g=self.solvent_sigma_g,
                    ddp=ddp,
                )
            )
            if loss_solvent_layer_mean is not None:
                pred["solvent_layer_mean"] = pred_solvent_layer_mean
                loss = loss + self.solvent_center_weight * loss_solvent_layer_mean
        return loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"charges_weight={self.charges_weight:.3f}, atomic_dipole_weight={self.atomic_dipole_weight:.3f}, "
            f"potential_weight={self.potential_weight:.3f}, "
            f"fermi_level_weight={self.fermi_level_weight:.3f}, "
            f"fermi_residual_reg_weight={self.fermi_residual_reg_weight:.3f}, "
            f"density_3d_weight={self.density_3d_weight:.3f}, "
            f"density_3d_samples={self.density_3d_samples}, "
            f"potential_1d_profile_weight={self.potential_1d_profile_weight:.3f}, "
            f"solvent_center_weight={self.solvent_center_weight:.3f}, "
            f"potential_axis={self.potential_axis}, potential_sign={self.potential_sign:.3f})"
        )


class WeightedEnergyForcesL1L2Loss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_absolute_error_energy(ref, pred, ddp)
        loss_forces = mean_normed_error_forces(ref, pred, ddp)
        return self.energy_weight * loss_energy + self.forces_weight * loss_forces

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )
