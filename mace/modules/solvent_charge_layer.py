import math
from typing import Optional

import torch


POTENTIAL_FROM_DIPOLE_SCALE_EV_PER_EANG_A2 = (
    4.0 * math.pi * 27.211386245988 / 1.8897261258369282
)


def _gaussian_1d(
    z_grid: torch.Tensor, centers: torch.Tensor, sigma: float
) -> torch.Tensor:
    sigma_t = z_grid.new_tensor(float(max(sigma, 1.0e-12)))
    dz = z_grid[:, None] - centers[None, :]
    return torch.exp(-0.5 * torch.square(dz / sigma_t)) / (
        math.sqrt(2.0 * math.pi) * sigma_t
    )


def _gaussian_d1_1d(
    z_grid: torch.Tensor, centers: torch.Tensor, sigma: float
) -> torch.Tensor:
    g = _gaussian_1d(z_grid, centers, sigma)
    sigma_t = z_grid.new_tensor(float(max(sigma, 1.0e-12)))
    dz = z_grid[:, None] - centers[None, :]
    return -(dz / (sigma_t * sigma_t)) * g


def _gaussian_d2_1d(
    z_grid: torch.Tensor, centers: torch.Tensor, sigma: float
) -> torch.Tensor:
    g = _gaussian_1d(z_grid, centers, sigma)
    sigma_t = z_grid.new_tensor(float(max(sigma, 1.0e-12)))
    dz = z_grid[:, None] - centers[None, :]
    sigma2 = sigma_t * sigma_t
    return ((dz * dz) / (sigma2 * sigma2) - 1.0 / sigma2) * g


def _residual_plane_density_from_radial_coefficients(
    z_grid: torch.Tensor,
    radial_coefficients: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    atomic_density_sigmas: list[float],
    axis: int,
) -> torch.Tensor:
    if axis != 2:
        raise NotImplementedError("baseline center density currently supports axis=2")
    area = _cell_cross_section_area(cell, axis).to(dtype=z_grid.dtype)
    z_atom = positions[:, axis].to(dtype=z_grid.dtype)
    coeff = radial_coefficients.to(dtype=z_grid.dtype)
    if coeff.dim() != 3:
        raise ValueError("radial_coefficients must have shape [n_atoms, n_sigmas, n_coeffs]")
    if coeff.shape[1] != len(atomic_density_sigmas):
        raise ValueError(
            f"radial coefficient count {coeff.shape[1]} does not match "
            f"number of sigmas {len(atomic_density_sigmas)}"
        )
    rho = z_grid.new_zeros(z_grid.shape)
    for i_s, sigma in enumerate(atomic_density_sigmas):
        c = coeff[:, i_s, :]
        rho = rho + (
            _gaussian_1d(z_grid, z_atom, float(sigma)) * (c[:, 0] / area)[None, :]
        ).sum(dim=1)
        if c.shape[1] > 1:
            dipole_component = c[:, 2]
            rho = rho + (
                _gaussian_d1_1d(z_grid, z_atom, float(sigma))
                * ((-dipole_component / area)[None, :])
            ).sum(dim=1)
        if c.shape[1] >= 9:
            rho = rho + (
                _gaussian_d2_1d(z_grid, z_atom, float(sigma))
                * (((c[:, 6] / 3.0) / area)[None, :])
            ).sum(dim=1)
    return rho


def _interp_crossing(
    z_grid: torch.Tensor, values_01: torch.Tensor, level: float
) -> torch.Tensor:
    level_t = values_01.new_tensor(float(level))
    below = torch.nonzero(values_01 <= level_t, as_tuple=False).view(-1)
    if below.numel() == 0:
        idx = torch.argmin(torch.abs(values_01 - level_t))
        return z_grid[idx]
    idx_hi = int(below[0].item())
    if idx_hi <= 0:
        idx = torch.argmin(torch.abs(values_01 - level_t))
        return z_grid[idx]
    idx_lo = idx_hi - 1
    v_lo = values_01[idx_lo]
    v_hi = values_01[idx_hi]
    z_lo = z_grid[idx_lo]
    z_hi = z_grid[idx_hi]
    dv = v_hi - v_lo
    if torch.abs(dv) < 1.0e-12:
        return 0.5 * (z_lo + z_hi)
    t = (level_t - v_lo) / dv
    t = torch.clamp(t, 0.0, 1.0)
    return z_lo + t * (z_hi - z_lo)


def _interp_last_descending_crossing(
    z_grid: torch.Tensor, values_01: torch.Tensor, level: float
) -> torch.Tensor:
    level_t = values_01.new_tensor(float(level))
    cond = (values_01[:-1] >= level_t) & (values_01[1:] < level_t)
    idx = torch.nonzero(cond, as_tuple=False).view(-1)
    if idx.numel() == 0:
        return _interp_crossing(z_grid, values_01, level)
    i = int(idx[-1].item())
    z_lo = z_grid[i]
    z_hi = z_grid[i + 1]
    v_lo = values_01[i]
    v_hi = values_01[i + 1]
    dv = v_hi - v_lo
    if torch.abs(dv) < 1.0e-12:
        return 0.5 * (z_lo + z_hi)
    t = (level_t - v_lo) / dv
    t = torch.clamp(t, 0.0, 1.0)
    return z_lo + t * (z_hi - z_lo)


def _truncated_gaussian_mean(
    center: torch.Tensor,
    sigma: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    sigma_t = center.new_tensor(float(max(sigma, 1.0e-12)))
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    a = (lower - center) / sigma_t
    b = (upper - center) / sigma_t
    phi_a = torch.exp(-0.5 * torch.square(a)) / math.sqrt(2.0 * math.pi)
    phi_b = torch.exp(-0.5 * torch.square(b)) / math.sqrt(2.0 * math.pi)
    cdf_a = 0.5 * (1.0 + torch.erf(a * inv_sqrt2))
    cdf_b = 0.5 * (1.0 + torch.erf(b * inv_sqrt2))
    norm = torch.clamp(cdf_b - cdf_a, min=1.0e-12)
    return center + sigma_t * (phi_a - phi_b) / norm


def _cell_cross_section_area(cell_g: torch.Tensor, axis: int) -> torch.Tensor:
    if cell_g.dim() == 1:
        if cell_g.numel() == 9:
            cell_g = cell_g.view(3, 3)
        elif cell_g.numel() == 3:
            if axis == 0:
                return torch.clamp(torch.abs(cell_g[1] * cell_g[2]), min=1.0e-12)
            if axis == 1:
                return torch.clamp(torch.abs(cell_g[0] * cell_g[2]), min=1.0e-12)
            return torch.clamp(torch.abs(cell_g[0] * cell_g[1]), min=1.0e-12)
    if axis == 0:
        v1 = cell_g[1, :]
        v2 = cell_g[2, :]
    elif axis == 1:
        v1 = cell_g[0, :]
        v2 = cell_g[2, :]
    else:
        v1 = cell_g[0, :]
        v2 = cell_g[1, :]
    return torch.clamp(
        torch.linalg.norm(torch.cross(v1, v2, dim=-1), dim=-1), min=1.0e-12
    )


def _axis_box_length(cell_g: torch.Tensor, axis: int) -> torch.Tensor:
    if cell_g.dim() == 1:
        if cell_g.numel() == 9:
            cell_g = cell_g.view(3, 3)
        elif cell_g.numel() == 3:
            return torch.clamp(torch.abs(cell_g[axis]), min=1.0e-12)
    return torch.clamp(torch.linalg.norm(cell_g[axis, :], dim=-1), min=1.0e-12)


def require_explicit_valence_electrons(
    atomic_numbers: torch.Tensor, explicit_valence: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if explicit_valence is None:
        raise ValueError(
            "atomic_valence_electrons must be specified explicitly in the config "
            "with the same order as atomic_numbers"
        )
    if explicit_valence.numel() != atomic_numbers.numel():
        raise ValueError("atomic_valence_electrons must match atomic_numbers length")
    return explicit_valence.to(
        dtype=torch.get_default_dtype(), device=atomic_numbers.device
    )


def _atomic_dipole_xyz_from_density_coefficients(
    density_coefficients: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    if density_coefficients.shape[1] > 1:
        return density_coefficients[:, 1:4][:, [2, 0, 1]]
    return torch.zeros_like(positions)


def _compute_density_crossings_from_mace_multipoles(
    density_coefficients: torch.Tensor,
    positions: torch.Tensor,
    node_valence_electrons: torch.Tensor,
    batch: torch.Tensor,
    cell: torch.Tensor,
    density_smearing_width: float,
    inward: float,
    outward: float,
    axis: int,
    n_grid: int,
    normalized_level: Optional[float] = None,
    width_delta: float = 0.2,
    density_threshold: Optional[float] = None,
    width_density_threshold: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    charges = density_coefficients[:, 0]
    atomic_dipole = _atomic_dipole_xyz_from_density_coefficients(
        density_coefficients, positions
    )
    nelec = node_valence_electrons - charges
    centers = []
    widths = []
    low_level = max(float(normalized_level or 0.0) - float(width_delta), 0.01)
    for g in range(num_graphs):
        mask = batch == g
        pos_g = positions[mask]
        if pos_g.numel() == 0:
            centers.append(charges.new_tensor(0.0))
            widths.append(charges.new_tensor(0.0))
            continue
        cell_g = cell[g] if cell.dim() == 3 else cell.view(-1, 3, 3)[g]
        area_g = _cell_cross_section_area(cell_g, axis)
        l_axis = _axis_box_length(cell_g, axis)
        z_grid = torch.linspace(
            0.0,
            float(l_axis.detach().cpu().item()),
            int(n_grid),
            dtype=positions.dtype,
            device=positions.device,
        )
        z_atom = pos_g[:, axis]
        mono = _gaussian_1d(z_grid, z_atom, density_smearing_width)
        dip = _gaussian_d1_1d(z_grid, z_atom, density_smearing_width)
        rho = (mono * (nelec[mask] / area_g)[None, :]).sum(dim=1)
        rho = rho + (dip * (-atomic_dipole[mask, axis] / area_g)[None, :]).sum(dim=1)
        z_top = torch.max(z_atom)
        z_lo = torch.clamp(z_top - inward, min=0.0)
        z_hi = torch.clamp(z_top + outward, max=l_axis)
        win = (z_grid >= z_lo) & (z_grid <= z_hi)
        if int(win.sum().item()) < 2:
            centers.append(z_top)
            widths.append(charges.new_tensor(0.0))
            continue
        z_win = z_grid[win]
        rho_win = rho[win]
        if density_threshold is None:
            rho_min = torch.min(rho_win)
            rho_max = torch.max(rho_win)
            span = torch.clamp(rho_max - rho_min, min=1.0e-12)
            rho_01 = (rho_win - rho_min) / span
            z_level = _interp_crossing(z_win, rho_01, float(normalized_level or 0.5))
            z_low = _interp_crossing(z_win, rho_01, low_level)
        else:
            z_level = _interp_last_descending_crossing(
                z_win, rho_win, float(density_threshold)
            )
            if width_density_threshold is None:
                z_low = z_level
            else:
                z_low = _interp_last_descending_crossing(
                    z_win, rho_win, float(width_density_threshold)
                )
        centers.append(z_level)
        widths.append(torch.clamp(z_low - z_level, min=0.0))
    return torch.stack(centers, dim=0), torch.stack(widths, dim=0)


def compute_ze_crossing_from_mace_multipoles(
    density_coefficients: torch.Tensor,
    positions: torch.Tensor,
    node_valence_electrons: torch.Tensor,
    batch: torch.Tensor,
    cell: torch.Tensor,
    density_smearing_width: float = 1.0,
    level: float = 0.5,
    width_delta: float = 0.2,
    inward: float = 5.0,
    outward: float = 5.0,
    axis: int = 2,
    n_grid: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _compute_density_crossings_from_mace_multipoles(
        density_coefficients=density_coefficients,
        positions=positions,
        node_valence_electrons=node_valence_electrons,
        batch=batch,
        cell=cell,
        density_smearing_width=density_smearing_width,
        inward=inward,
        outward=outward,
        axis=axis,
        n_grid=n_grid,
        normalized_level=level,
        width_delta=width_delta,
        density_threshold=None,
        width_density_threshold=None,
    )


def compute_density_threshold_crossing_from_mace_multipoles(
    density_coefficients: torch.Tensor,
    positions: torch.Tensor,
    node_valence_electrons: torch.Tensor,
    batch: torch.Tensor,
    cell: torch.Tensor,
    density_smearing_width: float = 1.0,
    density_threshold: float = 0.15,
    width_density_threshold: Optional[float] = None,
    inward: float = 5.0,
    outward: float = 5.0,
    axis: int = 2,
    n_grid: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _compute_density_crossings_from_mace_multipoles(
        density_coefficients=density_coefficients,
        positions=positions,
        node_valence_electrons=node_valence_electrons,
        batch=batch,
        cell=cell,
        density_smearing_width=density_smearing_width,
        inward=inward,
        outward=outward,
        axis=axis,
        n_grid=n_grid,
        normalized_level=None,
        width_delta=0.0,
        density_threshold=density_threshold,
        width_density_threshold=width_density_threshold,
    )


def compute_density_threshold_crossing_from_baseline_profile(
    z_values: list[torch.Tensor],
    raw_neutral_values: list[torch.Tensor],
    radial_coefficients: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    cell: torch.Tensor,
    atomic_density_sigmas: list[float],
    density_threshold: float,
    width_density_threshold: Optional[float],
    inward: float,
    outward: float,
    axis: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    if axis != 2:
        raise NotImplementedError("baseline center density currently supports axis=2")
    centers = []
    widths = []
    num_graphs = len(z_values)
    for g in range(num_graphs):
        mask = batch == g
        pos_g = positions[mask]
        if pos_g.numel() == 0:
            centers.append(positions.new_tensor(0.0))
            widths.append(positions.new_tensor(0.0))
            continue
        cell_g = cell[g] if cell.dim() == 3 else cell.view(-1, 3, 3)[g]
        area_g = _cell_cross_section_area(cell_g, axis).to(dtype=positions.dtype)
        z_grid = z_values[g].to(dtype=positions.dtype, device=positions.device)
        raw_neutral = raw_neutral_values[g].to(
            dtype=positions.dtype, device=positions.device
        )
        if z_grid.numel() < 2:
            centers.append(torch.max(pos_g[:, axis]))
            widths.append(positions.new_tensor(0.0))
            continue
        dz = torch.abs(z_grid[1] - z_grid[0])
        height = torch.clamp(dz * float(z_grid.numel()), min=1.0e-12)
        volume = area_g * height
        rho_neutral = raw_neutral / volume
        rho_residual = _residual_plane_density_from_radial_coefficients(
            z_grid=z_grid,
            radial_coefficients=radial_coefficients[mask],
            positions=pos_g,
            cell=cell_g,
            atomic_density_sigmas=atomic_density_sigmas,
            axis=axis,
        )
        rho = rho_neutral - rho_residual
        z_top = torch.max(pos_g[:, axis])
        z_lo = torch.clamp(z_top - inward, min=torch.min(z_grid))
        z_hi = torch.clamp(z_top + outward, max=torch.max(z_grid))
        win = (z_grid >= z_lo) & (z_grid <= z_hi)
        if int(win.sum().item()) < 2:
            centers.append(z_top)
            widths.append(positions.new_tensor(0.0))
            continue
        z_win = z_grid[win]
        rho_win = rho[win]
        z_level = _interp_last_descending_crossing(
            z_win, rho_win, float(density_threshold)
        )
        if width_density_threshold is None:
            z_low = z_level
        else:
            z_low = _interp_last_descending_crossing(
                z_win, rho_win, float(width_density_threshold)
            )
        centers.append(z_level)
        widths.append(torch.clamp(z_low - z_level, min=0.0))
    return torch.stack(centers, dim=0), torch.stack(widths, dim=0)


def compute_oh_structure_crossings(
    positions: torch.Tensor,
    node_attrs: torch.Tensor,
    batch: torch.Tensor,
    cell: torch.Tensor,
    atomic_numbers: torch.Tensor,
    sigma_o: float = 0.5,
    sigma_h: float = 0.7,
    level: float = 0.5,
    width_delta: float = 0.2,
    axis: int = 2,
    n_grid: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    element_index = torch.argmax(node_attrs, dim=-1)
    z_per_node = atomic_numbers[element_index]
    is_o = z_per_node == 8
    is_h = z_per_node == 1
    low_level = max(float(level) - float(width_delta), 0.01)
    z_o_all = []
    w_o_all = []
    z_h_all = []
    w_h_all = []
    for g in range(num_graphs):
        mask_g = batch == g
        pos_g = positions[mask_g]
        if pos_g.numel() == 0:
            zero = positions.new_tensor(0.0)
            z_o_all.append(zero)
            w_o_all.append(zero)
            z_h_all.append(zero)
            w_h_all.append(zero)
            continue
        cell_g = cell[g] if cell.dim() == 3 else cell.view(-1, 3, 3)[g]
        l_axis = _axis_box_length(cell_g, axis)
        z_grid = torch.linspace(
            0.0,
            float(l_axis.detach().cpu().item()),
            int(n_grid),
            dtype=positions.dtype,
            device=positions.device,
        )
        z_graph = pos_g[:, axis]

        def _cross(mask_species: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
            z_species = z_graph[mask_species[mask_g]]
            if z_species.numel() == 0:
                z_top = torch.max(z_graph)
                return z_top, positions.new_tensor(0.0)
            gprof = torch.exp(
                -0.5
                * torch.square(
                    (z_grid[:, None] - z_species[None, :])
                    / z_grid.new_tensor(float(max(sigma, 1.0e-12)))
                )
            ).sum(dim=1)
            gmax = torch.max(gprof)
            if float(gmax.detach().cpu().item()) <= 1.0e-12:
                z_top = torch.max(z_species)
                return z_top, positions.new_tensor(0.0)
            gprof = gprof / gmax
            z_level = _interp_last_descending_crossing(z_grid, gprof, level)
            z_low = _interp_last_descending_crossing(z_grid, gprof, low_level)
            return z_level, torch.clamp(z_low - z_level, min=0.0)

        z_o, w_o = _cross(is_o, sigma_o)
        z_h, w_h = _cross(is_h, sigma_h)
        z_o_all.append(z_o)
        w_o_all.append(w_o)
        z_h_all.append(z_h)
        w_h_all.append(w_h)
    return (
        torch.stack(z_o_all, dim=0),
        torch.stack(w_o_all, dim=0),
        torch.stack(z_h_all, dim=0),
        torch.stack(w_h_all, dim=0),
    )


def compute_interface_qm_summaries(
    charges: torch.Tensor,
    atomic_dipole: torch.Tensor,
    positions: torch.Tensor,
    node_attrs: torch.Tensor,
    batch: torch.Tensor,
    atomic_numbers: torch.Tensor,
    z_o_anchor: torch.Tensor,
    z_h_anchor: torch.Tensor,
    axis: int = 2,
    lambda_o: float = 0.6,
    lambda_h: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    element_index = torch.argmax(node_attrs, dim=-1)
    z_per_node = atomic_numbers[element_index]
    is_o = z_per_node == 8
    is_h = z_per_node == 1
    qsum_o_all = []
    qsum_h_all = []
    msum_o_all = []
    msum_h_all = []
    for g in range(num_graphs):
        mask_g = batch == g
        z_atom = positions[mask_g, axis]
        q_atom = charges[mask_g]
        m_atom = atomic_dipole[mask_g, axis]
        is_o_g = is_o[mask_g]
        is_h_g = is_h[mask_g]

        def _weighted_sum(
            mask_species: torch.Tensor, anchor: torch.Tensor, lam: float
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if int(mask_species.sum().item()) == 0:
                zero = charges.new_tensor(0.0)
                return zero, zero
            weights = torch.exp((z_atom[mask_species] - anchor) / float(max(lam, 1.0e-12)))
            qsum = torch.sum(weights * q_atom[mask_species])
            msum = torch.sum(weights * m_atom[mask_species])
            return qsum, msum

        qsum_o, msum_o = _weighted_sum(is_o_g, z_o_anchor[g], lambda_o)
        qsum_h, msum_h = _weighted_sum(is_h_g, z_h_anchor[g], lambda_h)
        qsum_o_all.append(qsum_o)
        qsum_h_all.append(qsum_h)
        msum_o_all.append(msum_o)
        msum_h_all.append(msum_h)
    return (
        torch.stack(qsum_o_all, dim=0),
        torch.stack(qsum_h_all, dim=0),
        torch.stack(msum_o_all, dim=0),
        torch.stack(msum_h_all, dim=0),
    )


def predict_potential_from_dipole_and_solvent_layer(
    dipole: torch.Tensor,
    total_charge: torch.Tensor,
    center: torch.Tensor,
    cell: torch.Tensor,
    sigma_g: float = 0.85,
    axis: int = 2,
    potential_sign: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cells = cell
    if cells.dim() == 2 and cells.shape[1] == 3:
        cells = cells.view(-1, 3, 3)
    dipoles = dipole.view(-1, 3)
    v1 = cells[:, 0, :] if axis != 0 else cells[:, 1, :]
    v2 = cells[:, 1, :] if axis == 2 else cells[:, 2, :]
    if axis == 0:
        v1, v2 = cells[:, 1, :], cells[:, 2, :]
    elif axis == 1:
        v1, v2 = cells[:, 0, :], cells[:, 2, :]
    else:
        v1, v2 = cells[:, 0, :], cells[:, 1, :]
    area = torch.clamp(
        torch.linalg.norm(torch.cross(v1, v2, dim=-1), dim=-1), min=1.0e-12
    )
    l_axis = torch.stack([_axis_box_length(cells[i], axis) for i in range(cells.shape[0])])
    layer_mean = _truncated_gaussian_mean(
        center=center,
        sigma=sigma_g,
        lower=torch.zeros_like(center),
        upper=l_axis.to(center.dtype),
    )
    solvent_charge = -total_charge.view(-1)
    explicit_mu = dipoles[:, axis]
    solvent_mu = solvent_charge * layer_mean
    scale = dipoles.new_tensor(POTENTIAL_FROM_DIPOLE_SCALE_EV_PER_EANG_A2)
    sign = dipoles.new_tensor(float(potential_sign))
    explicit_potential = sign * scale * explicit_mu / area
    solvent_potential = sign * scale * solvent_mu / area
    total_potential = explicit_potential + solvent_potential
    return total_potential, explicit_potential, solvent_potential
