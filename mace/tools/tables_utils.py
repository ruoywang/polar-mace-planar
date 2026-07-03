import logging
from typing import Dict, List, Optional, Tuple

import torch
from prettytable import PrettyTable

from mace.tools import evaluate


def custom_key(key):
    """
    Helper function to sort the keys of the data loader dictionary
    to ensure that the training set, and validation set
    are evaluated first
    """
    if key == "train":
        return (0, key)
    if key == "valid":
        return (1, key)
    return (2, key)


def _extra_metric_specs(table_type: str) -> List[Tuple[str, str, float, str]]:
    if "RMSE" in table_type:
        return [
            ("rmse_q", "RMSE Q / e", 1.0, "8.4f"),
            ("rmse_atomic_dipole", "RMSE Atomic Mu / eA", 1.0, "8.4f"),
            ("rmse_potential", "RMSE Potential / eV", 1.0, "8.4f"),
            ("rmse_fermi_level", "RMSE Fermi / eV", 1.0, "8.4f"),
            ("rmse_density_3d", "RMSE Density3D / e/A^3", 1.0, "8.5f"),
            ("rmse_potential_1d_profile", "RMSE Phi1D / eV", 1.0, "8.5f"),
            ("rmse_solvent_center", "RMSE Solv Center / A", 1.0, "8.4f"),
        ]
    if "MAE" in table_type:
        return [
            ("mae_q", "MAE Q / e", 1.0, "8.4f"),
            ("mae_atomic_dipole", "MAE Atomic Mu / eA", 1.0, "8.4f"),
            ("mae_potential", "MAE Potential / eV", 1.0, "8.4f"),
            ("mae_fermi_level", "MAE Fermi / eV", 1.0, "8.4f"),
            ("mae_density_3d", "MAE Density3D / e/A^3", 1.0, "8.5f"),
            ("mae_potential_1d_profile", "MAE Phi1D / eV", 1.0, "8.5f"),
            ("mae_solvent_center", "MAE Solv Center / A", 1.0, "8.4f"),
        ]
    return []


def _format_optional_metric(metrics: dict, key: str, scale: float, fmt: str) -> str:
    value = metrics.get(key)
    if value is None:
        return ""
    return format(value * scale, fmt)


def create_error_table(
    table_type: str,
    all_data_loaders: dict,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    output_args: Dict[str, bool],
    log_wandb: bool,
    device: str,
    distributed: bool = False,
    skip_heads: Optional[List[str]] = None,
) -> PrettyTable:
    if log_wandb:
        import wandb
    skip_heads = skip_heads or []
    table = PrettyTable()

    if table_type == "TotalRMSE":
        base_field_names = [
            "config_type",
            "RMSE E / meV",
            "RMSE F / meV / A",
            "relative F RMSE %",
        ]
    elif table_type == "PerAtomRMSE":
        base_field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
        ]
    elif table_type == "PerAtomRMSEstressvirials":
        base_field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            "RMSE Stress (Virials) / meV / A (A^3)",
        ]
    elif table_type == "PerAtomMAEstressvirials":
        base_field_names = [
            "config_type",
            "MAE E / meV / atom",
            "MAE F / meV / A",
            "relative F MAE %",
            "MAE Stress (Virials) / meV / A (A^3)",
        ]
    elif table_type == "TotalMAE":
        base_field_names = [
            "config_type",
            "MAE E / meV",
            "MAE F / meV / A",
            "relative F MAE %",
        ]
    elif table_type == "PerAtomMAE":
        base_field_names = [
            "config_type",
            "MAE E / meV / atom",
            "MAE F / meV / A",
            "relative F MAE %",
        ]
    elif table_type == "DipoleRMSE":
        base_field_names = [
            "config_type",
            "RMSE MU / mDebye / atom",
            "relative MU RMSE %",
        ]
    elif table_type == "DipoleMAE":
        base_field_names = [
            "config_type",
            "MAE MU / mDebye / atom",
            "relative MU MAE %",
        ]
    elif table_type == "DipolePolarRMSE":
        base_field_names = [
            "config_type",
            "RMSE MU / me A / atom",
            "relative MU RMSE %",
            "RMSE ALPHA e A^2 / V / atom",
        ]
    elif table_type == "EnergyDipoleRMSE":
        base_field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "rel F RMSE %",
            "RMSE MU / mDebye / atom",
            "rel MU RMSE %",
        ]
    else:
        base_field_names = ["config_type"]

    extra_specs = _extra_metric_specs(table_type)
    table.field_names = base_field_names + [label for _, label, _, _ in extra_specs]

    for name in sorted(all_data_loaders, key=custom_key):
        if any(skip_head in name for skip_head in skip_heads):
            logging.info(f"Skipping evaluation of {name} (in skip_heads list)")
            continue
        data_loader = all_data_loaders[name]
        logging.info(f"Evaluating {name} ...")
        _, metrics = evaluate(
            model,
            loss_fn=loss_fn,
            data_loader=data_loader,
            output_args=output_args,
            device=device,
        )
        if distributed:
            torch.distributed.barrier()

        del data_loader
        torch.cuda.empty_cache()
        if log_wandb:
            wandb_log_dict = {
                name + "_final_rmse_e_per_atom": metrics.get("rmse_e_per_atom", 0.0) * 1e3,
                name + "_final_rmse_f": metrics.get("rmse_f", 0.0) * 1e3,
                name + "_final_rel_rmse_f": metrics.get("rel_rmse_f", 0.0),
            }
            for key, _, scale, _ in extra_specs:
                if metrics.get(key) is not None:
                    wandb_log_dict[f"{name}_final_{key}"] = metrics[key] * scale
            wandb.log(wandb_log_dict)

        row = None
        if table_type == "TotalRMSE":
            row = [
                name,
                f"{metrics['rmse_e'] * 1000:8.1f}",
                f"{metrics['rmse_f'] * 1000:8.1f}",
                f"{metrics['rel_rmse_f']:8.2f}",
            ]
        elif table_type == "PerAtomRMSE":
            row = [
                name,
                f"{metrics['rmse_e_per_atom'] * 1000:8.1f}",
                f"{metrics['rmse_f'] * 1000:8.1f}",
                f"{metrics['rel_rmse_f']:8.2f}",
            ]
        elif table_type == "PerAtomRMSEstressvirials" and metrics["rmse_stress"] is not None:
            row = [
                name,
                f"{metrics['rmse_e_per_atom'] * 1000:8.1f}",
                f"{metrics['rmse_f'] * 1000:8.1f}",
                f"{metrics['rel_rmse_f']:8.2f}",
                f"{metrics['rmse_stress'] * 1000:8.1f}",
            ]
        elif table_type == "PerAtomRMSEstressvirials" and metrics["rmse_virials"] is not None:
            row = [
                name,
                f"{metrics['rmse_e_per_atom'] * 1000:8.1f}",
                f"{metrics['rmse_f'] * 1000:8.1f}",
                f"{metrics['rel_rmse_f']:8.2f}",
                f"{metrics['rmse_virials'] * 1000:8.1f}",
            ]
        elif table_type == "PerAtomMAEstressvirials" and metrics["mae_stress"] is not None:
            row = [
                name,
                f"{metrics['mae_e_per_atom'] * 1000:8.1f}",
                f"{metrics['mae_f'] * 1000:8.1f}",
                f"{metrics['rel_mae_f']:8.2f}",
                f"{metrics['mae_stress'] * 1000:8.1f}",
            ]
        elif table_type == "PerAtomMAEstressvirials" and metrics["mae_virials"] is not None:
            row = [
                name,
                f"{metrics['mae_e_per_atom'] * 1000:8.1f}",
                f"{metrics['mae_f'] * 1000:8.1f}",
                f"{metrics['rel_mae_f']:8.2f}",
                f"{metrics['mae_virials'] * 1000:8.1f}",
            ]
        elif table_type == "TotalMAE":
            row = [
                name,
                f"{metrics['mae_e'] * 1000:8.1f}",
                f"{metrics['mae_f'] * 1000:8.1f}",
                f"{metrics['rel_mae_f']:8.2f}",
            ]
        elif table_type == "PerAtomMAE":
            row = [
                name,
                f"{metrics['mae_e_per_atom'] * 1000:8.1f}",
                f"{metrics['mae_f'] * 1000:8.1f}",
                f"{metrics['rel_mae_f']:8.2f}",
            ]
        elif table_type == "DipoleRMSE":
            row = [
                name,
                f"{metrics['rmse_mu_per_atom'] * 1000:8.2f}",
                f"{metrics['rel_rmse_mu']:8.1f}",
            ]
        elif table_type == "DipoleMAE":
            row = [
                name,
                f"{metrics['mae_mu_per_atom'] * 1000:8.2f}",
                f"{metrics['rel_mae_mu']:8.1f}",
            ]
        elif table_type == "DipolePolarRMSE":
            row = [
                name,
                f"{metrics['rmse_mu_per_atom'] * 1000:.2f}",
                f"{metrics['rel_rmse_mu']:.1f}",
                f"{metrics['rmse_polarizability_per_atom'] * 1000:.2f}",
            ]
        elif table_type == "EnergyDipoleRMSE":
            row = [
                name,
                f"{metrics['rmse_e_per_atom'] * 1000:8.1f}",
                f"{metrics['rmse_f'] * 1000:8.1f}",
                f"{metrics['rel_rmse_f']:8.1f}",
                f"{metrics['rmse_mu_per_atom'] * 1000:8.1f}",
                f"{metrics['rel_rmse_mu']:8.1f}",
            ]

        if row is not None:
            row.extend(
                _format_optional_metric(metrics, key, scale, fmt)
                for key, _, scale, fmt in extra_specs
            )
            table.add_row(row)

    return table
