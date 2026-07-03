## 2026-06-11 - Restrict primary_valid_metric to loss only

Scope:
- Updated only `/scratch/08384/tg876840/tmp/9-final/.venv_polar_mace_latest`.

Files changed:
- `mace/tools/arg_parser.py`

Complete line-by-line patch:
- Saved in `/scratch/08384/tg876840/tmp/9-final/CHANGELOG_PRIMARY_METRIC_PATCH.diff`.

Behavior:
- Removed `rmse_potential` and `rmse_solvent_center` from the allowed
  `primary_valid_metric` choices.
- `primary_valid_metric` now only accepts `loss`.

Validation:
- `python -m py_compile ../9-final/.venv_polar_mace_latest/lib/python3.9/site-packages/mace/tools/arg_parser.py`
  passed.

## 2026-06-11 - Add Fermi-level residual loss

Scope:
- Updated only `/scratch/08384/tg876840/tmp/9-final/.venv_polar_mace_latest`.

Files changed:
- `mace/cli/run_train.py`
- `mace/data/utils.py`
- `mace/modules/extensions.py`
- `mace/modules/loss.py`
- `mace/tools/arg_parser.py`
- `mace/tools/default_keys.py`
- `mace/tools/model_script_utils.py`
- `mace/tools/scripts_utils.py`
- `mace/tools/tables_utils.py`
- `mace/tools/train.py`

Complete line-by-line patch:
- Saved in `/scratch/08384/tg876840/tmp/9-final/CHANGELOG_FERMI_PATCH.diff`.
- This diff was generated against a temporary baseline equal to the copied `7-all_method`
  code plus the previously recorded shift patch, so it contains only the Fermi-level
  residual/baseline changes from this step.

Behavior:
- Added `fermi_level_key`, defaulting to `fermi_level`, so xyz info fields can be mapped
  into `AtomicData.fermi_level`.
- Added `fermi_level_weight`, default `0.0`; the old `potential_weight` still means the
  original scalar potential-difference loss and is not silently redefined.
- Added train-split fitting of `fermi_level_baseline` when `fermi_level_weight > 0`:
  `baseline = mean_train(fermi_level + potential)`, matching
  `fermi_level_pred = baseline - potential_pred + fermi_level_residual`.
- Added a small graph-level residual head:
  `fermi_level_residual = MLP(mean(node_feats_out))`.
- The final residual-head layer is zero-initialized, so the initial prediction is exactly
  `baseline - potential_pred`.
- Added model outputs:
  `fermi_level_pred`, `fermi_level_residual`, and `fermi_level_baseline`.
- Added Fermi-level loss inside `WeightedEnergyForcesElectrostaticsLoss`.
- Added `RMSE_fermi` to epoch logging and `RMSE Fermi / eV` / `MAE Fermi / eV` to error tables.
- Saved `fermi_level_baseline` in extracted model configs and passed it into PolarMACE construction.

Validation:
- `python -m py_compile` passed for all changed Python files.
- Direct parser/loss construction check passed for:
  `--loss energy_forces_electrostatics --fermi_level_weight 1.0`.

## 2026-06-10 - Restore train-split solvent-center mean-shift fitting

Scope:
- Updated only `/scratch/08384/tg876840/tmp/9-final/.venv_polar_mace_latest`.

Files changed:
- `mace/cli/run_train.py`

Complete line-by-line patch:
- Saved in `/scratch/08384/tg876840/tmp/9-final/CHANGELOG_SHIFT_PATCH.diff`.
- The diff compares the copied `7-all_method/.venv_polar_mace_latest/.../run_train.py`
  against the modified `9-final/.venv_polar_mace_latest/.../run_train.py`.

Behavior:
- Replaced the old configured-only `solvent_center_mean_shift` path with train-split fitting.
- Method selection uses only:
  - `density_3d_weight > 0`: density-derived shift.
  - `charges_weight > 0`: partition-derived shift.
  - both enabled: raise `ValueError`.
  - neither enabled: use `0.0 Å`.
- Potential weights are not used to decide the method.
- Density-derived fitting requires full-grid `density_3d_file` plus `potential_1d_profile_file`;
  it computes the target solvent-layer mean from DFT density/profile/potential data and computes
  `z_e50` from the DFT total electron density profile.
- Partition-derived fitting computes the target solvent-layer mean from partition charges,
  atomic dipoles, scalar potential difference, total charge, and cell area, then computes `z_e50`
  from the corresponding partition electron-density approximation.
- The fitted value is assigned to `args.solvent_center_mean_shift` before model construction, so
  both SCF compensation initialization and the final solvent-center head use the same fitted
  constant.

Validation:
- `python -m py_compile ../9-final/.venv_polar_mace_latest/lib/python3.9/site-packages/mace/cli/run_train.py`
  passed.
- Direct import check passed:
  `from mace.cli.run_train import _fit_train_solvent_center_mean_shift,
  _fit_density_solvent_center_mean_shift, _fit_partition_solvent_center_mean_shift`.

## 2026-06-09 - Remove density_radial_coefficients output

Scope:
- Updated only `/scratch/08384/tg876840/tmp/8-charge_grid/7-all_method/.venv_polar_mace_latest`.

Files changed:
- `mace/modules/extensions.py`
- `mace/modules/loss.py`

Changes:
- Removed the forward-pass `density_radial_coefficients` output and its `_radial_charge_to_electron` construction.
- Changed 3D density loss to require `charge_density_radial_coefficients`.
- Changed 1D potential-profile loss to require `charge_density_radial_coefficients`.
- Removed fallback from residual density/profile losses to total-electron-density coefficients so residual targets cannot silently use the wrong physical quantity.

Reason:
- Current `density3d_net_grid` and 1D potential-profile reconstruction targets are baseline-subtracted residual/net density quantities.
- The removed `density_radial_coefficients` represented a GTO total-electron-density口径 and was not appropriate for these residual targets.

Validation:
- `python -m py_compile` passed for updated `extensions.py` and `loss.py`.
- `rg` confirmed no model output named `density_radial_coefficients` remains in `extensions.py`.

Exact code changes:

1. Deleted this complete method from `mace/modules/extensions.py`:

```python
    def _radial_charge_to_electron(
        self, radial_flat: torch.Tensor, node_valence_electrons: torch.Tensor
    ) -> torch.Tensor:
        radial = self._radial_flat_to_blocks(radial_flat)
        electron = radial.clone()
        charge_l0 = radial[:, :, 0]
        total_atomic_charge = charge_l0.sum(dim=-1)
        total_atomic_electrons = (
            node_valence_electrons.to(radial.dtype) - total_atomic_charge
        )
        radial_weights = torch.softmax(-charge_l0, dim=-1)
        electron[:, :, 0] = radial_weights * total_atomic_electrons.unsqueeze(-1)
        return electron
```

2. Removed this forward-pass construction from `mace/modules/extensions.py`:

```python
        density_radial_coefficients = self._radial_charge_to_electron(
            charge_density_radial_mul_ir, node_valence_electrons
        )
```

The surrounding code is now:

```python
        element_index = torch.argmax(data["node_attrs"], dim=-1)
        node_valence_electrons = self.atomic_valence_electrons[element_index]
        charge_density_radial_coefficients = self._radial_flat_to_blocks(
            charge_density_radial_mul_ir
        )
        density_width_threshold = None
```

3. Removed this output entry from the returned dictionary in `mace/modules/extensions.py`:

```python
            "density_radial_coefficients": density_radial_coefficients,
```

The surrounding output block is now:

```python
            "node_feats": node_feats_out,
            "density_coefficients": charge_density_mul_ir,
            "charge_density_radial_coefficients": charge_density_radial_coefficients,
            "spin_density": spin_density_mul_ir,
```

4. Replaced the coefficient selection at the start of `density_3d_residuals` in `mace/modules/loss.py`.

Old code:

```python
    density_coefficients = pred.get("density_radial_coefficients")
    if density_coefficients is None:
        density_coefficients = pred.get("charge_density_radial_coefficients")
    if (
        density_coefficients is None
        or _batch_get(ref, "sample_id") is None
        or _batch_get(ref, "atomic_numbers") is None
    ):
        return None
```

New code:

```python
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
```

5. Replaced the coefficient selection at the start of `potential_1d_profile_residuals` in `mace/modules/loss.py`.

Old code:

```python
    density_coefficients = pred.get("density_radial_coefficients")
    if density_coefficients is None:
        density_coefficients = pred.get("charge_density_radial_coefficients")
    if (
        density_coefficients is None
        or _batch_get(ref, "sample_id") is None
        or pred.get("solv_center") is None
    ):
        return None
```

New code:

```python
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
```

## 2026-06-06 03:24 - Sync selected c-test changes to 7-all_method

Scope:
- Updated only `/scratch/08384/tg876840/tmp/8-charge_grid/7-all_method/.venv_polar_mace_latest`.
- Did not modify `c-test`.

Backup:
- Original files saved in `/scratch/08384/tg876840/tmp/8-charge_grid/7-all_method/backups/code_sync_20260606_032403/`.

Files changed:
- `mace/modules/extensions.py`
- `mace/modules/loss.py`

Changes synced from `c-test`:
- Replaced the old analytic Gaussian compensation cross-energy with the 1D periodic Poisson compensation energy path:
  `compensation_cross_energy` -> `compensation_periodic_1d_energy`.
- Zero-initialized `lr_source_maps[*].linear`.
- Zero-initialized final `linear_2` parameters in `field_dependent_charges_maps`.
- Changed explicit density / 1D potential-profile loss coefficient priority to use `density_radial_coefficients` first, then fallback to `charge_density_radial_coefficients`.

Explicitly not synced:
- The SCF-pre solvent center prediction block:
  `solvent_head_inputs_pre`, `solvent_center_residual_pre`, `comp_explicit_potential_base`.
- Therefore outer `7-all_method` still uses the original `comp_center_init = comp_z_base + solvent_center_mean_shift` path.

Validation:
- `python -m py_compile` passed for updated `extensions.py` and `loss.py`.
- `rg` confirmed no `global_charge_energy`, `add_global_charge_energy`, `zero_init_field_response`, or `zero_init_solvent_center` remains.
- Diff versus `c-test` confirms the remaining source difference in `extensions.py` is the excluded SCF-pre center prediction block.

## 2026-06-09 04:28 - Rename extended electrostatics loss

Scope:
- Updated only `/scratch/08384/tg876840/tmp/8-charge_grid/7-all_method/.venv_polar_mace_latest`.

Backup:
- Original files saved in `/scratch/08384/tg876840/tmp/8-charge_grid/7-all_method/backups/loss_rename_20260609_042840/`.

Files changed:
- `mace/modules/loss.py`
- `mace/modules/__init__.py`
- `mace/tools/scripts_utils.py`
- `mace/tools/arg_parser.py`

Changes:
- Restored `energy_forces_dipole` / `WeightedEnergyForcesDipoleLoss` to the original semantic scope: energy, forces, and dipole only.
- Added `energy_forces_electrostatics` / `WeightedEnergyForcesElectrostaticsLoss` for the extended PolarMACE electrostatics losses:
  charges, atomic dipoles, scalar potential, 3D density, 1D potential profile, and solvent-center layer-mean loss.
- Added `energy_forces_electrostatics` to the allowed loss choices.
- Exported `WeightedEnergyForcesElectrostaticsLoss` from `mace.modules`.
- Removed stale `solvent_sigma_g` keyword passing from `weighted`, `dipole`, and generic SWA `WeightedEnergyForcesLoss` construction.

Validation:
- `python -m py_compile` passed for `loss.py`, `__init__.py`, `scripts_utils.py`, and `arg_parser.py`.
- Direct construction via `get_loss_fn` was checked for:
  `weighted`, `energy_forces_dipole`, and `energy_forces_electrostatics`.
- Confirmed `energy_forces_dipole` constructs `WeightedEnergyForcesDipoleLoss` with only `energy_weight`, `forces_weight`, and `dipole_weight`.
- Confirmed `energy_forces_electrostatics` constructs `WeightedEnergyForcesElectrostaticsLoss` with the extended electrostatics terms.

## 2026-06-09 04:49 - Remove external solvent-center target file

Scope:
- Updated only `/scratch/08384/tg876840/tmp/8-charge_grid/7-all_method/.venv_polar_mace_latest`.

Backup:
- Original files saved in `/scratch/08384/tg876840/tmp/8-charge_grid/7-all_method/backups/remove_center_file_20260609_044955/`.

Files changed:
- `mace/modules/loss.py`
- `mace/tools/train.py`
- `mace/tools/scripts_utils.py`
- `mace/tools/arg_parser.py`
- `mace/cli/run_train.py`
- `mace/tools/default_keys.py`

Changes:
- Removed the `solvent_center_dipole_file` training argument and all active source-code uses of the external `solvent_layer_mean_A` lookup table.
- Changed `solvent_center_weight` so the compensating-layer mean target is inferred per batch from reference scalar potential, partition charges, atomic dipoles, total charge, cell area, and the same `potential_sign` convention used by scalar-potential training.
- Updated validation metrics so `RMSE_solvent_layer_mean` uses the same inferred target rather than an external file.
- Removed the `SOLVENT_CENTER = "solvent_center"` default key so `DefaultKeys.keydict()` no longer creates `solvent_center_key`.
- Disabled and renamed the old train-split fit from xyz `solvent_center`; `run_train.py` now uses only the configured `solvent_center_mean_shift` and does not read xyz `solvent_center`.

Validation:
- `python -m py_compile` passed for `loss.py`, `train.py`, `scripts_utils.py`, `arg_parser.py`, `run_train.py`, and `default_keys.py`.
- A direct synthetic check recovered a known `solvent_layer_mean_A = 18.0` from constructed `potential`, `charges`, `atomic_dipole`, `total_charge`, and cell area.
- Direct `get_loss_fn` construction confirmed `WeightedEnergyForcesElectrostaticsLoss` no longer has a `solvent_center_dipole_file` attribute.
- `DefaultKeys.keydict()` was checked and no longer contains `solvent_center_key`.
- `rg` confirmed no active source-code references to `solvent_center_dipole_file`, `solvent_center_key`, `SOLVENT_CENTER`, `solvent_layer_mean_A`, or xyz `solvent_center` reads remain under `.venv_polar_mace_latest/lib/python3.9/site-packages/mace`.
# 2026-06-24: l=2 disable change reverted

- The temporary code/config change that disabled `atomic_multipoles_max_l > 1` was reverted. Training configs again allow `atomic_multipoles_max_l: 2`, and `.venv_polar_mace_latest/lib/python3.9/site-packages/mace/modules/extensions.py` no longer raises on `l=2`.
- Rationale for the revert: `l=2` may still be needed for explicit density/profile training. The bad result was from using unconstrained `l=2` coefficients in an only-potential-difference diagnostic, not from the existence of `l=2` support.
- Kept only the diagnostic cleanup in `3-charge_test/scripts/compare_sample84_split_neutral_ml.py`: this sample-84 only-diff comparison zeros coefficient indices `k>=4` before reconstructing density, so the specific diagnostic output does not use the known-bad unconstrained `l=2` part of that old model.
- Removed the erroneous `L2_REMOVAL_CODEX.diff` file.
## 2026-06-27 sync no-reg-full charge-problem code to top-level

- Updated top-level `.venv_polar_mace_latest` to match the tested no-reg/full code variant from `4-charge_problem/code_variants/no_fermi_residual_controls/site-packages/mace`.
- Changed `mace/cli/run_train.py`: when neither `density_3d_weight` nor `charges_weight` is enabled, training now uses the configured `solvent_center_mean_shift` instead of forcing `0.0`; if `solvent_center_weight > 0` without density/charge information, training raises an error.
- Changed `mace/tools/arg_parser.py`: default `solvent_center_mean_shift` is now `0.7` Angstrom.
- Changed `mace/modules/extensions.py`: added `learn_solvent_center_residual`; when false, `solvent_center_residual` is set to exactly zero through `solvent_raw_shift * 0.0`.
- Changed `mace/tools/model_script_utils.py`: `learn_solvent_center_residual` is set from `solvent_center_weight > 0`.
- Removed the zero initialization blocks for `lr_source_maps` and `field_dependent_charges_maps`, matching the tested no-reg/full variant.
- This top-level code still has no `fermi_residual_detach_features` or `fermi_residual_reg_weight` options.
- Exact patch record saved in `CHANGELOG_SYNC_NO_REG_FULL_PATCH.diff`.
