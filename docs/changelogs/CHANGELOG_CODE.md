# Code Change Log

## 2026-06-28

### Add Fermi residual L2 regularization to top-level `9-final` code

- Files:
  - `.venv_polar_mace_latest/lib/python3.9/site-packages/mace/modules/loss.py`
  - `.venv_polar_mace_latest/lib/python3.9/site-packages/mace/tools/arg_parser.py`
  - `.venv_polar_mace_latest/lib/python3.9/site-packages/mace/tools/scripts_utils.py`
- Reason:
  - The `p1e6_reg01_full_120` NiN test used an explicit L2 penalty on the learned `fermi_level_residual`.
  - The top-level `9-final` code had Fermi-level loss but no way to regularize the residual term.
- Change:
  - Added config/parser option `fermi_residual_reg_weight` with default `0.0`.
  - Added `mean_squared_error_fermi_residual(...)` in the electrostatics loss.
  - Added `fermi_residual_reg_weight * mean(fermi_level_residual^2)` to `WeightedEnergyForcesElectrostaticsLoss`.
  - Passed `fermi_residual_reg_weight` through normal and SWA electrostatics loss construction.
  - Did not add or restore `fermi_residual_detach_features`; the top-level code still uses the existing Fermi residual head path.
- Exact code changes:
  - `loss.py`
    ```diff
     def mean_squared_error_fermi_level(
         ref: Batch,
         pred: TensorDict,
         ddp: Optional[bool] = None,
     ) -> Optional[torch.Tensor]:
    @@
         raw_loss = torch.square(ref_fermi.view(-1) - pred_fermi.view(-1))
         return reduce_loss(raw_loss, ddp)
     
     
    +def mean_squared_error_fermi_residual(
    +    pred: TensorDict,
    +    ddp: Optional[bool] = None,
    +) -> Optional[torch.Tensor]:
    +    residual = pred.get("fermi_level_residual")
    +    if residual is None:
    +        return None
    +    raw_loss = torch.square(residual.view(-1))
    +    return reduce_loss(raw_loss, ddp)
    +
    +
     def weighted_mean_squared_error_explicit_potential(
    ```
    ```diff
             atomic_dipole_weight=0.0,
             potential_weight=0.0,
             fermi_level_weight=0.0,
    +        fermi_residual_reg_weight=0.0,
             density_3d_weight=0.0,
    ```
    ```diff
             self.register_buffer(
                 "fermi_level_weight",
                 torch.tensor(fermi_level_weight, dtype=torch.get_default_dtype()),
             )
             self.register_buffer(
    +            "fermi_residual_reg_weight",
    +            torch.tensor(fermi_residual_reg_weight, dtype=torch.get_default_dtype()),
    +        )
    +        self.register_buffer(
                 "density_3d_weight",
                 torch.tensor(density_3d_weight, dtype=torch.get_default_dtype()),
             )
    ```
    ```diff
             if self.fermi_level_weight > 1e-12:
                 loss_fermi = mean_squared_error_fermi_level(ref, pred, ddp)
                 if loss_fermi is not None:
                     loss = loss + self.fermi_level_weight * loss_fermi
    +        if self.fermi_residual_reg_weight > 1e-12:
    +            loss_fermi_residual = mean_squared_error_fermi_residual(pred, ddp)
    +            if loss_fermi_residual is not None:
    +                loss = loss + self.fermi_residual_reg_weight * loss_fermi_residual
             if self.density_3d_weight > 1e-12:
    ```
    ```diff
                 f"potential_weight={self.potential_weight:.3f}, "
                 f"fermi_level_weight={self.fermi_level_weight:.3f}, "
    +            f"fermi_residual_reg_weight={self.fermi_residual_reg_weight:.3f}, "
                 f"density_3d_weight={self.density_3d_weight:.3f}, "
    ```
  - `arg_parser.py`
    ```diff
         parser.add_argument(
             "--fermi_level_baseline",
             help="internal Fermi-level baseline in eV; normally fitted from the train split",
             type=float,
             default=0.0,
         )
         parser.add_argument(
    +        "--fermi_residual_reg_weight",
    +        help="L2 regularization weight on the learned Fermi residual in eV",
    +        type=float,
    +        default=0.0,
    +    )
    +    parser.add_argument(
             "--density_3d_weight",
             help="weight of partition-free 3D CHGCAR point density loss",
             type=float,
             default=0.0,
         )
    ```
  - `scripts_utils.py`
    ```diff
             atomic_dipole_weight=getattr(args, "atomic_dipole_weight", 0.0),
             potential_weight=getattr(args, "potential_weight", 0.0),
             fermi_level_weight=getattr(args, "fermi_level_weight", 0.0),
    +        fermi_residual_reg_weight=getattr(
    +            args, "fermi_residual_reg_weight", 0.0
    +        ),
             density_3d_weight=getattr(args, "density_3d_weight", 0.0),
    ```
    ```diff
             atomic_dipole_weight=swa_atomic_dipole_weight,
             potential_weight=swa_potential_weight,
             fermi_level_weight=getattr(args, "fermi_level_weight", 0.0),
    +        fermi_residual_reg_weight=getattr(
    +            args, "fermi_residual_reg_weight", 0.0
    +        ),
             density_3d_weight=getattr(args, "density_3d_weight", 0.0),
    ```
- Verification:
  - `python -m py_compile` passed for all three changed files.
  - `python -m mace.cli.run_train --help` lists `--fermi_residual_reg_weight`.
  - Constructing `WeightedEnergyForcesElectrostaticsLoss(fermi_level_weight=1.0, fermi_residual_reg_weight=0.1)` prints `fermi_residual_reg_weight=0.100`.

### Make Fermi residual regularization default to 0.1 but gate it by Fermi loss

- Files:
  - `.venv_polar_mace_latest/lib/python3.9/site-packages/mace/modules/loss.py`
  - `.venv_polar_mace_latest/lib/python3.9/site-packages/mace/tools/arg_parser.py`
- Reason:
  - The residual regularization should be available by default for Fermi training.
  - It should not penalize an otherwise unused residual output when `fermi_level_weight == 0`.
- Exact code changes:
  - `loss.py`
    ```diff
    -        fermi_residual_reg_weight=0.0,
    +        fermi_residual_reg_weight=0.1,
    ```
    ```diff
    -        if self.fermi_residual_reg_weight > 1e-12:
    +        if self.fermi_level_weight > 1e-12 and self.fermi_residual_reg_weight > 1e-12:
                 loss_fermi_residual = mean_squared_error_fermi_residual(pred, ddp)
                 if loss_fermi_residual is not None:
                     loss = loss + self.fermi_residual_reg_weight * loss_fermi_residual
    ```
  - `arg_parser.py`
    ```diff
             "--fermi_residual_reg_weight",
    -        help="L2 regularization weight on the learned Fermi residual in eV",
    +        help="L2 regularization weight on the learned Fermi residual in eV; only active when fermi_level_weight > 0",
             type=float,
    -        default=0.0,
    +        default=0.1,
         )
    ```
- Training behavior:
  - If `fermi_level_weight == 0`, `fermi_residual_reg_weight` is ignored even though its default value is `0.1`.
  - If `fermi_level_weight > 0`, the residual regularizer is active with default weight `0.1` unless overridden in config.
- Verification:
  - `python -m py_compile` passed for `loss.py` and `arg_parser.py`.
  - Constructing `WeightedEnergyForcesElectrostaticsLoss()` prints `fermi_residual_reg_weight=0.100`.
  - Source check confirms the forward condition is `self.fermi_level_weight > 1e-12 and self.fermi_residual_reg_weight > 1e-12`.

## 2026-06-09

### Restore missing `run_train.py` imports

- Files:
  - `.venv_polar_mace_latest/lib/python3.9/site-packages/mace/cli/run_train.py`
- Reason:
  - `3-partition` failed during Python startup with `NameError: name 'List' is not defined`, then `NameError: name 'ast' is not defined`.
  - Neighboring working copies (`6-compensate`, `7-all_method/c-test`, `3-polar`) all import these names; the top-level `7-all_method` copy was inconsistent.
- Change:
  - Restored `import ast`.
  - Restored `from typing import List, Optional`.
- Exact code change:
  ```diff
  +import ast
   import glob
  ...
  -from typing import Optional
  +from typing import List, Optional
  ```
- Training behavior:
  - No intended model or loss behavior change.
- Verification:
  - `python -m py_compile mace/cli/run_train.py`
  - `python -m mace.cli.run_train --help`

### Derive solvent center target from active electrostatic method

- Files:
  - `.venv_polar_mace_latest/lib/python3.9/site-packages/mace/modules/loss.py`
  - `.venv_polar_mace_latest/lib/python3.9/site-packages/mace/tools/train.py`
  - `.venv_polar_mace_latest/lib/python3.9/site-packages/mace/tools/arg_parser.py`
- Reason:
  - `potential_1d_profile_file` should not store or provide a precomputed solvent center.
  - Density/profile and partition methods need mutually exclusive center targets computed inside the loss from their own physical targets.
- Change:
  - Added density/profile-derived solvent layer mean target computed inside loss from full-grid residual density, fixed 1D profiles, reference potential difference, and total charge.
  - Kept partition-derived solvent layer mean target from reference partition charges, atomic dipoles, reference potential difference, and total charge.
  - Added ambiguity check: density/profile center target cannot be combined with partition charge or atomic-dipole supervision.
  - Evaluation metrics now use the same center target selection as training loss.
- Exact code changes:
  - `loss.py`
    - Added `Density3DGridTargets.plane_average(...)`.
    - Added `reference_solvent_layer_mean_from_density(...)`.
    - Added `_predicted_solvent_layer_mean(...)`.
    - Added `solvent_layer_mean_residuals_from_density(...)`.
    - Added method selector `solvent_layer_mean_residuals(...)`.
    - Added wrapper `weighted_mean_squared_error_solvent_layer_mean(...)`.
    - Changed `WeightedEnergyForcesElectrostaticsLoss` to load `density_3d_targets` when needed for center target:
      ```diff
      -if density_3d_file is not None and float(density_3d_weight) > 1.0e-12
      +if density_3d_file is not None and (
      +    float(density_3d_weight) > 1.0e-12
      +    or (
      +        float(solvent_center_weight) > 1.0e-12
      +        and potential_1d_profile_file is not None
      +    )
      +)
      ```
    - Changed `potential_1d_profile_targets` loading so center loss can use it even if `potential_1d_profile_weight == 0`.
    - Changed solvent center loss call from partition-only:
      ```diff
      -weighted_mean_squared_error_solvent_layer_mean_from_partition(...)
      +weighted_mean_squared_error_solvent_layer_mean(
      +    potential_targets=self.potential_1d_profile_targets,
      +    density_targets=self.density_3d_targets,
      +    use_density_center=self.use_density_center_target,
      +    use_partition_center=self.use_partition_center_target,
      +    ...
      +)
      ```
  - `train.py`
    - Changed metric calculation import and call from partition-only `solvent_layer_mean_residuals_from_partition(...)` to shared `solvent_layer_mean_residuals(...)`.
  - `arg_parser.py`
    - Updated help text for `solvent_center_weight`; no new center file argument was added.
- Training behavior:
  - `solvent_center_weight > 0` with `density_3d_file + potential_1d_profile_file` uses density/profile-derived target.
  - `solvent_center_weight > 0` without density/profile files uses partition-derived target.
  - No external center file is read.
- Verification:
  - `python -m py_compile` on changed files.
  - Function-level check for sample 128 gave density/profile-derived `target_layer_mean = 16.7135 A`, close to previous saved `6-compensate` value `16.7176 A`.
  - Partition path construction check selected partition center target.

### Enable dipole output for electrostatics loss without total dipole target

- Files:
  - `.venv_polar_mace_latest/lib/python3.9/site-packages/mace/cli/run_train.py`
- Reason:
  - `3-partition` has no total `dipole` target by design, but `energy_forces_electrostatics` still needs the model to compute dipole-like electrostatic outputs for potential and compensating-center losses.
  - The previous compute flag logic only enabled dipole output for `loss: energy_forces_dipole`, causing `get_loss_fn` to fail with `AssertionError`.
- Change:
  - Treat `loss: energy_forces_electrostatics` like `energy_forces_dipole` for PolarMACE compute flags: `compute_dipole=True`, energy/forces controlled by their weights.
- Exact code change:
  ```diff
  -elif args.model == "PolarMACE" and args.loss == "energy_forces_dipole":
  +elif args.model == "PolarMACE" and args.loss in (
  +    "energy_forces_dipole",
  +    "energy_forces_electrostatics",
  +):
       args.compute_dipole = True
  ```
- Training behavior:
  - Allows partition electrostatic training without storing total `dipole` in xyz.
- Verification:
  - `python -m py_compile mace/cli/run_train.py`
