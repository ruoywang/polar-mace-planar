# Nonlinear Poisson-Boltzmann solvent model (`--solvent_model pb`)

Branch `pb-solvent`, 2026-07-03.

## What

Adds a second implicit-solvent compensation model next to the planar
truncated-Gaussian layer: a laterally averaged ionic charge profile
rho_ion(z) obtained from a full 3D nonlinear Poisson-Boltzmann solve per
structure, using the validated pure-Python solver of `cep-dip-python-pb`
(cal_18 validation: PHI RMSE 3.0e-3 eV vs VASP/CEP-DIP).

The planar model remains the default (`--solvent_model planar`) and its
code path is byte-identical (jellium regression
`tools/jellium_check/test_periodic_fix.py` passes unchanged).

## How it plugs in

The planar layer enters PolarMACE at exactly three points; the PB profile
reuses all three with the truncated Gaussian replaced by rho_ion(z):

1. **SCF field features** — `_slab_compensation_profile_features` /
   `periodic_profile_layer_potential_field_nodes` (same periodic G!=0
   convention as the jellium fix), plus the slab dipole-correction feature
   with `solvent_mu_override` = q_ion * layer_mean from the profile.
2. **Energy** — `_slab_compensation_periodic_1d_energy_radial(...,
   comp_profile=...)`; the exact 1D periodic cross+self energy is valid
   for any laterally uniform rho_ion(z).
3. **Observables** — `predict_potential_from_dipole_and_solvent_mu`;
   `solv_center` becomes the PB profile mean (the learned center-residual
   head is kept in the graph with zero weight for DDP).

One PB solve per forward, from the **pre-SCF** density, reused post-SCF
(the PB analogue of the planar pre/post-SCF center inconsistency, known
issue 3). The solve is **detached**: no gradients flow through the solver
(first pass; implicit-function gradients are future work).

## Solver inputs (mace/modules/pb_solvent.py)

- Electron density for the cavity: neutral-atom Gaussian baseline
  (Z_val at `atomic_multipoles_smearing_width`) minus the model's net
  density (radial coefficients evaluated on the PB grid via
  `_gto_density_at_points_axis2_pbc`). Verified sign convention: a
  `density3d_net_grid` integrates to `total_charge` (physics sign).
- Solute potential: Hartree of that density plus Gaussian nuclei
  (`--solvent_pb_nuclear_sigma`, default 0.4 A) whose reciprocal-space
  tail matches the POTCAR local-pseudopotential Z/G^2 term exactly; no
  POTCAR dependency. Differences vs the VASP solute are confined inside
  the cavity.
- `q_sol = -total_charge` (VASP electron-count convention; cal18 has
  q_sol=+1 for the total_charge=-1 NiN system).
- Dipole-correction fix-steps as in the validated driver
  (`--solvent_pb_fixsol_steps`, default 2) with coarse-grid warm start.
- Returned profile is physics-sign: integral = -total_charge (verified
  to 1e-6 on the synthetic test).

## New training options

`--solvent_model {planar,pb}`, `--solvent_pb_config` (JSON solvation
block, VASPsol names), `--solvent_pb_repo`, `--solvent_pb_grid_spacing`
(default 0.25 A), `--solvent_pb_fixsol_steps`, `--solvent_pb_tol`,
`--solvent_pb_nuclear_sigma`, `--solvent_pb_coarse_init`. Checkpoint
config extraction defaults to `solvent_model="planar"` for old models.

## Validation and cost

- `tools/pb_check/test_pb_solvent_synthetic.py`: NiN-like cell, synthetic
  net density, cal18 solvation block — q_ion = -total_charge to 1e-6,
  ionic charge localized in the ion-accessible region, charge-conserving
  resampling. Full solve 7.2 s on 8 CPU threads at 0.25 A spacing
  (583k grid points; the 81 s cal_18 figure is at 0.088 A / 10.8M points).
- Requires the rebuilt `_pb_fast` C extension (17 kernels); the committed
  .so in cep-dip-python-pb was stale (1 kernel) and was rebuilt in place
  (git-ignored there).

## Known approximations (to revisit against reference fields)

1. Gaussian neutral-atom baseline underestimates the exponential density
   tail at the cavity boundary vs real CHGCAR density.
2. No POTCAR partial-core (dencor) charge in the cavity density.
3. Pre-SCF density feeds the solve; post-SCF energy reuses the profile.
4. Detached solve: interface/profile carries no gradient.
5. Solvation block (incl. SOL_Z0/SOL_Z1 z-window) is a fixed per-run JSON,
   appropriate while all structures share the cal18-family cell.
