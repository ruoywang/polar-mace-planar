# polar-mace-planar


**Reports:** https://ruoywang.github.io/polar-mace-planar/ (training and method
pages; solver-level reports at https://ruoywang.github.io/cep-dip-python-pb/)
Modified **PolarMACE** for machine-learning constant-potential electrochemistry:
charged metal/water slabs whose net electrode charge is compensated by a
**planar Gaussian implicit-solvent charge layer** (a learned surrogate for the
VASPsol/CEP-DIP Poisson-Boltzmann ionic charge RHOION).

This repository tracks only the modified `mace` Python package. It is the
source of truth for the code that was previously edited in place inside the
run venvs on TACC Lonestar6.

## Method summary (planar compensation layer)

For each training structure the explicit subsystem (slab + adsorbates +
explicit water) carries a net charge `total_charge` (typically ~-1 e in the
present datasets). The model places a Gaussian charge sheet of width
`solvent_sigma_g` (0.85 Å) carrying `-total_charge` at

```
z_plane = z_e + solvent_center_mean_shift + residual(features), |residual| <= 1.5 Å
```

where `z_e` is the last descending crossing of the reconstructed 1D electron
density (threshold `solvent_density_threshold`, e.g. 0.15 e/Å³, or a
normalized `solvent_ze_level` crossing). The layer contributes:

- **SCF field features**: Gaussian-sheet potential/field convolved with the
  receiver widths, plus a slab dipole-correction field, fed into the PolarMACE
  charge-equilibration loop (`extensions.py`).
- **Energy**: an exact 1D periodic (G!=0) cross + self electrostatic energy of
  the layer against the explicit density, plus the slab dipole-correction
  delta from explicit-only to total dipole (`_slab_compensation_periodic_1d_energy_radial`,
  `_slab_dipole_correction_delta`).
- **Observables/losses**: potential difference from the total dipole, Fermi
  level (`baseline - potential + residual head`), 3D CHGCAR point-density loss,
  1D potential-profile loss reproducing VASP's LDIPOL/cdipol correction, and
  solvent layer-mean center targets (density-derived or partition-derived).

New training options are documented in `mace/tools/arg_parser.py`
(`--solvent_*`, `--atomic_density_sigmas`, `--atomic_valence_electrons`,
`--fermi_level_*`, `--density_3d_*`, `--potential_1d_profile_*`,
`--element_charge_baseline`, loss `energy_forces_electrostatics`).

## Repository layout

```
mace/                    modified package source (installed layout)
docs/changelogs/         original in-place modification logs and patch diffs
configs/                 training configs of the 1-Au and 2-NiN runs
requirements-freeze.txt  pip freeze of the 9-final run venv
```

## Provenance

- Commit 1 (`Baseline`) is the unmodified PolarMACE as installed in
  `/scratch/08384/tg876840/tmp/3-polar/.venv_polar_mace_latest`
  (upstream `ACEsuit/mace@667eee4e58d23a38ff5a75122109ec2025809649`,
  mace-torch 0.3.15).
- Commit 2 (`9-final state`) is the working tree of
  `/scratch/08384/tg876840/tmp/9-final/.venv_polar_mace_latest` as of
  2026-07-02, i.e. the code used by the `1-Au/*` and `2-NiN/*` training runs
  (configs in `configs/`).
- Dependency `graph_longrange` 0.4.0 is **unmodified** (identical in both
  venvs) and is not vendored here; it is pinned to
  `WillBaldwin0/graph_electrostatics@01350365c4b62e4282a9fe07d8319a3c7b75ab8b`
  (MIT).
- Environment: Python 3.9.13 venv with `include-system-site-packages = true`
  over the conda toolchain in `$WORK/conda` (torch, e3nn, ase come from
  there); see `requirements-freeze.txt`.
- Both `mace` (MIT) and `graph_longrange` (MIT) licenses permit this private
  modified copy; upstream license headers are retained in the sources.

## Known issues (code review 2026-07-02)

1. **Feature-level jellium inconsistency (charged systems) — FIXED.** In the
   SCF field features the explicit density uses the periodic G!=0 evaluator
   (implicit uniform neutralizing background), while the compensation plane
   used the *isolated* erf potential (no background). For `total_charge != 0`
   the sum is not the potential of the neutral total system: a spurious
   quadratic potential with curvature -k*q_plane/V remains (verified at
   machine precision on 2-NiN sample 0: 2.45 eV spread across the atoms;
   `tools/jellium_check/verify_jellium.py`). The *energy* decomposition is
   exact and was never affected. Fixed by
   `periodic_gaussian_layer_potential_field_nodes` (1D periodic G!=0 layer
   potential with receiver-width damping) selected via
   `--solvent_plane_feature_convention` (default `periodic`; `isolated`
   reproduces the legacy behavior, and configs extracted from old checkpoints
   default to `isolated`). Validation: `tools/jellium_check/test_periodic_fix.py`
   (spurious curvature -0.0196 -> -1.3e-10 eV/Å²; autograd field consistency
   4e-16; legacy path byte-identical). Models trained before the fix must be
   retrained to benefit.
2. **`solvent_center_mean_shift` defaults to 0.7 Å** in `arg_parser.py` and is
   silently used whenever neither `density_3d_weight` nor `charges_weight` is
   active (e.g. the `*fermi*` runs). Should be explicit in configs.
3. SCF compensation features use the pre-SCF center without the learned
   residual, while the final energy/potential use the post-SCF center with the
   residual (bounded inconsistency, <= ~1.5 Å by construction).
4. Interface crossing searches are detached; the interface position receives
   no gradient through the density coefficients (by design).
5. `weighted_mean_squared_error_dipole` compares the reference explicit dipole
   against the predicted *total* dipole; only safe because
   `WeightedEnergyForcesElectrostaticsLoss` forces `dipole_weight = 0`.
6. The final radial-collapse path in `extensions.py` bypasses
   `charges_to_mul_ir`; correct only without cuEquivariance layouts.

## Roadmap

- Fix issue 1 (periodic-consistent plane features) as the next commit(s).
- Replace/augment the planar layer with a real Poisson-Boltzmann solvent
  charge model (planned; this repo will grow alongside).
