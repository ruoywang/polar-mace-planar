# 任务 → 代码版本台账

规则(2026-07-26 起):每个训练/评估作业启动时,作业脚本把所用代码树的
`git rev-parse HEAD` 写进 run 日志;本文件记录每个实验目录用的是哪个
commit。新实验一律登记,旧实验按已知信息回填(未知处如实标注)。

### the answer: the channel is charge-dependent but 121x too weak

40 epochs finished (3430114 epochs 0-37, 3430182 epochs 37-39, both COMPLETED).
Endpoint epoch 39: 10.33 meV/atom, 30.18 meV/A, 0.1317 eV, fermi 0.0843 eV.
That endpoint is NOT comparable to gate_bl's 11.75 / 32.75 / 0.1227 -- 39
epochs against 33, and gate_le's own 33->39 gain is 9%, so the two causes are
not separable there. The only same-epoch comparison remains epoch 33.

`le_channel_probe.py` on the finished model, 21 pairs, every one passing the
ablation gate (|discrepancy| 2.7e-11 eV against a total energy of -1306 eV,
tolerance 1e-6 eV derived from the size of the paired quantity).

CORRECTION TO THE PARAMETER FIGURE. The channel holds 76545 TRAINABLE
parameters, 7.98% of the model's 958718 -- not the 1.52% recorded above, which
summed the checkpoint's whole state_dict and so counted buffers in both
numerator and denominator (5110243 tensors against 958718 parameters; the
channel's own 77825 against 76545, the 1280 difference being four output_mask
buffers). The argument that "a 1.52% channel cannot move aggregate metrics" is
withdrawn; it is also no longer needed, since the channel's output change is
now measured directly.

q_in IS CHARGE-DEPENDENT, which the earlier audit never tested:

| channel input | shape | max abs diff | relative |
|---|---|---|---|
| charges_0 | (207, 54) | 2.07e-03 | 0.36% |
| charges_induced | (207, 54) | 3.51e-01 | 56.0% |
| field_feats | (207, 8) | 4.55 | 27.6% |
| node_feats | (207, 576) | 6.94e-17 | 4.1e-16 |

Exactly as the source predicted: charge reaches the channel through the field
feedback (extensions.py:2562), so charges_induced moves 56% while charges_0,
cloned at 2444 before that feedback, moves 0.36%. node_feats at 4.1e-16
reproduces the earlier charge-blind result, so this channel is the only path by
which charge information reaches the energy at all.

BUT ITS RESPONSE IS TWO ORDERS TOO SMALL. 20 held-out val pairs:

| quantity | value |
|---|---|
| eps_Delta mean / rmse | +4.8716 / 4.9295 eV |
| d(le) mean / rmse | -0.0401 / 0.0405 eV |
| d(le)/dN | -0.0382 eV/electron |
| eps/dN needed | +4.6339 eV/electron |
| share of the gap covered | -0.82% |

The sign is right -- d(le) is negative, the direction the pair needs -- and it
is uniform across every pair (-0.73% to -0.90%, 20 of 20), but the magnitude is
1/121 of what is required.

THE 12.5% IMPROVEMENT CANNOT BE CREDITED TO THE CHANNEL. On the same 20 pairs
gate_bl gave eps rmse 5.621861 and bias +5.570253; gate_le gives 4.9295 and
+4.8716, i.e. 12.3% and 12.5% better. On sid1/601 alone Delta E model moved
from -0.465482 to -1.203373, eps from +5.635 to +4.897, 13.1% better. But the
channel's own contribution to that 0.7379 eV is 0.0415 eV, 5.6%; the other 94%
came from the rest of the model, which differs by 6 epochs and a different
random trajectory. Only the 0.82% is attributable.

WHERE THE ERROR ACTUALLY SITS. Per-frame signed errors, 20 val pairs:

| | mean | rms |
|---|---|---|
| charged frames | +17.67 meV/atom | 17.97 |
| neutral frames | -5.87 meV/atom | 5.98 |

Opposite signs in 20 of 20 pairs, and the charged state carries 3.0x the
magnitude -- not the even split I estimated. This is why a paired error of
4.87 eV = 23.5 meV/atom can hide behind an aggregate RMSE_E_per_atom of 10.33:
the two states' errors cancel in the aggregate metric. (The identity
err_charged - eps = err_neutral is definitional, not a finding, and is not
offered as one.)

READING. The architecture is not the obstacle: charge information is present in
the channel's inputs at 56% relative amplitude, the channel consumes it, and
the term it produces has the right sign. What is missing is any pressure in the
loss to make the paired difference right -- the aggregate energy metric is
already satisfied by errors that cancel between the two states. Turning the
native channel on and training it for 40 epochs therefore does not fix the
charging energy, and the a = -5.2745 eV/electron constant remains the only
measured thing that does.

### the head CAN fit the pair; the fit it finds is not compatible with energies or forces

Head-only fit, everything else frozen, target = dE_DFT - dE_rest (mean -4.9117
eV on val against the current head's -0.0401). 160 train / 20 val NiN44 pairs,
splits never straddling, gates: cached inputs replay the head exactly
(0.000e+00), and perturbing the head's weights moves its own output by
1.198e-11 eV against a control of 6.222e-12 -- the wobble is the stateful PB
solver, not the head, so caching is exact. Optimiser shown to act: max|grad|
4.156e-01 at step 0 with 20/20 parameter tensors non-zero, and 20/24 state_dict
entries moved (the 4 unmoved are output_mask buffers).

| model | fitted on | train rmse / bias / mean-rem | val rmse / bias / mean-rem |
|---|---|---|---|
| current head | no fit | 5.0137 / +4.9719 / 0.6458 | 4.9295 / +4.8716 / 0.7532 |
| a*dN refit | train | 0.2134 / -0.0038 / 0.2134 | 0.1981 / +0.0249 / 0.1965 |
| a*dN, a=-5.2745 | none | 0.6957 / -0.6615 / 0.2153 | 0.6388 / -0.6156 / 0.1705 |
| head, fixed step 4000 | train | 0.1127 / -0.0012 / 0.1127 | 0.1359 / -0.0062 / 0.1357 |
| head, val-selected | train+val | 0.1127 / -0.0012 / 0.1127 | 0.1359 / -0.0062 / 0.1357 |

The last two rows are identical: the best val step WAS the pre-declared
endpoint, so no val selection occurred here. The val curve was still falling at
step 4000 (0.1403, 0.1382, 0.1359), so this is a fixed-budget number and not a
converged one. On val the head beats the refitted constant on the MEAN-REMOVED
column by 30.9% and on 15 of 20 pairs, so its advantage is not bias-only -- the
opposite of what the 3-pair smoke showed, where mean-removed was 0.0700 against
the constant's 0.0660. The 3-pair reading was a sample-size artefact. Also: the
constant refits to a = -4.6632 eV/electron on this residual, and forcing the
earlier -5.2745 gives val rmse 0.6388, so that constant is per-model, not a
property of the system.

So the ~0.04 eV response after 40 joint epochs is NOT this head's ceiling.
What the table does not separate is whether the extra capability comes from
geometry information or from a non-linear response to charge.

THEN INSTALLING IT. Gates first, both per-item and both with a failure exit,
against a declared 1e-3 eV (100x below the 0.1359 eV being interpreted) with an
A-vs-A control arm measuring the solver floor at |dE| 5.593e-11 eV:
  gate 1, per frame: E_new - E_old equals le_new - le_old, worst 9.229e-10 eV
          over 40 frames -- installation changed only that term
  gate 2, per pair: the live model's own dE against dE_rest(cached) + the
          fitted head's cached prediction, worst 1.173e-09 eV -- the cached fit
          transfers exactly
Installed paired error against DFT: rmse 0.1359, bias -0.0062, mean-removed
0.1357 eV, i.e. exactly the fit's own number.

| | charged A | charged B | neutral A | neutral B |
|---|---|---|---|---|
| absolute energy, mean meV/atom | +17.67 | -78.13 | -5.87 | -78.10 |
| absolute energy, rms meV/atom | 17.97 | 78.37 | 5.98 | 78.33 |
| force rmse, eV/A | 0.0292 | 0.7278 | 0.0330 | 0.7070 |
| worst single force error, eV/A | 0.2264 | 9.9954 | 0.6470 | 10.6306 |

The fit bought a correct dE by adding about -17.4 eV to BOTH frames: the head's
output goes from -0.78 eV in both states to about -20 (charged) and -15
(neutral). After installation the two states' errors are nearly EQUAL (-78.13
against -78.10 meV/atom, a 0.03 meV/atom gap = the -0.0062 eV bias), so the
paired difference is right and a common -16.17 eV offset is wrong. Forces
degrade 25x and 21x with worst-case errors of 10 eV/A.

THE COMMON MODE IS A REAL C(R), NOT A CONSTANT, and that is what the forces
prove rather than the shift itself. A shift in the two-state mean of le is NOT
by itself evidence of harm: a correct paired correction applied asymmetrically
already moves that mean by a*dN/2, and dN varies per frame, so a non-zero shift
with a non-zero spread is expected. Measured: shift -17.3912 eV with spread
1.0199 eV against the a_fit*dN/2 reference of -2.4434 eV with spread 0.3204 eV,
so the shift is 7.1x the reference and its spread 3.2x. And a pure constant
cannot change forces at all, so the 22x force degradation is itself the proof
that the term the head added depends on geometry.

SCOPE. This refutes the OBJECTIVE, not the head. Nothing in (dE_LE - target)^2
constrains the common level or the forces, so a solution like this was always
admissible. Whether a head can satisfy the pair AND the absolute energies AND
the forces at once is untested: the absolute-energy version of this fit is
cheap on the same cache (E_new = E_rest_cached + le_new against the label),
while a force-aware fit is not, because dle/dR needs a full-model backward per
step rather than cached inputs.

### three-arm force-weight experiment: the force weight is not the obstacle

From epoch 39 (EMA weights, _pb1d_epoch 39), only local_electron_energy
updated, full coordinate-derivative path with compute_force=True and nothing
cached, identical starting weights / loaded Adam state / lr / sample order
across arms, original per-atom normalisation and whole-model clipping, no added
paired loss. 16 train pairs spanning dN +0.795..+1.375, 100 steps, recorded at
0/20/50/100, scored on the same 20 val pairs. A = (1, 100) the run's own
weights, B = (1, 20), C = (1, 0). 20 sits below both measured crossover points
(forces_weight 32.4 at epoch 25, 87.6 at epoch 39).

| step 100 change | A (1,100) | B (1,20) | C (1,0) |
|---|---|---|---|
| pair_rmse [eV] | +3.49e-03 | +4.16e-03 | +2.07e-03 |
| pair_bias [eV] | +3.47e-03 | +4.12e-03 | +2.05e-03 |
| pair_mr [eV] | +4.12e-04 | +5.49e-04 | +2.80e-04 |
| E_chg_mean [meV/at] | -3.8198 | -3.5585 | -3.5052 |
| E_neu_mean [meV/at] | -3.8366 | -3.5784 | -3.5151 |
| F_chg [eV/A] | -1.77e-05 | -1.83e-05 | -9.96e-06 |
| F_neu [eV/A] | +1.73e-05 | +2.18e-05 | +1.10e-05 |

(baselines: pair_rmse 4.929487 eV, pair_bias 4.871605, pair_mr 0.753193,
E_chg_mean +17.6658 meV/at, E_neu_mean -5.8685, F_chg 2.924980e-02,
F_neu 3.304054e-02.)

NO ARM IMPROVES THE PAIRED ENERGY. All three move it slightly the wrong way,
by 0.04-0.08% of a 4.93 eV gap, and not monotonically in the force weight (B is
worse than A at step 100), so there is no force-weight trend in the pair at
all.

THE INTERVENTION DID TAKE EFFECT, which is what makes the null result readable:
the force metrics respond to the weight in the expected order, A improving
F_chg by 1.77e-05 against C's 9.96e-06, i.e. the arm with forces on improves
its own metric about 1.8x more.

WHAT THE HEAD ACTUALLY DID, in every arm including C with forces entirely off:
it moved the COMMON energy level and left the difference pinned.

| arm | common shift | differential | ratio |
|---|---|---|---|
| A | -3.8282 meV/at | +0.0168 | 228x |
| B | -3.5685 | +0.0199 | 179x |
| C | -3.5102 | +0.0099 | 354x |

The decomposition closes: differential x 207 / 1000 reproduces the pair_bias
change to 0.7-0.9% in every arm. And because the two states' errors have
opposite signs (+17.67 and -5.87 meV/at), a common downward shift necessarily
helps the charged frame and hurts the neutral one -- E_chg_rms falls by 3.43 to
3.74 while E_neu_rms rises by 3.47 to 3.79.

The trajectory is also not monotone: the common shift is -9.79 meV/at at step
20, -8.52 at 50 and -3.82 at 100 for arm A, so 100 steps is a transient in
which the common level oscillates while the pair stays flat.

READING, by the criteria fixed in advance: this is the third case. Even with the
force constraint removed entirely, the energy supervision moves the common
level 354x more than the paired difference. The force weight is therefore not
the obstacle, and the next target is how the energy head distinguishes the two
charge states -- not a weight.

This is consistent with, but stronger than, the local gradient picture
(||u_sum||/||u_diff|| = 5800 at this state): the realised 354x is what survives
Adam's per-coordinate rescaling and the loss structure, measured on held-out
pairs rather than inferred from a norm ratio.

CAVEATS kept explicit. The checkpoint holds EMA weights while the Adam moments
belong to the raw trajectory, so this is a controlled short experiment from the
closest available state, not a replay. 100 steps is not training, and the
non-monotone trajectory means a longer run could differ. What is excluded is
narrow and specific: lowering the force weight, on its own, from this state,
does not make the paired charging energy move.

### energy-force consistency: a stable z-axis derivative gap of 70-350 meV/A

2 NiN44 val pairs, charged and neutral each, both baseline paths, h =
0.005/0.01/0.02, 6 largest-|F_DFT| atoms per frame. MACE_PB1D_DFORCE unset
throughout, solvent flags on in every arm, so each arm's FD is the slope of its
own energy.

GATES, all passed and all measured rather than assumed:
  E(no_grad) == E(grad) to 2e-12..3e-10 eV in all 8 cells
  every displaced forward performed 2 PB solves (no cache short-circuit; the
    per-sid profile cache is empty and geometry-independent anyway)
  solver provenance: n_outer 8 of a cap of 12, 0 of 24 at the cap, rms_last
    max 5e-12 against tol 1e-3, every exit reason 'tol' for both the fixed-point
    and the Newton loop -- converged on the criterion, not on the cap
  the planar warm-up path cannot fire here: it is gated on training=True and
    every forward is training=False

| sid | state | path | auto-FD | auto-DFT | FD-DFT | h-drift | x | y | z |
|---|---|---|---|---|---|---|---|---|---|
| 28 | charged | train | 171.55 | 39.72 | 200.76 | 11.37 | 20.18 | 19.99 | 295.77 |
| 28 | charged | deploy | 347.93 | 40.46 | 334.97 | 25.15 | 30.80 | 19.13 | 601.55 |
| 628 | neutral | train | 329.52 | 27.71 | 329.11 | 11.37 | 23.75 | 20.71 | 569.88 |
| 628 | neutral | deploy | 69.91 | 25.89 | 77.84 | 11.37 | 30.62 | 18.93 | 115.61 |
| 30 | charged | train | 187.46 | 35.78 | 179.50 | 12.45 | 31.10 | 25.97 | 322.18 |
| 30 | charged | deploy | 322.74 | 39.50 | 334.43 | 23.16 | 27.08 | 16.67 | 558.08 |
| 630 | neutral | train | 327.19 | 33.01 | 314.04 | 12.40 | 22.32 | 25.60 | 565.67 |
| 630 | neutral | deploy | 68.25 | 36.42 | 79.64 | 12.40 | 28.37 | 22.38 | 112.55 |

(meV/A; the last three columns are the per-component rmse of auto - FD.)

THE GAP IS REAL: auto-FD is 68-348 meV/A against a step-size drift of 11-25,
a median factor of 14. This is the first row of the reading table -- the
derivative disagrees with that path's own forward energy.

IT IS THE SOLVENT AXIS: the z component carries 113-602 meV/A while x and y sit
at 20-31, a median factor of 17. x and y are themselves comparable to the drift,
so no gap is established there.

F_auto IS THE ACCURATE ONE: auto-DFT is 26-40 meV/A in all 8 cells while FD-DFT
is 78-335. The force the model returns matches DFT; the slope of its own energy
does not.

TWO CONSEQUENCES OF THAT, stated carefully.
  The FD-DFT and auto-FD columns are NOT independent here. Because auto is
  close to DFT, FD-DFT is numerically almost the same quantity as auto-FD in
  every cell. This test yields one finding, not two.
  Since FD measures the true slope and auto measures the graph's, a position
  dependence that is detached from the graph appears in FD and not in auto. So
  the energy carries a z-dependence of order 100-600 meV/A that the autograd
  does not propagate, and removing it (i.e. using auto) is what agrees with
  DFT. A plausible reading is that forces_weight 100 trained the autograd force
  onto DFT while the undifferentiated part of the energy stayed unconstrained
  and drifted -- that is a hypothesis, not something this test establishes.

PATH AND CHARGE DEPENDENCE, reproducible across both geometries: for the
charged frames the deployment path has the larger gap (348, 323 against 172,
187); for the neutral frames the training path does (330, 327 against 70, 68).
Two of two in each direction.

LIMITS. Six largest-|F_DFT| atoms per frame, chosen to expose a gap: this does
not establish that every atom's derivative is correct, and no full scan was
run. A consistency test also cannot separate an energy-expression approximation
from trained-parameter error, so nothing here assigns the gap a cause. Next
step is the term-by-term localisation the user specified: 1-D solvent response,
dipole correction, 3-D residual charge, cavity, baseline coupling.

### derivative gap, located by term and inside E_bl  (code e815a47)

Per-term split, 8 cells (2 NiN44 val pairs x charged/neutral x training-cache
and deployment baselines), h = 0.01 A, 6 largest-|F_DFT| atoms, D_k = F_auto,k
- F_FD,k. Four closures in every cell: terms sum to the model's energy
(3.4e-12 eV), per-term autograd forces sum to the returned force (2.3e-11
eV/A), per-term FD sums to total FD (4.1e-08 meV/A), and sum_k D_k = D_total
component by component (4.1e-08 meV/A).

GRAPH-DISCONNECTED TERMS (autograd force exactly zero, FD non-zero):
baseline_coupling_energy_g in all 8 cells; cavity_energy_g in the 4
training-cache cells only; e0 by construction. No term is "in the graph but
not reaching positions".

E_bl IS THE LARGEST SINGLE-TERM GAP IN ALL EIGHT CELLS, by RMS over all 18
components: 340.5-414.1 meV/A against the next term at most 184.1.

FIXING IT ALONE, measured as RMS(D_total - D_bl) against RMS(D_total):

| cell | RMS(D_total) | RMS(D_total - D_bl) | change |
|---|---|---|---|
| 28 charged train | 171.55 | 195.05 | +13.7% |
| 28 charged deploy | 347.93 | 66.80 | -80.8% |
| 628 neutral train | 329.52 | 28.09 | -91.5% |
| 628 neutral deploy | 69.91 | 9.74 | -86.1% |
| 30 charged train | 187.46 | 215.94 | +15.2% |
| 30 charged deploy | 322.74 | 63.27 | -80.4% |
| 630 neutral train | 327.19 | 15.99 | -95.1% |
| 630 neutral deploy | 68.25 | 8.98 | -86.8% |

It lowers the total gap in six cells and RAISES it by 13.7% and 15.2% in the
two charged training-cache cells, because there E_bl partly cancels the other
missing responses -- solute electrostatics alone is 154.4 and 152.1 there.
These are two different statements (largest single term; effect of removing
it) and are recorded as such, not collapsed into "dominant in six of eight".

THE TWO RESPONSES INSIDE E_bl, by finite difference at fixed partner using
diagnostic exports captured at the e_bl_raw site (closures: recomputation vs
site value 1.7e-16 eV, vs baseline_coupling_energy_g 1.7e-16 eV, FD_rho +
FD_phi vs full FD 0.012 meV/A):

| cell | Phi_b d(rho_solv)/dR, z sum | rho_solv d(Phi_b)/dR, z sum | d(Phi_b) share |
|---|---|---|---|
| 28 charged train | -3299.87 | +0.00 | 0.00% |
| 28 charged deploy | +1035.24 | -24.36 | 1.29% |
| 628 neutral train | -3207.38 | +0.00 | 0.00% |
| 628 neutral deploy | +240.72 | -4.82 | 1.30% |
| 30 charged train | -3642.10 | +0.00 | 0.00% |
| 30 charged deploy | +2723.16 | -35.18 | 1.79% |
| 630 neutral train | -3152.24 | +0.00 | 0.00% |
| 630 neutral deploy | +393.28 | -9.34 | 2.18% |

The share column is max|FD_phi| over max|FD_full| across all 18 components
(for 28 deploy: 14.02 / 1086.45), NOT a ratio of the signed z sums, which
cannot reproduce it. Frozen per-sid baseline: d(Phi_b)/dR is exactly zero by
that path's definition, as expected in 4/4. Runtime baseline: 1.3-2.2%, as
expected non-zero in 4/4. The missing derivative is therefore almost entirely
Phi_b d(rho_solv)/dR -- the solvent charge's response to atomic displacement
through the PB solve -- with the baseline-potential response a 1-2% correction
that exists only on the deployment path.

Mechanism in source: pb1d_backend.py forms
e_bl_raw = ((rho_ion_z + rho_bound_z) * (-pbz_s)).sum() * dz * area and then
e_bl_t = e_bl_raw if live_resp else e_bl_raw.detach(), with live_resp set only
by MACE_PB1D_DFORCE. Under the default the value enters the energy while
neither the force nor the energy loss can pass a gradient through it.

### second-order check: the force-loss gradient to solve-reaching parameters is wrong under the production adjoint  (code c7f5229, job 3432527)

Directional FD in parameter space along a fixed random unit direction v, per
group, sid 28 (charged) and 628 (neutral), training-cache path, LIVE_POS=1,
_pb1d_epoch=39. Autograd at GRAD_PASSES=1 (the analytic adjoint with a frozen
coupled Jacobian J_c, what trains) and =0 (fully unrolled, kept in the code as
the exact baseline). L_F = mean (F - F_DFT)^2, the training form. Every
perturbed forward re-solved (2 solves each, counted).

CONTROL PASSED: local_electron_energy does not reach the solve and agrees at
both orders in both frames (v.dE 1.1e-6 / 1.3e-4 relative, v.dL_F 3.0e-3 /
2.0e-6, all within the FD's own drift). The script is sound.

FIRST ORDER (v.dE), identical at gp=1 and gp=0 in every row as IFT predicts:

| group | sid 28 rel err | sid 628 rel err |
|---|---|---|
| pb1d_head | 8.2e-06 | 5.1e-08 |
| products (trunk) | 7.6e-03 | 1.7e-02 |
| field_dependent_charges_maps | 2.9e-01 | 2.5e-01 |

The density-coefficient maps carry a 25-29% FIRST-order energy-gradient error
at BOTH gp settings with FD drift of 2.4e-6 / 3.6e-3, so it is not the solver.
It is an upstream truncation on the density path, and the candidate named in
the source is the scheme-C detached SCF profile features: the density maps
change the solvent, the solvent enters node_feats through prof_feat, and that
path is cut by design. Quantified here, not fixed; outside the E_bl scope.

SECOND ORDER (v.dL_F), the user's concern, confirmed and large:

| group | frame | auto gp=1 | auto gp=0 | FD | gp=1 rel err | gp=0 rel err |
|---|---|---|---|---|---|---|
| pb1d_head | 28 | -4.629e-05 | +2.692e-05 | +2.720e-05 | 2.70 (sign wrong) | 1.0e-02 |
| pb1d_head | 628 | -8.459e-06 | -3.530e-05 | -3.530e-05 | 0.76 | 5.7e-07 |
| density maps | 28 | +2.284e-03 | -1.010e-04 | -6.588e-05 | 35.7 (sign wrong) | 0.53 |
| density maps | 628 | +2.890e-03 | -8.272e-05 | -8.177e-05 | 36.3 (sign wrong) | 1.2e-02 |
| products | 28 | -8.039e-06 | -1.068e-06 | -9.606e-07 | 7.37 | 0.11 (FD drift 0.17) |
| products | 628 | +8.207e-05 | -6.785e-07 | -1.210e-06 | 68.8 (sign wrong) | 0.44 |

Under the production adjoint the force-loss gradient to pb1d_head is 76-270%
wrong (sign wrong on the charged frame), to the density-coefficient maps 36x
wrong with the wrong sign in both frames, and to the trunk 7-69x wrong with the
wrong sign on the neutral frame. The unrolled graph recovers pb1d_head to
1e-2 / 6e-7 and the density maps to 1.2% on the neutral frame; its residual
errors (53% on sid 28 density maps, 11-44% on the trunk, where the trunk FD
itself drifts 17%) are consistent with the same upstream scheme-C truncation
and with FD noise on values of order 1e-6, and are not attributable to J_c.

WHAT THIS MEANS. forces_weight is 100. For every parameter group that reaches
the PB solve, the training signal from the force loss has been pointing in a
direction that is not the gradient of the force loss -- in sign as well as
size on the groups measured. The energy gradient (first order) is exact for
pb1d_head, 1-2% off for the trunk, and 25-29% off for the density maps.

REMEDY CANDIDATES, to be decided on measured cost: (a) GRAD_PASSES=0, the
existing exact unrolled path -- its time and memory at training scale are
being measured; (b) make J_c theta-differentiable and iterate the differentiable
step so phi_star carries first-order theta dependence -- more code, second-order
correct to the needed order, cheaper if (a) is not affordable.

### LIVE_POS acceptance: deployment path closed, training-cache path partial  (code d51cc7b, jobs 3432525/3432532)

Value-identity gate PASSED first: same frames, switch off vs on in one process,
both paths, with and without compute_force, 16 cells: worst forward-value
difference 2.567e-10 eV over the total energy and every term. Forces moved by
253-1831 meV/A max, as intended.

Per-term RMS(D_k) over all 18 components, meV/A, LIVE_POS 0 -> 1:

| cell | D_total | E_bl | solute ES | comp 1D | slab dip | local_e |
|---|---|---|---|---|---|---|
| 28 train | 171.5 -> 22.8 | 366.3 -> 54.8 | 154.4 -> 154.4 | 56.5 -> 4.5 | 6.3 -> 63.0 | 14.7 -> 14.7 |
| 28 deploy | 347.9 -> 6.3 | 414.1 -> 5.0 | 3.4 -> 3.4 | 67.4 -> 0.7 | 1.1 -> 1.4 | 0.4 -> 0.4 |
| 628 train | 329.5 -> 70.9 | 354.3 -> 37.7 | 42.0 -> 42.0 | 181.8 -> 2.5 | 125.4 -> 2.1 | 16.0 -> 16.0 |
| 628 deploy | 69.9 -> 6.5 | 77.6 -> 0.5 | 1.8 -> 1.8 | 11.8 -> 0.0 | 1.3 -> 0.0 | 0.4 -> 0.4 |
| 30 train | 187.5 -> 9.0 | 402.6 -> 92.1 | 152.1 -> 152.1 | 96.6 -> 20.5 | 12.7 -> 52.4 | 15.2 -> 15.2 |
| 30 deploy | 322.7 -> 7.2 | 385.2 -> 4.6 | 2.2 -> 2.2 | 62.6 -> 1.0 | 0.8 -> 0.8 | 0.3 -> 0.3 |
| 630 train | 327.2 -> 75.5 | 340.5 -> 32.2 | 41.4 -> 41.4 | 184.1 -> 2.0 | 128.8 -> 2.1 | 16.2 -> 16.2 |
| 630 deploy | 68.3 -> 5.8 | 75.7 -> 0.5 | 2.0 -> 2.0 | 11.8 -> 0.0 | 1.5 -> 0.1 | 0.4 -> 0.4 |

(cavity 0-10.2 and solvent3D 4.3-21.9 essentially unchanged; energy head
1.3-1.5 is the connected-term floor.)

DEPLOYMENT PATH: ACCEPTED. D_total 5.8-7.2 and E_bl's own D_k 0.5-5.0, both
at or below the 11-25 meV/A step-size drift measured in step 1. The returned
force is now the derivative of the energy on this path to FD resolution.

TRAINING-CACHE PATH: PARTIAL. D_total falls 4-21x (171.5->22.8, 329.5->70.9,
187.5->9.0, 327.2->75.5) but E_bl's own D_k is still 32-92 RMS with max|D|
98-281, above the drift. The acceptance criterion -- E_bl's autograd force
equal to its FD -- is met on deployment and NOT yet on the training path.

Three terms keep a gap ONLY on the training path and all three couple solvent
charge to solute potential: E_bl (32-92), solute electrostatics (152-154 on
the charged frames, 41-42 neutral, UNTOUCHED by LIVE_POS), and the slab dipole
correction, which got WORSE on the charged frames (6.3->63.0, 12.7->52.4)
while closing on the neutral ones (125->2). On deployment the same three sit
at 0.5-5.0, 1.8-3.4 and 0.0-1.4. So there is a truncation specific to the
frozen-baseline path that pf_in did not reach, and the slab dipole's growth
says a live path there now carries a derivative that disagrees with FD rather
than merely missing one. Not localised yet. The frozen baseline itself is not
the candidate: its zero position response is by that path's definition and FD
sees the same zero.

Diagnostic queued: the same split under GRAD_PASSES=0 on the four training
cells, to separate "the analytic adjoint's first-order position gradient is
incomplete on this path" from "the truncation is outside the solver".

### remedy input: the exact unrolled solver graph is affordable  (job 3432543)

One training-style step per frame (training=True, compute_force=True, the
run's loss, backward), three NiN44 train frames, LIVE_POS=1, epoch 39:

| GRAD_PASSES | frame | fwd+loss+bwd s | peak GiB | loss |
|---|---|---|---|---|
| 1 (first, warm-up) | 1 | 47.17 | 12.54 | 2.699734 |
| 1 | 2 | 1.40 | 12.74 | 2.570729 |
| 1 | 3 | 4.00 | 12.93 | 2.559273 |
| 0 | 1 | 5.48 | 12.99 | 2.699734 |
| 0 | 2 | 3.46 | 12.99 | 2.570729 |
| 0 | 3 | 1.07 | 12.99 | 2.559273 |
| 1 (repeat) | 1 | 1.60 | 12.93 | 2.699734 |
| 1 | 2 | 3.59 | 12.93 | 2.559273 |
| 1 | 3 | 1.02 | 12.93 | 2.559273 |

Loss values identical to six decimals between the two settings, as they must
be: only the gradient construction differs. Peak memory 12.99 against 12.93
GiB, +0.06 GiB (0.5%) -- the 1-D solver's unrolled graph is small next to the
3-D grids. Time per frame 1.07-5.48 s (mean 3.34) against 1.02-3.59 s (mean
2.07) on the repeat pass; with three frames and a 5.48 s first-use outlier the
ratio is bounded at ~1.6x and is probably less. At the measured 2.3 min/epoch
of the full-PB stage that is at most ~3.7 min/epoch.

So the fully unrolled graph, already in the code and FD-verified there, is an
affordable remedy for the second-order defect: no new math, values unchanged,
exact where the analytic adjoint was wrong in sign.

### the original training path: pb1d_head received EXACTLY ZERO gradient from energy and forces  (job 3432543 b)

Same second-order check, same two frames and four groups, but on the original
training configuration: MACE_PB1D_LIVE_POS unset (the default), training-cache
path, epoch 39. Control exact at both orders (1.7e-4 / 2.4e-6 and 1.3e-4 /
1.7e-7). Every perturbed forward re-solved.

| group | quantity | autograd (gp=1 = gp=0) | FD | missing |
|---|---|---|---|---|
| pb1d_head, sid 28 | v.dE | **0.000000e+00** | +4.118880e-03 | 100% |
| pb1d_head, sid 28 | v.dL_F | **0.000000e+00** | +9.958086e-08 | 100% |
| pb1d_head, sid 628 | v.dE | **0.000000e+00** | -4.056353e-03 | 100% |
| pb1d_head, sid 628 | v.dL_F | **0.000000e+00** | +4.993448e-08 | 100% |
| density maps, sid 28 | v.dE | 3.454577e-03 | 4.854779e-02 | 93% |
| density maps, sid 628 | v.dE | 2.521807e-02 | 4.611300e-02 | 45% |
| density maps, 28 / 628 | v.dL_F | 4.68e-06 / -9.14e-06 | 6.55e-06 / -1.32e-05 | 29% / 31% |
| products, 28 / 628 | v.dE | -6.160e-02 / -5.881e-02 | -6.140e-02 / -5.665e-02 | 0.3% / 3.8% |
| products, 28 / 628 | v.dL_F | -7.860e-06 / 3.051e-06 | -7.726e-06 / 2.798e-06 | 1.7% / 9.1% |

The autograd columns are identical at GRAD_PASSES 1 and 0 in every row: with
nothing flowing from the solve into the energy, the adjoint's construction is
irrelevant, and every discrepancy here is an upstream truncation.

WHY IT IS EXACTLY ZERO. pb1d_head's only route to the energy is through the
solve output -- p_off -> phi -> rho_solv -> {compensation_periodic_1d_energy,
E_bl, slab dipole}. Under the default all three are detached from the graph:
prof_energy at extensions.py:1902, e_bl_raw in the backend, solvent_dipole_e at
extensions.py:2957. So the energy and force losses could not reach the head at
all; its 8733 Adam steps were driven only by the profile/observable losses
(potential_1d_profile, rhob_1d, rho1d, solvent3d, fermi). That is consistent
with an Adam step count advancing on a zero-valued (not None) gradient.

The density-coefficient maps lost 45-93% of their energy gradient the same way
-- the solvent-coupling energy is a larger share on the charged frame -- and
29-31% of their force-loss gradient. The trunk was largely unaffected because
its dominant paths (interaction energy, direct electrostatics) never went
through the solve.

WHAT THIS ANSWERS. The charging energy lives in the solvent-coupling terms.
The parameters that shape the solvent response were structurally disconnected
from the energy label. This is the mechanism behind the earlier finding that
the native head's charging response stayed at 0.04 eV under joint training:
not a competition the force term won, but a channel the energy loss never
reached.

Under LIVE_POS=1 (job 3432527, earlier entry) the same head's energy gradient
becomes exact to 8e-6 / 5e-8; the second-order defect that appears there is the
separate frozen-J_c issue, remedied by GRAD_PASSES=0 at +0.06 GiB.

### the adjoint is not the training-path residual: GRAD_PASSES=0 changes nothing  (job 3432559)

Same 8 cells, LIVE_POS=1, GRAD_PASSES 1 -> 0. Every term's RMS(D_k) and every
D_total identical to the printed 0.1 meV/A (E_bl 54.8->54.8, 37.7->37.7,
92.1->92.1, 32.2->32.2 on the training cells; solute ES 154.4->154.4;
slab dipole 63.0->63.0, 52.4->52.4). The analytic adjoint's FIRST-order
position gradient is therefore exact on both paths, as IFT predicts and as the
parameter-space check already showed. The residual training-path gap is
upstream of the solver.

Where to look, stated as a candidate to be TESTED, not a conclusion: the
frozen-baseline branch computes the solvent-center quantities (z_e50, w_e50,
solv_center) at extensions.py:2665-2670 from radial_coefficients.detach() and
positions.detach(). If that center enters the 1-D solvent placement, its FD
response is present and its autograd response absent, on that branch only --
which would touch exactly the three residual terms (E_bl, solute
electrostatics, slab dipole). A generic scan for outputs that move under FD
but carry no grad_fn decides this without guessing.

### the training-path residual, localised: the SCF charges' position response, truncated behind scheme C  (jobs 3432586, 3432602)

MEASURED (partial-truncation scan: for every connected output and a fixed
random contraction, autograd d(v.out)/dR_z at the largest-|F| atom against the
FD of the same scalar, h = 0.01 A, LIVE_POS=1):

| output | 28 train auto / FD | 628 train auto / FD | 28 deploy auto / FD | 628 deploy auto / FD |
|---|---|---|---|---|
| explicit_potential(_base) | 1.77e-2 / 1.79e-1 | -1.7e-3 / 3.98e-1 | -1.77e-2 / -1.13e-2 | -1.7e-3 / 2.26e-2 |
| explicit_dipole | -1.59e-2 / -1.57e-1 | 6.1e-3 / -1.71e-1 | -2.6e-3 / -4.5e-3 | 4.9e-3 / -9.9e-3 |
| density_coefficients (SCF charges) | -2.24e-2 / -6.45e-2 | -- | 1.07e-2 / 1.22e-2 | -- |
| electrostatic_energy | -1.25e-2 / 1.75e-1 | 1.47e-2 / 6.59e-2 | 1.25e-2 / 2.00e-2 | 1.47e-2 / 2.00e-2 |
| compensation_slab_correction | -6.08e-2 / 1.58e-2 | -1.50e-1 / -1.53e-1 | -9.1e-3 / -6.1e-3 | 2.1e-3 / 2.3e-3 |
| baseline_coupling E_bl | 3.84e-1 / 4.51e-1 | -3.91e-1 / -4.37e-1 | -8.41e-1 / -8.31e-1 | -2.22e-1 / -2.23e-1 |
| solvent_potential | -2.171 / -2.200 | -2.168 / -2.181 | 1.93e-2 / 2.06e-2 | -3.07e-2 / -3.13e-2 |
| solvent_profile_features | -2.3e-5 / -3.2e-5 | -8.6e-5 / -8.2e-5 | 3.6e-6 / 2.1e-6 | 2.3e-5 / 2.5e-5 |

(eV/A for energies and potentials; dipole in e; the profile row is a random
contraction of 1024 features.)

On the training path explicit_potential and explicit_dipole have a position
response 10-40x larger than on deployment, and autograd captures <= 10% of it
(sid 628: essentially zero, wrong sign). electrostatic_energy and the slab
correction inherit that with the wrong sign on the charged frame; E_bl inherits
10-15%. solvent_potential and solvent_dipole -- the solve's own outputs -- are
within 1.3% on both paths: the solve is not the carrier. The first scan
(3432586) flagged cavity_energy_g as detached-but-moving on the training path
only (FD 3.9-6.5e-3), which localises its 4-10 meV/A residual.

READ FROM SOURCE:
  explicit_potential_base = predict_potential_from_dipole_and_solvent_layer(
      dipole=explicit_dipole, center=solv_center [or z_e50], cell=cell.detach())
                                                          (extensions.py 2762, 2885, 2895)
  explicit_dipole = sum_i positions_i * charge_coeff_i   (no detach inside)
  so the dipole's missing response is the SCF charges' missing response;
  density_coefficients is 65% truncated on the training path against 12% on
  deployment. explicit_potential_base also enters graph_feats_global (2775)
  and the interface pooling uses z_axis = positions[:, axis].detach() (2749)
  and z_center = z_e50.detach() (2750); solv_center / z_e50 come from
  compute_density_threshold_crossing_from_baseline_profile with
  radial_coefficients.detach() and positions.detach() (2665-2670); the SCF
  profile features are prof_feat = (...).detach() (1893, scheme C, by design).
  On the frozen-baseline path n_e = neutral_v(frozen, reference geometry) -
  net(displaced), so a displaced atom's neutral reference stays behind.

INFERRED, not proven: that last construction gives the electron density an
artificial position response on the training path that is absent on
deployment (where neutral_v follows pf_in through the runtime tables), and it
propagates through the detached SCF-feature and centre paths into the charges.
That is consistent with the 5-15x larger FD responses on the training path and
with autograd missing most of them, but the share attributable to each detach
site has not been measured.

CONSEQUENCE FOR ACCEPTANCE. E_bl's autograd force equals its FD on the
deployment path (0.5-5.0 meV/A) and does NOT on the training-cache path
(32-92), and the remaining carrier is the scheme-C SCF-feature truncation plus
the detached centre and z-axis inputs -- all shared with solute electrostatics
and the slab correction. Reconnecting them means a differentiable stage-1 solve
and live centre/interface geometry; the source keeps them detached
deliberately, so that is a design decision, not a bug fix, and is not taken
here.

### two corrections to earlier entries, and the stage-1 reconnection  (code a2a2523)

CORRECTION 1. The zero-gradient finding (job 3432543 b) explains why pb1d_head
did not adjust to the energy target. Extending it to local_electron_energy's
0.04 eV charging response was a step too far: that is a different head, and
the earlier gradient audit measured a non-zero energy gradient on it. How much
of the ~5 eV charging error comes from these truncations is for the post-fix
joint training to show, not for this audit to assert.

CORRECTION 2. The partial-truncation scan drew its random projection vector
from a generator that advanced between paths, so each path's autograd-vs-FD
comparison used a DIFFERENT v. The per-path AD-FD disagreements stand; the
"65% vs 12%" is not a fraction of the whole charge response lost and is not a
quantitative cross-path comparison. It supports fixing the proven truncation;
it does not judge the frozen baseline's physics.

WHAT IS RECONNECTED, in computational order, all under MACE_PB1D_LIVE_POS
(values unchanged; the user's configuration is LIVE_POS=1, GRAD_PASSES=0,
DFORCE unset, keeping the existing baseline):
  stage 1 (extensions.py _pb1d_stage1): want_grad follows the switch. Under
    fresh_stage1 this pass is a fresh prior-only solve on the current geometry
    every forward -- it never reads a previous encounter -- so keeping its
    graph changes no algorithm. Its density input comp_charge_density is live
    at its source and was cut only by want_grad=False.
  the SCF's solvent-profile features (2165): take profile_features_grad, the
    live twin the code already computes at 1907-1910, instead of the detached
    prof_feat of 1893.
  the stage-1 dipole into the slab-correction features (2184): passed live
    instead of re-detached.
  Read and confirmed detach-free on these inputs:
    periodic_profile_layer_potential_field_nodes, the slab-correction feature
    function, and their entry into the field-dependent charges.

NOT touched, next by dependency order: comp_center_init (2140), the stage-1
planar centre built from detached density and positions; then the post-SCF
centre / interface-pooling detaches (2665-2670, 2749-2750), which sit after the
charges form and cannot explain their missing response; and cavity_energy_g,
detached-but-moving on the training path only.

ACCEPTANCE reuses the existing scripts in one job (3432680): value identity
under GRAD_PASSES=0, the 8-cell term split on both paths, the second-order
parameter check (pb1d_head must now receive gradient; the density maps' 25-29%
first-order error was attributed to this truncation), and the training-step
cost -- stage 1 now carries a graph, so the +0.06 GiB figure is void.

### stage-1 reconnection: acceptance  (code a2a2523, job 3432680, LIVE_POS=1 GRAD_PASSES=0)

1. VALUE IDENTITY: passed. Switch off vs on, both paths, with and without
   compute_force, 16 cells: energies and every energy term identical below
   1e-9 eV. The original checkpoint's paired charging energy and its ~5 eV
   error are therefore unchanged by construction.

2. PER-TERM FORCE vs FD, RMS over 18 components (meV/A), before -> after:

| cell | D_total | E_bl | solute ES | slab dip | comp 1D | local_e | cavity | solv3D |
|---|---|---|---|---|---|---|---|---|
| 28 train | 22.8 -> 7.5 | 54.8 -> 0.0 | 154.4 -> 0.1 | 63.0 -> 0.0 | 4.5 -> 0.0 | 14.7 -> 0.0 | 4.5 -> 4.5 | 9.0 -> 9.4 |
| 628 train | 70.9 -> 20.1 | 37.7 -> 0.0 | 42.0 -> 0.1 | 2.1 -> 0.0 | 2.5 -> 0.0 | 16.0 -> 0.0 | 8.4 -> 8.4 | 20.4 -> 13.5 |
| 30 train | 9.0 -> 5.9 | 92.1 -> 0.2 | 152.1 -> 0.1 | 52.4 -> 0.0 | 20.5 -> 0.1 | 15.2 -> 0.0 | 10.2 -> 10.2 | 6.2 -> 13.3 |
| 630 train | 75.5 -> 10.8 | 32.2 -> 0.0 | 41.4 -> 0.1 | 2.1 -> 0.0 | 2.0 -> 0.0 | 16.2 -> 0.0 | 3.6 -> 3.6 | 14.5 -> 9.8 |
| 28 deploy | 6.3 -> 6.3 | 5.0 -> 3.3 | 3.4 -> 0.1 | 1.4 -> 0.0 | 0.7 -> 0.5 | 0.4 -> 0.0 | 0.1 -> 0.0 | 4.6 -> 4.6 |
| 628 deploy | 6.5 -> 5.5 | 0.5 -> 0.5 | 1.8 -> 0.1 | 0.0 -> 0.0 | 0.0 -> 0.1 | 0.4 -> 0.0 | 0.0 -> 0.0 | 4.9 -> 5.0 |
| 30 deploy | 7.2 -> 7.0 | 4.6 -> 2.8 | 2.2 -> 0.1 | 0.8 -> 0.0 | 1.0 -> 0.4 | 0.3 -> 0.0 | 0.1 -> 0.0 | 6.2 -> 6.1 |
| 630 deploy | 5.8 -> 4.7 | 0.5 -> 0.5 | 2.0 -> 0.1 | 0.1 -> 0.0 | 0.0 -> 0.1 | 0.4 -> 0.0 | 0.0 -> 0.0 | 4.3 -> 4.3 |

   E_bl ACCEPTED on both paths: max|D| 0.03-0.77 meV/A on the training path,
   1.5-8.7 on deployment (within the 11-25 step-size drift; the deployment
   residual on the charged frames is of the size of the rho_solv d(Phi_b)/dR
   piece measured earlier, 14-19 max). The SCF charges' position response is
   restored: solute electrostatics 41-154 -> 0.1, slab dipole 52-63 -> 0.0,
   comp 1D -> <= 0.1, local_electron 15-16 -> 0.0, all on the training path
   where they had stayed after LIVE_POS alone. Remaining, both paths: cavity
   (training path only, 3.6-10.2, the known detach) and solvent3D (4.3-13.5,
   altered by the fix -- 20.4 -> 13.5, 6.2 -> 13.3 -- but not closed: a
   partial truncation of its own). D_total is now those two.

3. SECOND ORDER, read per the user's frame:
   regression -- pb1d_head stays exact: v.dE 4.4e-05 / 1.5e-07, v.dL_F at
     GRAD_PASSES=0 1.1e-03 / 3.1e-07. Not credited to this patch.
   not a criterion -- density maps' first-order energy-gradient error is
     UNCHANGED to four digits (0.2923 / 0.2463 before and after):
     field_dependent_charges_maps is applied after the recursion, downstream
     of stage 1, so this 25-29% is a downstream truncation and the next item.
     A parameter derivative is cut where a parameter-dependent quantity is
     detached; the centres built from detached density (comp_center_init at
     2140; z_e50 / solv_center at 2665-2670) fit that.
   trunk v.dL_F at GRAD_PASSES=0: 11% -> 1.2%, 44% -> 4.4%. Control exact.

4. COST, stage 1 in the graph: peak 16.45 GiB at GRAD_PASSES=0 (16.33 at 1),
   against 12.99 / 12.93 before -- +3.4 GiB (+26%), fine on 40 GB; time per
   frame mean 3.65 s (gp=0) / 2.35 s (gp=1) against 3.34 / 2.07, i.e. +10-15%.
   The full training loss ROSE with energies identical (frame 1: 2.699734 ->
   3.737508). The expected cause is the force term -- the reconnected force
   is farther from DFT, as anticipated -- but the observable terms were not
   individually gated; loss_terms_identity.py does that before any A/B.

NEXT, by dependency: comp_center_init (2140) -> post-SCF centres (2665-2670,
2749-2750) -> cavity (training path) -> solvent3D's partial truncation; then
the joint A/B with LIVE_POS=1, GRAD_PASSES=0, DFORCE unset.

### cavity and solvent3D reconnection: acceptance  (code 20a0fbb; audits b0ace4d, 28190fc, 51cd604; jobs 3432886, 3432998, 3433193, 3433228; LIVE_POS=1 GRAD_PASSES=0)

WHAT CHANGED (20a0fbb, pb1d_backend.py _stage2_energy): ne_cav = ne2 stays live
under LIVE_POS (it was ne2.detach() on the training path, so the cavity and the
envelopes saw no position response through the SCF charges), and delta_b /
delta_i are no longer detached before the solvent3D energy unless neither
MACE_S3D_LIVE_DELTA nor LIVE_POS is set. Values untouched by construction (1).

1. VALUE IDENTITY against a MEASURED floor (3433193; 16 cells = 4 sids x
   train/deploy x force off/on). The fixed 1e-9 eV gate of the earlier runs sat
   below the iterative solve's own run-to-run scatter, so the gate is now 10x
   the on-vs-on floor measured in the same job. Worst off-vs-on |dE| 7.31e-10
   eV (sid 28 train, force=0), worst on-vs-on floor 9.08e-10 eV, largest
   per-cell ratio 5.8, threshold 9.08e-9 eV. Energies -1306.584202658 /
   -1305.015926664 / -1306.408789702 / -1304.511542449 eV on the training
   path, identical off and on to the digits shown.

2. PER-TERM FORCE vs FD (3432886), RMS over all 18 components, meV/A, h =
   0.01 A, before (a2a2523, job 3432680) -> after (20a0fbb):

| cell | D_total | cavity | solv3D | E_bl | energy head |
|---|---|---|---|---|---|
| 28 train | 7.53 -> 1.47 | 4.51 -> 0.00 | 9.37 -> 0.00 | 0.02 -> 0.02 | 1.47 -> 1.47 |
| 628 train | 20.10 -> 1.47 | 8.41 -> 0.01 | 13.54 -> 0.00 | 0.01 -> 0.01 | 1.47 -> 1.47 |
| 30 train | 5.87 -> 1.35 | 10.24 -> 0.00 | 13.32 -> 0.01 | 0.18 -> 0.18 | 1.35 -> 1.35 |
| 630 train | 10.78 -> 1.31 | 3.63 -> 0.01 | 9.75 -> 0.01 | 0.02 -> 0.02 | 1.35 -> 1.35 |
| 28 deploy | 6.28 -> 3.35 | 0.00 -> 0.00 | 4.63 -> 0.00 | 3.30 -> 3.30 | 1.47 -> 1.47 |
| 628 deploy | 5.54 -> 1.49 | 0.00 -> 0.00 | 4.96 -> 0.00 | 0.54 -> 0.54 | 1.47 -> 1.47 |
| 30 deploy | 7.00 -> 2.76 | 0.00 -> 0.00 | 6.10 -> 0.00 | 2.78 -> 2.78 | 1.35 -> 1.35 |
| 630 deploy | 4.68 -> 1.40 | 0.01 -> 0.01 | 4.30 -> 0.00 | 0.52 -> 0.52 | 1.35 -> 1.35 |

   Closures in every cell: sum of terms = model energy to 3.4e-12 eV; sum of
   per-term autograd forces = returned force to <= 4.4e-8 eV/A; sum of
   per-term FD = total FD to <= 3.9e-8 meV/A; sum_k D_k = D_total component
   by component to <= 9.0e-6 meV/A. Cavity and solvent3D are CLOSED on both
   paths; the training-path D_total drops 4-14x. E_bl is unchanged in all
   eight cells, as it must be (this patch does not touch it). What remains on
   the training path, 1.31-1.49, is the energy-head (interaction energy) term:
   identical in all four cells of a geometry (charged/neutral, train/deploy),
   so it carries no solvent dependence; whether it is FD truncation at h =
   0.01 A on the short-range network was not measured for that term alone.
   Deployment keeps E_bl on the charged frames at 3.30 / 2.78 (max |D| 8.7 /
   7.5), the rho_solv d(Phi_b)/dR piece recorded under LIVE_POS acceptance;
   deployment is not the training path and was not the target here.

3. DENSITY-MAP PARAMETER GRADIENT, split by energy term (3433228 at 51cd604;
   the same section in 3432886 crashed because E0 has no grad_fn -- the script
   now sets AD_k := 0 for such a term and prints which). Same random direction
   v as the second-order check (FD(E) identical: 4.854780e-02 / 4.611300e-02),
   so this is the direct before -> after of the 29.2% / 24.6% first-order
   error recorded in the stage-1 entry:

   sid 28, field_dependent_charges_maps (75816 params): AD 4.854760e-02 vs FD
     4.854780e-02, D_total -1.93e-07 = -4.0e-6 relative (before: auto
     3.435951e-02, -29.2%). Per term: E_bl -2.8e-07, cavity +6.1e-08, comp 1D
     +4.5e-08, solv3D -1.3e-08, the rest <= 6e-09; sum_k D_k = D_total to
     8.4e-12; every D_k is below that term's own FD drift.
   sid 628, same group: AD 4.594609e-02 vs FD(eps 1e-4) 4.611300e-02, D_total
     -1.67e-04 = -0.36% (before: auto 3.475703e-02, -24.6%). 100.0% of it is
     the solv3D term, whose FD_k drifts 1.66e-2 relative between eps = 1e-4
     and 3e-5; against the eps = 3e-5 FD of the same direction (4.594602e-02,
     job 3432680's table) autograd agrees to 1.5e-6 relative. The residual is
     the FD reference's truncation on that term, not the derivative.
   products (126336 params; v drawn second here, so a different direction
     from the second-order check): D_total -2.05e-08 (-1.7e-6 relative) /
     +2.14e-10 (+1.5e-8 relative); the energy head's own gradient through
     products, 1.35507e-02, agrees to 6e-13.
   So the 25-29% first-order error WAS the cavity + solvent3D detach
   (ne2.detach() and delta_b/delta_i.detach() in _stage2_energy), downstream
   of stage 1 as the previous entry predicted, and it is closed. The
   second-order quantity v.dL_F is being re-measured at 51cd604 with the same
   script (job 3433245); before, at GRAD_PASSES=0: density maps 26.8%,
   products 1.2%, pb1d_head 0.11%, control 0.018%.

4. LOSS-TERM IDENTITY (3432998, code 28190fc: both arms scored on ONE sampled
   batch; the first version rebuilt the batch per arm, and the unseeded
   sample-point draw made density_3d / solvent3d look changed). LIVE_POS off
   -> on, three NiN44 train frames, training=True, GRAD_PASSES=0: all eleven
   non-force terms identical to <= 5.2e-11 (fermi_level, frame 1); the force
   term 0.0803 -> 3.1629, 0.0662 -> 2.0967, 0.0793 -> 1.9089 (x39, x32, x24);
   TOTAL 0.660 -> 3.742, 0.666 -> 2.696, 0.698 -> 2.528. Controls (frame 3,
   off): same batch twice, density_3d 1.2e-15, solvent3d 2.0e-13, forces
   1.1e-12; rebuilt batch, density_3d 7.3e-2, solvent3d 7.5e-2, forces
   2.1e-12 -- the rebuilt-batch numbers are the sampling noise behind the
   earlier false reading. The loss rise recorded in the stage-1 entry
   (2.6997 -> 3.7375 on frame 1) is therefore the force term and nothing
   else: the reconnected force -- the actual slope of the energy the model
   returns -- is much farther from DFT than the truncated force was. That is
   the state of this checkpoint, trained on a force that was not its energy's
   derivative; it is not a defect of the patch.

5. COST with the full graph (3432886; three NiN44 train frames, epoch-39
   path, batch 1, fwd + loss + bwd): peak 20.58 GiB at GRAD_PASSES=0, 20.46
   at 1; before this patch 16.45 / 16.33 (stage 1 in the graph); before
   stage 1 12.99 / 12.93. Cavity + solvent3D add +4.1 GiB; the whole
   reconnection is +7.6 GiB (+58%) over the LIVE_POS-only graph, for a
   207-atom frame on a 40 GB A100. The 339-atom frames (160 of the 640
   training frames) were NOT measured and must be before any training job.
   Time per frame, mean of three (noisy, first call included): 3.86 s at
   gp=0 against 3.65 before.

All commits pushed from a compute node (3433229; gh cannot start on the login
node under its 300-thread limit while this session's daemon holds ~228).

NEXT: the joint A/B retraining agreed with the user -- arm A: LIVE_POS unset,
GRAD_PASSES default (1); arm B: LIVE_POS=1, GRAD_PASSES=0; DFORCE unset in
both; same code, seed, config, data order, loss weights and epoch budget; the
native electron head trained in both; no new correction term -- after a GPU
gate that measures arm B's peak memory on the 339-atom frames and one full
3-GPU DDP epoch on the post-warmup path.

### second order after the cavity/solvent3D reconnection: re-measured, then split by term  (jobs 3433245 @51cd604, 3433255 @ce2c1a8; LIVE_POS=1 GRAD_PASSES=0)

1. SAME SCRIPT AS BEFORE (second_order_check.py), same v, both frames,
   relative error of autograd against FD (eps 1e-4 unless noted):

| group | quantity | sid 28 before -> after (gp=0) | sid 628 before -> after (gp=0) |
|---|---|---|---|
| density maps | v.dE | 29.2% -> 3.9e-6 | 24.6% -> 0.36% (vs eps 3e-5 FD: 1.5e-6) |
| density maps | v.dL_F | 26.8% -> 7.5% | n/a: FD(L_F) drifts 103% between eps 1e-4 and 3e-5 |
| pb1d_head | v.dE / v.dL_F | 4.4e-5 / 0.11% -> 1.3e-5 / 0.039% | 1.2e-8 / 7.5e-7 |
| products | v.dE / v.dL_F | 0.76% / 1.16% -> 5.6e-7 / 0.029% | 3.4e-9 / 0.37% (FD drift 0.43%) |
| local_e (control) | v.dE / v.dL_F | 2.4e-4 / 3.0e-5 | 1.3e-4 / 8.5e-7 |

   At gp=1 (the analytic adjoint) the second-order numbers stay wrong, as
   before: density maps 8955%, pb1d_head 1329% / 450%, products 9.9% / 574%.
   GRAD_PASSES=0 stays the training setting. The absolute values moved
   because the cavity/solvent3D reconnection changed the force and hence L_F
   (sid 28 FD v.dL_F 1.411e-04 -> 6.143e-05).

2. THE REMAINING 7.5% SPLIT BY TERM (floss_grad_by_term.py: g = dL_F/dF
   frozen, Q_k = g . F_k, AD_k = v.grad Q_k, FD_k central; sum_k AD_k =
   v.dL_F exactly, sum_k FD_k = FD(L_F) + O(eps^2); eps_rel 1e-4, 3e-5,
   1e-5). Closures: sum_k AD_k vs direct 5.9e-11 / 1.3e-11; sum_k FD_k vs
   FD(L_F) 2.4e-9 / 4.8e-11 at eps 1e-5; sum_k D_k = D_total to 5.9e-11.

   sid 28, density maps: D_total -4.48e-06 = -7.31% of FD at eps 1e-5
   (-7.54% at 1e-4, so stable). Per term (D_k at eps 1e-5, share):
     cavity     AD -5.88278e-05  FD -5.43441e-05  D -4.484e-06  -100.1%  (drift 1.9e-3)
     E_bl       +6.12781e-04     +6.12773e-04     +7.9e-09      +0.2%
     solute ES  +7.04856e-05     +7.04880e-05     -2.4e-09      -0.1%
     comp 1D, slab dip, solv3D, local_e: |D_k| <= 8e-10, each below its drift.
   sid 628, density maps: at eps 1e-4 the FD is unusable (solv3D FD_k
   +3.73e-03 against -1.13e-04 at 1e-5: a 34x drift in that one term); at
   eps 3e-5 and 1e-5 it is smooth (FD(L_F) -9.861e-05 / -1.074e-04).
   D_total -3.58e-06 = -3.34% of FD at eps 1e-5, again 100.0% cavity:
     cavity     AD -3.66013e-04  FD -3.62430e-04  D -3.583e-06  -100.0%  (drift 4.7e-3)
     every other term |D_k| <= 9e-11.
   Control group local_electron_energy: only its own term has AD_k != 0
   (+1.18617e-05 / +2.49079e-05, matching FD to 8e-5 / 2e-7); all other
   FD_k are 1e-9..1e-12 (solver noise); D_total +0.06% / -0.00%.

   READING. The whole remaining second-order gap is the cavity term: its
   force's parameter derivative is 8.3% (sid 28) and 1.0% (sid 628) off,
   while its VALUE closes to 1e-12 eV, its FORCE closes to 0.00-0.01 meV/A
   (term split) and its FIRST-order parameter derivative closes to 1.8e-6
   (param_grad_by_term). The other nine terms are exact at second order.
   The 1e-30 floor under sqrt(|grad s_cav|^2) and the two clamps in
   create_cavity_torch are the candidate operations; the clamps sit where
   the shape function's slope is ~1e-51 (N_MIN 1e-4, SIGMA_K 0.6), so a kink
   reading is unlikely on paper. cavity_second_order_probe.py (job 3433298,
   code fda741c: env-gated live export of the chain's intermediates and an
   area-floor knob, defaults unchanged) measures AD vs FD stage by stage --
   ne, s_vdw, s_cav, |grad|^2, area at floors 1e-30/1e-20/1e-12, the term
   itself -- and the plateau / clamp populations at theta.

### the cavity second-order gap is the 1e-30 floor under sqrt(|grad s_cav|^2)  (job 3433298, code fda741c)

Staged AD vs FD of g . F_S for a scalar S at each stage of the cavity chain,
same frozen g and direction v as the term split, eps_rel 3e-5 and 1e-5,
3,000,000 grid points per frame:

| stage | sid 28 AD-FD rel (drift) | sid 628 AD-FD rel (drift) |
|---|---|---|
| sum w.ne (raw density, before the log) | 1.03 (1.28) -- unreadable, see below | 0.99 (0.66) -- unreadable |
| sum w.s_vdw | 8.5e-5 (8.0e-5) | 1.6e-6 (1.3e-5) |
| sum w.s_cav | 1.8e-6 (1.5e-6) | 3.6e-8 (2.1e-7) |
| sum w.|grad s_cav|^2 | 1.7e-5 (2.1e-5) | 6.5e-8 (5.3e-7) |
| area, floor 1e-30 (the backend's) | 8.25e-2 (2.9e-4) | 9.9e-3 (5.4e-3) |
| area, floor 1e-20 | 5.3e-2 (2.6e-4) | 1.2e-2 (5.5e-3) |
| area, floor 1e-12 | 2.7e-4 (8.4e-4) | 3.0e-4 (1.0e-3) |
| E_cav term | 8.25e-2 (2.9e-4) | 9.9e-3 (5.4e-3) |

The E_cav row reproduces the term split (AD -5.882783e-05 vs FD
-5.434416e-05 on sid 28). Everything up to and including |grad s_cav|^2
closes; the sqrt with its 1e-30 floor does not, and the SAME gradients with
a 1e-12 floor close to within the FD drift on both frames. The FD values
themselves are floor-independent to 1e-3 (-2.1215 / -2.1215 / -2.1224 on
sid 28), so the true response of the regularised area does not care about
the floor; only autograd's second derivative does.

WHY. d^2 sqrt(x^2 + a) / dx^2 = a / (x^2 + a)^{3/2}: with a = 1e-30 the
curvature at a plateau point with |grad| ~ 1e-13 is ~1e13, and the mixed
term (d grad/d theta . d grad/dR) / |grad| is amplified by 1/|grad| there.
The plateau population: |grad|^2 < 1e-20 on 0.033% / 0.032% of the grid
(about 1000 points, minimum 6e-28 / 5e-26 -- FFT round-off on the saturated
regions), < 1e-12 on 25% / 22%. A thousand points with 1/|grad| ~ 1e10-1e13
amplification are the 8.25% / 0.99%. This is noise in dF/dtheta, not a
response of the physics; the FD (a finite step) never sees it.

The two clamps are NOT it: clamp(ne, min=0) is active on 30.7% / 32.0% of
the grid and the log floor on 46.9% / 48.2%, which is why the raw-density
stage's FD is unreadable (kinks under a random +-1 weighting), but the shape
function's slope at the floor (x = log 1e-4 = -9.2, SIGMA_K 0.6) is ~1e-51,
and s_vdw already closes.

REMEDY, pending the floor scan (job queued: floors 1e-20 / 1e-18 / 1e-16 /
1e-14 / 1e-12 with the value change of area and E_cav at each): set
MACE_PB1D_AREA_EPS (added in fda741c, default 1e-30 = unchanged) to the
smallest floor at which the second order closes, in BOTH arms of the A/B --
in arm A the cavity has no parameter gradient at all (ne_cav detached), so
the floor changes only its value there, by the same amount as in arm B.
Value cost bound: <= (number of points with |grad| < sqrt(floor)) x
sqrt(floor) x TAU dV; at 1e-12 that is <= 7.5e5 x 1e-6 x 2.56e-5 = 1.9e-5 eV
on E_cav = 3.94 eV, measured exactly by the scan.

### arm-B GPU gate before the joint A/B  (job 3433252, code 6ec75bf; run dir 3-residual_3D/ab_deriv_gate)

Arm B = MACE_PB1D_LIVE_POS=1, MACE_PB1D_GRAD_PASSES=0, DFORCE unset; gate_le
config with warmup_encounters 0 and max_num_epochs 1 so the one epoch runs on
the full-PB path from step 1. Nothing here is about accuracy.

1. PER-FRAME COST on the 339-atom frames (sids 202/203/204, all charged --
   train.xyz has no neutral NiN88 frame; the 160 NiN88 frames are sids
   202-400), gate_le epoch-39 model, batch 1, fwd + loss + bwd: peak 23.92 /
   23.93 / 23.92 GiB at GRAD_PASSES=0, 23.80 / 23.80 / 23.81 at 1; against
   20.58 / 20.46 for the 207-atom frames. So +3.3 GiB for the larger cell;
   time 1.7-6.1 s per frame (noisy).
2. ENV REACHES EVERY RANK: srun -n 3 printed LIVE_POS=1 GRAD_PASSES=0
   DFORCE=unset on ranks 0, 1, 2.
3. ONE 3-GPU DDP EPOCH on the full-PB path, from scratch: run_train rc 0,
   no NaN, checkpoint and model written. nvidia-smi memory.used sampled every
   2 s (3624 samples): peak 26791 / 27767 / 27441 MiB on GPUs 0/1/2 of 40 GB
   -- 12 GB headroom. Epoch 0 took 16.0 min (10:00:55 -> 10:16:54) against
   gate_le's first full-PB epoch of 4.5 min (epoch 30, 04:23:20 -> 04:27:51)
   and 4.5-5.2 min thereafter: 3.1-3.6x slower than arm A's post-warmup
   epochs. This is a cold-model number (a from-scratch solve at epoch 0 is
   not the solve at epoch 30 after warm-up); the A/B run itself will give the
   per-epoch cost after warm-up. The end-of-training evaluation (train +
   valid + test error tables, forces by autograd through the full graph) took
   18.7 min (10:17:02 -> 10:35:45) against gate_le's 11 min.

BUDGET IMPLICATION. Arm A (gate_le's own arithmetic): 8 min startup + 30 x
2.1 + 10 x 5.2 + 11 min eval = 2 h 14 min -- already over the 2 h dev wall,
which is why gate_le needed a resume. Arm B: 8 + 30 x 2.2 + 10 x (8-16) +
19 = 2.8-4.4 h. On gpu-a100-dev (MaxJobsPU 1, 2 h) that is 3-4 resumed
segments per arm, run one after another; on gpu-a100-small (MaxJobsPU 3,
48 h) both arms run uninterrupted side by side, ~2.5 h and ~4.5 h. The
production-queue choice is the user's. Both arms' directories, configs and
job scripts are in place (3-residual_3D/ab_deriv_A, ab_deriv_B; job.sh
takes MAXEP for a split protocol); nothing submitted.

## gate_le: does the NATIVE local_electron_energy channel work? (2026-09-11)

Run 3430114, gpu-a100-dev, 2 h wall, `timeout 6900`, started 03:06:14. Config
differs from gate_bl in exactly two lines: `max_num_epochs 34 -> 40` and
`add_local_electron_energy: True`. From scratch, seed 123, so it is directly
comparable to gate_bl's from-scratch run.

TEST GATE PASSED. `add_local_electron_energy=True` appears in the run's own
Namespace (not only in the yaml), epoch 0 landed at 03:17 with no Traceback and
no OOM, and the epoch-10 checkpoint carries 24 `local_electron_energy.*`
tensors totalling 77825 parameters = 1.52% of the model's 5110243, with the
mlp's last layer at |w|_rms 3.52e-02, i.e. trained rather than left at init.

| epoch | gate_le E / F / pot | gate_bl E / F / pot |
|---|---|---|
| 0  | 38.05 / 668.15 / 2.5433 | 137.54 / 597.56 / 3.9916 |
| 2  | 35.61 / 472.52 / 3.0838 | 25.51 / 402.01 / 2.1159 |
| 5  | 47.56 / 124.23 / 1.6983 | 63.25 / 135.95 / 1.6609 |
| 8  | 26.45 / 64.15 / 1.5968 | 20.52 / 65.31 / 1.4909 |
| 10 | 24.80 / 55.62 / 1.4444 | 20.36 / 57.12 / 1.4340 |

(meV/atom, meV/A, eV.) These early epochs are NOT a verdict: the two runs start
from different random states and their initial losses differ by a factor of
8200 (158610 vs 1.30e9), so the epoch-0..10 gap is initialisation noise. The
comparable points are epoch 33 (gate_bl: 11.75 meV/atom, 32.75 meV/A, 0.1227
eV) and the endpoint.

WHAT THE EARLIER CHARGE-BLIND FINDING DOES AND DOES NOT COVER. The reconcile
audit compared the READOUT INPUT FEATURES between sid1 and sid601 (identical to
1.9e-16 relative, d(inter_e) exactly +0.000000 eV). It did not compare the
atomic charges, so it is silent about this channel. From field_blocks.py:723
the channel is

    q_in = charges_induced + charges_0
    le   = mlp( [ <node_feats, W_q q_in> , <node_feats, W_v field_feats> ] )

so charge-blind node_feats are contracted against a charge-carrying vector, and
whether le is charge-dependent reduces to whether q_in differs between the two
states. total_charge enters the charge construction only through the solvent
(extensions.py:178) and reaches the atomic charges only via the field feedback
at 2562; charges_0 is cloned at 2444, i.e. BEFORE that feedback, so the two
inputs have different exposure and must be reported separately.

`experiments1d/audits/le_channel_probe.py` measures this instead of arguing it:
a forward_pre_hook captures charges_0, charges_induced, field_feats and
node_feats exactly as the channel sees them, a forward_hook captures its
output, and every frame is run twice -- once normally, once with the output
forced to zero -- so the captured sum is checked against the change in the
model's own reported energy before anything is interpreted. It needs the
`.model` architecture pickle, which mace writes only at the end of training, and
refuses with the current checkpoint list until then.

### the run did not finish, and the cause was my own time budget

3430114 reached epoch 37 of 40 and was killed at 05:01:17 by its own
`timeout 6900` (sacct: step 3430114.0 CANCELLED, ExitCode 0:9, elapsed
01:55:02), five minutes inside the 2 h allocation. mace writes models/*.model
only on normal completion, so the endpoint model the probe needs does not
exist.

THE DEFECT IS THE ESTIMATE, not the pace. Epoch cost is bimodal:
`solvent_pb1d_warmup_encounters=30` puts epochs 0-29 on the cheap warmup path
and epochs 30+ on the full self-consistent 1-D PB solve. Measured in both runs
at the same boundary, with the loss dropping 4.20 -> 0.95 there as well:

| | cheap epoch | expensive epoch |
|---|---|---|
| gate_le | 2:04 (19->20) | 5:14 (36->37) |
| gate_bl | 2:00 (28->29) | 4:33 (29->30) |

The budget came from gate_bl's AVERAGE pace, 34 epochs in 1h27m = 2.56
min/epoch. That average does not transfer: gate_bl paid for 4 expensive epochs
out of 34, gate_le needed 10 out of 40. Segmented arithmetic gives 8 min
startup and initial eval + 30*2.1 + 10*5.2 = 1h54m against a 1h55m cap, i.e.
the run was always going to finish within a minute of the wall. Any future wall
estimate here must be segmented, and must check that the cheap/expensive epoch
mix is the same before reusing a measured pace.

3430182 resumes from `checkpoints/s3d_gate_le_run-123_epoch-37.pt`
(`restart_latest: True`; mace takes the epoch from the filename, the checkpoint
carries model + optimizer + lr_scheduler) and runs epochs 37-39 only, budgeted
from the measured expensive-epoch pace rather than an average: 3*5:15 = 16 min
plus ~8 min overhead, `timeout 3300` against a 1 h allocation. It writes to
run_finish.log with `>>` because the first run's run.log is the only copy of
the epoch-0..37 history.

| epoch | gate_le E / F / pot | gate_bl E / F / pot |
|---|---|---|
| 33 | 11.37 / 32.73 / 0.1450 | 11.75 / 32.75 / 0.1227 |
| 34 | 11.26 / 32.37 / 0.1520 | -- (34-epoch run ended at 33) |
| 35 | 10.97 / 31.74 / 0.1624 | -- |
| 36 | 11.21 / 31.43 / 0.1453 | -- |
| 37 | 10.75 / 30.97 / 0.1548 | -- |

At the one comparable epoch, 33, gate_le is 3.2% better on energy, 0.06% on
force, 18.2% WORSE on potential, with total loss within 0.8% (0.82303 vs
0.81640). The energy advantage is not an effect: the signed le-minus-bl gap on
energy runs +3.1% (ep 20), +1.8% (24), -0.3% (28), -5.4% (30), -1.6% (32),
-3.2% (33), so it changes sign repeatedly and a 3% difference is not resolvable
against trajectory noise between two from-scratch runs. There are no replicate
seeds, so no run-to-run spread can be quoted -- the claim here is only that
+-3% sits inside the observed swing, not that the channel helps or hurts. The
aggregate metrics were never the criterion anyway: a channel holding 1.52% of
the parameters could fix the paired charging energy completely and move these
three numbers by almost nothing.

## 工作树快照(2026-07-26 登记)

| 工作树 | commit | 用途 |
|---|---|---|
| pmp-mix | 3bfe707 | mix150 混合训练(400+199,solvated 门控) |
| pmp-rhob80 | b9903cc | rho_b 监督系列 + 身份探针旧代码参照 |
| pmp-prod400 | eb271c7 | pb1d 400-epoch 生产 |
| pmp-prod150 | a68e44a | pb1d 150-epoch 生产 |
| pmp-prod | 2eda78a | 早期生产试跑 |
| pmp-base | ad0ee85 | pb-solvent 分支基线(= origin/pb-solvent) |

## 实验登记

| 实验目录 | 代码版本 | 数据 | SLURM 作业 | 状态 |
|---|---|---|---|---|
| exp_pb1d_mix150 | pmp-mix @ 3bfe707 | data/NiN-mix (599) | gate 3319618-23(判定 bug 空转);训练 3319659-61+3319748 | 完成 150/150(终评过) |
| 1-train_all(c-MACEsol 根) | pmp-trainall @ 43214d1 | 600 帧完整包 480/60/60(中性 20/20 对称;2 场 baseline_cache、表/json 已裁死重) | 预检 3323980 过(身份+5ep);生产 a100 3324023 | 进行中 |
| exp_pb1d_mix400 | pmp-mix @ 3bfe707 | data/NiN-mix (599) | a100 normal 3320693(13.7h) | 完成 400/400:NiN44 0.063/0.041/0.042,NiN44vac 0.076/0.041/0.039,rho_b 1.06e-4 |
| exp_pb1d_mix400d | pmp-mix @ 3bfe707 | data/NiN-mix (599) | dev 链(用户指示 ep258 处停用,火力转编译/MD) | 已停 |
| exp_pb1d_rhob400a 等 rb* 系列 | pmp-rhob80 @ b9903cc | train-data (400) | 3303xxx–3311xxx | 完成(终局见 rhob 记忆) |
| exp_pb1d_prod400 | pmp-prod400 @ eb271c7 | train-data (400) | 3292xxx–3296xxx | 完成 |
| exp_pb1d_prod150* | pmp-prod150 @ a68e44a | train-data (400) | — | 完成 |
| exp_jit_verify | pb-1d @ HEAD(见作业日志) | NiN-mix val | dev 3321473 | 编译数值一致性+速度 |
| exp_md_smoke | pb-1d @ HEAD(见作业日志) | NiN-mix val 帧40(中性) | dev 3321474 | 真空 Langevin 200 步 smoke |
| exp_runtime_baseline | pb-1d @ HEAD | train-data/baseline_cache (400) | 登录节点 | 完成:反解(盲验 1e-4)+ 1D 剖面表(溶剂窗 3-8e-4 eV)+ 接线;门:L2 potential 差 ≤0.011 eV(模型误差 0.063),L3 无 sid 溶剂 MD 471 ms/步 |
| exp_md_gcmd | pb-1d @ HEAD | mix400 模型 + NiN-mix val 帧0(无 sid) | dev 3323089-3323141 | 完成:500 步恒电位 MD,mu 后半程 −3.336±0.180(目标 −3.360),469 ms/步→暖启动 359 ms/步(scheme-C 复用,冷检 ~1e-11、偶发 6e-3 力偏离≪模型误差);2000 步 mu −3.381±0.201;显存平 1.6 GiB(修复句柄泄漏) |
| exp_neutral_rerun (VASP) | 不涉及本仓库代码;VASP=$WORK/CEP-DIP 自编译 | 0-44_neutral | neu* 3318xxx;cal_194 终解:dev 3323433(ALGO=All,EDIFF 1e-5,124 步收敛);回填 3323561 | 5/5 收敛,数据集 200/200(sid 594 入 train,splits 冻结) |
| 编译沙箱 exp_neutral_prep/jit_sandbox | 主仓库 pb-1d @ 2cf9ba8 | stub 缓存 | 登录节点 | COMPILE OK |

## 未精确回填的

- prod150/prod400 之前的探索性 exp_pb1d_* 目录:当时未记版本,只能按
  工作树指针近似;此后不再发生(启动即写版本)。

## 2026-08-03 charge_density_1d supervision (3-train_add1Dcharge)
- Code: pb-1d-charge1d @ 369176e (new loss: plane-averaged 1-D net density
  vs density_3d grid plane average, valid-window masked; metric rmse_charge_density_1d).
- Reference cache: density1d_net_cache.npz (600 sids, nz=500; signal rms
  0.0194 e/A^3 in-window; z grids asserted identical to potential cache).
- Weight derivation (contribution parity, rho_b methodology; probe.o3334372):
  trained train_all model on val split -> rms 1.7408e-3 e/A^3, mse 3.030e-6;
  converged potential-family contributions: fermi 2.70e-3 / Phi1D 2.40e-3 /
  rho3d 1.64e-3 / rho_b 3.12e-3, median 2.553e-3;
  weight = 2.553e-3 / 3.030e-6 = 842 -> 800.
- Gate: 8-epoch dev run, job 3334406 (exp_add1dq/gate).
- Production dir: 3-train_add1Dcharge (config ready, weight 800, 400 epochs).

## 2026-08-03 Bader supervision (4-bader)
- Labels: critic2 YT on CHGCAR (ref AECCAR0+AECCAR2); REF_charges = ZVAL-Ne,
  atomic_dipole = -M_electron (a.u.->eA, critic2 x,z,y order handled).
  Charged 400: reused 3-partition critic2 cache (copied, originals untouched).
  Neutral 200: computed fresh (inputs copied into 4-bader, zero writes to
  source dirs; job 3335225 on gpu-a100 CPUs).
- Verification battery (all PASS): coverage 600/600; charge closure
  max 5e-5 e; geometry identity < 1e-5 A; charged labels identical to
  3-partition originals (5e-9); element stats physical; neutral closure 0.
- Package: 4-bader/data = full copy (splits identical to 1-train_all,
  + REF_charges/atomic_dipole arrays).
- Config: train_all recipe + charges_weight 1.0 + atomic_dipole_weight 1.0
  (3-partition precedent); rhob weight 1.0 (new normalised semantics);
  NO charge_density_1d (single-variable). 500 epochs.
- Code: pb-1d-charge1d @ 71c26b9 (same as add1Dcharge experiment).

## 2026-08-12 4-TTF: corrected-pbc rerun of 3-train_add1Dcharge (COMPLETE)
- Data: neutral 200 frames pbc TTT->TTF (surgical line edit, 160/19/21 lines);
  config identical to 3-train_add1Dcharge except charge_density_1d weight 10->1
  with the x10 baked into the code coefficient (c98a843, exactly equivalent).
- Job 3357895, single 24h window (~16.5 h), 500 epochs, seed 123, code @c98a843.
- vs 3-train_add1Dcharge at epoch 499 (valid): density 0.0442->0.0425 (-4%),
  Phi1D 0.0402->0.0371 (-8%), rho_b 1.09e-4->0.95e-4 (-13%), F 17.9->16.8 (-6%),
  potential 0.0629->0.0724 (+15%; test-set solvated frames -23% — mixed, noise-level).
- Verdict: TTT impact real but modest; 4-TTF is the corrected-convention
  reference baseline going forward.

## 2026-08-12 occ-aug head + fresh stage-1 gate (exp_occupancies/gate34, COMPLETE)
- Two mainline switches under test vs 4-TTF (same data/config/seed 123, 34 ep,
  3-GPU a100, job 3361829, code @74f364b + dc85e5e):
  (1) OccAugHead: equivariant linear readout of CHGCAR PAW augmentation
      occupancies (600-structure cache occ_aug_cache.npz), pure auxiliary
      supervision, occ_aug_weight 1.0;
  (2) solvent_pb1d_fresh_stage1: initial P_z from pre-SCF density every
      forward, no cross-epoch cache (train/eval alike; cache/ = 0 files, verified).
- rc=0, zero fallback/warning lines.
- Timing: warmup 90 s/ep (4-TTF 75, +20%; eval-side fresh stage-1) with one
  transient 330 s band (ep 11-14, no solver warnings, unattributed);
  PB segment 164/155/184/197 s (mean ~175) vs 4-TTF ~105 s -> +67%,
  dominated by per-step fresh stage-1 (occ head is linear; warmup would
  show the same hit if it were the head).
- Metrics ep30-33 avg (valid, vs 4-TTF same epochs): potential 0.218 vs 0.250,
  fermi 0.130 vs 0.209, Phi1D 0.140 vs 0.164, density_3d ~-7%, F/E/rho_b par.
  No degradation; PB entry visibly smoother (no stale-cache cold start).
- RMSE_occ_aug 0.042 (ep0) -> 0.0094 (ep33), ~2% of signal RMS, still falling.
- Decision pending (user): +67% PB-epoch cost vs "adopt if speed cost small";
  500-ep production projects ~24 h (at the 24 h a100 wall).

## 2026-08-15 6-no_old: 500-ep production, occ-aug head + fresh stage-1 (COMPLETE)
- User dir 6-no_old; gate34 config with max_num_epochs 500, seed 123,
  3-GPU a100, job 3362946, code @6912832 (same modules as gate).
- COMPLETED in 19h06m, rc=0, zero fallback lines, cache/ = 0 files (verified).
- Timing (production log): warmup 86 s/ep (+15% vs 4-TTF 75), PB segment
  mean 139 s/ep (+32% vs 105; max 303 s, only 12 ep > 250 s), total +16%
  vs 4-TTF 16.5 h. Early-PB cost (gate saw +67%) relaxes as the model
  converges and the fresh stage-1 initial guess improves.
- Final error table (valid, vs 4-TTF): potential 0.0665 vs 0.0713 (-7%),
  fermi 0.0359 vs 0.0369 (-3%), Phi1D 0.0360 vs 0.0374 (-4%),
  density_3d 0.0431 vs 0.0425 (+1%), rho1d par, E 3.5 vs 4.2 meV,
  F 17.1 vs 16.8. Epoch-499 rhob_1d 1.02e-4 vs 0.95e-4 (+7%).
- Test: fermi better on all three systems, Phi1D better on all three,
  potential better on 2/3; occ_aug 0.0045-0.0052 (no overfit vs valid 0.0049).
- occ_aug RMSE 0.00491 ~ 1% of signal RMS.
- Verdict: both switches adopted at production quality; 6-no_old supersedes
  4-TTF as the reference model (fresh stage-1 = train/eval/MD path identical).

## 2026-08-16 8-band: DFT vs fully-ML CHGCAR band comparison (COMPLETE)
- 6 structures (median frame per config type, test+train), 3 variants each:
  dft / ml_dftocc / ml_full. Non-SCF ICHARG=11, Gamma-M-K-Gamma 24 kpts,
  explicit NBANDS (448/688), VASPsol+LDIPOL as source. Model 6-no_old.
- Assembly (exp_band/): ml_electron = baseline - GTO residual on the full
  168x168x500 grid; aug occupancies replaced per-atom (byte-identical
  round-trip validated). Build checks: totals vs NELECT ~1e-4 e,
  residual RMSE 0.039-0.045 (= training metric), occ RMSE ~1% of signal.
- Result (window = 24 bands at fermi, each run self-fermi-aligned):
  occ prediction HARMLESS (ml_full - ml_dftocc = 1-15 meV, slightly better);
  train == test (no generalization gap); dispersion nearly exact
  (k-residual 0.06 eV after per-band shift removal); errors are per-band
  rigid shifts: window RMSE 0.42-0.46 (neutral) / 0.76-0.87 (charged),
  fermi diff +0.9 to +2.3 eV. Au precedent: 0.10 / +0.45 with 2.6x better
  3D density (0.017 vs 0.045 e/A^3) -> band quality is density-limited.
- Root causes isolated: (1) 31% of grid points slightly negative (water/
  cavity boundary worst, -0.069) distort the VASPsol density-based cavity;
  clamping negatives + renorm fixes fermi diff 1.59 -> 0.48 but window
  stays ~0.83; (2) remaining per-band shifts = local electrostatics from
  atomic-scale density error. 1D plane-averaged electrostatics verified
  fine (Poisson on delta-rho: +-0.15 eV). No-solvent diagnostic protocol
  is unusable (charged slab non-SCF does not converge without LSOL).
- Open (user): adopt clamping into the recipe + rerun ML variants; better
  3D density (probe line?) as the route to band-quality parity.

## 2026-08-18 9-larger_3d: density_3d weight sweep 10/50/200 + neutral band checks (COMPLETE)
- Three 500-ep productions (jobs 3368866/67, 3370058), config = 6-no_old except
  density_3d_weight (raw MSE term; w=1 is normalized-weight 1/215, signal_ms 0.00466).
- Endpoints (valid, vs 6-no_old 0.0431/0.0643/0.0351/0.0353):
  w10:  density 0.0359 (-17%) pot 0.0671 (+4%)  fermi 0.0335 (-5%)  Phi1D 0.0376 (+7%)
  w50:  density 0.0317 (-26%) pot 0.0862 (+34%) fermi 0.0453 (+29%) Phi1D 0.0454 (+28%)
  w200: density 0.0290 (-33%) pot 0.1180 (+84%) fermi 0.0444 (+26%) Phi1D 0.0610 (+73%)
  -> 1D-metric balance point near w10 (near-free density gain).
- Band checks (2 neutral cases sid486/454, fully-ML CHGCAR, DFT band refs reused
  from 8-band). RECIPE INCIDENT: unclamped w50 CHGCAR made the non-SCF
  eigensolver diverge (ghost wells at negative-density pockets, dE runaway);
  a SOFT clamp with positive vacuum floor (eps/2 = 5e-5) also failed to
  converge (floor >> physical vacuum density). HARD clamp max(rho,0) +
  renormalize converges everywhere -- adopted for all variants incl. a
  re-done w1 baseline (clamped-recipe-consistent table):
  window RMSE (test44v/train44v): w1 0.297/0.191, w10 0.388/0.318,
  w50 0.348/0.267, w200 0.345/0.170; fermi diff: w1 -0.33/-0.62,
  w10 -0.48/-0.63, w50 -0.24/-0.35, w200 +0.02/-0.20.
  Unclamped for reference: w1 0.452/0.418 (+0.88/+0.97), w10 0.266/0.311
  (+0.24/+0.38).
- Reading: clamping the w1 baseline captures most of the band-window gain;
  after clamping there is no clean weight trend at 2-case statistics.
  Weight helps absolute fermi placement monotonically (w200 best).
  Band-level conclusions need more cases; 1D-level conclusion (w10 sweet
  spot) is solid.

## 2026-08-30 exp_residual3d_probe: residual-3D solvent-charge representation probe (COMPLETE)
- Question: can "1D-broadcast baseline + envelope x atom-centered-GTO residual"
  represent the 3-D RHOB/RHOION labels? Model w200 (9-larger_3d), 9 frames
  (val 28/62/394/208/401/433, train 1/202/403), 300k fit + 100k eval points,
  per-frame lstsq ceilings (no training). Dir: claude/2-1D_PB/exp_residual3d_probe.
- Labels pass all audits (CONTCAR match 1e-13, ion integral = q_tot, rb
  integral ~0, plane avgs == dft_solvent1d_ref to 1e-10).
- DATA GOTCHA: NiN-mix neutral frames (sid 401-600) are the VACUUM calcs
  with solvated=0 AND pbc=TTT — NiN-mix is the UNFIXED original (the TTT
  incident's corrected copy lived only in 4-TTF/data, which w200 actually
  trained on via symlinks and which is now DELETED; both surviving bundles,
  NiN-mix and 1-train_all/data, are still TTT — verified line-by-line).
  w200 thus trained neutrals as TTF slabs WITHOUT solvent (solvated=0);
  PB never ran on them either way. Probe forces solvated=1 + pbc=TTF ->
  runtime-baseline path solves cleanly (rms ~1e-12, n_outer 7, q_ion=0).
  Geometries == 5-44_neutral_withsolv, so the withsolv 3-D labels are valid.
  Future residual-3D training must swap neutral frames to withsolv labels
  + solvated=1 + pbc TTF; NiN-mix's formal TTT fix is still pending (user).
- bound: plane average removes only 5-7% (lateral-dominated; region signal
  2.6-3.6e-3, after 1D 2.5-3.5e-3 e/A^3); envelope fit ceiling leaves 21-28%
  (gradS == s(1-s) envelopes; sigma .25-1 == .5-2 > 1-3; l=2 required:
  l0/l1/l2 = 1.21/0.87/0.65e-3 on sid 28); bare-basis control 2-3x worse
  (confirms envelope is essential); far (>6 A) content only 1-2%.
- ion: 1D broadcast removes 60-65%; shape-mod s_ion/S_ion baseline adds
  ~10-15% on NiN88 only; envelope fit leaves 27-40% of the remainder;
  neutral-frame ion signal ~5e-5 (negligible, head should output ~0).
- Ridge tradeoff: lam 1e-10 -> 1e-7 costs 2-4% rms while |c|max drops
  1e4 -> 4e2-2e3 -> no 1D-era-style unlearnability; mild ridge tames it.
- Energy scale: integral[bound residual x lateral cvhar3 fluctuation] =
  +2.3..+3.4 eV (charged) / +0.9 eV (neutral); after best fit 0.3-0.7 /
  -0.2 eV. Lateral solvent electrostatics is eV-scale -> supports the
  stage-2 energy/force term.
- Verdict: representation adequate; next level = frozen-trunk head training
  (cross-structure learnability of the modulation coefficients).

## 2026-08-30 TTT pbc fix executed everywhere (user directive: never again)
- User directive: fix TTT to TTF now, and this error must never recur.
- Surgical line edits (tools1d/fix_pbc_ttf.py; solvated=0 rows only, pbc
  field only; backups + log in claude/2-1D_PB/ttt_fix_backup/): NiN-mix
  160/19/21, 1-train_all/data 160/20/20, 5-only_fermi val/test 20/20,
  exp_neutral_prep/neutral_draft.xyz 200 (the assembly source). 840 lines.
- VERIFIED: NiN-mix's 200 fixed neutral header lines are byte-identical to
  the surviving corrected copy (claude/3-charge_probe/data_neutral) — the
  fix reproduces the deleted 4-TTF correction exactly. ASE round-trip OK.
- 7-cpmace was WRONGLY edited first, then reverted byte-identical from
  backup: it is the vanilla CP-MACE comparison bundle, deliberately all-TTT
  (solvated=1 charged frames are TTT, no Fermi/total_charge keys). Blanket
  fixes across bundles with different conventions are exactly the incident
  class — the new audit caught it immediately.
- Remaining TTT after final sweep: 7-cpmace (intentional), ttt_fix_backup/
  (the backups), exp_md_smoke/smoke_traj.xyz (output artifact). Zero in any
  data-source bundle.
- Recurrence prevention: (1) extract_neutral_set.py now sets pbc TTF
  explicitly (ase defaults to TTT); (2) NEW GATE tools1d/
  audit_bundle_conventions.py — run on every new/adopted bundle before use
  (pbc TTF, solvated presence/consistency, charged->solvated=1, sid
  uniqueness, per-config_type info-key inventory drift). NiN-mix and
  1-train_all/data PASS.
- Dangling-link repair: all 62 symlinks that pointed into the deleted 4-TTF
  now point at data/NiN-mix (post-fix == 4-TTF content); real
  density1d_net_cache.npz restored into NiN-mix from exp_2iter_gate's copy.
  Older dangling links into long-deleted 3-train_add1Dcharge /
  6-larger_chargew (archived exp dirs) left as-is.

## 2026-08-31 exp_residual3d_head: frozen-trunk head learnability (level 2, COMPLETE)
- Setup: all 539 frames prepped (labels via fast np.fromfile reader, verified
  byte-identical to the probe reader; w200 frozen forward stashes feats/
  envelopes/1-D baselines; 300k fixed points/frame). Head = LayerNorm +
  MLP 1152-512-256-54 (2ch x 3sig[.5,1,2] x l<=2), zero-init out,
  OUT_SCALE 100, AdamW 1e-3 cosine, 15k steps x 2 frames x 8192 pts,
  0.061 s/step (~15 min). Jobs 3403509/3403510.
- HELD-OUT (59 val frames): bound head/res1d = 0.47 mean (0.40-0.54, flat
  across charged AND neutral); ion (charged) 0.58 (0.45-0.73). The head
  halves the lateral residual on unseen structures out of the box.
- vs per-frame lstsq ceiling (same basis): head/ceiling 1.7-2.3 (bound),
  1.5-2.4 (ion) — a 2x capacity/feature gap, not a learnability failure.
- Energy diag: charged bound coupling +2.3..3.4 -> +0.6..1.6 eV (halved);
  neutral -> -0.4..-0.9 (sign flip, similar magnitude).
- KNOWN ISSUE: neutral-frame ion channel gets WORSE (2x, adds ~1e-4 where
  signal ~5e-5): global variance normalization gives neutral frames no vote.
  Fix candidates: charge-gated ion output or per-frame loss weighting.
- Verdict: cross-structure learnability CONFIRMED; route is sound. Next
  levers to close the 2x gap: equivariant readout (invariant-MLP is the
  probe shortcut), bigger head/longer training, richer features.

## 2026-08-31 exp_residual3d_head level 2b: head done properly (COMPLETE)
- Two architectures, same protocol (25k steps x 4 frames x 8192 pts, AdamW
  cosine, ion output gated by q_tot; jobs 3403715/3403716):
  (a) equi: structured readout from the irreps blocks of the mixed feats
      (scalars+block norms -> gate MLP; l=1/2 coefficients = gated linear
      channel mixes of the matching blocks);
  (b) mlp: plain MLP control, 1152-1024-512-256-54.
- HELD-OUT (59 val): bound rms/res1d equi 0.464 / mlp 0.471 / level-2 small
  mlp 0.473 -> FLAT. head/ceiling(bound) ~2.0 both. Architecture, capacity
  (2-4x), and sample count (3.3x) all move nothing on bound.
- ion (charged) improved 0.584 -> 0.489 (equi) / 0.464 (mlp); neutral ion
  ratio exactly 1.00 by the q gate (level-2's 2x degradation eliminated).
- Energy diag: charged bound coupling mean |E| ~0.9-1.0 eV (res1d ~2.3-3.4,
  per-frame fit ceiling 0.3-0.7).
- READING: the remaining 2x bound gap is INFORMATION-limited, not
  architecture-limited — the frozen trunk features do not carry the
  frame-specific lateral solvent detail (and part of the per-frame lstsq
  ceiling is unreachable for any transferable model). Lever to close it:
  joint training (residual loss backprops into the trunk) = the planned
  stage-2 integration; or accept ~0.47 and integrate as-is.

## 2026-09-05 stage-2 solvent energy terms (branch pb-s3d-energy, clone pmp-s3denergy)
- Motivation: twin diagnostic (2026-09-04) — DFT E_solv-E_vac = +2.944 =
  A_cav +3.839 + A_solv -1.011 + SCF relax +0.116; the model's solvent
  energy path is 1-D electrostatic only (+0.037, A_solv part wrong-signed).
  Sizing run (solvent_energy_3d_vs_1d.py, job 3416856): linear-response
  E_int_3D/2 = -0.758 covers 74% of A_solv; the 1-D level gives +0.050
  (wrong sign, 7% amplitude) — the term must be 3-D.
- Code @5503789 (default-off flags, pure energy additions):
  * solvent_cavity_energy: E_cav = TAU * int|grad s_diel3| dV from the live
    cavity (TAU from the solvation json = the DFT setting, 0.009 eV/A^2).
  * solvent3d_energy: E_3d = int[delta*(-cvhar3)] + int[delta*phi(rho_1d)]
    + 0.5*int[delta*phi(delta)], delta = env*m from the solvent3d head on
    the PB grid, per-channel charge-conservation projection
    (delta -= (int delta / int env) * env). Lagged-SCF convention mirrors
    the 1-D compensation energy: solvent state detached, cvhar3 live.
  * save_latest_every: rolling restart checkpoint every N epochs (plateau-
    starved dev chains; never deletes the best checkpoint).
  * Supervision loss unchanged (detached baselines/envelopes stay).
- Unit factor (measured): grid GTO assembly = point evaluator * volume
  (ratio/V = 1.0003; periodic-image tails 3e-4).
- Guards: solvent3d_energy requires head + weight>0; cavity requires pb1d.
- NOTE: flags change the energy of every solvated frame by ~+3 eV — only
  fresh trainings or restarts of runs that already had the flags on.

## 2026-09-05 neutral-frame fallback defect + twin validation (fixed code @be722aa)
- DEFECT FOUND: the pb-solvent3d line missed trainall's 2026-08-31 fix
  2b149d6 (neutral-safe layer_mean). Old code: layer_mean = moment/1e-12 on
  q_ion~0 -> health gate rejected EVERY neutral solvated frame -> silent
  planar fallback through the whole s3d gate + 500-ep production. Fallout:
  the solvent3d head never trained/scored on neusol frames (published
  bound x0.51 is charged-frames-only), neusol Phi1D/rho_b supervision saw
  planar outputs. train800 is clean (ran post-fix code). "Zero fallback"
  ledger checks were blind: fallbacks only print with MACE_PB_DEBUG.
- Fix ported verbatim (3-branch layer_mean + mu from ion_dipole_t);
  backend otherwise diff-clean vs trainall.
- Twin validation, gate_w1dev model, fixed code (job 3416946, 2 pairs,
  MACE_PB_DEBUG=1, ZERO fallbacks):
    flags off: dE = +0.252 +/- 0.001 (healthy-solve baseline of this model)
    flags on:  dE = +3.914; e_cav = +3.863 +/- 0.05 vs DFT A_cav
    3.839 +/- 0.055 (0.6%, live model density); e_s3d = -0.20 (right sign;
    small because the head is charged-frames-trained = neutral extrapolation;
    retraining with the fix closes toward the -0.9 label value).
- Unit battery (jobs 3416874/86/900/10): grid=point*V (3e-4), Poisson
  helper = sizing script (0.1%), l0_inv sign fixed (+net_phys*V), projection
  1e-19, E_cav formula bitwise vs VASPsol++ print (433.82=433.82).
- Smoke gate submitted: gate_s2e (3 ep, 3-GPU DDP, warmup 0, both flags,
  save_latest_every 1).
- RECOMMENDATION: production rerun on fixed code with the energy terms,
  folding all fixes into one run (user decision).

## 2026-09-05 34-ep paired gate PASS (gate_s2e34, job 3417982, code @d77d161)
- Same recipe/seed as gate_w1_dev; three new flags (neutral fix baked in the
  branch, solvent3d_energy + solvent_cavity_energy + save_latest_every 5).
- ep33 paired vs w1: F 33.3/34.0, pot 0.148/0.148, fermi 0.113/0.107,
  Phi1D 0.123/0.120, density/occ par, solvent3d_b 1.081/1.069e-3 par
  (now pooling neusol points too), solvent3d_i 0.86/1.00e-4 BETTER.
  E 11.1 vs 8.5 meV: understood transient - the energy terms switch on at
  ep30 (warmup end) adding ~+3.6 eV to every solvated frame; 4 PB epochs
  cannot re-equilibrate E0s (production has 470).
- Fallbacks: 15, ALL in transition epoch 30 (mu_bound trips on first fresh
  solves, mixed charged+neusol), zero in ep31-33 - visible via the new
  always-on counter. Rolling ckpts epoch-25/33 + best coexist correctly.
- Epoch time 3.3-4.9 min (first PB epoch heaviest) - production compatible.
- READY FOR PRODUCTION RERUN (user decision): recipe = prod500 config
  + solvent3d_energy + solvent_cavity_energy + save_latest_every 5,
  code pb-s3d-energy @d77d161, dual-lane as before.

## 2026-09-05 s3d production RERUN v2 SUBMITTED (user-approved, dual-lane)
- Recipe: prod500 config + solvent3d_energy + solvent_cavity_energy +
  save_latest_every 5; code pb-s3d-energy @d77d161; seed 123; 500 epochs.
- Lanes: prod500v2 (a100 3418149, 36h) + prod500v2_dev (2h chain,
  link 1 = 3418156, warden-managed with zero-epoch autopsy + user-dev
  priority). Cross-stop at Epoch 499; rolling ckpts cap restart loss at 5 ep.
- Sentinel v3: epoch-gap 18min cap, log staleness, stderr, ghost jobs,
  fallback-count alarm (>60 = persistent fallbacks beyond the transition).

## 2026-09-06 review fixes: complete gradients + force validation (@f3e370b)
- User review found real gaps in the "fully live" claim, all fixed:
  (1) call-site pos detach (extensions) meant anchors got no explicit force
  (user's synthetic check: autograd 0 vs FD 0.02072); positions now enter
  live, every pre-existing solve consumer detaches itself. (2) E_cav/env
  now differentiate through the density: ONE checkpointed rebuild
  (net -> cavity -> envelopes -> delta -> couplings). (3) slab-correction
  energy passes gradient through the residual dipole (1-D part lagged).
  (4) supervision-consistent projection: the loss gets its own DC ratio
  (frozen envelopes, live m) so its backward is the exact derivative of
  its forward; the energy keeps the fully-live ratio (values identical).
  (5) runtime-mode energy regenerates the baseline LIVE (rt tables are
  pure-torch structure factors). (6) NaN fix: eps floor on both |grad S|
  (sqrt(0) on saturated plateaus has infinite backward - caught by the
  first live-force run).
- Force validation (job 3418548, gate_w1dev model, neusol frame):
  * tier(ii) runtime mode, FULL field refresh per FD displacement:
    autograd vs FD gap = +0.003/+0.019/+0.002 eV/A (Ni/C/H, 0.2-1.4%) -
    the remaining lagged-1D-solve share, same scale as the 20 meV/A force
    RMSE. Energy-force consistency HOLDS on the MD path.
  * tier(i) cached-baseline surface: FD-autograd gap up to 1.2 eV/A =
    the measured size of the frozen per-sid baseline approximation
    (training-path forces are supervised by DFT labels, not this
    derivative; same situation as the whole 1-D lineage).
  * runtime vs cached baseline energy: 0.3 meV agreement.
- Final 34-ep gate on @f3e370b: job 3418577 (pending). Production resubmit
  after it passes.

## 2026-09-06 blowup root cause: MEASURED and fixed (out_scale 100->10)
- Evidence chain (all measured, no assumptions):
  * Gate matrix: value-only terms PASS (3417982); E_cav-live-only PASS
    (3419003: ep33 pot 0.169/E 11.4/s3d_b 1.12e-3, = the passing gate);
    both-live FAIL twice (3418577 oscillation, 3418909 explosion to
    pot 8.5 eV); warmup-0 both-live FAIL (3419006, explodes from ep2,
    loss 1405) -> E_3d feedback is the destabilizer, timing-independent.
  * Forensics on the blown run's own pre-shock ckpt (ep28, head==0),
    job 3419243: gradient anatomy - E_3d->head 4.3/11.8 (charged/neusol)
    vs point-loss->head 2.2/4.8; force-loss->head 9.85 (2nd-order d2E/dRdW,
    largest channel); E_cav->density 9.4-10 but x energy-residual only
    0.013-0.045 (harmless, matches cav-only PASS).
  * Dynamics: Adam's zero-moment first step moves every head weight by
    exactly lr -> with OUT_SCALE=100 the residual field jumps ~100x label
    scale -> SELF-ENERGY (quadratic in delta) explodes: step-1 e_self
    +95.5 eV; energy/force losses then thrash the head; never settles in
    real training (800 frames/3 ranks).
  * Scale law verified quantitatively (job 3419252): step-1 self-energy
    95.5 / 0.88 / 0.25 eV at scale 100 / 10 / 5 = (lr*scale)^2 exactly.
- FIX (physics-invariant reparameterization): head out_scale = 10 as an
  instance attribute (old pickles fall back to class 100, so trained
  models keep their calibration). Energy expression unchanged; only the
  optimizer geometry. @a319387.
- Final 34-ep gate with the fix: job 3419262 (pending). Criteria: pot
  trajectory converging (0.22->0.15 class), ep33 s3d_b ~1.1e-3, E ~11 meV.

## 2026-09-08 migration to UT workstation tmi-a77203 (gate_bl_repro)
- Machine: 2 x RTX 4090 (24 GB), no SLURM; conda env pmp39 (py3.9, torch 2.2.1+cu121).
  Code pb-s3d-energy @ b4eb669 (bundle == GitHub head). Data = migration_kit
  bundle800 (symlinked); density3d manifest rewritten to local absolute paths.
- Reproduction gate: ~/jobs/1-3DPB/gate_bl_repro, recipe = jobs/gate_bl
  (34 ep, seed 123). world_size=2 (one rank per GPU, NCCL) -> effective batch
  2 instead of the LS6 run's 3 (320 vs 214 steps/epoch); ep33 numbers are
  expected close, not identical. First attempt kept world_size=3 (ranks 0,2 on
  GPU0, gloo backend via claude/run_train_gloo.py, repo untouched): Initial
  valid loss 1301722262.544966 vs LS6 1301722262.544977 (rel 1e-14) and
  PB1D-FALLBACK 5/rank as on LS6, but OOM on GPU0 at the first backward
  (2 x ~11.2 GiB > 23.5 GiB).
- Environment check without training: LS6 gate_bl ep33 checkpoint re-evaluated
  with eval_offset_ab.py -> identical to cohbl.o3422505 (RMSE 11.65;
  cohort bias NiN44 +20.00 / neusol -6.91 / vac -7.14 / NiN88 -5.32).
- Reference for ep33: E 11.75 / F 32.75 / pot 0.1227 / fermi 0.0912 / Phi1D 0.1086.
- RESULT (EXIT 0, 34 ep in 2 h 29 min; warmup 3 min 21 s/ep, PB epochs
  ~12 min/ep vs LS6 4 min -> fp64 throughput-bound on 4090, GPUs 100%):
  ep33 valid E 10.58 / F 32.06 / pot 0.1759 / fermi 0.1226 / Phi1D 0.1174 /
  s3d_b 0.001203 (LS6: 11.75 / 32.75 / 0.1227 / 0.0912 / 0.1086 / 0.001169).
  Cohort bias NiN44 +18.23 / neusol -6.51 / vac -6.20 / NiN88 -4.46
  (LS6 +20.00 / -6.91 / -7.14 / -5.32); pair diff -0.31 (LS6 +0.23).
  Test E per cohort 20.4 / 6.6 / 6.1 / 4.4 (LS6 22.3 / 7.1 / 7.0 / 5.3).
  User ruling: this ep33 is the local baseline; LS6 numbers are direction
  reference only. pot/fermi ep32->33 rebound (0.1337->0.1759) attributed to
  effective-batch 3->2 trajectory jitter.

## 3426061  cavity_compare (raw vs lateral separation)   code 7ae44ae
Cancelled 3426020/3426042 and replaced them: the earlier version binned only
the LATERAL charge (rho_solv - <rho_solv>_xy), which is non-zero inside a
closed cavity purely to cancel the plane average. Reading "28.5% of the
lateral charge sits where the cavity is closed, therefore the cavity is
wrong" is RETRACTED (user correction 2026-09-09).
Now measured, raw and lateral kept in separate tables:
  Q1-raw   RHOB binned by s_diel, RHOION binned by s_ion, full solvent by
           s_diel (native DFT grid, no interpolation). No requirement that
           the bound charge vanish at small s_diel: it is -div P with a
           non-local convolution.
  Q1-lat   the old lateral binning, relabelled.
  Q2       cavity agreement classes (both open / both closed / model closed
           DFT open / model open DFT closed).
  Q2-raw   per class: TRUE raw solvent charge vs the model's broadcast 1-D
           background vs its 3-D residual, with cross energies. Separates
           "cavity wrong" (real charge where the model's cavity is closed)
           from "1-D background + envelope-restricted residual cannot work
           together" (real charge ~0, background not, |delta| small).
  Q3       density comparison, with the caveat recorded in the output that
           the DFT density goes through the MODEL's cavity parameters, so it
           isolates the density input only.
Backend: MACE_S3D_EXPORT_DELTA (diagnostics only, off in production) exports
the exact energy-side residual on the ENERGY grid; d_sup_* live on the
upsampled supervision grid and are the wrong field for this.

## 3426068  cavity_compare, raw/lateral separated   code 4b61668   2m32s
Cavity is essentially RIGHT: the two cavities disagree on 0.44% (charged) /
0.53% (neutral) of the cell. So "the cavity prediction is wrong" is not the
main story.
Where the true raw solvent charge actually is (native grid, no interpolation):
  by s_diel : 83% of int|rho| in the 5% of volume with 0.01 < s_diel < 0.90
              (the dielectric transition shell). Only 12% at s_diel > 0.90.
  by distance: 74.6% of int|rho| 1.5-2.5 A from the nearest solute atom
              (10.9% of volume), carrying raw cross -6.549 eV of a -5.963 eV
              total; mean s_diel there is only 0.18.
              0-1.5 A holds 18.5% of int|rho| and +0.811 eV, OPPOSITE sign.
  CORRECTS the earlier model-grid claim "90% of the lateral charge within
  1.5 A": on the native grid it is 26.8% within 1.5 A and 61.5% at 1.5-2.5 A.
  The earlier figure was substantially an interpolation artifact.
Same-potential cross energy by cavity class (model potential, DFT charge):
  charged  true -5.233 eV vs model -3.544 eV -> 32% short, of which
           both-closed 1.221 eV (72%) and model-closed-DFT-open 0.458 eV (27%,
           and the model has the WRONG SIGN there: +0.120 vs -0.338).
  neutral  true -0.990 eV vs model -1.300 eV -> 31% OVER, with the sign of the
           error flipped and misallocated between classes (both-open -0.189
           vs -0.578 under, both-closed -1.319 vs -0.515 over).
Representation vs cavity, per the user's criteria: NEITHER in strict form.
  The residual is NOT excluded from the closed region (|delta| 0.839 e there),
  and the model's net charge per class is close to true (+0.532 vs +0.603 in
  both-closed). The failure is the DISTRIBUTION: right amount of charge, 37%
  less attraction. Verified alongside: residual plane-sum 2.65e-15 / 4.07e-16
  and 1-D background net matches the solver exactly (+1.000 / +0.000).
Q3 density: the mismatch class has model n_e higher than DFT by +0.0046 /
  +0.0052 e/A^3 against the NC_K = 0.015 threshold -- a threshold-grazing
  density error on a small volume. Caveat printed in the output: the DFT
  density is put through the MODEL's cavity parameters, so this isolates the
  density input and cannot alone rule out a recipe/parameter mismatch.

## 3426137  envelope_alignment v1 -- CANCELLED before running, superseded
Four defects, all found by the user in review, all real:
 (a) "the cavity is excluded" was premature. The classification called
     s_diel <= 0.5 "closed", so the charge-carrying transition region (mean
     s_diel ~ 0.18) was inside "closed"; agreement of that binary label says
     nothing about whether the CONTINUOUS s_diel and its GRADIENT agree, and
     the bound charge is -div P, hence gradient-dependent. A 0.43%-of-volume
     mismatch carrying 27% of the gap cannot be dismissed as small. Neither
     exonerated nor convicted.
 (b) ARITHMETIC ERROR in the reporting: int|bg + delta| is not int|bg| +
     int|delta|. "The model already puts about the right amount of charge
     there" rested on 0.592 + 0.839 ~ 1.407 and is withdrawn. The sum field
     must be formed before taking the absolute value; the script never
     produced that quantity.
 (c) the charge-share / envelope-share ratio cannot confirm or refute a
     mechanism: the envelope is not a prediction of the charge, and large
     coefficients do not imply large self-energy (a functional of the final
     distribution). The script also compared RHOB+RHOION against the BOUND
     channel's env_b, remixing the two charge types just separated.
 (d) "the earlier 90% was an interpolation artifact" is wrong. The earlier
     figure was 90% of the lateral charge within 2.5 A (17.8% + 10.9% of
     volume), not within 1.5 A; the native grid gives 26.8% + 61.5% = 88.2%
     within 2.5 A, which agrees. No artifact; I misread my own table.

## 3426230  envelope_alignment v2 (corrected accounting)   code (this commit)
 T1 bound and ionic kept apart, each against its own reference (RHOB /
    RHOION): net and int|.| of the model's TOTAL channel field, formed as
    bg + delta BEFORE the absolute value; int|bg| and int|delta| printed as
    components and never added.
 T2 bound split into plane-average and lateral on both sides, so the part of
    the gap owned by the 1-D pipeline is separated from the part the 3-D
    residual is responsible for.
 T3 in the regions that carry the cross-energy gap, the CONTINUOUS cavity
    value and its gradient magnitude, model vs DFT, plus correlations inside
    the DFT transition shell -- the test the binary classification skipped.
 T4 position diagnostic only, no pass/fail: RHOB against env_b and env^0.5,
    RHOION against s_ion, with the native-grid |grad s| distribution so the
    interpolation smoothing cannot contaminate it.
Backend: MACE_S3D_EXPORT_DELTA now exports delta_b and delta_i SEPARATELY
(delta_grid kept as their sum); merging them made it impossible to compare
against RHOB and RHOION as distinct references.

## 3426002  neutral_feasible_amp   code 4b61668   5m07s
Neutral verdict: NO capacity obstruction. All three bases hit cross = ref and
self = ref exactly with point error inside the cap; only the amplitude window
separates them:  A rms 0.316 (cap 0.320) |q| 1.205 -> misses the +-20% window
by 0.5%;  A+s_diel rms 0.218 (cap 0.274) |q| 1.214 -> misses by 1.4%;
A+env^0.5 rms 0.168 (cap 0.214) |q| 1.147 -> PASS.
Confirms the user's correction: the earlier "S_ref above the family's
attainable range" for basis A was a limit of the nu-parametrisation, not of
the subspace. Predicted rms/ref 0.3155 offline, measured 0.316.
Also: the other candidates in each basis have |q| INSIDE the window
(0.824-1.067) but point errors 0.83-0.98, far over cap -- amplitude and point
error trade off along the feasible manifold, and this sampling (24 random
null-space directions, keep 4 by lowest point error) only ever selected for
point error. So basis A's FAIL is a statement about the sampling and about a
threshold I chose, not about capacity.
Contrast with charged: min attainable self is 2.28-7.79 x the reference --
factors, not percent. The neutral/charged asymmetry is real, not a threshold
artifact.

## Workstation cross-run (two RTX 4090, 24 GB), 2026-09-09
Second machine, auxiliary; LS6 stays primary (user ruling). Numbers below were
produced there on a clean worktree at 028702a, interpreter conda pmp39
(Python 3.9.25, torch 2.2.1+cu121), and cross-checked here by arithmetic:
the T2 split reconstructs T1 exactly and bound+ionic reconstructs T3 to 1e-4.

Non-additivity of the absolute integral, quantified: int|bg| 2.028 +
int|delta| 1.673 = 3.700 against the true int|bg+delta| = 3.238 e, i.e. the
naive sum overstates by 0.463 e = 14.3% (charged; 0.18 e neutral). That is
the size of the error in the withdrawn claim.

BOUND channel, charged: model carries 4.7% MORE absolute charge than DFT
(3.238 vs 3.094 e) and delivers 44.2% of the attraction (-1.454 vs -3.286
eV). Correctly founded this time.

Split of the 1.832 eV missing bound attraction, two independent ways:
  by channel (T2): plane-average 0.525 eV (28.7%), lateral 1.307 eV (71.3%).
    Plane-average magnitude ratio is 1.004 while its ENERGY ratio is 0.496 --
    right amount of charge, half the coupling. Reading the split on magnitude
    alone says "purely lateral" and is wrong. The standing constraint is that
    the 3-D fit owns the lateral part only, so this 0.525 eV must be split off
    before any basis-capacity work.
  by shell (T1): 0-1 A +0.785 (42.8%), 1-1.5 A -0.567, 1.5-2 A +1.362 (74.3%),
    2-2.5 A +0.508, 2.5-3 A -0.149, 3-3.5 A -0.167, 3.5-5 A +0.295,
    5-inf -0.235. Heavy cancellation; the two dominant positives are the
    0-1 A and 1.5-2 A shells.

The representation defect the user predicted 2026-09-09 is now measured.
Sorting the charged bound shells by envelope weight: where the residual has
weight (bg/del 0.3-0.6, shells 1-2.5 A) the magnitude lands within 27% of the
reference; where it has almost none (bg/del 10.6 at 0-1 A, |del| = 0.000 at
3.5-5 A and beyond) the model's charge is 3 to 10 times the reference and is
almost purely 1-D background. At 0-1 A the model puts 0.176 e where DFT has
0.029 e (6.1x), of which 0.169 e is background against 0.016 e of residual,
and it costs +0.817 eV of coupling against DFT's +0.032. Envelope-void shells
sum to about +0.53 eV, close to the 0.525 eV plane-average deficit measured
independently.

Frame asymmetry explained by the same defect with opposite sign: at 0-1 A the
charged frame's background is net +0.131 e against a +15 V interior potential
(spurious repulsion, +0.785 eV of gap) while the neutral frame's is net
-0.056 e (spurious attraction, -0.392 eV) -- and the neutral bound gap is
only -0.309 eV in total, so that one shell exceeds the whole gap. Neutral
plane-average energy ratio is 1.020, lateral 1.648.

Displacement, not just scaling (plane_profile_audit): charged bound profile
shifted -0.423 A toward the slab, correlation 0.875, best single scale 0.870
leaving 48.5% residual -- so misplaced, not merely scaled. Charged ionic is
close to merely scaled (shift -0.461 A, corr 0.977, scale 0.969, residual
21.2%). Neutral bound shift +0.437 A, corr 0.706, residual 70.8%. Neutral
ionic row is noise (4e-3 e) and is not a finding. In z the 0.525 eV sits in
four 1.5 A bins at 15-21 A (+0.569 eV, cumulative peak +0.666) with -0.141 eV
returned beyond 21 A.

Cavity is NOT clean, and the binary threshold was blind to it: in the DFT
transition shell 0.01<s<0.9 the s_diel correlation is 0.887 with mean ratio
0.850, and |grad s_diel| correlation 0.725 with ratio 0.759 (neutral 0.861 /
0.825 and 0.657 / 0.734). The threshold test had said 0.44% mismatch by
volume. Interpolating the DFT cavity onto the coarser model grid lowers its
gradient, i.e. biases the ratio UP, so the deficit is not a smoothing
artefact.

IONIC, charged: net exact, |q| 1.000 vs 1.006, but over-attracts 7.5%
(-2.091 vs -1.946 eV) with charge pulled inward -- 3.5-5 A holds 0.146 e
against DFT's 0.054.

env^0.5 retraction: it does NOT pull envelope weight inward. Shares move OUT
of the 1.5-2 A shell (20.6% vs env_b's 27.2%) into 2.5-3 A and beyond 5 A, so
it is a broadening. Whatever makes A+env^0.5 the only basis to pass on the
neutral frame, it is not inward migration.

Two open anomalies, neither attributable to basis capacity:
 A. env_b puts 45.8% (charged) / 46.4% (neutral) of its weight beyond 5 A
    from any atom, and the native |grad s_diel| agrees at 44.1% / 44.0%, so
    it is genuine weight, not the weighted mean being dragged by the 59% of
    the cell out there. A saturated switch should carry almost none there.
 B. At z 6.0-7.5 A inside the slab the DFT plane average is zero to 3e-13
    while the model puts -9.2e-06 / -8.8e-06 e/A^3, and the +15.3 / +16.5 V
    interior potential turns that into -0.074 / -0.067 eV of spurious
    coupling -- nearly identical in both frames, hence structural. The
    neighbouring bins carry the opposite sign, so it is charge-neutral
    ringing.
Both are measured by envelope_void_audit (this commit).

Gate defect found by running the smoke test twice on identical input:
pb_rms_last does not reproduce (3.56e-13 then 6.89e-13; 6.07e-13 then
1.57e-12), 3.3e-5 and 9.6e-5 relative against the 1e-8 floor, so it would
fail the 1e-6 gate however correct the port is, while every energy term was
bit-identical. Cancellation-limited quantities are now reported by order of
magnitude, not gated.

Convergence provenance now exported and clean on both frames: fixed-point
exits on tolerance after 7 steps (minimum 5, cap 60), Newton on tolerance
with 7-8 total outer iterations (cap 12 per call). The 80-round-idling
concern is closed for this path with evidence rather than by assertion.

## CORRECTION to the cross-run entry above: the gated fields are not bit-identical
Reported by the workstation session 2026-09-09, correcting its own earlier
claim which I had propagated into commit messages ea430d5 onward and into the
NONGATED rationale in smoke_test.py. Two runs on the SAME machine with
identical code and identical input differ on every float except q_tot and the
integer-valued provenance fields:
  e_bl (sid 1)                2.2e-09 relative
  rho_layer_z_absint (sid 1)  1.3e-10
  delta_plane_max (sid 1)     1.1e-10
  comp_1d (sid 1)             8.5e-11
  e_s3d (sid 1)               4.6e-11
  e_xsol (sid 1)              1.4e-11
  everything else             <= 1e-12
The 1e-6 gate is unaffected (2.2e-09 is 2.7 orders inside it) and SMOKE PASS
stands, but: the tolerance must NOT be tightened below about 1e-7 on an
assumption of determinism, and a failure at the 1e-8 level would be this
jitter rather than a port defect. It also explains the worst gated deviation
moving 1.72e-09 -> 2.75e-10 between two runs with no code change: the same
e_bl field both times. The exit reason and iteration counts ARE exactly
reproducible across runs and machines and remain gated.

## Reruns at 40866f6 (workstation): no substantive number moved
V1 identical to the digit on both frames, including the floor lines
+0.2780 / +0.2390 / +0.0036 / -0.1587 eV charged and -0.4078 / -0.4089 /
-0.4399 / -0.4681 neutral, and the 3e-01-1e+00 bin at +1.896 eV on 4.06% of
volume and 78.67% of the envelope. V3 identical: 2.699e-07 near the Nyquist
against DFT 7.641e-05, interior slice 1.241e-02 e against 3.346e-08 e costing
-0.0674 eV (neutral 1.028e-02 e, -0.0550 eV). Shift-scan BEST-by-residual
lines unchanged: charged bound +0.075 A / 0.8711 / 48.3%, charged ionic
+0.400 A / 0.9911 / 1.4%, neutral bound +0.150 A / 0.6773 / 64.9%. Outside
the three fixed lines the only difference anywhere is 1e-15-level jitter in
the bulk gradient bins. Neutral ionic is now skipped by the tightened guard.
Smoke at 40866f6: SMOKE PASS, worst 2.75e-10 over 52 gated numbers.

## Second wording defect of mine, fixed in this commit
envelope_void_audit printed the WITHDRAWN creeping-switch hypothesis as an
assertion immediately above the two lines that refute it: the model carries
1.06% of its gradient weight inside its own plateau region against DFT's
1.15% -- less, not more (neutral 1.07% against 1.16%). The block is a test of
the plateau LEVEL only and is now labelled as such; the far-field envelope
share is accounted for in full by the second dielectric interface. Those
numbers also close that account from the other side: 1.06% in the plateau
plus about 45.8% beyond z = 37.5 A leaves nothing unexplained.

## Provenance and a caveat on the jitter figure (2026-09-09)
Two methodological points from the workstation session.

Job provenance. The LS6 job scripts already printed the mace HEAD, so the
"yours is not pinned" premise was only half right -- but the half that stands
matters twice over: they printed no pb-repo HEAD, and the script that actually
runs is a COPY in a scratch directory, so a HEAD line is a proxy and not a
guarantee that the executed file matches the committed one. Both job scripts
now print the mace HEAD, the pb HEAD, the sha256 of the file being executed,
and an explicit IDENTICAL/DIFFERS comparison against the committed copy, so
the log certifies itself. Checked at patch time: both copies IDENTICAL.

The jitter figure. 2.2e-09 must be quoted as "at least 2.2e-09 observed", not
as the jitter floor: it is one field from two runs on one machine and one
card. Enough to rule out determinism and to make a sub-1e-7 tolerance
imprudent; not a characterised bound. Treating it as one would repeat, in a
new form, the very mistake it corrects. Characterising it needs repeats across
runs and both cards, which the workstation can do cheaply if wanted.

## NONGATED decade spreads, workstation gated run at 40866f6 (last numbers in)
  sid 1    pb_rms_last        1.417724204e-12 vs A100 2.687221277e-13  0.72 decades
           pb_fix_res         6.191158697e-12 vs      3.044675623e-12  0.31
           pb_newton_rms_last same as rms                              0.72
  sid 601  pb_rms_last        1.204409717e-12 vs      1.135462900e-12  0.03
           pb_fix_res         2.917670550e-10 vs      2.894502416e-10  0.00
           pb_newton_rms_last same as rms                              0.03
DGEMM in that run 1173 GFLOP/s. The 0.72 decades on sid 1 is the widest
spread either side has seen on that field, against 0.15 in the earlier gated
run, which strengthens the "at least 2.2e-09 observed, NOT a characterised
bound" wording rather than weakening it. No gated number and no conclusion
changes. Full per-z tables from plane_profile_audit are in the pushed logs on
kit-results-4080 at 388e7dc, not only in summary form.

## Close-out 2026-09-09: 3426543 / 3426544 / 3426545 cancelled, not run
Cancelled on the user's prompting, and the reasoning is that the premise for
keeping them had already been met. They were kept as independent-hardware
cross-checks at a time when no cross-machine validation of the diagnostic
chain existed. Job 3426230 then reproduced the workstation's
envelope_alignment output digit for digit on both frames, which validates the
chain these three scripts share -- DFT file parsing, cavity construction,
spectral gradients, interpolation, binning. A second and third script through
that same validated chain has low marginal value: the expected output is
"identical", at a cost of another hour of queue.
Specifically: 3426543 (void audit) and 3426544 (shift scan) would have been
repeats, and the workstation's own rerun at 40866f6 already showed V1 and V3
identical to the digit against its earlier run. 3426545 would have produced
only a warmed A100 big-FFT benchmark value -- a hardware number, not physics,
already excluded from the gate and documented as absent in the reference JSON.
Nothing in the record depends on any of the three.

## THREE DOWNGRADES to the final summary (user review 2026-09-09)
All three are the same category error in different clothes: converting a
recovered fraction, or a currently-absent quantity, into a causal share or an
impossibility claim.

(a) "+0.278 eV is an unrecoverable floor" -- WITHDRAWN. A small envelope and a
    currently small residual do not imply that no coefficients could correct
    that region: the coefficients multiply the envelope and env_b < 1e-4 is
    not zero. "The cost would be self-energy" is a separate argument and one
    already conceded not to be a theorem. Correct statement: the EXISTING gap
    in that region is +0.278 eV, 15.2% of the 1.832 eV bound gap.

(b) "the bound channel is three fifths position, two fifths shape" --
    WITHDRAWN, and the original log contradicts it. I converted "shifting by
    the measured +0.450 A recovers 62.1% of the cross-energy deficit" into a
    causal apportionment. The profile-best shift is +0.075 A with the residual
    moving only 48.5% -> 48.3%, and at +0.450 A the residual RISES to 53.5%:
    the shift that recovers most of the energy makes the shape fit worse. So
    no share of the defect "is position". What stands is the IONIC channel,
    where the evidence is much stronger: both criteria agree at +0.40/+0.45 A
    and the residual collapses 21.2% -> 1.4% at scale 0.9911.

(c) "Gibbs ringing is refuted" -- DOWNGRADED to cause undetermined. Ringing
    from a band-limited reconstruction appears at the band limit of that
    reconstruction, which can sit well below the DFT grid's Nyquist, and the
    band compared here is defined by the profile length. Less high-frequency
    power than the reference does not exclude it. No new job for this.

## Plan fixed by the user 2026-09-09: density check first, then substitution
Not starting: repeat validation, long training, basis extension.

Step 1, density check, with the criteria corrected by the user: use the actual
parameters, the actual grid and the FULL cavity generation pipeline, pointwise,
and aggregate afterwards -- do NOT substitute a mean density into a switch
formula. Reproducing 0.9444 proves only that the computation path matches.
Support for "the density tail causes the plateau deficit" requires that
replacing ONLY the suspect tail restores the plateau. A failure to reproduce
means finding which step differs, not declaring the recipe or parameters
wrong. This must not turn into a large waiting project.

Step 2, substitution experiment, and it must keep a refit-on-the-ORIGINAL-
cavity control. With potential, 1-D background and grid fixed, a 2x2:
  envelope source | original coefficients | jointly refit by the same method
  model cavity    | baseline              | is the present representation usable
  DFT cavity      | effect of substitution| is it fittable after substitution
Every cell reports point error, cross energy, self energy and charge
amplitude together. If only DFT-cavity-with-refit succeeds, that cannot be
attributed to "just the loss function"; if both fail, that does not prove
full-basis capacity is insufficient. The 1-D solver's cavity substitution is
recorded SEPARATELY, checking whether the ionic layer displacement and the
0.525 eV improve, so it is not mixed with the lateral 1.307 eV.

## STEP 1 RESULT: the density-tail hypothesis is NOT supported
Single run on the workstation, mace 2a2edb2, pb 9b3b9ba, executed file sha256
a72d8b74cb502abfb61ae17f9a1ce06610a59bf2cff9690fd9721e769a2f73ff verified
identical to the committed file. No cross-check (repeat validation ruled out).
Log on kit-results-4080 at 1e54c24.

Part A, path consistency: max, mean and 99th-percentile pointwise difference
all exactly 0.000e+00, with the self-measured call-to-call floor also
0.000e+00 -- within one run this path is bitwise reproducible, unlike the
between-run energy scalars, so the jitter concern does not apply here. The
workstation checked the code before trusting an exact zero: s_re is an
independent create_cavity_torch call and s_used is the captured forward-pass
cavity, so it is a genuine recompute. Ceilings 0.9447 both. Per the user's
criterion this proves the computation path only and is NOT evidence.

Part C, the decisive one: the tail swap closes 0.0% of the plateau gap in
BOTH directions on BOTH frames. Model grid with the DFT tail inside the mask
stays at 0.9444 (target 1.0000); native grid with the model tail inside the
mask stays at 1.0000 (target 0.9444). The plateau moves by under 6e-09 while
the two plateaus differ by 0.0556. The swap is real in code (torch.where then
a fresh create_cavity_torch), so this is not a no-op. By the user's criterion
-- support requires that replacing ONLY the suspect tail restores the plateau
-- there is no support for the hypothesis.

Part D: the suspect tail IS real. Model n_e inside the mask is about 4x DFT's
at the mean (2.109e-05 vs 5.206e-06 charged) and 4.7x at the 99th percentile.
But every percentile through the 99th is at least three orders below
NC_K = 0.015 and even the maxima are below it, which is consistent with the
swap doing nothing there. Parameters in use: NC_K=0.015, SIGMA_K=0.6,
TAU=0.009.

Part B contributed nothing; see the defect below.

WHAT IS LOCATED: not the model's own cavity construction (part A is bitwise),
and not n_e inside the mask (swapping it either way moves the plateau by under
6e-09 while the plateaus still differ by 0.0556).
WHAT IS NOT LOCATED: where the difference is carried. Both densities are deep
below threshold throughout the mask and the pipeline is non-local, so the mask
is not where the cavity in that region is decided -- a non-local pipeline can
set s inside the mask from density OUTSIDE it, and the parameter list shows
I_NLOC_SOL, LNLDIEL and LNLION in use. The workstation flags "swap outside the
mask, or widen the mask until the plateau moves" as the next measurement, not
as the answer, and did not write it. No pronouncement on the recipe or the
parameters.

## Two defects in density_tail_test, fixed in this commit
1. The max(signed, 1e-30) overflow AGAIN, in part C's r2 -- the identical
   pattern I had already been corrected on in profile_shift_scan, rewritten
   from scratch in a new file. It printed -1.9e23% (neutral -5.7e23%) where
   the correct value is +0.0% in both. Now a signed denominator with an
   abs > 1e-12 guard printing "n/a" instead of a number.
2. Part B offered two MUTUALLY EXCLUSIVE readings and the second voids the
   first: the within-bin spread is 0.39-0.49 on a quantity bounded in [0,1],
   so the pipeline is not a pointwise function of n_e, and under a non-local
   map two fields with different spatial structure give different binned means
   with the map identical. The table therefore separates nothing. It now
   decides from the measured spread and prints that verdict rather than
   leaving the reader to choose. Note for the record: the user's instruction
   not to substitute a mean density into a switch formula was more far-reaching
   than I understood -- the map is not local at all, so no "plug in a density"
   reasoning was valid in either direction from the start.

## STEP 2 RESULT: the cavity-substitution 2x2, all four cells
LS6 gpu-a100-dev, jobs 3426779 (MACE_CAVSRC=model) and 3426780 (dft), code
890eb3e, script sha256 6975b5347f6caaf1 verified IDENTICAL to the committed
copy in both logs. 8m19s and similar; peak GPU 10.96 and 11.01 GiB, which
confirms the memory arithmetic and that a 24 GB card was never the limit.
Self-check 0.000e+00 both ways on both frames.

Cross-machine check, free of charge: the model-envelope cells reproduce the
workstation's incomplete run exactly (charged original coefficients cross
-0.9350 / 0.4171x, self +0.6118 / 0.8959x, |q| 1.7805 / 0.7010x), and the
refit cells reproduce the original joint_subspace_solve numbers exactly
(charged A minself 3.153, rms 0.965, self 3.153, |q| 1.694; neutral A 0.155,
0.286, 0.847, 1.096). The copy is faithful, so "the same joint objective"
holds.

CHARGED, reference lateral cross -2.2417, self +0.6829, |q| 2.5398 e
  envelope      | original coefficients          | refit, basis A
  model cavity  | cross 0.4171x self 0.8959x     | minself 3.153x INFEASIBLE
                | |q| 0.7010x                    | rms 0.965 (cap 0.294) FAIL
  DFT-rebuilt   | cross 1.0455x self 2.7520x     | minself 1.606x INFEASIBLE
                | |q| 1.1533x                    | rms 1.127 (cap 0.248) FAIL

NEUTRAL, reference lateral cross -0.4649, self +0.6601, |q| 2.1822 e
  model cavity  | cross 1.6441x self 0.9183x     | minself 0.155x feasible
                | |q| 0.7657x                    | rms 0.286 self 0.847 FAIL
  DFT-rebuilt   | cross 5.2788x self 3.3379x     | minself 0.150x feasible
                | |q| 1.3859x                    | rms 0.187 self 1.000 PASS

READING, with the user's two advance rulings applied.
Direct substitution, charged: the cross energy goes from 0.4171x to 1.0455x
of the reference by swapping the envelope alone with the coefficients
untouched -- from 42% to 105% of the attraction. But it does NOT come alone:
self-energy goes 0.8959x -> 2.7520x and |q| 0.7010x -> 1.1533x. An
energy-only reading would call this a fix; the four-quantity requirement is
what shows it is not. The charge produced is nearly three times too
self-interacting.
Direct substitution, neutral: much WORSE, cross 1.6441x -> 5.2788x, self
0.9183x -> 3.3379x. So direct substitution with the trained coefficients is
not an improvement in general; it is a redistribution that happens to suit
one frame.
Refit, charged: the obstruction HALVES but survives -- the minimum attainable
self-energy at cross = reference drops 3.153x -> 1.606x. Both cells FAIL. By
the user's ruling this does not prove full-basis capacity is insufficient; it
speaks for basis A and this 8-vector family.
Refit, neutral: FAIL -> PASS on all four conditions with the same method and
the same basis. By the user's ruling this may NOT be attributed to "just the
loss function".

WHAT IS NOT MARGINAL and what is. The neutral PASS/FAIL flip itself grazes
thresholds I chose: the model-cavity cell fails the self condition by 0.003
(0.847 against a 0.15 window) and the DFT-cavity cell passes with |q| 1.195
against a 1.20 limit and rms 0.187 against a 0.195 cap. The flip should not
be quoted as a large effect. What IS substantial and threshold-free: the
neutral fit quality improves throughout -- plain rms 0.213 -> 0.130,
constrained rms 0.286 -> 0.187 -- and the self-energy lands exactly at the
reference. On the charged frame the plain fit also improves (0.196 -> 0.165)
while the constrained fit worsens (0.965 -> 1.127), and the minself halving
is the largest single change.

A CAVEAT THAT MATTERS FOR THE PAUSED PLATEAU QUESTION. The DFT-rebuilt cavity
has s_diel ceiling 0.9447, IDENTICAL to the model's, because it is built from
the interpolated DFT density on the model grid. So the substitution changed
the cavity's spatial structure, not the plateau level, and the large charged
cross-energy change came from structure alone. On this evidence the plateau
level is not what governs the charge fitting. Flagged once, not chased: the
user has paused that line. Also note this says the ceiling on the model grid
is grid-determined rather than density-determined, which is a datum for that
question whenever it resumes.
Standing caveats: the DFT density carried onto the coarser model grid is
smoothed, so this is not the native-grid DFT cavity; basis A and an 8-vector
family only; and the 1-D cavity substitution is NOT in this experiment -- it
needs a re-solve and is to be compared separately, and it is not written.

Defect in my own instrumentation: the host-RAM sampler reported rss_GB 0.0
throughout because it read $$ inside the subshell, which is the subshell's
own PID rather than the python process. The MemAvailable column is valid
(238 GB flat); the process RSS column measured nothing, which is the one
thing I added it for.

## 1-D CAVITY SUBSTITUTION (job 3426811, code 9216d28): ISOLATION FAILED
Script sha256 e856ba84093bf38f, verified IDENTICAL to the committed copy.
Both solves converged on their criteria on both frames (fixed-point tol after
7 steps of a 60 cap; Newton tol with 8 and 7 outer of a 12 cap), so nothing
below is a convergence artefact, and max |ne_dft - ne_model| = 1.618 e/A^3
confirms the substitution took effect.

THE CHECK I BUILT FOR THIS FIRED. "solute potential identical across the two
solves" reads max |d cvhar3| 4.142e-01 eV on the charged frame and 4.265e-01
on the neutral one, not zero. So the two solves did NOT share the same solute
potential and the intervention did not isolate the cavity, which is what the
design required. Verified beforehand and still true: closure_from_fields uses
its n_e_density argument only in tp.create_cavity_torch. What I did not
account for is that after the solve the model side responds -- constant
potential and charge equilibration make the solute answer a changed solvent
-- or, alternatively, that my wrapper captures the last of several closure
calls and the two runs took different paths through them. Which of those it
is I have NOT determined. Either way this run cannot attribute anything to
the cavity, and the numbers below answer a different question: rebuild the
cavity from the DFT density and re-solve EVERYTHING.

As that different question, on one fixed potential taken from run 1:
  charged BOUND  coupling 0.496 -> 0.436 x DFT; displacement -0.423 -> -1.419
                 A; residual 48.5% -> 52.3%; int|.| 2.0275 -> 2.2710; the
                 0.525 eV gap becomes 0.588 eV, 11.9% WORSE
  charged IONIC  coupling 1.074 -> 1.084; displacement -0.461 -> -0.522 A, so
                 the ~0.42 A displacement does NOT shrink; residual 21.2% ->
                 24.5%; gap +0.144 -> +0.165, 13.9% worse
  neutral BOUND  coupling 1.020 -> 1.684; displacement +0.437 -> -1.394 A;
                 correlation 0.706 -> 0.469; residual 70.8% -> 88.3%; int|.|
                 0.377 -> 0.802
  neutral IONIC  4e-3 e of charge, correlation about zero, shift +4.5 A --
                 noise, and it should have been suppressed
So on this run "the more accurate cavity also corrects the background charge
position" is not supported and the movement is in the opposite direction --
limited, because of the failed isolation, to "after the solute potential
moved with it".

TWO PRESENTATION DEFECTS OF MINE
- The neutral BOUND line printed "-3254.9% of it closed": arithmetically
  right, meaningless, because the denominator is a near-zero baseline
  (+0.0107 eV). My guard tests abs(den) > 1e-12, which 0.0107 passes. A
  percentage needs a denominator that is large compared with the effect, not
  merely non-zero. The absolute values are the statement: +0.0107 ->
  +0.3591 eV.
- No negligible-channel guard on the neutral ionic row, although I had added
  exactly that guard to profile_shift_scan the same afternoon after the
  workstation found it there. Building a guard and not carrying it to the next
  script is the same failure as not having built it.

## THE 1-D CAVITY CONTROL, PROPERLY ISOLATED (job 3426875, code 56fe2d8)
Script sha256 97dfd0e2563a33d4, IDENTICAL to the committed copy. 3m16s.
One full model call; the low-level solver called directly twice from the
captured inputs, differing ONLY in the three cavity-derived arguments
(s_ion, a1, p_off). Solve grid nz_s 600 (upsample f=2).

ALL FOUR GATES PASS, both frames, with exact zeros:
  G1 baseline reproduces the model's own profiles: max abs 0.000e+00 for
     both bound and ionic -- bitwise, not merely close
  G2 same tensor objects True; max |d cvhar_z| 0.000e+00 eV; q_sol identical
  G3 max |d dphi/dz| 0.000e+00 eV/A; mean cvhar_z +0.000000 in both groups,
     so the reference zero is shared
  G4 both solves exit on their criteria
Cavity input differs as intended: max |ne_dft - ne_model| 1.618 / 1.613
e/A^3. So the control the previous attempt failed to establish now holds and
the numbers can be read.

The trap avoided: a1 = plane_mean(a3_scr) and a3_scr depends on the cavity AND
on phi_sol, so the substituted a1 must NOT come from a second full model call.
It is recomputed by calling closure_from_fields with the DFT density and RUN
1's phi_sol.

CHARGED FRAME, reference bound coupling -1.0955 eV, ionic -1.9362 eV
  BOUND  coupling 0.501x -> 0.395x; ABSOLUTE gap -0.5468 -> -0.6625 eV, i.e.
         0.116 eV WORSE; profile error 48.6% -> 65.6%; displacement by
         residual +0.075 -> -0.425 A
  IONIC  coupling 1.074x -> 1.085x; ABSOLUTE gap +0.1441 -> +0.1641 eV;
         profile error 1.4% -> 1.2%; displacement by residual +0.400 ->
         +0.450 A
NEUTRAL FRAME, reference bound coupling -0.5182 eV
  BOUND  coupling 1.031x -> 1.507x; ABSOLUTE gap +0.0160 -> +0.2628 eV, much
         worse; profile error 66.1% -> 57.1%, BETTER; displacement by
         residual +0.150 -> +0.375 A
  IONIC  skipped by the negligible-channel guard (reference coupling
         +0.000550 eV on 0.0041 e). The guard was carried over this time.

ANSWERS to the three intended questions, control established:
1. The ionic layer displacement does NOT shrink. It stays at 0.40 -> 0.45 A
   by residual (0.450 -> 0.500 by coupling). The cavity is not the source of
   that displacement.
2. The bound profile error moves in OPPOSITE directions on the two frames:
   worse on charged (48.6% -> 65.6%), better on neutral (66.1% -> 57.1%).
3. The absolute cross-energy gap gets worse everywhere: charged bound
   -0.547 -> -0.663 eV, charged ionic +0.144 -> +0.164, neutral bound
   +0.016 -> +0.263.
So substituting a DFT-density-built cavity, with the solute side rigorously
fixed, does not improve any of the three. The only improvements are in
profile SHAPE (neutral bound, and marginally the charged ionic) while the
corresponding couplings worsen -- shape and energy again moving oppositely,
the same pattern the shift scan showed.

TWO THINGS NOT TO MISREAD, both mine to flag:
- These numbers live on the SOLVE grid (nz_s 600) with the 1-D cvhar_z
  potential, whereas the earlier 0.525 eV figure came from the model grid
  with the plane-averaged 3-D potential. The reference bound coupling here is
  -1.0955 against -1.0429 there. So compare WITHIN this run, model cavity
  against DFT cavity, and do not set 0.5468 against 0.525 as though they were
  the same measurement.
- The charged bound "best-by-coupling" shift for the DFT cavity reads +1.500
  A, which is exactly the edge of the +-1.5 A scan window. That value is
  CENSORED, not measured, and must not be quoted as a shift. The
  by-residual column (-0.425 A) is interior and valid.

## 2x2 OF CAVITY SOURCE AGAINST THE LEARNED CORRECTION (job 3426892, code c80e0bd)
Script sha256 23ab439d31035cf4, IDENTICAL to the committed copy. 3m11s. All
four gates PASS on both frames, now covering all four solves (G1 bitwise
0.000e+00, G2 same tensor objects with max |d cvhar_z| 0.000e+00, G3 max
|d dphi/dz| 0.000e+00 with mean cvhar_z +0.000000 in every group, G4 all four
exit on their criteria).

Added because the first pass kept the OLD learned correction while swapping
the cavity (p_off = new prior + old delta_p). delta_p was trained against the
model's own cavity, so that was a mismatched combination and the earlier
conclusion "the cavity is not the source of the displacement" was too strong.

Size of the removed term: delta_p rms 9.6013e-04 against prior rms 3.8408e-02
on the charged frame, so 2.5% of the prior; 8.1472e-04 against 4.0056e-02,
2.0%, on the neutral one.

CHARGED BOUND, reference coupling -1.0955 eV
  cell                        gap (eV)   profile   shift (A)
  model cavity, delta_p ON     -0.5468     48.6%      +0.075
  DFT cavity,   delta_p ON     -0.6625     65.6%      -0.425
  model cavity, delta_p OFF    -0.8033     59.5%      +0.100
  DFT cavity,   delta_p OFF    -0.8595     66.9%      -0.425
CHARGED IONIC, reference -1.9362 eV: all four cells at +0.1441 / +0.1641 /
  +0.1441 / +0.1642 eV, profile 1.4 / 1.2 / 1.4 / 1.2%, shift +0.400 / +0.450
  / +0.400 / +0.450 A. The learned correction does not touch this channel.
NEUTRAL BOUND, reference -0.5182 eV
  model cavity, delta_p ON     +0.0160     66.1%      +0.150
  DFT cavity,   delta_p ON     +0.2628     57.1%      +0.375
  model cavity, delta_p OFF    +0.0051     55.5%      +0.100
  DFT cavity,   delta_p OFF    +0.2610     59.4%      +0.350
NEUTRAL IONIC: all four skipped by the negligible-channel guard.

WHAT IT SETTLES, against the user's three readings.
- The 2.5%-of-prior correction is NOT cosmetic on the charged bound channel:
  removing it costs 0.257 eV of coupling (-0.5468 -> -0.8033) and 10.9 points
  of profile error, taking the coupling from 0.501x to 0.267x of reference.
- NO compensation relationship is unmasked. With the correction removed the
  DFT cavity is still WORSE than the model cavity on both frames: charged by
  -0.0562 eV and 7.4 profile points, neutral by +0.2559 eV. So the earlier
  negative for the rebuilt cavity was not an artefact of the mismatched
  correction.
- The charged frame therefore lands on the user's THIRD reading: with the
  correction removed both cavities are clearly wrong (0.267x and 0.215x of
  reference, profile 59.5% and 66.9%), so the next place to look is the
  solute potential or the 1-D closure approximation.
- The neutral frame lands on the FIRST reading, and only for the shape:
  removing the correction improves the profile error 66.1% -> 55.5%, 10.6
  points, while the gap improves only +0.0160 -> +0.0051 eV, which is a small
  absolute change on an already small gap. So on the neutral frame the
  learned correction is actively degrading the shape.
- The ionic displacement is explained by NEITHER: it sits at 0.400-0.450 A in
  all four cells and the correction changes its gap by 1e-4 eV. That is
  stronger than the previous run's statement and points the same way.

So the same correction acts in opposite directions on the two frames --
rescuing 0.257 eV on charged, costing 10.6 profile points on neutral -- which
is the same charged-against-neutral and shape-against-energy opposition that
has recurred throughout this investigation.

STILL LIMITED: this says a cavity rebuilt from the DFT density, with this
recipe and on this grid, does not help. It does not say the cavity is
correct. Both cavities still plateau at 0.9447 and the interpolation-smoothing
caveat stands. And the DFT-cavity "best-by-coupling" shift reads +1.500 A in
both delta_p states, exactly the edge of the +-1.5 A scan window: censored,
not measured, and not to be quoted as a shift. The by-residual column is
interior and valid.

## IONIC ORDER OF OPERATIONS (job 3426981, code a600938), charged frame only
Script sha256 cce1f42bfb641fc2, IDENTICAL to the committed copy. No model, no
re-solve; native DFT grid (168,168,500) throughout, so no interpolation.
params: LION True, LNLION True, theta_b 4.360394e-01, ZBETA 3.894110e+01,
n_max 2.762136e-03, invBETA 2.567981e-02.

A. THE 3-D PATH REPRODUCES RHOION EXACTLY, not approximately.
Sign measured rather than assumed: phi = -PHI_raw needs offset c = -0.000000
eV and gives pointwise int|d rho| 0.000000 e, 0.00% of int|rho_ref|, max
deviation 1.311e-12 e/A^3. phi = +PHI_raw gives 209.61% and is rejected.
So in one shot this validates the formula, the parameters, the cavity recipe
(s_ion from create_cavity_torch on the DFT density), the sign convention, the
units AND the reference zero -- and the required offset is ZERO, i.e. the
stored PHI already carries the correct electrolyte reference. No gauge
adjustment is needed anywhere, and the mean must not be subtracted.

THE SENSITIVITY IS THE OTHER HALF OF THE RESULT and it is severe:
  c -0.20 eV -> total +9.260 e      c -0.05 -> +6.978
  c  0.00    -> +0.999995 (target)  c +0.05 -> -5.830   c +0.20 -> -9.254
A 0.05 eV error in the potential changes the total ionic charge by a factor of
about 7. That is exactly ZBETA * 0.05 = 1.95 in the sinh argument and
exp(1.95) = 7.0, so the number is the expected exponential and not an
artefact. The ionic channel is exponentially sensitive to the potential
reference.

B. AVERAGING FIRST IS NOT THE PROBLEM.
  case                              total (e)   int|.| (e)   L1 vs ref   max dev
  DFT reference RHOION              +0.999995     0.999995    0.000000   0
  A: 3-D pointwise, then averaged   +0.999995     0.999995    0.000000   1.3e-12
  B: averaged first, same offset    +1.005279     1.005279    0.005326   2.4e-05
  B: averaged first, own offset     +0.999995     0.999995    0.009432   2.4e-05
The 1-D path's own neutrality offset is c1 +0.000035 eV against -0.000000,
itself a consequence of the order of operations. Its error is 0.0053 e in L1
with the shared offset, 0.0094 e with its own, against a 1.0 e total -- so
0.5 to 0.9%, and the largest plane-profile deviation is 2.4e-05 e/A^3. It
produces essentially no displacement.

READING, by the user's own criteria: A reproduces and B ALSO essentially
reproduces, so the averaging approximation is NOT the priority, and the
indicated next place is the model's potential and its boundary handling. The
sensitivity above makes that direction quantitative rather than a guess: a
sub-0.1 eV error in the model's solute potential is enough to move the ionic
charge by a large factor, and an order-of-operations error of 0.5% cannot
explain a 0.4 A layer displacement or the 7.4% ionic coupling gap
(+0.144 eV against a -1.936 eV reference).

DEFECT IN MY OUTPUT, the third of this class: the line "order-of-operations
penalty: L1 of the 1-D path is 12321156.54x the 3-D path's" divides by the
3-D path's L1, which is exactly zero because it reproduces. Arithmetically
fine, meaningless as printed. The absolute L1 values are the statement. Last
time I wrote in this ledger that a percentage needs a denominator large
compared with the effect rather than merely non-zero -- and then did not
apply it here.

## IONIC 2x2: TOTAL POTENTIAL AGAINST IONIC SWITCH (job 3427096, code 2f8b614)
Direct substitution, charged frame, no re-solve. Script sha256 b733e123...,
IDENTICAL to the committed copy (job 3427013 was the same script before the
sign fix; its 2x2 was correctly refused by its own gate).

MY BUG, caught by the gate in 3427013: the solver works in the
ELECTRON-ENERGY convention -- which is why the backend writes
rho_ion_z = -(n_ion/volume), n_work being odd -- while the DFT side here uses
-PHI_raw, i.e. PHYSICAL. Feeding the solver's phi straight in crossed the two
conventions and produced exactly the negated ionic charge (net -1.0000 e
against the model's +1.0000). Fixed by negating the model's total potential,
and the script now IDENTIFIES the convention by measurement: physical gives
6.505e-19, flipped gives 2.520e-03.

GATES both PASS: model phi + model s_ion reproduces the model's own rho_ion to
6.505e-19 e/A^3; DFT phi + DFT s_ion reproduces RHOION on this grid to
L1 0.58% (the 0.42% above the native-grid 0.00% is the averaging error plus
the band-limited 500 -> 600 resample).

  cell                     net (e)  int|.|    cross      gap       L1   shift  resid
  model phi, model s_ion   +1.0000  1.0000  -2.0802  +0.1417  0.18687  +0.375   1.4%
  DFT phi,   model s_ion   +1.1317  1.1317  -2.3314  +0.3929  0.13218  +0.375   0.9%
  model phi, DFT s_ion     +0.8863  0.8863  -1.7399  -0.1986  0.11403  +0.025   1.3%
  DFT phi,   DFT s_ion     +1.0053  1.0053  -1.9531  +0.0146  0.00583  +0.025   0.7%

THE READING SET FOR THIS RUN DOES NOT FIRE, and the answer is the other way
round. Swapping the TOTAL POTENTIAL alone leaves the ionic layer displacement
exactly where it was, +0.375 -> +0.375 A. Swapping the ionic SWITCH alone
collapses it, +0.375 -> +0.025 A. So with the potential held fixed the
displacement is carried by s_ion, not by the total potential, and there is no
direct evidence here for concentrating on how the model produces that
potential.

THAT IS NOT A CONTRADICTION of job 3426875/3426892, where substituting the
cavity WITH a re-solve left the displacement at 0.400 -> 0.450 A. The
difference is the re-solve: here the potential is held, so this measures the
DIRECT sensitivity; there the solve was allowed to respond and the resulting
potential moved the layer back. Together they say the layer position is a
self-consistent outcome of switch and solve together, and neither input owns
it. Both are measurements of different things, and this run measures direct
sensitivity only.

AND THE ENERGY SHOWS A CANCELLATION that makes single-factor attribution
invalid: |gap| is 0.1417 eV for the model pair, but 0.3929 with the DFT
potential alone and 0.1986 with the DFT switch alone -- BOTH single swaps are
worse -- and 0.0146 with both. So the model's total potential and its ionic
switch each carry an error and those errors partially cancel in the coupling
energy. Repairing either one in isolation would make the energy worse.

THE AVERAGING QUESTION, now on comparable quantities rather than an L1 against
a displacement, which is what the previous reading did wrongly:
  A: 3-D pointwise, then averaged   shift -0.000 A, resid 0.0%, gap -0.0000 eV
  B: averaged first, then 1-D       shift +0.025 A, resid 0.7%, gap +0.0146 eV
So averaging introduces displacement +0.025 A, profile error +0.7 points and
coupling -0.0146 eV, against the model's own +0.375 A and +0.1417 eV. That is
7% of the displacement and 10% of the gap. And in L1 the model's ionic profile
is 0.18687 e off the reference, 19% of the 1.0 e total, against the averaging
error's 0.58%. Averaging is not the main cause, now measured on like against
like.

## s_ion SUBSTITUTION WITH RE-SOLVE (job 3427122, code 7fe861f): IMPROVEMENT HOLDS
The single control the user specified. Only s_ion is replaced -- with the DFT
switch built exactly as in job 3427096 (native-grid cavity, plane-averaged,
band-limited 500 -> 600), fed in unchanged -- every other input held at the
baseline as the same tensor objects, and the solve re-run self-consistently.
Three gates PASS: baseline re-solve reproduces the model's own rho_ion to
0.000e+00; every other input identical with q_sol +1.000000; both solves exit
on their criteria.

  case                              net (e)    cross    eV/e    shift   resid      L1
  DFT reference                     +1.0000  -1.9385  -1.9385  -0.000    0.0%  0.00000
  baseline, model s_ion, re-solved  +1.0000  -2.0802  -2.0802  +0.375    1.4%  0.18687
  DFT s_ion, NO re-solve            +0.8863  -1.7399  -1.9630  +0.025    1.3%  0.11403
  DFT s_ion, RE-SOLVED              +1.0000  -1.9555  -1.9555  -0.000    1.0%  0.01764

Total charge restored first, as required before reading anything: 0.8863 ->
1.0000 e, matching the reference exactly.
The improvement HOLDS and strengthens. Displacement +0.375 -> -0.000 A, better
than the no-re-solve +0.025. Coupling per unit charge -2.0802 -> -1.9555
against the reference -1.9385, so the error falls from 0.1417 to 0.0170 eV/e,
88%; with the charge back at exactly 1 e the absolute gap is the same 0.0170
eV. L1 0.18687 -> 0.01764 e, 91%. Profile residual 1.4% -> 1.0%.

By the user's reading: investigate and fix along the ionic switch's GENERATION
PATH. This is the first clear "fix this" pointer in the investigation.

IT ALSO SETTLES THE EARLIER APPARENT CONTRADICTION, in the direction opposite
to the attribution I had made and withdrawn. Jobs 3426875/3426892 showed the
displacement NOT shrinking under a re-solve (0.400 -> 0.450 A), but they
substituted s_ion AND a1 AND p_off with the DFT density first interpolated
through the model grid. Replacing only s_ion, natively built, and re-solving
takes the displacement to zero. So that null result came from the other two
substitutions and/or the interpolation route, NOT from the re-solve. Now
demonstrated rather than asserted.

LIMITS, unchanged: a1 and p_off still carry the model's own cavity values, so
this is a controlled single-factor intervention and NOT a physically
consistent better solvent model. It does not explain WHY the model's s_ion
differs -- that is the generation path to look at. One frame. And the residual
1.0% profile error and 0.0170 eV gap are what remains after the switch is
fixed, so the switch accounts for roughly 88-91% of this frame's ionic-channel
error and not all of it.

## WHOLE-SYSTEM CHECK AFTER THE s_ion FIX (workstation, code a137d48)
Ran on the 4090 in about 15 seconds, no kill (peak RSS 1.9 GB, peak GPU 4445
MiB of 24564, host MemAvailable never below 241.9 GB). Executed file sha256
b5b39227... verified identical to the commit. Log on kit-results-4080 at
9adffd4; arrays in ion_switch_resolve_arrays.npz. LS6 job 3427287 was kept as
a fallback and cancelled unrun. New standing preference from the user: quick
tests go to the workstation first.
Convention check PASS: l0_inv reconstruction of phi - phi_sol leaves 6.916e-12
eV after removing the G=0 constant. All three gates PASS, including the
baseline re-solve reproducing the model's own rho_ion bitwise.

Cross-machine agreement for free: the [RESULT] table reproduces LS6 job
3427122 on every figure -- +0.375 -> -0.000 A, 0.1417 -> 0.0170 eV/e,
0.18687 -> 0.01764 e, total charge restored to +1.0000.

  case                    net (e)   chg L1   chg max     cross     self    total   d total
  DFT reference           +1.0000  0.00000  0.00e+00   -3.0243  +1.5237  -1.5006   +0.0000
  baseline, model s_ion   +1.0000  0.63739  2.38e-03   -2.6289  +1.4718  -1.1571   +0.3435
  DFT s_ion, re-solved    +1.0000  0.63760  2.38e-03   -2.6272  +1.4700  -1.1572   +0.3434

THE IONIC GAIN DOES NOT REACH THE WHOLE SYSTEM.
1. Charge sum: L1 0.63739 -> 0.63760, WORSE by 0.00021 e, and the max
   deviation does not move at all, while the ionic channel's own L1 improved
   10.6-fold.
2. Total energy: gap 0.3435 -> 0.3434, a change of 0.0001 eV, 0.03%.
3. Potential profile: all three measures worse by 0.8-1.8% (L1 2.7322 ->
   2.7548, max 0.2915 -> 0.2967, rms 0.11093 -> 0.11286).

A CORRECTION to the workstation's reading, verified by arithmetic: it wrote
"cross improves by 0.0017 and self worsens by 0.0018, so they cancel". BOTH
terms get worse -- |cross gap| 0.3954 -> 0.3971 and |self gap| 0.0519 ->
0.0537. The total is nearly unchanged because the SIGNED errors move in
opposite directions (+0.0017 and -0.0018), not because a gain is cancelled by
a loss. The distinction matters: this is not "the ionic gain was absorbed", it
is "both terms retreated slightly and the total happened to offset".

THE MUTUAL BOUND-ION TERM is the largest movement in the block and it goes
the wrong way: reference -1.3112, baseline -1.2824 (0.0288 off), substituted
-1.4111 (0.0999 off, and on the other side) -- 3.5x further from the
reference, overshooting. Computing the self-energy as the sum of the two
separate self-energies would not show this term at all, which is exactly why
the user required the summed field.

AN INFERENCE REFUSED, correctly, by the workstation and recorded here: the
sum's L1 rose 0.00021 e while the ionic channel's fell 0.16923 e, and it is
tempting to conclude the bound channel degraded by about 0.169 e to absorb it.
That does not follow -- L1 of a sum is not the sum of L1s -- so the bound
channel's own change cannot be recovered from these two numbers. Same shape as
int|a+b| against int|a|+int|b| and as reading a magnitude ratio as an energy
claim, both of which this project has already been caught by.

VERDICT by the user's decision tree: the ionic branch is PAUSED. The ionic
channel improves 10.6-fold in isolation, the aggregate gain is 0.03% of the
energy gap, and the charge sum, the mutual term and all three potential
measures move slightly the wrong way. Focus returns to the bound charge and
its lateral distribution.
OPERATIVE CAVEAT: the substituted group's bound charge comes from its OWN
re-solve, so these tables show the NET of the ionic improvement and the bound
channel's response to it, and that net is zero in energy. Separating the two
needs the bound channel scored on its own, which this run does not do.

CORRECTION, supplied by the user immediately after the above was written, and
it overturns the last paragraph. The bound channel CAN be scored on its own
from the numbers already in hand, because the cross coupling is linear in the
charge: cross = int rho phi with rho = rho_bound + rho_ion, so the couplings
of the two channels ADD and the bound term is simply (total - ionic):

  coupling error, model - reference (eV)   baseline   after the s_ion swap
    bound                                   +0.5371          +0.4141
    ionic                                   -0.1417          -0.0170
    total                                   +0.3954          +0.3971

BOTH channels improved -- bound by 0.1230, ionic by 0.1247 -- and their errors
have OPPOSITE signs: the bound channel under-attracts, the ionic over-attracts.
Each moving closer to the reference therefore removed part of a mutual
cancellation and left the total slightly worse. My own reading two paragraphs
up, "both terms get worse", is arithmetically correct about the TOTALS and
wrong at the channel level, which is the level that carries the physics. The
result must NOT be read as "the bound charge was degraded to absorb the ionic
gain": nothing here supports that, and the decomposition contradicts it.

What still does not follow from the coupling: whether the bound DISTRIBUTION
improved. Coupling is one scalar projection of the profile onto the solute
potential; two different profiles can share it. That requires comparing the
arrays directly, with no shift and no scale.

SCOPE OF THE WHOLE-SYSTEM TABLE, which it failed to state: every number in it
is the PLANE-AVERAGED 1-D electrostatic total. None of the lateral 3-D error
is in any of them, so nothing in that table bears on the 3-D fit, whose only
responsibility remains the lateral shortfall.

## 2026-09-10  bound_response_test.py -- driven wrong, or responds wrong?
code 940549c (parent 3f1518d added the script, b947b7e the whole-system entry
above). Frame sid 1 only. One model forward to capture the solver kwargs, then
n_b = B @ phi + nb_off evaluated directly with a1 and p_off HELD FIXED at the
captured baseline and only the driving potential swapped, model -> DFT. No
self-consistent re-solve, so nothing else can move. Compared against the
plane-averaged DFT RHOB directly, no shift and no scale.

The branch, set by the user in advance: if the bound distribution visibly
recovers, the priority becomes how the total potential is generated; if it
stays clearly wrong, the priority is the bound response itself -- the cavity,
the 1-D closure and the learned correction. Either way the point is to
separate "the potential driving it is wrong" from "its response to the
potential is wrong", which is more targeted than extending the basis or
retraining, and any later fix is still judged on aggregate charge, potential
and energy together.

Three things measured rather than assumed, after this chain's history: the
relative sign of PHI_raw and the solver's phi, by correlation with the mean
removed, with both signs reported; whether a constant offset in phi can reach
the bound charge at all, via |B @ 1| against |B @ phi|; and how much the two
driving potentials actually differ, since a near-identical pair would make the
swap vacuous. Plus a cross-check that this run's baseline bound coupling gap
reproduces the +0.5371 eV obtained above by the different route.

Ran on the 4090 workstation, ~10 s, code 52ea73c after one round of
correction (first run 940549c). All gates PASS. Provenance certified: sha256
of the executed file IDENTICAL to the commit, peak RSS 2.0 GB.

GATES AND CONVENTIONS, all measured:
  n_b = B @ phi + nb_off reproduces the model's own bound charge to exactly
    0.000e+00 -- the response really is that linear map.
  max |B @ 1| 1.219e-10 against max |B @ phi| 4.128e+03, ratio 2.95e-14. B
    annihilates constants outright, so the electrolyte reference zero CANNOT
    reach the bound charge at all. That closes the reference-zero question for
    this channel by measurement rather than by reading the formula.
  sign by correlation, mean removed: +PHI_raw +0.9995, -PHI_raw -0.9995.
    Separates cleanly; +PHI_raw is the solver's convention, as expected from
    physical = -out["phi"] and physical = -PHI_raw both having been measured.
  room to act: the two driving potentials differ by rms 0.11093 eV, 0.10304
    with the constant removed. Not vacuous.
  CROSS-CHECK: the baseline bound coupling gap comes out +0.5370 eV here
    against the +0.5371 obtained by total-minus-ionic from the whole-system
    run. Two routes, same quantity, agreeing to 0.0001 eV.

[RESULT] response fixed, driving potential swapped, no re-solve
                          case   net (e)   int|.|     cross       gap        L1   max dev    shift   resid
            DFT reference RHOB   +0.0000   2.0193   -1.0857   +0.0000   0.00000  0.00e+00   -0.000    0.0%
     model response, model phi   +0.0000   2.0302   -0.5487   +0.5370   0.80003  2.38e-03   +0.050   48.7%
       model response, DFT phi   +0.0000   8.5965   +4.4975   +5.5832   7.56925  2.05e-02   +0.825   85.0%
No row's displacement is at the +-2.0 A scan edge. The coupling flips sign
from attraction to repulsion; int|.| goes 4.3x too large; L1 rises 9.5-fold.

VERDICT on the user's branch: the bound distribution does NOT recover when
the correct potential is supplied -- it gets substantially worse. By the rule
set in advance, the priority is therefore the bound RESPONSE itself: the
cavity, the 1-D closure and the learned correction, not the generation of the
total potential.

WHAT THE TEST IS AND IS NOT. It tests the response OPERATOR, not the fixed
point: a correct linear response fed the correct driver must return the
correct answer whether or not it is self-consistent, so "no re-solve" is what
makes the reading possible rather than what spoils it. The one real limit is
that plane-averaging does not commute with the response -- <a1 E> is not
<a1><E> -- so some of the discrepancy belongs to the 1-D closure's averaging
order rather than to a1 or p_off being wrong. That is inside the user's own
list, so it does not change the branch, but it does mean the sub-part is not
yet isolated. The 7% figure measured for the ionic channel's averaging order
must NOT be imported here; the bound channel's is unmeasured.

A STRUCTURAL FACT that came out of the mechanism block and sharpens the
direction: nb_off/V has sum|.| 7.2847 and B@phi_model/V has 7.3768, against a
result of 0.14266 -- each term is 51.7x the bound charge it produces, their
summed magnitude 102.8x. So the model reaches the RIGHT bound-charge magnitude
(int|.| 2.0302 e against the reference 2.0193 e) as a small residue of two
much larger terms, and that residue is calibrated to the model's own
potential: supply a different one and the magnitude goes 4.3x too large. Since
nb_off is a FIXED array with no phi dependence, nothing compensates.

The equivalent statement needs care, and the workstation was right to push on
my first phrasing of it. "a1 E is close to -p_off up to a constant, so the
model's total polarization is nearly flat in z" is true but buries the
content: n_b is essentially -dP/dz, so n_b being small makes P nearly flat
almost by construction, and stated that way the finding sounds trivial. The
non-trivial part is that the two CONTRIBUTIONS to dP/dz are each about fifty
times their own sum -- sum|B@phi/V| 7.3768 and sum|nb_off/V| 7.2847 against
0.14266 -- so a1 E and p_off each vary strongly in z and their variations
cancel to 2%. That is what makes separating a1 from p_off worth doing.

And flat is not small: P is nearly constant in z, but its actual VALUE is set
by that constant, which is a separate unknown from its flatness and is exactly
what any integral formulation of P has to pin down.

TWO CORRECTIONS I MADE TO THE WORKSTATION'S FIRST READING, both then confirmed
by measurement. It read the table as a broken cancellation and argued a 1%
perturbation of B@phi explained the whole degradation, so the swap was
ill-posed. The conclusion (do not read the dichotomy off the raw table) was
right; the mechanism was wrong twice over:
  1. nb_off is IDENTICAL in both arms, so it cancels exactly out of the
     difference -- rho(dft) - rho(model) = -(B @ dphi)/V, with no nb_off in
     it. Verified at 8.361e-16. A near-cancellation common to both arms cannot
     amplify the difference between them.
  2. "1.03% smaller in aggregate" is a difference of L1 norms, not the norm of
     the difference -- int|a+b| against int|a|+int|b| yet again. The triangle
     inequality alone bounded the real perturbation at 6.3-10.1% of B@phi; it
     measures 7.1%, six to ten times the quoted figure. And the result moves
     1.00x the perturbation: the amplification is exactly none.
What stands is that B @ dphi is simply large next to a small bound charge --
0.5206 in density units, which is 7.41 e in L1 against a reference int|.| of
2.0193 e, so the swap changes the bound charge by 3.7x the entire reference.
That is a statement about the bound charge being small, not about tuning.

THE OPERATOR GAIN IS FALLING, NOT RISING, so the "double derivative amplifies
the top of the band" reading was backwards in direction, not merely in degree.
B applied to unit cosines is BAND-PASS, peaking near mode 60 (k about 8.4/A)
and falling fifteen orders of magnitude by mode 300 (2.37e-15): the two
Gaussian smoothings in B = V WB D diag(a1) D WB beat the two derivatives
comfortably. So the 1.81x difference between the potentials in the top half of
the band cannot drive anything -- the gain there is 1e-15. Nearly the entire
potential difference sits in modes 1-30 (rms 0.10272 of 0.10304), where the
gain is still rising.

THE STAGED SWAP FAILED AS DESIGNED and only its endpoint is readable. Swapping
only dphi below a cutoff gives L1 27.6, 23.6, 36.7, 40.8, 24.8, 16.0, 8.58,
7.58, 7.57 for cutoffs 5, 10, 20, 30, 50, 75, 100, 150, 250 -- every partial
swap is far WORSE than the complete one, worst of all at modes 1-30, which is
where essentially the whole potential difference lives. The cause is measured:
the per-band L1s of B @ dphi sum to 6.12412 against an actual 0.52056, an
11.8-fold cancellation BETWEEN bands, so the bands are not independently
substitutable and those band shares are shares of the uncancelled sum, not
contributions to the net effect (the same non-additivity as above -- nobody
should quote 46.7% as "half the effect"). Consequence: there is no turn point
to read, the design's intended reading is unavailable, and what survives is
the endpoint -- the full swap is the BEST of all of them and still 9.5x worse
than the baseline. "Nothing helps at any length scale" is robust; the shape of
the curve carries no separate reading and none is drawn from it.

DISMISSED BY MEASUREMENT: my grid-mismatch worry. Modes 251-300, which the DFT
grid cannot populate after the 500->600 upsample, contribute 0.00000.

THE FLIPPED-SIGN ROW IS NOT A SIGN CONTROL, as the workstation showed:
rho(-phi_dft) = (B @ phi_dft - nb_off)/V and B @ phi_dft is approximately
-nb_off pointwise, so it returns about -2 nb_off/V. Measured sum|.| 14.5805
against 2 x 7.2847 = 14.5693, agreeing to 0.07%. It measures the offset and
says nothing about the response; relabelled as a probe.

BANKED, and it needs no substitution at all: the baseline bound profile has a
48.7% shape residual at a displacement of only +0.050 A, with L1 0.80003 e
against int|.| 2.0193 e. Against the ionic channel's 1.4% residual and
0.18687 e, the bound channel carries 4.3x the 1-D charge error and 3.8x the
coupling gap. The 1-D error lives in the bound channel, and it is SHAPE, not
position. Plane-averaged 1-D throughout; no lateral 3-D error is in any of it.

NEXT CHEAPEST CANDIDATE, written down but NOT executed and awaiting the
user's decision: compare the polarization itself rather than its second
derivative.

  P_model(z) = a1 * E + p_off
  P_DFT(z)   = -integral_0^z <RHOB>_xy dz'   constant fixed by P -> 0 in vacuum

Integration has gain 1/k where B has the band-pass gain measured above,
peaking at 4.67 and dying to 2.37e-15, so this removes the operator that made
the last comparison unreadable -- a genuine improvement in the observable, not
a different view of the same one. It also separates a1 from p_off directly,
which is the point.

Two guards the workstation added to the design, both free and both about the
constant, which is the weak part of the formulation:
  P -> 0 must hold at BOTH ends. The s_diel profile has no dielectric below
  about z = 15 A and above about z = 42 A, so fixing the constant at one end
  makes the value at the other end a CHECK rather than an assumption. It
  should come out consistent, since RHOB's net is +0.0000 on this frame, but
  if it does not then the constant is not well defined and nothing downstream
  is readable.
  a1 E and p_off must each be reported alongside their sum, since separating
  them is the whole purpose and only the sum is currently known to be well
  behaved.

## 2026-09-10  three corrections from the user, and the design that replaces mine
code ead0a4a. The user rejected the polarization-integral test I proposed and
replaced it. All three corrections are accepted; two of them retract things I
had written into the entry above and into the report to the user.

CORRECTION 1 -- the 52-fold cancellation does NOT point at p_off. Withdrawn.
The prior in the code is ALREADY a covariance, read straight out of
closure_from_fields:

  prior = plane_mean(a3_scr * ez_scr) - plane_mean(a3_scr) * plane_mean(ez_scr)

Its job is to repair the average-first-then-multiply error, so

  a1 * E + prior = plane_mean(a3 * E)

BY CONSTRUCTION. The two terms are therefore SUPPOSED to cancel strongly, and
the size of the cancellation says the covariance is comparable to the mean
product -- a statement about lateral inhomogeneity, not a defect. The
arithmetic (7.2847, 7.3768, 0.14266) stands; the attribution does not. Noting
also that the covariance is built from the SCREENED VACUUM field ez/eps3 out
of phi_sol, while a1 in the solver multiplies the self-consistent smoothed
total field -- that pairing is a heuristic (E_total is approximately E_vac/eps
and is not known before the solve), and it is a separate candidate from
"p_off is wrong".

CORRECTION 2 -- integrating RHOB gives W_B P, not P. The charge is the
divergence of the SMOOTHED polarization, n_b = -V * WB @ D @ P, so the
integral of the plane-averaged RHOB recovers W_B P and cannot be set against
an unsmoothed a1*E + p_off. Fixing the constant in the vacuum does not remove
that difference. My proposed test was wrong on this point, and both of the
guards the workstation added to it are moot because the constant no longer
carries any argument: the integral is demoted to a cross-check that attributes
nothing.

CORRECTION 3 -- the branch conclusion must stay narrow. What is supported:
the model's bound charge has close to the right TOTAL but a clearly wrong
SHAPE, and swapping in the DFT total potential alone does not repair it. What
is NOT supported and which I over-claimed: that the generation of the total
potential is excluded. Also recorded: the top few frequencies contributing
0.00000 excluded only that band of the potential DIFFERENCE; grid error in the
cavity and in the a1 construction is still open.

THE DESIGN THAT REPLACES MINE, in the user's three steps. The point is that
one result cannot fix two unknowns -- P = a1*E + p_off with P and E known
still admits many pairs -- so the two coefficients need an INDEPENDENT
reference, and the DFT fields can supply it:

  1. Compute the 3-D polarization from the DFT native density and the DFT
     total potential through the published path and confirm it reproduces
     RHOB. No DFT re-solve. A failure means the 3-D response, the cavity or
     the parameters are wrong and nothing downstream is readable.
  2. From those same fields take A_ref = plane_mean(a3) and
     prior_ref = plane_mean(a3 E_z) - A_ref plane_mean(E_z). These are the
     CORRECT a1 and p_off, and exactly so: plane-averaging the 3-D
     construction gives n_b = -V WB_1d D plane_mean(P_z), the lateral
     derivatives dropping out, so the 1-D reduction is EXACT when the two
     coefficients take these values.
  3. On one grid, one sign convention and one smoothing, compare the model's
     a1, its prior and its learned delta_p separately, and check that
     together they give the right polarization and the right charge.

That is what separates a wrong mean response from a wrong covariance
background from a correction that failed to repair the difference.

Six gates, three of them the identities the design rests on: the covariance
making the 1-D product exact; the 1-D operator on plane_mean(P_z) equalling
the plane average of the 3-D charge (correction 2 made numerical); the 1-D
driving field equalling the plane average of the 3-D one; the two cells
agreeing; p_off = prior + delta_p as captured; and the script's own 1-D chain
reproducing the model's bound charge. The ref/ref cell of the 2x2 is a closure
check that must return the reference, not a result.

Two extras inside the user's scope: a1's error split into its pure cavity part
resp_unit*(plane_mean(s_diel)_model - plane_mean(s_diel)_ref) with the
remainder attributable to the saturation heuristic; and delta_p projected onto
the correction it was supposed to supply, prior_ref - prior_model, so "did the
learned correction point the right way" is a number. Plus the share of the
reference coefficients' spectral energy above the model closure grid's Nyquist
(the model's coefficients come from 100x100x300 and carry no mode above 150),
which is the grid error correction 3 leaves open.

Ran on the 4090, 12 s, peak GPU 1377 MiB of 24564, peak RSS 1.20 GB, no
kill. First run at ead0a4a, re-run at 0ac7b09 after three fixes; every number
reproduced to the digit. Provenance certified, sha256 IDENTICAL to the commit.
All gates PASS.

THE BIG ONE, STEP 1: the published 3-D construction REPRODUCES the DFT bound
charge on this frame. From the DFT native density and the DFT total potential:
net +0.000000 e against +0.000000, int|.| 2.0229 against 2.0193 e, L1 0.00365
e = 0.18% of the reference int|.|, correlation +0.999991. So the 3-D response,
the cavity recipe and the parameters are right, and the entire bound-channel
error lives in the 1-D REDUCTION. That is the largest narrowing this chain has
produced.

The substantive identity gate also passes: the 1-D operator on
plane_mean(P_z) equals the plane average of the 3-D charge to 6.939e-18
against a profile max of 2.933e-03, and the 1-D driving field equals the plane
average of the 3-D one to 9.770e-14 against 3.054e+01. So the reduction is
exact when the two coefficients take their reference values, as designed.

STEP 3a AND 3a-2 -- and the split is the result, because it says the two
coefficients fail in DIFFERENT ways:

  quantity                          ref rms   model rms    err rms   rel    corr
  a1    (mean response)          2.8973e-01  2.8167e-01 2.3947e-02 0.083 +0.9935
  prior (covariance background)  2.4731e-02  3.8408e-02 1.3765e-02 0.557 +0.9987
  p_off = prior + delta_p        2.4731e-02  3.7477e-02 1.2850e-02 0.520 +0.9985

  quantity                       best scale    err rms  after rescale  removed
  a1    (mean response)              0.9692 2.3947e-02     2.2216e-02     7.2%
  prior (covariance background)      1.5510 1.3765e-02     1.9411e-03    85.9%
  p_off = prior + delta_p            1.5132 1.2850e-02     2.0037e-03    84.4%

  a1's error is SHAPE. Its best single gain is 0.9692, within 3% of unity, and
  it removes only 7.2% -- no scalar fixes it. 77.7% of that error in rms is
  the pure cavity part, resp_unit*(plane_mean(s_diel)_model - _ref), and a
  cavity that plateaus at the wrong value and turns on in the wrong place is
  exactly a shape error.

  prior's error is GAIN. It is 1.5510x TOO LARGE -- over-supplied, the
  opposite of the natural guess -- and that one number removes 85.9% of the
  error, so its shape is essentially right.

STEP 3b -- the learned correction points the right way and is about fourteen
times too weak. needed rms 1.3765e-02, supplied 9.6013e-04, projection
+0.0668, correlation +0.9553, and it removes 6.6% of the gap as it stands.
Rescaled by 13.72x it would remove 71.0%. The rescale residual is the number
that matters, not the correlation: two profiles both localised at the
dielectric interface correlate highly whatever their detail, so +0.9553 alone
was confoundable. It survives the test -- but 71.0% against prior's own 85.9%
says the shape is right in the main and not in the detail, so a gain change on
that head buys most of the correction and not all of it.

Coherent picture of p_off: the covariance background is 1.55x over-supplied,
the learned correction is aimed correctly at reducing it and is under-powered,
and it moves the effective gain only from 1.5510 to 1.5132 -- 7% of the excess.

THE PLATEAU, now with direct evidence and with its limits stated.
plane_mean(s_diel) max is 0.9447081 on the model's grid against 0.9999851 on
the DFT native grid, same recipe and same parameters. On its own that line
establishes nothing; it is the earlier density tail test -- same plateau for
both densities in both directions, moving it by under 6e-09 -- that excludes
density and leaves the grid the cavity is built on. Three limits, all from the
workstation and all written in rather than smoothed over: the two grids differ
LATERALLY as well (100x100 against 168x168) and plane_mean averages
laterally, so a z effect is not separated from a lateral one; "resolution"
here means the grid the cavity is built on, including the density it is a
nonlinear function of, not the resolution of the plane average; and STEP 3c's
result below does NOT rule out a grid explanation, because representable and
correctly computed are different claims for a nonlinear function of a
grid-resolved density.

STEP 3c -- grid REACH is closed, negatively. The share of the reference
coefficients' spectral energy above the model closure grid's cut is 0.00% for
A_ref and 0.00% for prior_ref, so the 300-in-z grid can represent the
reference coefficients entirely. Grid reach is not the a1 error. Read together
with the plateau: the grid can carry the right answer and still compute the
wrong one.

STEP 3d, THE 2x2 -- its intended reading is unavailable, and this is the third
single-factor intervention in the chain defeated by a coupling identity.

        a1    p_off   int|.|       gap        L1    shift   resid
         -        -   2.0193   +0.0000   0.00000   -0.000    0.0%
     model    model   8.5965   +5.5832   7.56925   +0.825   85.0%
       ref    model  37.5740  +39.0290  36.94743   +1.250   93.4%
     model      ref  34.9159  -33.4487  33.69750   +1.275   95.9%
       ref      ref   2.0228   -0.0029   0.00369   -0.000    0.4%

The closure check passes: ref/ref returns to the reference at L1 0.00369 e.
But BOTH single swaps are 3.5 to 3.9 times WORSE than the baseline while both
together give 0.00369, so neither names a faulty coefficient. The reason is
measurable on the reference side alone: a1*E has rms 2.4447e-02 and prior_ref
2.4731e-02 against a sum of 1.6706e-03, a 14.6-fold cancellation, because
a1*E + prior = plane_mean(a3*E) holds by construction on BOTH sides. The two
coefficients are complementary halves of one decomposition, and a reference
half paired with a model half breaks a cancellation the reference itself
relies on. Named as a pattern, since it is now the third instance after the
staged potential swap and the density tail swap: when the two things being
separated satisfy an exact identity together, substituting one of them is not
a controlled intervention.

What the rows DO give, since each row's deviation from ref/ref is exactly one
error array: the p_off error alone costs 36.94743 e of L1, the a1 error alone
33.69750 e, both together 7.56925 e -- the two coefficient errors cancel about
4.7-fold. Reported as a magnitude and NOT as a finding about independent
errors: the model's own closure satisfies the same identity with its own
fields and its bound-charge total is close to the reference, which largely
forces the two errors to oppose.

CONSEQUENCE THAT MUST TRAVEL WITH THE DIAGNOSIS -- it is not a repair list,
and this is measured rather than argued (code 99ee87f, the fifth row of the
2x2):

     model prior/1.551   -0.0000  35.7184  -35.7095  34.52551  1.52e-01   +1.325   96.1%

Dividing prior by its own best gain 1.551 -- the single change that removes
85.9% of its rms error -- gives charge L1 34.52551 e against the baseline's
7.56925 e, a factor of 4.56. So fixing the covariance gain ALONE makes the
bound charge substantially WORSE, because it removes a compensation the wrong
a1 currently relies on. The two halves have to move together, and any change
still gets judged on aggregate charge, potential and energy.

TWO THINGS THE FACTOR ALONE HIDES:
  the coupling gap does not merely grow, it FLIPS SIGN and grows 6.4-fold,
  +5.5832 eV to -35.7095 eV. "4.56x worse in L1" invites the reading that the
  shortfall deepens; it is the opposite -- the gain-only fix overshoots past
  the reference into over-attraction.
  and the gain-only fix is slightly worse than substituting the reference
  p_off outright, 34.52551 against 33.69750 e. So a partial move inside the
  coupled pair is not even MONOTONE toward the full move. Fourth instance of
  the structural theme: partial moves inside a coupled pair do not
  interpolate.

THE ASYMMETRY, which is what actually shapes any repair: the two halves are
not equally tractable and a1 is the binding constraint. prior's error is 85.9%
removable by one scalar, so a joint change has a cheap, well-specified handle
on that half. a1's is 7.2% removable by ANY scalar and 77.7% pure cavity, so
its half has no scalar handle at all and needs the cavity, or the grid the
cavity is computed on. And there is no repair ORDER either -- the 4.7-fold
cancellation read forwards says any single-coefficient change is a regression
on the aggregate, so "fix a1 first" is not available as a plan.

METHOD NOTE, stated as the positive rule and generalising past this chain.
Measure each piece against its own reference and leave the identity intact;
reach for substitution only when the pieces are not bound by one. Three
substitution designs in this chain failed that way and I wrote all three,
while the one thing that separated anything was the scale-plus-residual
decomposition, which never disturbed the identity. Separately: a bound and a
measurement are STAGES, not alternatives. The triangle inequality is the cheap
sign test run before committing, and its output is a decision about whether to
measure, not a result to report. Publishing a ratio of aggregates was skipping
the first stage; publishing the 28.5-38.9 e interval as the answer would have
been stopping at it. Both are the same error -- treating a stage as the
destination. Concretely here: the bound gave the sign for free and contained
the answer, while the point estimate available from the same inputs was 28.9 e
against a measured 34.53 e, 16% low and sitting exactly on its own lower edge.

THREE CORRECTIONS TO MY OWN SCRIPT, all found by the workstation:
  the covariance gate's 1e-18 absolute threshold was unachievable -- one
  float64 epsilon on terms of order 0.2 is 4.28e-17 and the measured 1.279e-17
  is 0.30 epsilon. Now relative. The workstation was right to ignore a gate
  whose number contradicts its own label rather than obey it. The gate was
  also weaker than either of us said: prior_ref is DEFINED as
  Pz_ref - A_ref*Ez_ref, so it holds by construction and can only measure
  round-off.
  the polarization cross-check carried a sign error. rho_b = +WB D P, so the
  running integral is +W_B P; the minus made the two arrays exact negatives,
  caught from a ratio of 1.99986 at correlation -0.999910. Corrected they
  agree to a relative rms of 1.34%, which confirms n_b = -V WB D P.
  and my "about 0.01%" for that agreement was wrong by a factor of 149: 0.01%
  was the correlation DEFICIT 1-corr = 9.0e-05, while the relative rms
  difference is 1.9835e-05/1.4789e-03 = 1.34%. For nearly parallel arrays the
  relative difference goes as sqrt(2(1-corr)), and sqrt(2*9.0e-05) = 1.34e-02
  exactly. A correlation is never a relative error -- it is a relative error
  squared and halved. Same family as the difference-of-norms slip.

## 2026-09-10  poff_exact_target.py -- hold a1 fixed, get the compensation right
code e48faec. The user set a staged plan with a stop rule, and step 1 of it
dissolves the obstruction that ended the previous entry.

THE MOVE. With a1 HELD at the model's value there is only ONE unknown left in
P = a1 E + p_off, and it is exactly determined:

  P_off*   = plane_mean(P_DFT) - a1_model * plane_mean(E_DFT)
  delta_p* = P_off* - prior_model

The 2x2 failed because a1 and p_off are complementary halves of one
decomposition, so substituting either alone breaks a cancellation the
reference itself relies on. Defining the target WITH the model's own a1 makes
the pair exact by construction and leaves nothing to break. This is the
user's, not mine.

A CORRECTION IT CARRIES, and it retracts a verdict from the entry above:
prior_ref = Pz_ref - A_ref*Ez_ref is the correct p_off for the REFERENCE a1,
NOT for the model's. Scoring delta_p against it therefore charged delta_p with
a1's error as well, so the "delta_p is right in shape and about fourteen times
too weak" verdict was measured against the wrong target. Step 1 re-decides it
against delta_p*. The 13.72x and the 71.0% from that entry are suspended until
it does.

Also recorded from the user: the branch conclusion stays where it was -- the
model's bound charge has close to the right total and a clearly wrong shape,
and swapping the DFT total potential alone does not repair it. Nothing further
is claimed, and the generation of the total potential remains not excluded.

EXPLICITLY OUT OF SCOPE this round, per the user: no grid change, and no
scaling up of the correction coefficients. The question is only whether
getting the compensation right, with everything else as it is, improves the
aggregate.

  STEP 1  arrays only. P_off* and delta_p*, how far the target moved from the
          old one, and the amplitude-versus-shape split of the current delta_p
          against the CORRECT target -- by best-scale-plus-residual, which is
          the one decomposition in this chain that has separated anything,
          rather than by substitution.
  STEP 2a wiring check, labelled as such: P_off* was built at the DFT field,
          so reproducing the reference there proves the algebra and nothing
          physical. The residual should be the 500->600 operator
          discretisation plus the 0.18% by which the 3-D reconstruction
          differs from stored RHOB.
  STEP 2b the actual test. P_off* in, RE-SOLVE self-consistently with every
          other input the same tensor object, judged on aggregate charge sum
          L1 against RHOB+RHOION, bound and ion L1 separately, cross coupling,
          and the FULL self-energy on the SUM field so the bound-ion mutual
          term is included, plus the potential profile and both solver exit
          reasons. Gated on the baseline re-solve reproducing the model's own
          bound charge.
  STEP 3  not run. Only if step 2 improves: can the existing head produce
          delta_p* within its own basis and its actual clipping.

THE USER'S STOP RULE, to be applied and not softened: if the aggregate does
not improve, STOP -- the compensation is not the binding problem and the next
place to look is the other inputs and the self-consistent coupling, not
training. A bound-channel-only improvement is NOT success; that is the trap
the s_ion round set, where the ionic channel improved 10.6-fold and the
aggregate moved 0.03%.

ONE FREE NUMBER THAT MAY DECIDE STEP 3 IN ADVANCE. The head is
delta_p = G_0.2 * [w_env(z) * sum_k c_k B_k(u)] with K = 8 zones and
c_k = c_max*tanh(...) at c_max = 0.25 -- a HARD bound. So c_absmax is printed:
at the bound, no training can supply a larger correction whatever delta_p*
asks for; far below it, the amplitude is a learning outcome rather than a
representational limit.

Ran on the 4090 across six iterations (e48faec first, final 04c384b), ~12-15
s each, provenance certified IDENTICAL every time, every repeated number
reproducing to the digit. Gate 1.301e-18, both solves exit on tolerance.

VERDICT: STEP 2 FAILS ON ITS OWN TERMS. Step 3 does not run.

STEP 1 -- holding a1 fixed shrinks the target 3.74-fold and REVERSES last
entry's shape verdict.
  P_off* rms 3.5170e-02; prior_model 3.8408e-02; p_off_model 3.7477e-02
  delta_p* (needed) 3.6777e-03; delta_p (supplied) 9.6013e-04
  old target rms 1.3765e-02 -> new 3.6777e-03, ratio 0.267, their mutual
    correlation +0.9177
So the head is short by 3.83x, NOT the 14x recorded above -- that figure was
measured against prior_ref, the correct p_off for the REFERENCE a1, so it
charged delta_p with a1's error. But the shape comes out WORSE against the
correct target, correlation +0.8580 against +0.9553, and gain-alone recovery
falls from 71.0% to 49.7%. Fixing the target weakened "right shape in the
main", it did not strengthen it. Both figures follow from the correlation,
since the residual after ANY optimal gain is sqrt(1 - corr^2).

A PRESENTATION FAULT CORRECTED IN PASSING, and it mattered: the split
originally scaled the TARGET down to fit the head, which is not a repair, and
reported 83.3% removed against a denominator of the raw difference. Scaling
delta_p UP against a denominator of what is needed gives 49.7%. Only the
second is actionable and the first would have been read as the headline.

HEAD SATURATION, and it was UNREACHABLE at first: delta_stats is spread into
self.last_diagnostics (pb1d_backend.py line 510), not into solve_graph's
returned dict, so out.get("c_absmax") was always None and an "if is not None"
swallowed the whole block SILENTLY -- worse than the bug. Once read:
c_absmax 0.007093, 2.8% of the hard bound c_max = 0.25; scaling every c_k by
the 3.83x shortfall reaches 10.9%, comfortably inside. The missing amplitude
is a LEARNING OUTCOME, not a representational limit, so step 3 would never
have been blocked by clipping in either direction. That bounds the amplitude
question only; the 49.7% leaves shape open.

STEP 2a, WIRING ONLY: bound-charge L1 with P_off* is 0.00365 e against
7.56925 e for the model's own p_off, on int|.| 2.0193 e. That 0.00365 is
numerically the 3-D reconstruction's own 0.18% floor, so the algebra is right
and nothing physical follows.

STEP 2b -- SELF-CONSISTENT RE-SOLVE
                      case   net (e)   chg L1  bound L1   ion L1     cross      self     total   d total
             DFT reference   +1.0000  0.00000   0.00000  0.00000   -3.0243   +1.5237   -1.5005   +0.0000
     baseline, model p_off   +1.0000  0.63739   0.80003  0.18687   -2.6289   +1.4718   -1.1572   +0.3434
   exact P_off*, re-solved   +1.0000  0.47303   0.63902  0.19199   -3.0020   +1.4869   -1.5151   -0.0145

  BETTER: charge sum L1 -25.8%, bound L1 -20.1%, cross gap +0.3953 -> +0.0223,
          self gap -0.0520 -> -0.0368, total gap +0.3434 -> -0.0145 eV (96%)
  WORSE:  ion L1 +2.7%; potential rms 0.11093 -> 0.22339 eV (+101.4%), of
          which mean -0.04109 -> -0.12304 (3.0x) and mean-removed 0.10304 ->
          0.18645 (+81.0%); longest-wavelength (45 A) amplitude 0.08041 ->
          0.20436 eV (+154.2%); mutual bound-ion term 0.0288 -> 0.1261 eV from
          the reference (4.4x)

Applying the user's stop rule as written and not softened: it names aggregate
charge, the POTENTIAL, cross energy and the full self-energy, and two of those
went the wrong way -- the potential by a factor of two and the mutual term by
4.4x. The total energy gap closing 96% while the field agreement halves is the
compensating-error signature this work exists to remove, so the closed energy
gap cannot stand as the verdict, any more than the ionic channel's 10.6-fold
improvement could when the aggregate moved 0.03%. Same trap with the sign of
the surprise reversed.

WHY THE POTENTIAL WORSENED -- mechanism identified in the source, not guessed.
A first attempt assumed phi_sol was an untouched solute-side quantity so the
error would split into a fixed bracket plus l0_inv(solvent charge error). Its
own gate REFUTED that: the bracket differed between the two cases by 1.972e-01
eV, half its own magnitude, at both signs. The reason is in the code:

  phi_sol = cvhar_z + cvdip
  cvdip   = cdipol_potential_1d(nz, lz, ef_z, indmin)
  ef_z    = c_unit * d_mix        <- the SOLVENT CHARGE's own dipole

and cdipol_potential_1d returns (-e_comp*lz/nz) * ii * cutoff, a LINEAR RAMP
in the signed distance from the cell centre, tapered at the edge. cvhar_z is
fixed; phi_sol is not. p_off changes the solvent charge, which changes its
dipole, which drives a ramp into phi_sol. That is the dipole-feedback path and
it is the self-consistent coupling.

It also explains the band pattern, which fit NEITHER branch the table was
built to distinguish:
       modes    wavelength |   charge err rms: base -> P_off*  | potential err rms: base -> P_off*
         1-3     >= 15.0 A |   2.176e-05 -> 6.137e-06  better 3.5x |  6.789e-02 -> 1.611e-01  WORSE 2.4x
        4-10      >= 4.5 A |   9.869e-05 -> 6.330e-05  better      |  7.325e-02 -> 8.699e-02  WORSE
       11-30      >= 1.5 A |   1.913e-04 -> 1.957e-04  worse       |  2.404e-02 -> 3.262e-02  WORSE
      31-100      >= 0.5 A |   1.444e-04 -> 7.245e-05  better      |  8.053e-03 -> 1.294e-02  WORSE
     101-300      >= 0.1 A |   1.665e-05 -> 9.885e-06  better      |  6.197e-04 -> 7.410e-04  WORSE
The charge improves in four of five bands INCLUDING the lowest, 3.5-fold
there, and the potential worsens in all five. Not the 1/k^2 weighting trade
either branch assumed. A ramp is the longest-wavelength component -- the 45 A
amplitude is the largest single part of the change -- and a TAPERED ramp is
not a single Fourier mode, so it puts weight in every band at once.

THE EXACT DECOMPOSITION, gate closing at 3.545e-12 eV (8e-12 of the change):
                                           term   rms (eV)       mean   45 A amp
                  total change in the potential    0.13373   -0.08195    0.13003
                        of it, the G=0 constant    0.08195   -0.08195    0.00000
                 of it, the rest (mean removed)    0.10569   +0.00000    0.13003
              from dipole feedback (cvdip ramp)    0.06679   -0.00000    0.07438
              from the charge directly (l0_inv)    0.05925   +0.00000    0.07100
Only the last three combine, and they do, at an implied correlation of +0.4041
-- they reinforce rather than oppose. As variance shares of the total change:
G=0 constant 37.6%, dipole feedback 24.9%, direct charge 19.6%, their cross
term 17.9%.

The constant needed its own row because without it the table invited an
impossibility: 0.06679 + 0.05925 = 0.12604 is capped 6.1% BELOW the printed
0.13373, and solving for the correlation gives +1.2524. l0_inv annihilates the
constant so the direct term carries none of it. Fourth member of the
difference-of-norms family and the first in the presentation rather than the
reasoning. Cross-checked as real: the potential error's mean moved -0.04109 to
-0.12304, a difference of exactly the -0.08195 reported, and the mean-removed
part still worsened 81.0%, so neither part accounts for the degradation alone.

BOTH CANDIDATE INPUTS ARE EXONERATED BY MEASUREMENT, which makes the
conclusion stronger than the reading written in advance. That reading was "if
P_off* moves the dipole FURTHER from the reference the degradation is
explained". The dipole error goes -0.29449 -> -0.04943 e A (reference
16.70259, baseline 16.40810, P_off* 16.65316): 5.96x CLOSER. So at a charge
profile AND a dipole both closer to the reference,
cvhar_z + cvdip + l0_inv(charge) moves FURTHER from the DFT total potential.
Neither input is the problem; the construction of the potential is.

TWO OPEN ITEMS, NOT ONE, and this was dispatched as a question with the
criterion set in advance. The cdipol ramp is ii*cutoff and the taper breaks
its antisymmetry, so mean(cvdip) could have scaled with the dipole and the
constant would then belong to the same path. Measured: mean of the feedback
change is 0.0% of the total constant. So the tapered ramp produces NO mean and
the G=0 constant is a SEPARATE term.
  item 1: the potential construction -- the cdipol taper form and cvhar_z.
  item 2: the G=0 bookkeeping, unattributed.
THE TWO ARE NOT EQUAL IN SIZE AND THE NAMED ONE IS THE SMALLER: the identified
dipole-feedback mechanism is 24.9% of the variance while the unexplained
constant is 37.6%. Writing this up as "the mechanism is the dipole feedback"
would lose the larger term sitting beside it.

NO MECHANISM IS OFFERED for why the construction fails at correct inputs, and
one candidate is explicitly withheld: that the baseline's wrong dipole was
compensating an error in the ramp form. It fits, and it is the same shape as
the solute-side compensation story that failed its own gate this round, so it
needs its own gate before anyone states it.

VOID, and dropped from script and ledger: the 0.15884 eV "floor" on the
potential error and "the baseline sits below its own floor only by
cancellation". Both rested on the refuted bracket. Also void as a
discriminator: the sign of the correlation between the solvent term and the
bracket -- with bracket 0.15884, solvent 0.07288 and total 0.11093 the
correlation is FORCED to -0.788, because a total smaller than one of its parts
requires a negative sign. The discriminator has to be something the magnitudes
do not already fix.

NOT DONE, per the stop rule: step 3, and the confirmation on a second charged
frame and a neutral frame, which the user scheduled after step 2 succeeded.

## 2026-09-10  potential_rebuild.py -- can the potential be rebuilt at all?
code d40b5ec. Three corrections from the user first, because they withdraw the
previous entry's conclusion and two of its readings.

CORRECTION 1 -- "both inputs exonerated, so the construction must be wrong" is
WITHDRAWN. The full self-energy IMPROVED: its error went 0.0520 -> 0.0368 eV.
What worsened is the MUTUAL term inside it. And the cross energy improved
0.3953 -> 0.0223 eV. So the gain cannot be written off as error cancellation,
and my "the energy closing while the field degrades is the compensating-error
signature" was too broad -- it was true of the potential and not of the energy
terms. The sufficient reason to pause was the POTENTIAL alone, and only that.
The pause itself stands.

CORRECTION 2 -- the inputs are NOT excluded, and the exoneration was wrong
three ways:
  a better charge L1 and a better dipole do not imply that every spatial
  component which determines the potential is better, and a band rms does not
  establish it either;
  the dipole reported was the first moment of the SOLVENT charge, while the
  feedback is dip_z = val_ion_dipole_z + dsol_z - q*center_z and therefore
  contains the MODEL's SOLUTE dipole, which was never checked;
  and holding cvhar_z fixed across the two cases says nothing about whether
  cvhar_z is correct.

CORRECTION 3 -- the 24.9% and 37.6% are shares of the CHANGE the modification
caused, not shares of the DFT error, and were presented as if they located the
defect. And the solver fixes the potential's constant by IONIC
ELECTRONEUTRALITY: the residual's G=0 row is mean(n_b + n_ion) + q_sol = 0,
and n_ion depends on phi's mean through n_work, so the constant MUST move when
the shape moves. A G=0 change therefore implies no second bookkeeping error.
The "two items, and the named one is smaller" conclusion is WITHDRAWN -- the
0.0% measurement stands, the interpretation drawn from it does not.

SCOPE, also the user's: the `total` column in the previous entry is the 1-D
solvent electrostatic energy at a FIXED bare-solute potential. It contains
neither the dipole correction nor the G=0 term under discussion, so "96%
closed" must not be read as the full DFT energy being repaired.

THE CONTROL. Hold ONE solvent charge -- the DFT bound plus ionic -- and rebuild
the total potential twice, changing only the solute input:

  DFT solute potential and dipole    can all-reference inputs rebuild the DFT
                                     total potential at all?
  model solute potential and dipole  how much error does the swap add?

Each row computes its dipole correction from its OWN complete inputs, as the
solver does. Mean-removed shapes are compared; the constant is checked
separately against the electroneutrality condition rather than folded in.

  all-DFT rebuilds, the model's does not -> the SOLUTE INPUT is the priority
  all-DFT does not rebuild              -> potential ASSEMBLY, boundary and
                                           grid conventions
  both rebuild                          -> back to the residual charge error
                                           and the self-consistent response

THE DFT SOLUTE INPUT IS CONSTRUCTED INDEPENDENTLY, never back-derived from the
potential it is used to test. The backend builds cvhar3 = phi_base -
l0_inv(net_g) with net = neutral_v - n_e, i.e. l0_inv(cores - electrons).
Substituting the DFT density in the same assembly makes the reference fields
phi_base and neutral_v CANCEL out of the difference:

  cvhar_DFT = cvhar_model + l0_inv(n_e_DFT - n_e_model)

so only the two electron densities are needed and PHI never enters. The DFT
solute dipole comes from the same solute_dipole_z the model calls, with the
DFT profile, reported on both the closure grid and the native grid since that
function's sawtooth window is grid-size dependent.

GATED FIRST on the assembly reproducing the model's own converged potential
from its own inputs and its own solvent charge, mean-removed -- one check that
validates cvdip, l0_inv, indmin, c_unit, center_z, the dipole mixing and every
sign at once.

Two numbers expected to matter beyond the branch: the share of cvhar_DFT's
spectral energy above the model closure grid's cut at mode 150, which the
model's assembly cannot represent since cvhar is upsampled from 300 in z; and
the model's SOLUTE dipole error, the term correction 2 says was never checked.

Ran on the 4090, provenance certified IDENTICAL. The branch is answered
unambiguously.

THE IDENTITY WAS CHECKED IN SOURCE BEFORE RUNNING, not taken on faith:
pb1d_backend.py:388-390 gives n_e_values = neutral_v - net_values and
cvhar3 = phi_base - l0_inv(net_g), with neutral_v and phi_base coming from the
baseline tables, hence independent of the substituted density and cancelling
in the difference. The one precondition -- both arms using the same baseline
row -- holds, since baseline_index.json maps this geometry to a single row.

GATE: the assembly reproduces the model's own converged potential from its own
inputs and its own solvent charge, mean-removed max abs deviation 7.507e-12
eV, constant +1.020240 eV. PASS -- so cvdip, l0_inv, indmin, c_unit, center_z,
the dipole mixing and every sign are validated together.

[RESULT] one solvent charge (DFT bound + ionic, net +0.99999 e), two solute
inputs, mean-removed:
                        solute input        L1       max       rms  45 A amp    dipole
       DFT solute potential + dipole    0.0167    0.0171   0.00157   0.00001   -0.3653
     model solute potential + dipole    8.4556    0.4196   0.20944   0.23008   -0.9439

The all-DFT rebuild error is 0.00157 eV rms, 0.05% of the potential's own
3.35043 eV. Swapping in the model's solute input takes it to 0.20944 eV, a
FACTOR OF 133.

VERDICT on the user's branches: the SOLUTE INPUT is the priority. Two things
are exonerated by the same run -- the potential ASSEMBLY, which reproduces the
DFT total potential to 0.05% when fed DFT solute inputs, and GRID REACH, since
0.00% of cvhar_DFT's spectral energy sits above the model closure grid's
mode-150 cut.

THE CONSTANT lands correctly for the all-DFT row: the neutrality offset is
-1.062176 eV giving a mean of -1.06218 against the DFT potential's -1.06133,
agreeing to 0.85 mV. The model-solute row's is off by 0.138 eV. And the stored
DFT potential's own offset is -0.000849 eV, so the calibration is essentially
zero as it should be. The user's electroneutrality point is directly visible:
the constant tracks the shape rather than being independent of it.

THE TERM THE USER IDENTIFIED AS NEVER CHECKED IS THE LARGE ONE:
  electron profile: model int 660.99991, DFT 660.99999; difference rms 6.98500
  solute potential: cvhar_model rms 3.43934, cvhar_DFT 3.48212; difference rms
    0.12710 eV
  solute dipole: model -6.74120, DFT -6.16258 (300 grid) and -6.16260 (native
    500); MODEL MINUS REFERENCE = -0.57862 e A, i.e. the model's dipole is
    more negative by that amount (the workstation wrote +0.57862; the
    magnitude is right and the sign as stated here is model - reference)
That solute dipole error is about 12x the solvent dipole error the previous
round was optimising (-0.04943 e A after P_off*). And the result table's dipole
column difference, -0.9439 against -0.3653, is exactly -0.57860 -- so the
feedback term is carrying the solute dipole error, and the two measurements are
consistent. The electron COUNT matches to 1e-4 while the profile differs at
6.985 rms -- but stated that way it compares two quantities in different units
with no scale. Relative to their own references: count 1.2e-07 (8e-05 in 661),
density shape 0.411% (6.98500 against a profile rms of 1.698128e+03). The
shape error is therefore 3.4e+04 times the count error in relative terms, and
THAT pair is what makes this a shape error in the predicted density rather
than a charge error.

THE FOUR RELATIVE ERRORS, each against its own reference scale, in the order
the assembly applies them: electron count 1.2e-07, density shape 0.411%,
cvhar 3.650% (0.12710 against 3.482119), rebuilt potential 6.251% (0.20944
against 3.35043). Each is larger than the last, and l0_inv sits between the
second and third while the dipole correction sits between the third and
fourth. NO SHARES ARE ATTRIBUTED -- the two contributions were never
separated, and treating the printed parts of a total as a decomposition is
exactly what went wrong the round before.

A SIGN FAULT IN MY OWN SCRIPT, not just in the report: the output line
computed dip_DFT - dip_model and labelled it "the model's error", so its sign
and its label disagreed, and that is where the +0.57862 came from. Fixing only
the ledger would have left the next reader to repeat it from the log, which is
how this chain's four earlier sign errors propagated. The line now prints
MODEL MINUS REFERENCE with the convention named.

WHAT THIS DOES NOT SEPARATE, flagged rather than left implicit: the model's
solute input was swapped as a UNIT -- cvhar shape and dipole together -- so
which of the two dominates is not established, and both are non-trivial alone,
0.12710 eV rms in cvhar and +0.57862 e A in the dipole. But unlike a1/p_off
these are two SEPARATE inputs rather than two halves of one identity, so a
one-at-a-time swap IS a controlled intervention here and the obstruction that
defeated three earlier designs does not apply. Proposed to the user, not run.

## 2026-09-10  lateral_bound_swap.py -- what does fixing the lateral bound charge buy?
code 09d4389. The user moved this back to the MAIN LINE and specified one
experiment.

WHY THIS IS THE MAIN LINE. The bound coupling gap splits by channel as
plane-average 0.525 eV (28.7%) and LATERAL 1.307 eV (71.3%) -- the T2 channel
decomposition recorded earlier in this ledger. The 3-D fit's only
responsibility is that lateral shortfall, and every round since the s_ion
result has been inside the 1-D plane-averaged channel, i.e. the other 28.7%.
The question this line was opened for is what fixing the lateral part actually
buys the final energy.

THE EXPERIMENT. On the six already-checked frames (sid 61, 1, 28, 201, 353,
601) with the current checkpoint, substitute

    lat_DFT(r) = rho_b_DFT(r) - plane_mean(rho_b_DFT)(z)

for the model's own lateral bound residual. Held fixed: the solute electron
density, the 1-D background, the ionic charge and every other network output.
Every z-plane of the substituted field sums to zero -- the same per-plane
conserving form d_sup_b already has -- so the net charge, the 1-D profile and
the z-dipole are untouched BY CONSTRUCTION rather than by argument.

WHY THE ENERGY UPDATE IS A ONE-TERM CHANGE. The reconciliation identity gives
e_xsol = int delta*phi, e_self = 0.5 int delta*phi[delta] and
int rho_1d*phi[delta] = 0 by the per-plane projection, and
solvent3d_energy_g = e_xsol + e_self enters the total energy additively
(extensions.py). Since lat_DFT also has zero plane means the cross term stays
zero and comp and E_bl are unaffected, so the entire effect is

    dE = [cross + self](d_i + lat_DFT) - [cross + self](d_b + d_i)

with the self-energy on the SUM, which keeps the bound-ion interaction inside
it instead of dropping it between two separate self energies.

THREE GATES on exactly that chain: the model's own 3-D term reproduced through
THE Coulomb function, so solvent3d_energy_g really is cross+self; the
substituted field's plane means; and the 1-D/3-D cross term after the swap. If
the first fails, dE is not a one-term change and the table must not be read.

THE LIMITATION THAT TRAVELS WITH THE NUMBER, not after it: the model's energy
grid is coarser laterally than the DFT one (100x100 against 168x168), so the
swap can only inject the lateral structure that grid can represent. The share
of native lateral power above the model grid's lateral Nyquist is reported and
the cross energy is computed on BOTH grids. Agreement means truncation does
not matter for the energy; disagreement makes the improvement a LOWER BOUND on
what a perfect lateral fix would give.

THE READING, set by the user in advance:
  the charged frames' deviation drops clearly -> a basis for concentrating on
      how the 3-D bound charge is represented and trained
  the improvement is small or negative -> that part of the coupling gap can no
      longer be used to explain the total-energy plateau, and the remaining
      energy terms and their compensation have to be examined
  it improves partly -> record exactly how much is recovered and how much
      remains

Two things expected to matter beyond the table: the correlation between the
model's own d_sup_b and lat_DFT per frame, which says whether the model's
lateral charge points the right way at all and doubles as the sign check; and
the neutral frame, where the ionic channel is identically zero so its row
isolates the bound lateral effect with no ionic contribution.

The user was explicit that this does NOT show the network can learn the
reference charge. It settles whether the line is worth the work.

4090 SUBSET (KIT_FRAMES=n44, four frames, code c8560dd, provenance IDENTICAL,
peak RSS 2.33 GB, peak GPU 4811 MiB, no kill despite four 250 MB ASCII reads).
The six-frame LS6 run (job 3427615, all frames incl. the NiN88 cell) is
separate and adds the cell-size axis the subset cannot speak to.

ALL THREE GATES PASS: the model's own 3-D term through THE Coulomb function to
3.331e-16 eV, the substituted field's plane means to 8.743e-19 e/A^3, and the
1-D/3-D cross term after the swap to 9.711e-13 eV. So dE is a one-term change
and comp and E_bl are genuinely untouched.

TRUNCATION DOES NOT APPLY, so these are values and NOT lower bounds. Power
above the model grid's lateral Nyquist is 0.0% on all four frames, and the
cross energy on the model grid equals the native-grid one at ratio 1.000 every
time (-2.3991, -2.1808, -2.3886, -0.4309). The stated condition was that
agreement means truncation does not matter for the energy; it agrees exactly.

A MECHANISM I OFFERED FOR THAT AND THEN MEASURED, which does NOT hold as
stated. I claimed RHOB's Gaussian smoothing at sigma_b puts its cutoff below
the model grid's lateral Nyquist so "there is nothing up there to lose". With
sigma_b = A_K = 0.125 A (R_B is 0) and the model grid's lateral Nyquist at
k = pi*100/14.802 = 21.22 /A, the Gaussian's amplitude there is 0.0296, i.e. a
power suppression of about 1140x -- strong, but it SUPPRESSES rather than
annihilates. (At the DFT native cut, k = 35.66 /A, it is 2.4e-09 in power.) So
the parameter is consistent with the measured 0.0% and does not establish it:
the result also depends on the polarization's own spectrum, which this run did
not measure. "Nothing up there to lose" is withdrawn; what stands is the
measurement, 0.0% above the cut with the two cross energies matching at ratio
1.000.

[RESULT] final total energy error, meV/atom
             frame  atoms   original  substituted  improvement  fraction
     NiN44 q=-0.80    207    +12.987       +8.991       -3.996     30.8%
     NiN44 q=-1.00    207    +20.843      +15.209       -5.634     27.0%
     NiN44 q=-1.32    207    +27.309      +20.817       -6.492     23.8%
           neutral    207     -6.379       -4.499       +1.881     29.5%
  charged mean |err|            20.380       15.006       -5.374     26.4%

THE USER'S PARTIAL BRANCH, so both halves as instructed: 26.4% of the charged
error is RECOVERED and 73.6% REMAINS -- 15.006 meV/atom of an original 20.380.
The neutral frame, where the ionic channel is identically zero, gives 29.5%,
consistent with the charged frames rather than different, so the effect is not
an artefact of the ionic channel.

THE SIGN CHECK PASSES: correlation between the model's own d_sup_b and lat_DFT
is +0.8733, +0.8632, +0.8337, +0.8396 -- the model's lateral bound charge
points the right way on every frame, and it is not a convention error.

TWO TRENDS THE MEAN HIDES, and they carry more than the 26.4% does.
  The fraction recovered FALLS with charge: 30.8%, 27.0%, 23.8% across
  q = -0.80, -1.00, -1.32. The more charged the frame, the less of its error a
  perfect lateral bound charge explains.
  And the residual KEEPS THE CHARGE DEPENDENCE: after substitution the error
  is still monotone in |q| (+8.991, +15.209, +20.817) and grows slightly
  faster across the sweep than the original did, 2.32x against 2.10x. So a
  perfect lateral bound charge does not remove the charge-driven part of the
  total-energy error, and whatever carries the remaining three quarters is
  ALSO charge-driven. That is a statement about where the rest is.
26.4% with a falling trend and a charge-dependent residual is a different
object from 26.4% flat across the sweep, and neither reading should be taken
from the mean alone.

THE SAME TREND SEEN IN THE DERIVATIVE, with two phrasings of mine corrected by
the workstation before they were written down. The interval slopes, meV/atom
per unit |q|:
                    original  substituted  ratio
  |q| 0.80 -> 1.00     39.28        31.09  0.7915
  |q| 1.00 -> 1.32     20.21        17.52  0.8669
The two ratios differ by 9.5%, and the numbers here are exact to ~1e-9, so
that is real: the substitution scales the LOW-|q| slope down MORE than the
high-|q| one, which is the falling fraction-recovered trend seen in the
derivative rather than an independent fact. My "roughly the same factor" was
therefore wrong and slightly contradicted the trend beside it. What survives
is the part that matters: the charge dependence is NOT FLATTENED, since both
ratios are far from zero -- the swap reduces the dependence's amplitude and
leaves it in place.

Second phrasing corrected: "strongly sublinear in |q|" is not supported. It
rests on two intervals from three points, and the slope roughly HALVES between
the two charged intervals (0.515 original, 0.564 substituted) -- that is a
two-number observation, not a functional form, and the shape is left unnamed.

AND THE NEUTRAL FRAME IS EXCLUDED FROM THAT SEQUENCE, deliberately and for a
stated reason, because including it makes the slopes NON-MONOTONE (0 -> 0.80:
24.21 original, 16.86 substituted; then 39.28/31.09; then 20.21/17.52 -- it
rises then falls). The reason it is excluded: its ionic channel is identically
zero AND its total-energy error has the OPPOSITE SIGN (-6.379 against
+12.987), so it is not the |q| = 0 point of the charged sweep, it is a
different system. Stated explicitly because otherwise the first reader who
plots all four points gets a non-monotone curve and concludes the sublinearity
was invented.

### 2026-09-10  the six-frame run REVERSES the subset's reading
LS6 job 3427615, code 67504e7, provenance IDENTICAL (repo copy executed
directly, sha256 matched against the committed blob), all three gates PASS
(2.776e-16 eV, 9.437e-19 e/A^3, 9.708e-13 eV), truncation 0.0% and cross-energy
ratio 1.000 on all six frames.

[RESULT] final total energy error, meV/atom
             frame  atoms   original  substituted  |err| old  |err| new    d|err|  verdict  factor
     NiN44 q=-0.80    207    +12.987       +8.991     12.987      8.991    -3.996   BETTER   0.692
     NiN44 q=-1.00    207    +20.843      +15.209     20.843     15.209    -5.634   BETTER   0.730
     NiN44 q=-1.32    207    +27.309      +20.817     27.309     20.817    -6.492   BETTER   0.762
     NiN88 q=-1.00    339     -5.413      -11.515      5.413     11.515    +6.102    WORSE   2.127
     NiN88 q=-1.31    339     -2.778       -6.514      2.778      6.514    +3.736    WORSE   2.345
           neutral    207     -6.379       -4.499      6.379      4.499    -1.880   BETTER   0.705
       ALL charged                                    13.866     12.609    -1.257   BETTER   0.909   (+9.1% recovered)
         NiN44 only                                   20.380     15.006    -5.374   BETTER   0.736   (+26.4% recovered)
         NiN88 only                                    4.096      9.014    +4.918    WORSE   2.201   (-120.1%)

THE CELL-SIZE AXIS REVERSES THE SUBSET'S READING, which is exactly why the
user asked for six frames and exactly what the workstation flagged its four
could not say. NiN44 recovers 26.4%; the two NiN88 frames get 2.13x and 2.34x
WORSE; across all five charged frames the mean |error| improves only 9.1%.

THE MECHANISM RESTS ON THE SIGN, NOT ON THE MAGNITUDE -- and an earlier
phrasing of mine overstated it. dE per atom is NEGATIVE on all five charged
frames (-3.996, -5.634, -6.492, -6.102, -3.736 meV/atom), and that uniform
SIGN is the whole argument: a shift of uniform sign cannot repair a bias whose
sign DIFFERS between cells. NiN44 OVER-predicts (+12.987 to +27.309) so the
shift helps it; NiN88 UNDER-predicts (-5.413, -2.778) so the same shift hurts.
Hence the lateral bound charge error is NOT what makes the model's total-energy
error vary between cells.

What is NOT supported, and what I wrote first: that the shift is "similar
regardless of cell or charge". Its magnitude spreads 1.74x (3.736 to 6.492),
and its |q| dependence has OPPOSITE sign in the two cells -- rising within
NiN44 (3.996, 5.634, 6.492) and falling within NiN88 (6.102, 3.736). The
conclusion never needed magnitude uniformity, and that sentence was both
unsupported and stronger than the conclusion requires, which makes it the one
most likely to be quoted back as if it were the measurement.

A REPORTING FAULT OF MINE, AND THE SUBSET COULD HAVE EXPOSED IT -- an earlier
version of this entry said it could not, which was too generous to both of us.
The column I labelled "improvement" was the SIGNED change err1 - err0, which
coincides with an improvement only when the error is positive. The NiN88
frames have negative errors, so their negative changes printed as improvements
while |err| doubled. But the four-frame subset ALREADY contained the
inconsistency: the neutral frame's error is negative (-6.379 -> -4.499), so its
"improvement" printed as +1.881 while the three NiN44 rows printed gains as
NEGATIVE numbers. A column in which both a positive and a negative value mean
"better" is self-contradicting on its face, and it was on the page. Worse on my
side: my own verification script printed "improvement -3.996" next to
"fraction 30.8%" and "improvement +1.880" next to "fraction 29.5%" -- I
computed the fraction against |err|, which sidestepped the bug without my
noticing that the column beside it was wrong. Both of us handled the sign
structure well enough to get the neutral fraction right and neither flagged
the convention. Half-catch on both sides, and the author's half is the larger
one. The table now carries |err| before and after,
the change in |err|, an explicit verdict and the factor, per row, plus a
per-cell split -- because an aggregate over cells hides a sign difference
between them and here the sign IS the finding.

VERDICT on the user's branch -- LED BY THE PER-CELL SPLIT, not by the
aggregate. NiN44 recovers 26.4% on three frames; NiN88 gets 120% WORSE on two;
the two cells' errors have OPPOSITE SIGNS. The sign split IS the result, and
any mean over the two cells hides exactly what was discovered, so the 9.1%
aggregate is reported as a consequence rather than as the headline. It is
arithmetically fine and substantively misleading on its own.
The improvement is small in aggregate (9.1%) and NEGATIVE on one of the two
cells, which is the "small or worse" branch: that
part of the coupling gap can no longer be used to explain the total-energy
plateau, and the remaining energy terms and their compensation are the next
place to look. Recorded with both halves as the partial instruction requires:
NiN44 24-31% recovered, NiN88 120% worse, aggregate 9.1% recovered and 90.9%
remaining.

STILL TRUE and not weakened by this: the substitution is a genuine like-for-
like swap (all gates pass, truncation nil), the model's lateral bound charge
points the right way on every frame (correlations +0.8045 to +0.8733), and the
lateral channel really is 71.3% of the bound COUPLING gap. What the six frames
show is that closing that coupling gap does not translate into closing the
TOTAL ENERGY error, because the error's cell-to-cell variation lives elsewhere.

As the user set it, this does NOT show the network cannot learn the reference
charge. It shows that this line, on its own, does not buy the final energy.

ONE MORE FACT IN THAT TABLE, and it is not about the swap at all -- the
workstation's point and the most actionable thing here. The model was
originally about 5x BETTER on NiN88 (mean |err| 4.096) than on NiN44 (20.380),
with the two cells' errors in OPPOSITE DIRECTIONS. That cell-dependent,
sign-flipping bias is what the remaining error actually looks like, and it is
now the thing to explain. It was visible before this experiment and this
experiment only sharpened it, since a uniform-sign correction is exactly what
cannot address it.

## 2026-09-10  charging_reconcile.py -- the charging error is a missing mu integral
code 1a8a942 (executed 5b1db5b for steps 1-4, then all 200 pairs after the
leak fix; e07f7f7 did not run at all, see the defects below). Workstation,
provenance IDENTICAL on every run that produced numbers.

WHAT THIS CORRECTED FIRST. My A/B cache called E_model - node_energy the
"solvent remainder". It is not: the assembly puts the SOLUTE electrostatic
energy, the optional electron energy and the boundary/dipole corrections in
that same difference. And +5.635 eV is not a solvent number -- it is the error
in the whole system's charged-minus-neutral energy for the sid1/sid601 pair,
and the explicit part changes its electron distribution when charge is added.

THE HEADLINE IS NOT 5.635 eV. The model captures 7.6% of the DFT charging
energy on that pair: -0.465482 against -6.100577, a FACTOR OF 13.1. "+5.635
missing" reads as a correction to something roughly right; it is not roughly
right.

MEASURED FRAMING: NELECT 661 against 660, so this is an electron-ADDITION
energy; the label is energy(sigma->0) from the OUTCAR with NO reservoir term,
so canonical rather than grand canonical; E-fermi moves -7.3736 -> -3.8745.

STEP 4 KILLED THE A/B REFIT BEFORE IT RAN. Readout input features identical
between the two states at 1.9e-16 relative, d(inter_e) exactly +0.000000 eV.
The descriptor is CHARGE-BLIND, so the energy head contributes equally to both
states and cancels exactly in Delta E -- retraining it cannot move a paired
charging energy whatever the target. Both A/B jobs were cancelled unrun. The
workstation's framing before the measurement was the right one: the test was
not asking whether the features happen to agree but whether charge enters the
descriptor at all.

THE TEN TERMS, closure PASS at 4.320e-12, and there are TWO kinds of zero:
  electron energy (optional)      +0.000000   not enabled
  external field . dipole         +0.000000   not enabled
  e0 (atomic, per-species)        +0.000000   cancels: identical species
  energy head (inter_e)           +0.000000   cancels: identical features
  solute electrostatic            +2.893386
  1D solvent compensation         -1.669968
  slab dipole correction          -2.609291
  cavity energy                   +0.049728
  3D solvent energy               -0.164936
  baseline coupling E_bl          +1.035598
The six live terms sum to -0.465 while their absolute values sum to 8.423 --
an 18.1-fold cancellation -- and the missing 5.635 is 1.95x the largest single
live term. So the error is located in a GROUP OF SIX and no term is indicted
by its size.

THE DFT SIDE, by its own formula and not mapped onto the model's. dA_corr
-2.636389 against d(slab dipole correction) -2.609291 agrees to 1.03% -- a
CANDIDATE correspondence, not an identification; if real, that term is
essentially right and the six narrow to five. dA_cav -0.006904 against
d(cavity) +0.049728 is a NEGATIVE result worth stating: small in absolute
terms but OPPOSITE in sign, so the model's cavity term does not track the
DFT's across charging. Internal check: dA_corr = dEcorr + dEcorr_band to 1e-6
(+105.571730 - 108.208118), but those are two ~100 eV terms cancelling
40-fold and neither is quotable alone.

THE RELATION, over all 200 pairs. Four terms cannot contribute to Delta E, so
the model has NO channel for the added electron's own energy -- no integral of
mu dN. Crediting the trapezoid mean of the two Fermi levels times dN:
  split  pairs  eps_D rmse  eps_D bias  resid rmse  resid bias  explained
    val     20    5.621861   +5.570253    0.321732   -0.166124      94.3%
  train    160    5.717653   +5.676327    0.354029   -0.164801      93.8%
   test     20    6.054751   +6.034449    0.262612   -0.045151      95.7%
    ALL    200    5.742802   +5.701532    0.342807   -0.152969      94.0%
  slope +1.0251, correlation +0.8951
  ENDPOINT CONTROLS: charged 2.0644 (bias +1.9674), neutral 2.3118 (bias
  -2.2734), trapezoid 0.3428 (bias -0.1530) -- 6.0x and 6.7x worse.
The endpoint controls are the result: the missing quantity is an INTEGRAL over
electron count, not a single chemical potential. And pair 1 sits at percentile
49, so the pair the relation was identified from is representative rather than
the outlier a one-point fit invites.

READ 94% WITH ITS DENOMINATOR -- the workstation's caution and it changes what
may be claimed. eps_Delta is nearly a pure offset: rmse 5.7428 against bias
5.7015, so its own std is only 0.6872 eV. By measure over the 200:
  RMS ratio            94.0%
  offset removed       97.3%   (bias +5.7015 -> -0.1530)  <- STRUCTURAL
  variance about mean  80.1%
  std reduction        55.4%   (spread 0.6872 -> 0.3068)  <- FRAME BY FRAME
The relation does two things and only one is 94%: it identifies and removes a
missing ~5.7 eV TERM, 97.3% of the offset, which is the structural claim and
is strongly supported; of the frame-to-frame VARIATION it removes about half.
"94% explained" would be read as the remainder being 6% of what it was, and
frame by frame it is about 45% of the spread. Both numbers travel together.

TWO RESIDUAL FEATURES THAT ARE SYSTEMATIC AT n = 200, not scatter: the slope
is 1.0251, 2.5% above unity, and the residual bias is -0.153 eV with the range
-1.074 to +0.458, skewed negative. So the missing term is the chemical-
potential integral to within a few per cent AND there is a systematic
remainder beyond it.

NO SPLIT EFFECT (94.3 / 93.8 / 95.7), which supports the structural reading
over a learned one: a term that is ABSENT rather than mis-learned should not
care about the split.

AND THE 19-PAIR SUBSET WAS OPTIMISTIC, in the direction its own report
flagged: correlation +0.968 there against +0.8951 on all 200, residual RMSE
0.229 against 0.343.

DEFECTS FOUND IN MY OWN SCRIPTS THIS ROUND, all by the workstation:
  no availability guard -- dft_terms opened OUTCAR directly, so a machine with
  a partial payload would print four steps and then throw inside the fifth
  loop. The exact pattern lateral_bound_swap grew a selector for two rounds
  earlier, not carried over. SECOND instance of a fix failing to propagate to
  the next script, after sid 201 in FRAMES.
  a one-sided directory listing read as a PAIR CENSUS, twice and in opposite
  directions: once all charged sides present and neutral absent, once the
  reverse. The intersection on that machine is {1}, the pair the relation was
  derived from, so the run I dispatched would have tested a one-parameter
  relation on its own fitting point.
  step 5 did not need OUTCAR at all. The label, dN and both Fermi levels are
  in the xyz for all 800 frames, the xyz energy matches the OUTCAR sigma->0 to
  all printed digits and its Fermi carries MORE digits (-3.8745365801 against
  -3.8745), so the test was never payload-limited: 200 pairs across all three
  splits on either machine, which also removed the train/val labelling problem
  instead of managing it. My step-1 residual of -0.0110 eV was itself computed
  at the coarser Fermi precision.
  e07f7f7 DID NOT RUN: `_bk = PB.PB1DBackend.solve_graph` with no
  `import mace.modules.pb1d_backend as PB`. The _evict patch brought the
  reference and left the import behind. Fixed at 1a8a942.

THE LEAK WAS REAL AND MY DIAGNOSIS OF IT WAS DIFFERENT FROM THE FIRST
CANDIDATE. RSS grew 1.98 -> 8.09 GB over 38 forwards, ~161 MB each, GPU flat.
The workstation's candidate (_grid_for's unbounded _grids) is REFUTED: the
cells are identical across frames -- 1 distinct Lattice string over 60 train
frames -- so that key never varies. The actual sources: self._bl_ram
[sample_id] at ~24 MB a sample bounded only at MACE_PB1D_PRELOAD_MAX=512, and
baseline_cache.npy being mmap'd so every sample's row faults in pages that
count in RSS and are never dropped. Both grow strictly with distinct samples,
which is the monotone-no-plateau signature. MACE_PB1D_NO_PRELOAD=1 plus
per-pair eviction took it from 160 to 11 MB/forward, 14.5x -- reduced, not
removed, and the in-script RSS reporting is what made that readable at pair 20
instead of at a kill.
NOT A MISUSE OF THE CACHE BY ITS AUTHOR: 24 MB a sample bounded at 512 is a
deliberate design for TRAINING, where the same samples recur every epoch. A
one-pass audit over 400 distinct samples is the case it was never meant for,
so any future cohort-wide audit needs the same flag.

AND THE WORKSTATION'S KILLS ARE NOT ABOUT MEMORY. Across four kills there,
every background task reaching about 90 seconds was killed with "low memory"
whatever its RSS, while foreground runs under `timeout 570` completed every
time -- including this 400-forward one at 5.94 GB with 254.7 GB available,
after the same run had been killed in the background at 3.2 GB. Two of the
four had no real growth, one did, this one did not. The mitigation there is
the foreground, not a smaller job, so chunking for that machine's benefit is
the wrong remedy.

## 2026-09-10  charge_branch_fit.py -- the residual is a per-electron constant
code 164df9f. Workstation, cache reuse, provenance IDENTICAL, test SEALED.

THE FRAMING THE USER CORRECTED FIRST, and it is what makes the result mean
anything. mu = dE/dN, so int mu dN IS the whole charging energy and already
contains what the existing electrostatic and solvent terms contribute -- it is
not an extra additive term and adding it on top would double-count. The
apparent 94% was close to a tautology: eps_Delta = dE_model - int mu dN and
dE_model is only 7.6% of the truth, so eps_Delta is about -int mu dN whatever
the model does, and the trapezoid beating the endpoints 6-fold is a fact about
numerical integration. The "missing int mu dN term" reading is WITHDRAWN.
So the target became the residual dE_correction = dE_DFT - dE_existing, which
cannot double-count by construction.

CONVENTIONS, settled from source and settings rather than from a field name,
which is where the previous round overreached:
  0 of 800 INCARs set EFERMI_ref or any const-pot tag; its default in
  solvation.F is 0; L_const_pot is (EFERMI_ref < 0) hence FALSE; the OUTCAR
  prints EFERMI_ref = 0.000000;
  Ecorr_sol = A_corr - Ecorr_band - q_sol*EFERMI_ref IS added into TOTEN
  (electron_all.F:456, electron.F:590), so the solvation free energy is inside
  the label while the reservoir term, present in the formula, is identically
  zero;
  the label is energy(sigma->0), exactly VASP's ISMEAR=1 extrapolation of the
  printed TOTEN, ratio 1/3 to 3e-9;
  the stored Fermi is the raw OUTCAR E-fermi, and the reference zero is the
  RIGHT vacuum, pinned at +2.3e-4 and +2.0e-5 eV in the two states with a
  spread of 3.6e-6, so mu is on a unified scale across a pair to 2.2e-4 eV.
  The LEFT vacuum moves 3.46 eV, which is the slab responding to the charge.

[RESULT] val pairs, eV, each rung judged against the ROW ABOVE
                   form      rmse      bias  mean-removed   all three better?
        (no correction)    5.6219   -5.5703        0.7600   -
                   a*dN    0.1569   -0.0430        0.1509   YES
          a*dN + b*dN^2    0.1572   -0.0406        0.1519   NO: rmse and
                                                            mean-removed worse
     (a + w.h)*dN ridge    0.1000   -0.0041        0.0999   yes, but on a
                                                            contaminated score
  mu_bar*dN (CONTROL)      0.3217   +0.1661        0.2755   (extra information)

ESTABLISHED: a per-electron energy baseline is the first thing missing and it
is decisive -- ONE fitted parameter removes 99.2% of the bias and 80.1% of the
frame-to-frame spread. The fitted constant is a = -5.2885 eV per electron
against a mean mu_bar of -5.4425, so 0.154 eV less negative: close enough to
be the same physics, far enough that "the constant IS the mean chemical
potential" is not what the data says.

NO EXTRA GAIN FROM CURVATURE ON THE RESIDUAL, and the earlier wording
"REFUTED: charging curvature" over-reached. b*dN^2 fails on two of three
metrics, so the RESIDUAL of the existing model needs no curvature term. That
is NOT a statement about the real system: what is fitted here is the residual,
and the existing terms may already carry the physical curvature. The
observation is still sharp against one specific hypothesis -- that the missing
quantity is a trapezoid of a chemical potential linear in N, for which the
quadratic is the exact form -- but it says nothing about whether the system has
charging curvature.

THE CONTROL LOST, which is what proves the earlier "oracle"/"upper bound"
label was wrong: mu_bar*dN uses the true DFT Fermi levels of BOTH states and
still reaches only 0.3217 and 0.2755 against one fitted constant's 0.1569 and
0.1509. Rescaling does not rescue it -- measured scale 0.9723 gives 0.2779 and
0.2778, so rescaling helps the rmse and leaves the spread alone, and the
constant is still 1.84x better in spread. The 200-pair correlation of 0.8951
caps what any rescaling can do.

WHY A CONSTANT BEATS THE TRUE CHEMICAL POTENTIAL, stated through the direct
measurement after an earlier route was shown unsound:
  corr((mu_bar - a)*dN, dE_model) = +0.4253, so the model's EXISTING terms
  already track part of mu's variation and crediting the full integral
  over-corrects -- the user's definitional point, visible in the data;
  spreads (mu_bar-a)*dN 0.2096 and dE_model 0.3020 would give 0.3676 if
  independent, against a measured row-1 residual spread of 0.2159, so the
  cancellation cuts it 41%.
THE UNSOUND ROUTE, recorded because the conclusion survived it: arguing this
through the control's residual by treating Q = dE_DFT - mu_bar*dN as small is
wrong. Q's spread is 0.1837, which is 0.61 of dE_model's in SPREAD and 0.37 in
VARIANCE -- the two are not the same statement and this chain keeps swapping
them -- with corr(Q, dE_model) = +0.2735 and a mean of -0.5671 eV. The
control's residual lands near spread(dE_model) only because those two offset.
Right answer, wrong route.

UNSETTLED: whether the correction depends on configuration -- and the earlier
wording here over-stated the problem. Choosing alpha on VAL is STANDARD
PRACTICE and is not itself contamination; what was wrong was taking an
ordinary bootstrap CI on those same val points as evidence that the
improvement is SIGNIFICANT, with ten penalties scanned against twenty points.
That significance conclusion is withdrawn; the selection is not. Row 3 shows a
real sign of further improvement, val rmse 0.157 -> 0.100 with bias and
mean-removed error both improving as well, and it is worth an independent
check rather than being called undecidable.
  with alpha by CV inside train: alpha 1e-06 at the grid edge, val rmse
  0.1083, bootstrap of (row1 - row3) median +0.0488, CI [-0.0127, +0.1128] --
  includes zero;
  with alpha selected on val: alpha 1, val rmse 0.1000, CI [+0.0020, +0.1120]
  -- excludes zero.
The verdict FLIPPED on the selection, not on evidence. The lower bound moved
+0.0147 to land at +0.0020, which is 3.5% of the median, and the penalty
bought 5.3 of row 3's 36.3-point margin over row 1 -- the other 31.0 were
already there at the smallest penalty. So the ridge is not what makes row 3
look good; it is what pushed a marginal interval across zero.

THE SHUFFLE CONTROL AND THE BOOTSTRAP ANSWER DIFFERENT NULLS and must not be
read as agreeing. Shuffled features give 0.7408, far WORSE than row 1's
0.1569, so the shuffle null is "these 1153 features are noise" and its failure
mode is overfitting damage. Beating it establishes only that the features are
not noise; whether they beat knowing the electron count ALONE is a different
question that only the bootstrap addresses.

WHICH ROWS ARE CLEAN, since this decides what the seal is for: rows 1, 2 and 4
were fitted on train alone (or not fitted at all), so their val scores are
held-out estimates. ONLY row 3 had anything selected on val. So opening the
seal for form 1 adds precision to a number that is already clean, while for
form 3 it is the entire remaining evidence. One caveat that survives either
way: choosing WHICH FORM to deploy on val scores is itself a 3-way selection,
so the winner's val score is mildly optimistic even for form 1 -- much smaller
than row 3's ten-value penalty scan stacked on top, not zero.

FOUR DEFECTS OF MINE, all caught by the workstation before anything was
reported: a verdict line printing "excludes zero, so the gain is real" four
lines below its own caveat that the val score was selection-contaminated, when
the CI is computed on those same contaminated points; a table cell carrying a
verdict against no-correction and a parenthetical against the row above at
once, so the cell contradicted itself; the unsound route above; and the
oracle/upper-bound label on what is only a control.

### FINAL COMPARISON, test set opened once, both forms
LS6 job 3430107, code 56cda73, provenance IDENTICAL, 6m58s, 400 forwards with
MACE_PB1D_NO_PRELOAD (RSS 5.81 GB at pair 200, no kill).

Opening the test set once does NOT mean evaluating one model -- the user's
point, and it dissolved the false dilemma of form 1 against form 3. Both were
frozen before the test set was touched: form 1's coefficient, and form 3's
alpha = 1, its train-fitted standardisation and its weights.

                       form      rmse      bias  mean-removed
               form 1  a*dN    0.1507   -0.1057        0.1074
       form 3  (a + w.h)*dN    0.1557   -0.0557        0.1454
            form 3 - form 1   +0.0050   -0.0500       +0.0380
  form 3 closer on 10 of 20 pairs -- exactly a coin flip

FORM 3'S ADVANTAGE DOES NOT SURVIVE. It is 3.3% worse in rmse and 35% worse in
mean-removed error, better only in bias. On val it was 36% BETTER in rmse
(0.1000 against 0.1569); on test it is 3.3% worse. So the val margin was the
selection, and this is exactly what the held-back test was for. The per-pair
column is what makes it unambiguous: 10 of 20 is a coin flip, so form 3 was
not winning broadly and its val margin did not come from a real effect spread
across the set.

DECISION, per the reading fixed before the numbers were seen: KEEP FORM 1, and
stop tuning against this test set. The structure features are NOT established
to carry additional information. No further form will be proposed to reopen
these pairs.

FORM 1 ON TEST, the clean number, single coefficient a = -5.2745 eV per
electron fitted on train alone:
  residual before   rmse 6.0548   bias -6.0344   mean-removed 0.4954
  residual after    rmse 0.1507   bias -0.1057   mean-removed 0.1074
a 40-fold rmse reduction, 98.2% of the bias and 78.3% of the spread, on data
that touched no fitting and no selection. Form 1's test rmse (0.1507) is
slightly BETTER than its val rmse (0.1569), so it generalises; its bias is
larger on test and its spread smaller, both within what 20-pair samples give.

WHAT THIS DOES AND DOES NOT SETTLE. It answers the practical question: a
single uniform per-electron energy is enough, and making it configuration-
dependent is not justified by this evidence. It does NOT separate how much of
that constant absorbs density error from how much is electronic energy the
model never represents explicitly -- a fitted coefficient cannot be
apportioned that way, and nothing here attempts it.
