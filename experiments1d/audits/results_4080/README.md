# Workstation (2 x RTX 4090) run logs

Logs only. No field data. Produced on the user's UT workstation, which does not
share a filesystem with Lonestar6.

## Machine

- 2 x NVIDIA GeForce RTX 4090, 24564 MiB each, driver 570.153.02. NOT 4080/16 GB.
- One card used (CUDA_VISIBLE_DEVICES=0).
- conda env `pmp39`: Python 3.9.25, torch 2.2.1+cu121, ase 3.22.1, e3nn 0.4.4,
  scipy 1.13.1 — exact match to the required list.
- float64 DGEMM 1175 GFLOP/s; float64 FFT 100x100x300 0.88 ms;
  168x168x500 5.70-9.85 ms across runs.

## Data

No kit was copied. Everything is read from the read-only migration kit already
on the machine, `/home/rw28343/jobs/1-3DPB/0-ls6/migration_kit`:
`dft_audit_set/1-44_GCE/cal_1` (sid 1), `dft_audit_set/5-44_neutral_withsolv/cal_1`
(sid 601), `models/s3d_gate_bl2_run-123.model` + `_epoch-33.pt`,
`data/bundle800/`. `baseline_index.json` used unmodified: 600 entries,
sid 1 -> row 0, sid 601 -> row 0.

## Provenance per log

| log | mace HEAD | pb repo HEAD |
|---|---|---|
| `smoke_028702a.log` | 028702a | 9b3b9ba |
| `envelope_alignment_028702a.log` | 028702a + a 4-line path patch (KIT_DFT/KIT_PB_REPO/KIT_MACE_REPO), superseded by 2ad0d55 | 9b3b9ba |
| `smoke_afe513d.log` | afe513d | 9b3b9ba |
| `plane_profile_audit_afe513d.log` | afe513d | 9b3b9ba |

Run from a clean detached worktree, never from the user's own checkout — that
checkout carries an uncommitted `MACE_PB1D_DTYPE=float32` change to
`mace/modules/pb1d_backend.py` which the user has asked be left alone, and
which must not be in the path of any 1e-6 comparison.

## Gate status

`reference_values.json` was absent for every run here, so the 1e-6 relative
comparison is UNEVALUATED. Every number in these logs is a measurement, not a
pass or a fail.

Solver provenance (afe513d onwards): both loops exit on tolerance, never on
their cap — fixed point 7 steps against a cap of 60, Newton 7-8 total outer
iterations against a cap of 12 per call, on both frames.

Caution for whoever wires up the comparison: `pb_rms_last` is not reproducible
run to run on identical code and identical input. Two runs here gave
3.55960868e-13 and 6.89045931e-13 for sid 1, and 6.07368057e-13 and
1.56622252e-12 for sid 601. Against the 1e-8 denominator floor in
`smoke_test.py` those are relative differences of 3.3e-5 and 9.6e-5, both above
the 1e-6 tolerance, so comparing that field will report SMOKE FAIL regardless
of whether the port is correct. (An earlier revision of this paragraph added
"every energy term, by contrast, was bit-identical across the two runs". That
was inferred from the 8-decimal console printout and it is wrong -- see the
correction section below.)

## Update: gate evaluated (mace HEAD 060ef30, pb 9b3b9ba)

`smoke_060ef30_GATED.log` — **SMOKE PASS** against
`reference_values_a100.json` (NVIDIA A100-PCIE-40GB): worst relative
difference 1.72e-09 over 52 gated numbers, 0 over the 1e-6 tolerance. The
three cancellation-limited residuals are reported by order of magnitude as
intended (0.15, 0.25, 0.15 decades on sid 1; 0.21, 0.01, 0.21 on sid 601).
Solver provenance identical to the A100: both loops exit on tolerance on both
frames, fixed point 7 steps of a 60 cap, Newton 8 (sid 1) and 7 (sid 601)
outer of a 12 cap.

Note on the benchmark line: the reference's `fft64_168x168x500_ms` = 57.902 ms
was recorded at 2ad0d55, BEFORE the big-FFT warm-up was added at 3b289ba, so
it still contains cuFFT plan creation for a non-power-of-two shape. Our warmed
number is 4.58 ms, and smoke_test.py therefore prints "0.1x slower here",
which reads as this card being 10x faster at that FFT. That comparison is an
artefact of the stale reference field, not a throughput result. The reference
needs regenerating at 3b289ba or later before that line means anything. The
DGEMM comparison (1176 vs 9757 GFLOP/s, 8.3x) and the small FFT (0.88 vs
0.46 ms) are both warmed on each side and are the honest ones.

`envelope_void_audit_060ef30.log`, `profile_shift_scan_060ef30.log` — the two
diagnostics. Numbers only; no verdict attached. Two script defects found while
running them are described in the cross-session report and should be fixed
before these outputs are quoted: the "% of deficit recovered by shift alone"
figure overflows to -5.2e31% on the charged bound channel (correct value
98.9%), and the `s_diel > 0.99` summary line divides by an empty set and
prints nan because the model's switch saturates at 0.9447 and never reaches
0.99 at all.


## Correction: the gated fields are NOT bit-identical run to run

This file and the commit that added it claimed the energy terms were
bit-identical between two runs on this machine. That came from reading the
8-decimal console output, not the full-precision JSON, and it is wrong.
Comparing the JSON from two runs here, every float except `q_tot` and the
integer-valued solver-provenance fields differs:

| field | run-to-run relative difference |
|---|---|
| `e_bl` (sid 1) | 2.2e-09 |
| `rho_layer_z_absint` (sid 1) | 1.3e-10 |
| `delta_plane_max` (sid 1) | 1.1e-10 |
| `comp_1d` (sid 1) | 8.5e-11 |
| `e_s3d` (sid 1) | 4.6e-11 |
| `e_xsol` (sid 1) | 1.4e-11 |
| everything else | 1e-12 or below |

This does not weaken the gate: 2.2e-09 is 2.7 orders inside the 1e-6 tolerance,
and the worst deviation from the A100 reference across all 52 gated numbers was
1.72e-09 in one run and 2.75e-10 in another, both dominated by that same `e_bl`
field. But the tolerance must not be tightened below about 1e-7 on an assumption
of determinism, and a SMOKE FAIL at the 1e-8 level would be this jitter rather
than a real divergence.

## Update: fixes verified (mace HEAD 40866f6)

All three defects reported from this machine are fixed and re-measured here.
Every substantive table number is unchanged; the only differences against the
060ef30 logs are the three defective lines themselves plus 1e-15-level jitter in
the bulk gradient bins.

- shift scan overflow: now prints "recovering 98.9% of the -0.5253 eV deficit
  with shift ALONE" and names the signed deficit it divided by.
- negligible-reference guard: the neutral ionic channel is now SKIPPED, naming
  the +0.000556 eV coupling and 0.0041 e of reference charge that triggered it.
- the empty-set nan is gone, replaced by the switch-ceiling block.

`smoke_40866f6.log` — SMOKE PASS, worst relative difference 2.75e-10 over 52
gated numbers. The stale big-FFT key is out of the compared set, so the
misleading "0.1x slower here" line no longer appears.

One wording problem remains in `envelope_void_audit.py`'s new output. Its
summary line asserts "A switch that does not saturate has a non-vanishing
normalized gradient wherever it is still creeping", and the two lines printed
directly beneath it refute that reading: in its own plateau region the model
carries 1.06% of its gradient weight against DFT's 1.15% — less, not more. The
far-field envelope weight is the second dielectric interface at the far face of
the solvent slab (z = 37.5-42 A, 45.80% model against 44.12% DFT, matching the
T4 "beyond 5 A" columns of 45.79% and 44.11% to the last printed digit), not
bulk creep. The plateau level, 0.9447 against 1.0000 across the solvent region
on both frames, is a separate and real finding.

## Final state

`pb-s3d-energy` at **ca6c790** is the pinned final state of this exchange; every
log in this directory was produced by code that is an ancestor of it, and the
per-log HEADs above say which. Verified from this machine: ca6c790 is the branch
tip, and its `experiments1d/RUNS.md` carries the run-to-run jitter table, the
correction to the bit-identical claim, and the 0.72-decade NONGATED spread from
the 40866f6 run.

The three LS6 jobs that were queued at the end (3426543 void cross-check,
3426544 shift cross-check, 3426545 reference regeneration) were cancelled rather
than run. The reason: job 3426230 had already reproduced this machine's
`envelope_alignment` output digit for digit on both frames, which validates the
chain all three shared, so two would have been repeats and the third only a
warmed A100 big-FFT benchmark that is excluded from the gate anyway. That is why
`reference_values_a100.json` still carries the renamed-out
`fft64_168x168x500_ms_UNWARMED_2ad0d55` key with its note instead of a warmed
value, and why that one benchmark line is absent from the smoke output rather
than wrong.

Nothing in this directory is superseded by the cancellations.

## Dispatched job: density tail test (mace HEAD 2a2edb2)

`density_2a2edb2.log`. Provenance is in the log's first line: mace 2a2edb2,
pb 9b3b9ba, sha256 of the executed file
a72d8b74cb502abfb61ae17f9a1ce06610a59bf2cff9690fd9721e769a2f73ff, verified
IDENTICAL to the committed `experiments1d/audits/density_tail_test.py` at
2a2edb2. Single run, no cross-check — the LS6 copy was cancelled.

Result in one line: the tail swap closes **0.0% of the plateau gap in both
directions**, on both frames.

Two defects found, reported before any interpretation:

1. The signed-denominator overflow that was fixed in `profile_shift_scan.py` at
   40866f6 is present again here, in part C's `r2`:
   `max(float(s_used[M_m].mean()) - float(s_d[M_d].mean()), 1e-30)` evaluates
   `max(-0.0556, 1e-30)` and returns `1e-30`. The printed
   "-189773485814725950308352.0% (native grid)" and the neutral frame's
   "-571753788758400039911424.0%" are that division. Recovering the numerator
   from the printed value gives -1.898e-09 and -5.718e-09, so with the correct
   signed denominator both are **+0.0%** (3.4e-08 and 1.0e-07 of the gap).
   `r1` is unaffected because its denominator is positive.

2. Part B cannot answer the question it is posed. Its legend says "same n_e ->
   same s_diel in both columns means the switch is identical", but its own
   standard-deviation column reads 0.39 to 0.49 on a quantity bounded in [0,1],
   which by the legend's second sentence means the pipeline is not a pointwise
   function of n_e. If the map is non-local, two fields with different spatial
   structure will give different binned means under an IDENTICAL map, so the
   difference between the columns does not distinguish "different map" from
   "same map, different structure". The sd finding stands and voids the
   mean-comparison finding.

What the run does establish, without pronouncing on the recipe or the
parameters: part A's recomputed cavity is bit-identical to the one the model
actually used (max, mean and 99th-percentile difference all exactly 0.000e+00,
against a call-to-call floor also exactly 0.000e+00), so the model side is
self-consistent and the difference is not in how the model builds its cavity
from its own n_e. Inside the mask both densities sit below NC_K = 0.015 at
every percentile, the swap moves the plateau by under 6e-09, and the plateaus
still differ by 0.0556. So the plateau difference
is not carried by n_e inside the mask. Where it is carried is open; the
parameter line shows non-locality keys in use (I_NLOC_SOL, LNLDIEL, LNLION)
and part B's spread is consistent with a non-local pipeline, which is the next
thing to examine rather than a conclusion.


### Correction to the part D margin above

An earlier revision of the paragraph above said both densities sit "3 or more
orders below NC_K = 0.015 at every percentile up to the 99th". That is wrong,
and it overstated the margin at exactly the percentiles where the margin
matters. The actual ratios of NC_K = 0.015 to the measured percentile, charged
frame:

| percentile | model | ratio to NC_K | DFT | ratio to NC_K |
|---|---|---|---|---|
| 50th | 7.203e-08 | 2.1e+05 x | 1.612e-08 | 9.3e+05 x |
| 75th | 2.506e-06 | 5986 x | 1.536e-06 | 9766 x |
| 95th | 5.080e-05 | 295 x | 2.208e-05 | 679 x |
| 99th | 5.689e-04 | **26 x** | 1.219e-04 | 123 x |
| max | 4.174e-03 | **3.6 x** | 4.563e-04 | 33 x |

Neutral frame: model 99th 4.374e-04 (34x) and max 5.509e-03 (**2.7x**); DFT
99th 9.930e-05 (151x) and max 3.813e-04 (39x).

So "3 or more orders" holds only up to about the 75th percentile. At the 99th
the model density is 26x below the threshold and at its maximum only 3.6x below
(2.7x on the neutral frame). The claim that both densities are "deep below
threshold throughout the mask" is therefore too strong in the upper tail, which
is the part of the distribution a threshold argument depends on. What the
measurement supports is narrower: the swap moves the plateau by under 6e-09
while the plateaus differ by 0.0556, so the density inside the mask does not
carry the difference — that stands on the swap result directly and does not
need the threshold-margin argument at all.

## Dispatched job: cavity_substitution_2x2 — KILLED TWICE, and not for the stated reason

mace 890eb3e, pb 9b3b9ba, executed file sha256
6975b5347f6caaf1126e813ab36159d8efc438ab231f3e153c49778ac40d609b, verified
IDENTICAL to the commit. `MACE_CAVSRC=model` reached the same point both times
and was stopped by the harness with "the system is running low on memory".

The second attempt carried a 5-second sampler. **The machine was never low on
memory.** Twenty samples over 100 seconds, ending at the kill:

| quantity | value |
|---|---|
| process RSS | 1.54 GB, flat and unchanging across all 20 samples |
| host MemAvailable | 241.9 GB minimum, and RISING at the kill (254.19 -> 254.78 GB) |
| GPU | 10619 MiB steady, of 24564 MiB |
| host total | 251 GB, 1 GB swap, unused |

Nothing was growing on either the host or the card. No CUDA OOM, no traceback,
no Python error. There is no cgroup limit in play: `memory.max` and
`memory.high` on this session scope both read `max`.

The one number that is large is `memory.current` for the session scope, 99.6 GB,
which tracks the 104 GB of `buff/cache` this session has accumulated from
repeatedly reading the 257 MB ASCII DFT field files. That memory is reclaimable
page cache, which is why `MemAvailable` stayed at 254 GB. A monitor that reads
`memory.current` rather than `MemAvailable` would see 99.6 GB and conclude the
session is consuming a lot, and that is the only quantity measured here that is
consistent with the kill message. Stated as the measurement plus the one
hypothesis it fits, not as a diagnosis.

This is consistent with the peer session's report that the same machinery ran
to completion on Lonestar6 as `joint_subspace_solve` — 111 minutes, three
frames, three bases, no host-memory problem. It is not the script.

### What the incomplete run did establish

  [self-check] rebuilt-vs-exported residual: max abs 0.000e+00, relative 0.000e+00 -- OK
  ENVELOPE SOURCE: the model's own cavity (baseline column)
  channels in play: ['env_b', 's_ion', 's_diel', 'env^0.5']  (q_tot -1.000, ion channel live)
  reference lateral: cross -2.2417  self +0.6829  |q| 2.5398 e
  [ORIGINAL COEFFICIENTS, envelope = model] cross -0.9350 (0.4171 x ref)  self +0.6118 (0.8959 x)  |q| 1.7805 (0.7010 x)
  design matrix: 80000 x 22356 float32 = 7.15 GB

The self-check passing means the original-coefficient machinery reproduces the
model's own exported residual exactly, so that column is validated even though
the 2x2 was never completed. One of the four cells exists; the other three and
the whole `MACE_CAVSRC=dft` column do not.

The -2.2417 here against envelope_alignment's -2.2434 is not a discrepancy: this
script's reference lateral is RHOB + RHOION together while T2's row is the bound
channel alone, and the ionic lateral term appears on both sides at the same size
(+0.0017 eV on the reference, +0.0016 eV on the model).

Nothing was modified to get past this — NPTS was not lowered, nothing was
chunked, the script was not touched. Completing the run needs a decision about
how to launch it that is the user's to make, not something to work around.

## Dispatched job: ion_switch_resolve (mace HEAD a137d48)

`ionsw_a137d48.log`. Provenance in the log's first line: mace a137d48, pb
9b3b9ba, executed file sha256
b5b39227d3f8e558c2ce706b49dca4a87fd991f7f5e1494942328c8202e9d9cb, verified
IDENTICAL to the commit. Ran to completion in about 15 seconds; no kill this
time. Peak process RSS 1.9 GB, host MemAvailable never below 241.9 GB, peak GPU
4445 MiB of 24564. `ion_switch_resolve_arrays.npz` written, 65660 bytes.

All three gates PASS and the convention check reconstructs phi - phi_sol to
6.916e-12 eV after removing the G=0 constant.

**The ionic-channel improvement does not carry to the whole system.** The
substitution improves the ionic channel by a factor of 10.6 in L1 and removes
the displacement entirely, and the whole-system total energy gap changes by
0.0001 eV out of 0.3434 — 0.03% of it. Charge L1 for the sum and all three
potential-error measures move slightly the WRONG way.

| ionic channel, fixed scoring potential | net (e) | cross | eV per e | shift | resid | L1 |
|---|---|---|---|---|---|---|
| DFT reference | +1.0000 | -1.9385 | -1.9385 | -0.000 | 0.0% | 0.00000 |
| baseline, model s_ion, re-solved | +1.0000 | -2.0802 | -2.0802 | +0.375 | 1.4% | 0.18687 |
| DFT s_ion, NO re-solve | +0.8863 | -1.7399 | -1.9630 | +0.025 | 1.3% | 0.11403 |
| DFT s_ion, RE-SOLVED | +1.0000 | -1.9555 | -1.9555 | -0.000 | 1.0% | 0.01764 |

| whole system, bound + ionic | net (e) | chg L1 | chg max | cross | self | total | d total |
|---|---|---|---|---|---|---|---|
| DFT reference | +1.0000 | 0.00000 | 0.00e+00 | -3.0243 | +1.5237 | -1.5005 | +0.0000 |
| baseline, model s_ion | +1.0000 | 0.63739 | 2.38e-03 | -2.6289 | +1.4718 | -1.1572 | +0.3434 |
| DFT s_ion, re-solved | +1.0000 | 0.63760 | 2.38e-03 | -2.6272 | +1.4700 | -1.1572 | +0.3433 |

Mutual bound-ion term in the self-energy: reference -1.3112 eV, baseline
-1.2824, substituted -1.4111. The baseline sits 0.0288 eV from the reference
and the substituted case 0.0999 eV from it on the other side, so on this term
the substitution is about 3.5x further out than the baseline was.

Potential profile: baseline phi L1 2.7322 eV A, max 0.2915 eV, rms 0.11093;
substituted 2.7548, 0.2967, 0.11286. All three worse, by 0.8% to 1.8%.

A caution on reading the charge column, because this project has twice been
caught by exactly this. The sum's L1 rose by 0.00021 e while the ionic
channel's own L1 fell by 0.16923 e. It is tempting to conclude the bound
channel degraded by about 0.169 e, and that does NOT follow: L1 of a sum is not
the sum of L1s, so the bound channel's own change cannot be read off these two
numbers. What is measured is that the ionic channel improved 10.6-fold in
isolation and the sum did not improve at all. Separating the bound channel's
self-consistent response from the ionic improvement needs the bound channel
scored on its own, which this run does not do.

## Dispatched job: bound_response_test (mace HEAD 940549c)

`bndresp_940549c.log`. mace 940549c, pb 9b3b9ba, executed file sha256
4b8337d4424f0a8985722f564d266ff656aa178e1c5405f87c37e815c3c2ac5d, IDENTICAL to
the commit. Ran in about 10 s. Peak RSS 2.0 GB, MemAvailable never below
242.5 GB. `bound_response_test_arrays.npz` written.

Gates all PASS. `|B @ 1|` = 1.219e-10 against `|B @ phi|` = 4.128e+03, ratio
2.95e-14, so B annihilates constants and the electrolyte reference zero cannot
reach the bound charge — as predicted. Sign measured, not assumed: +PHI_raw
correlates +0.9995 and -PHI_raw -0.9995, a clean separation. Room to act: the
two driving potentials differ by rms 0.11093 eV, 0.10304 with the constant
removed. Cross-check passes: this run's baseline bound coupling gap is +0.5370 eV
against +0.5371 from the whole-system run by the total-minus-ionic route, a
difference of -0.0001 eV.

| case | net (e) | int&#124;.&#124; | cross | gap | L1 | max dev | shift | resid |
|---|---|---|---|---|---|---|---|---|
| DFT reference RHOB | +0.0000 | 2.0193 | -1.0857 | +0.0000 | 0.00000 | 0.00e+00 | -0.000 | 0.0% |
| model response, model phi | +0.0000 | 2.0302 | -0.5487 | +0.5370 | 0.80003 | 2.38e-03 | +0.050 | 48.7% |
| model response, DFT phi | +0.0000 | 8.5965 | +4.4975 | +5.5832 | 7.56925 | 2.05e-02 | +0.825 | 85.0% |
| control: DFT phi, flipped | -0.0000 | 207.4935 | +240.1045 | +241.1903 | 208.82518 | 9.50e-01 | +1.300 | 94.4% |

No row's displacement sits at the ±2.0 A scan edge.

### The table cannot answer the question it was built for

> **The mechanism given in this section is WRONG. The conclusion it supports
> survives, for a different reason. See "Correction: the mechanism was wrong"
> below, which is backed by measurement at 52ea73c.**

The bound charge is a near-total cancellation of two large terms, and only the
model's own self-consistent potential produces the cancellation. Recovered
exactly from the saved arrays by linearity — `rho(phi) = -(B@phi + C)/V`, so
`C/V = -(rho(phi_dft) + rho(-phi_dft))/2` and `B@phi_dft/V =
-(rho(phi_dft) - rho(-phi_dft))/2`; the reconstruction residual is 3e-17:

| term, sum&#124;.&#124; over z in density units | value |
|---|---|
| offset `nb_off/V` | 7.2847 |
| `B@phi_model/V` | 7.3768 |
| `B@phi_dft/V` | 7.3006 |
| result, model phi | 0.14266 |
| result, DFT phi | 0.60407 |

The baseline result is 1.93% of the term magnitude that produces it, a 103-fold
cancellation. `B@phi_dft` differs from `B@phi_model` by 1.03% in aggregate
magnitude, and that 1% perturbation takes the residue from 1.93% to 8.28% of the
term magnitude — which is the reported 9.5-fold L1 degradation. Spectrally the
two potentials are identical to within 1% below half-Nyquist (band ratios 1.00,
0.99, 0.99) and differ by 1.81x only in the top half, which is exactly the band
a double-derivative operator amplifies.

So a large degradation under substitution is what breaking a finely tuned
cancellation looks like whether the response is right or wrong, and the
dichotomy has a third branch neither of us anticipated: feeding a foreign
potential into a linear response that was converged with its own potential is
not a clean single-factor intervention. Here "no re-solve" is what makes the
comparison ill-posed rather than what keeps it controlled.

### The flipped-sign control measures only the offset

`rho(-phi_dft) = (B@phi_dft - C)/V`, and since `B@phi_dft ≈ -C` pointwise this
is approximately `-2C/V`. Measured: the control's sum&#124;.&#124; is 14.5805
against 2 x 7.2847 = 14.5693 for twice the offset, agreeing to 0.07%. So that
row's int&#124;.&#124; of 207.49 and cross of +240.10 are twice the offset term
and carry no information about the bound response at all.

No reading of the priority question is offered from this table, and nothing
here bears on the lateral 3-D error, which appears in none of these numbers.


### Correction: the mechanism was wrong (measured at 52ea73c)

Three errors of mine in the section above, all now measured rather than argued.
The conclusion — do not read the driven-versus-responds dichotomy off that table
— stands, but not for the reason I gave.

**1. The offset cannot amplify anything.** `nb_off` is identical in both arms, so
it cancels exactly out of the difference:
`rho(phi_dft) - rho(phi_model) = -(B @ dphi)/V`, with no `nb_off` in it.
Verified numerically: max abs difference between the two sides is 8.361e-16
e/A^3. A cancellation both arms share cannot amplify the difference between
them. The 103-fold figure is a real structural fact about the model — and to be
precise it is 51.7x against each term and 102.8x against their sum, which I
failed to distinguish — but it is not a mechanism for this comparison.

**2. I compared norms instead of taking the norm of the difference.** "B@phi_dft
is 1.03% smaller in aggregate" is `sum|a| - sum|b|`, not `sum|a-b|`. That is the
`int|a+b|` against `int|a|+int|b|` error, the same one this project has now been
caught by four times, and I made it two messages after flagging it in someone
else's work. Measured: `sum|B@dphi|/V = 0.5206`, which is **7.1%** of
`B@phi_model`, not 1.03%. The result moves by **1.00x** that perturbation — the
amplification is exactly none, as it must be, since the map from `dphi` to the
change in `rho` is precisely `-B/V`.

**3. My spectral reading was backwards.** I asserted the top of the band is
"exactly the band a double-derivative operator amplifies". B carries two Gaussian
smoothings as well as two derivatives, and the Gaussians win. Measured gain of B
on unit cosines:

| mode | k (1/A) | gain |
|---|---|---|
| 1 | 0.140 | 6.85e-03 |
| 5 | 0.698 | 9.89e-02 |
| 15 | 2.094 | 8.20e-01 |
| 30 | 4.189 | 2.66e+00 |
| 60 | 8.378 | **4.67e+00** |
| 100 | 13.963 | 1.85e+00 |
| 150 | 20.944 | 9.25e-02 |
| 200 | 27.925 | 8.00e-04 |
| 250 | 34.907 | 1.33e-06 |
| 300 | 41.888 | 2.37e-15 |

B is band-pass with a peak near mode 60 and falls by fifteen orders of magnitude
by mode 300. The 1.81x top-half difference between the two potentials therefore
cannot drive anything: the operator's gain there is 1e-15.

**What is actually true, and it does support the conclusion:** `B @ dphi` is
large next to a small true bound charge. `sum|B@dphi|/V` is 0.5206 against the
reference's `int|.|` of 2.0193 e, so a modest absolute perturbation is a large
relative error. That is a statement about the bound charge being small, not
about a tuned cancellation being broken.

### Where the perturbation lives, and the staged swap

| modes | rms dphi | sum&#124;B@d&#124;/V | share |
|---|---|---|---|
| 1-30 | 0.10272 | 2.86143 | 46.7% |
| 31-75 | 0.00792 | 2.57460 | 42.0% |
| 76-150 | 0.00149 | 0.68580 | 11.2% |
| 151-250 | 0.00048 | 0.00228 | 0.0% |
| 251-300 | 0.00001 | 0.00000 | 0.0% |

Two things to read carefully here. The band L1s sum to 6.12412 against the
actual `sum|B@dphi|/V` of 0.52056, an 11.8-fold cancellation BETWEEN bands, so
those shares are fractions of the uncancelled sum and must not be read as
contributions to the net effect — the same non-additivity as above. And the
grid-mismatch band 251-300, which is model-only content after the 500->600
upsample, contributes 0.00000: that concern is dismissed by measurement.

Staged swap, cutoff walked upward, same fixed response, no re-solve:

| cutoff | int&#124;.&#124; | gap | L1 | shift | resid |
|---|---|---|---|---|---|
| none (baseline) | 2.0302 | +0.5370 | **0.80003** | +0.050 | 48.7% |
| modes 1-5 | 28.5044 | -3.8967 | 27.62705 | -0.475 | 83.6% |
| modes 1-10 | 24.0294 | +3.8190 | 23.58153 | +0.475 | 92.6% |
| modes 1-20 | 36.7133 | +5.7979 | 36.73508 | +0.950 | 96.3% |
| modes 1-30 | 40.8309 | +5.6470 | 40.77592 | +0.900 | 96.6% |
| modes 1-50 | 24.9518 | +5.5748 | 24.83785 | +0.825 | 92.5% |
| modes 1-75 | 16.2669 | +5.5854 | 16.04677 | +0.800 | 88.0% |
| modes 1-100 | 9.3423 | +5.5832 | 8.58265 | +0.825 | 85.0% |
| modes 1-150 | 8.5995 | +5.5832 | 7.57510 | +0.825 | 85.0% |
| modes 1-250 | 8.5965 | +5.5832 | 7.56925 | +0.825 | 85.0% |
| full swap | 8.5965 | +5.5832 | 7.56925 | +0.825 | 85.0% |

**No cutoff beats the baseline at any length scale.** The best is modes 1-250 at
L1 7.56925 e against the baseline's 0.80003, still 9.5x worse. By the script's
own stated criterion that implicates the bound response itself — cavity, 1-D
closure, learned correction — rather than the generation of the total potential.

One feature worth flagging rather than smoothing over: the partial swaps are far
WORSE than the complete one, peaking at L1 40.78 for modes 1-30, and modes 1-30
is where essentially all of the potential difference lives (rms dphi 0.10272 of a
total 0.10304). So swapping the band that contains the entire difference is the
worst option of all. That is consistent with the 11.8-fold inter-band
cancellation measured above: the two potentials' band content is not
independently substitutable, and the staged swap therefore establishes "nothing
helps" robustly while its non-monotonic shape carries no separate reading.

All of this remains plane-averaged 1-D. No lateral 3-D error appears anywhere in
it.

### One thing that does not depend on the substitution

The baseline bound profile has a 48.7% shape residual at a displacement of only
+0.050 A, with L1 0.80003 e against `int|.|` 2.0193 e. Against the ionic
channel's 1.4% residual and L1 0.18687 e, the bound channel carries 4.3x the 1-D
charge error and 3.8x the coupling gap. The 1-D error is in the bound channel and
it is shape, not position.

## Dispatched job: bound_coeff_decompose (mace HEAD ead0a4a)

`bcd_ead0a4a.log`. mace ead0a4a, pb 9b3b9ba, executed file sha256
56ffb31ff1f70cc1660de16cddc39f567f8f32c57a72ddd5a8c74a914cd6af89, IDENTICAL to
the commit. Ran in about 12 s. Peak RSS 1.20 GB, host MemAvailable never below
242.8 GB, peak GPU 1377 MiB of 24564 — the 168x168x500 native grid cost almost
nothing. No kill.

### The reported gate FAILURE is spurious: a threshold below float64 resolution

The log says `[FAIL] A_ref*<E_z> + prior_ref = plane_mean(P_z): max abs
1.279e-17` and then `A GATE FAILED -- results below are printed but NOT to be
attributed`. That verdict is wrong. From the saved arrays, `max|a1*E|` is
1.928455e-01, and one float64 epsilon on that is **4.282e-17**. The measured
residual, 1.279e-17, is BELOW a single epsilon on the terms being summed, so the
identity is satisfied as exactly as double precision permits. Expecting ~1e-19
is unachievable at these magnitudes. The threshold needs to be relative, not
absolute; the identity itself holds and everything downstream is attributable.

Every other gate passes: the 1-D operator on `plane_mean(P_z)` equals the plane
average of the 3-D charge to 6.939e-18 against a profile max of 2.933e-03; the
1-D driving field equals the plane average of the 3-D one to 9.770e-14 against
3.054e+01; the two cells agree to 4.26e-16; `p_off = prior + delta_p` to exactly
0; and the script's own 1-D chain reproduces the model's bound charge to
4.992e-14.

A note for whoever re-derives this from the npz: recomputing the identity from
the saved 600-point `A_ref`, `E_dft` and `prior_ref` gives a residual of
8.077e-06, not 1.279e-17. That is a resampling artefact, not a contradiction —
`phi_dft_z` is saved at the native 500 and `E_dft` at 600, so the npz holds
quantities mapped to a common grid rather than the native-grid values the
identity was evaluated on. It cannot be verified to 1e-17 from the npz.

### STEP 1 passes strongly

3-D polarization from the DFT density and the DFT total potential, against DFT
RHOB plane-averaged: reconstructed net +0.000000 e and int|.| 2.0229 e against
the reference's +0.000000 and 2.0193, L1 0.00365 e = **0.18%** of the reference
int|.|, correlation **+0.999991**. The published 3-D response, the cavity and
the parameters reproduce the DFT bound charge. This also fixes the sign pair.

### STEP 3a, 3b, 3c

| quantity | ref rms | model rms | err rms | rel | corr |
|---|---|---|---|---|---|
| a1 (mean response) | 2.8973e-01 | 2.8167e-01 | 2.3947e-02 | 0.083 | +0.9935 |
| prior (covariance background) | 2.4731e-02 | 3.8408e-02 | 1.3765e-02 | 0.557 | +0.9987 |
| p_off = prior + delta_p | 2.4731e-02 | 3.7477e-02 | 1.2850e-02 | 0.520 | +0.9985 |

a1 is 8.3% off, and 77.7% of that error in rms is the pure cavity part
`resp_unit*(<s_diel>_model - <s_diel>_ref)`, the remainder the saturation
heuristic that evaluates the response at a screened vacuum field estimate rather
than the self-consistent one. The model's `prior` is 55.7% off and too LARGE,
not too small.

The learned correction points almost exactly the right way and is far too small:
needed rms 1.3765e-02, supplied `delta_p` rms 9.6013e-04 — about **1/14** of
what is needed — with projection +0.0668 onto the needed correction and
correlation **+0.9553**. It reduces the residual from 1.3765e-02 to 1.2850e-02,
a 6.6% reduction.

**STEP 3c closes the grid-reach question, negatively.** Share of the reference
coefficients' spectral energy above the model closure grid's cut: A_ref
**0.00%**, prior_ref **0.00%**. The model's 300-in-z grid can represent the
reference coefficients entirely, so grid reach is not the cavity/a1 error.

### STEP 3d: the 2x2's intended reading is unavailable, for a measurable reason

|  a1 | p_off | int&#124;.&#124; | gap | L1 | shift | resid |
|---|---|---|---|---|---|---|
| — | — | 2.0193 | +0.0000 | 0.00000 | -0.000 | 0.0% |
| model | model | 8.5965 | +5.5832 | 7.56925 | +0.825 | 85.0% |
| ref | model | 37.5740 | +39.0290 | 36.94743 | +1.250 | 93.4% |
| model | ref | 34.9159 | -33.4487 | 33.69750 | +1.275 | 95.9% |
| ref | ref | 2.0228 | -0.0029 | 0.00369 | -0.000 | 0.4% |

The closure check passes: ref/ref returns to the reference at L1 0.00369 e and
0.4% residual. But **neither single swap names a faulty piece — both make it
3.5 to 3.9 times worse**: 36.94743 and 33.69750 against the baseline's 7.56925,
while both together give 0.00369.

The reason is measurable on the reference side alone. From the saved arrays,
`a1*E` has rms 2.4447e-02 and `prior_ref` 2.4731e-02, and their sum
`plane_mean(P_z)` has rms 1.6706e-03 — a **14.6-fold cancellation**. Since
`a1*E + prior = plane_mean(a3*E)` holds by construction, the two coefficients
are complementary halves of one decomposition, and pairing a reference half with
a model half breaks the cancellation. That is a structural property of the
construction, not evidence about either coefficient. This is the third designed
single-factor intervention in this chain defeated by an identity that couples
the pieces, after the staged potential swap and the density tail swap.

### The cross-check has a sign error, and once corrected it confirms the relation

Reported: rms of the difference 2.9576e-03 against a `W_B P` rms of 1.4789e-03,
correlation **-0.999910**. The ratio is 2.9576/1.4789 = **1.99986**, i.e. exactly
2, which together with a correlation of -1 is the signature of two arrays that
are exact negatives of each other. So one side carries the wrong sign; corrected,
the two agree to about 0.01%. As printed it reads as a large disagreement when it
is in fact a confirmation of `n_b = -V * WB @ D @ P`.

Incidentally this run reproduces the earlier plateau finding by an independent
path: `sd_mean_ref` max is 0.9999851 against `sd_mean_model` max 0.9447081.

Robust statement unchanged: the bound charge has close to the right total and a
clearly wrong shape, and swapping the DFT total potential alone does not repair
it. The generation of the total potential is NOT excluded.

## Re-run at 0ac7b09: the amplitude/shape split

`bcd_0ac7b09.log`. mace 0ac7b09, pb 9b3b9ba, executed file sha256
eb4fa6453c596bb4ec75fa8a1ad999f87b4f6b9fbcabf2a1fa6a0d4d57707f84, IDENTICAL to
the commit. All gates now PASS, including the covariance identity at 1.279e-17,
reported as 0.30 float64 epsilon on the terms — the relative threshold fixes the
spurious FAIL. Everything from the previous run reproduces to the digit.

### The two coefficients fail in opposite ways

| quantity | best scale | err rms | after rescale | removed |
|---|---|---|---|---|
| a1 (mean response) | 0.9692 | 2.3947e-02 | 2.2216e-02 | **7.2%** |
| prior (covariance background) | 1.5510 | 1.3765e-02 | 1.9411e-03 | **85.9%** |
| p_off = prior + delta_p | 1.5132 | 1.2850e-02 | 2.0037e-03 | 84.4% |

This is the sharpest result of the run and it separates cleanly:

- **a1's error is SHAPE.** The best gain is 0.9692, within 3% of unity, and
  applying it removes only 7.2% of the error. No gain correction helps a1.
  Consistent with 77.7% of a1's error being the pure cavity part.
- **prior's error is almost pure GAIN.** The model's covariance background is
  about 1.55x the reference and correcting that single number removes 85.9% of
  the error. `p_off` inherits this at 84.4%.

The learned correction is a gain problem too, and less cleanly: rescaling
`delta_p` by 13.72x removes **71.0%** of what is needed, residual 3.9910e-03
against 1.3765e-02. So the +0.9553 correlation was not purely a localisation
artefact — but 71% is materially weaker than prior's own 85.9%, so the
correction's shape is right in the main and not in detail.

### The cavity plateau is set by resolution, not density

`plane_mean(s_diel)` max: model **0.9447081** on its 300-in-z grid, reference
**0.9999851** on the DFT native 500. Combined with the earlier density tail test
— which found the model grid gives the SAME plateau for both densities in both
directions, moving it by under 6e-09 — density is excluded and resolution is
what is left. This is the first direct evidence for it, and it depends on that
earlier test; on its own this line would not establish it.

Two limits on how far that goes. The two grids differ in lateral resolution
(100x100 against 168x168) as well as in z (300 against 500), and
`plane_mean` averages laterally, so this does not separate a z-resolution effect
from a lateral one. And "resolution" here means the grid the cavity is built on,
including the density field it is a nonlinear function of, not the resolution of
the plane average alone.

This does not conflict with STEP 3c's 0.00%, though the two are easy to read as
conflicting. STEP 3c says the reference coefficients contain no spectral content
above the model grid's cut, i.e. the answer is representable on 300 in z. The
plateau result says the value computed on that grid comes out different anyway,
because `s_diel` is a nonlinear function of a grid-resolved density. Representable
and correctly computed are not the same claim.

### The corrected cross-check, and its size

With the sign fixed: rms of the difference 1.9835e-05 against a `W_B P` rms of
1.4789e-03, correlation +0.999910. That confirms `n_b = -V * WB @ D @ P`.
One precision note, since the number was predicted as "about 0.01%": the
agreement is **1.34%** in rms (1.9835e-05 / 1.4789e-03), not 0.01%. The 0.01%
figure is the correlation deficit, 1 - 0.999910 = 9.0e-05, which is a different
quantity — for nearly parallel arrays the relative rms difference goes as
`sqrt(2 * (1 - corr))`, and `sqrt(2 * 9.0e-05)` = 1.34e-02, which is exactly
what was measured. Both say "agrees", but they differ by a factor of 134 and only
one of them is the relative error.

### The 2x2, with the reading the rows do support

Unchanged from the previous run to the digit. Neither single swap indicts its
coefficient, for the construction reason already recorded. What the rows do give
as magnitudes: the p_off error alone costs 36.94743 e in L1, the a1 error alone
33.69750 e, both together 7.56925 e, so the two errors cancel about 4.7-fold.
That cancellation is largely structural — the model's own closure satisfies
`a1*E + prior = plane_mean(a3*E)` with its own fields and its bound-charge total
is close to the reference, so the two errors are substantially forced to oppose —
and it is recorded as a magnitude, not as a finding about independent errors.
