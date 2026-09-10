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

## Re-run at 99ee87f: the covariance-gain-alone row, measured

`bcd_99ee87f.log`. mace 99ee87f, pb 9b3b9ba, executed file sha256
7b7327ce6974c18d1db2b11af03947939ee4776c017419cce975003085966517, IDENTICAL to
the commit. Every earlier number reproduces to the digit.

The new row replaces the bound with an evaluation:

|  a1 | p_off | int&#124;.&#124; | gap | L1 | shift | resid |
|---|---|---|---|---|---|---|
| model | model (baseline) | 8.5965 | +5.5832 | 7.56925 | +0.825 | 85.0% |
| model | prior / 1.551 | 35.7184 | **-35.7095** | **34.52551** | +1.325 | 96.1% |
| model | ref | 34.9159 | -33.4487 | 33.69750 | +1.275 | 95.9% |
| ref | ref | 2.0228 | -0.0029 | 0.00369 | -0.000 | 0.4% |

**L1 34.52551 e against the baseline's 7.56925 e, a factor of 4.56.** This lands
inside the 28.5-38.9 e window predicted from the triangle inequality, at about
58% of the way across it, and inside the predicted 3.8-5.1x range.

Worth recording that the bound was the right instrument and a point estimate
would not have been. Assuming the rescaling residual shared the direction of the
a1 error gave a point estimate of 28.9 e; the measured value is 34.53 e, so that
estimate would have been 16% low and would have sat exactly on the lower edge of
its own interval. The bound gave the sign immediately and cheaply and contained
the answer; the point estimate would have been published as a number and been
wrong.

Two details beyond the headline factor. The coupling gap does not merely grow, it
**flips sign and grows 6.4-fold**, from +5.5832 eV to -35.7095 eV, so this is an
overshoot past the reference rather than a further shortfall. And the gain-only
fix is slightly worse than substituting the reference `p_off` outright — L1
34.52551 against 33.69750 — because dividing by 1.551 removes 85.9% of the rms
error and the remaining 14.1% pushes a little further in L1 than the full
substitution does.

So: fixing the covariance gain alone makes the bound charge 4.56x worse in charge
L1 and flips the coupling gap. The diagnosis is not a repair list and not a
repair order. The two halves must move together, and they are not equally
tractable — prior's error is 85.9% removable by one scalar, a1's is 7.2%
removable by any scalar and 77.7% pure cavity, so a1 is the binding constraint
and needs the cavity or the grid it is computed on.

## Dispatched job: poff_exact_target (mace HEAD e48faec)

`poff_e48faec.log`. mace e48faec, pb 9b3b9ba, executed file sha256
76fa7ca7714acd95c4652aa9a86af0941368fb92f29a94c3f0e00056e4514b5b, IDENTICAL to
the commit. Gate passes: the baseline re-solve reproduces the model's own bound
charge to 1.301e-18 e/A^3. Both solves exit on tolerance.

### The requested c_absmax did NOT print, and the reason is a one-line bug

The number described as "free here and decisive for step 3" is absent from the
output, with no message, because `if c_am is not None:` silently skipped. The
cause: the script reads `out.get("c_absmax")` from `solve_graph`'s return dict,
but `pb1d_backend.py` spreads `delta_stats` — which is where `c_absmax` and
`dp_rms` live — into **`self.last_diagnostics`** at line 516, not into the
returned dict. So the key is never in `out`.

The fix is one line and the wrapper already holds what it needs: read
`self.last_diagnostics["c_absmax"]` after the call instead of `out.get(...)`.
`w_env_solve` and `u_solve` did capture, which is why the omission is easy to
miss — most of that key list worked.

### STEP 1: holding a1 fixed shrinks the target 3.7-fold, and reverses last round's shape verdict

| quantity | rms |
|---|---|
| P_off* (the exact target) | 3.5170e-02 |
| prior_model | 3.8408e-02 |
| p_off_model | 3.7477e-02 |
| **delta_p\* (needed)** | **3.6777e-03** |
| **delta_p (supplied)** | **9.6013e-04** |

Old target rms 1.3765e-02, new 3.6777e-03, ratio 0.267, mutual correlation
+0.9177. So scoring `delta_p` against `prior_ref - prior_model` last round did
charge it with a1's error, exactly as diagnosed. The head is short by a factor
of 3.83, **not the 1/14 reported last round**.

| delta_p vs | best scale | err rms | after rescale | removed | corr |
|---|---|---|---|---|---|
| delta_p* (correct) | 0.2256 | 2.8886e-03 | 4.8304e-04 | 83.3% | **+0.8580** |
| the old target | 0.0668 | 1.2850e-02 | 2.7837e-04 | 97.8% | +0.9553 |

and separately: rescaling `delta_p` by 3.31x leaves 1.8503e-03 of 3.6777e-03,
so **a pure gain change removes 49.7%**.

Those two percentages answer different questions and the actionable one is the
smaller. The 83.3% row scales the TARGET down by 0.2256 to fit `delta_p`, and its
denominator is the raw difference 2.8886e-03; the 49.7% scales `delta_p` UP by
3.31x to fit the target, with the denominator being what is needed. Both follow
from corr = +0.8580 — the residual fraction after any optimal gain is
`sqrt(1 - corr^2)` = 0.514 — so no gain choice can recover more than about half.
Against the correct target the shape agreement is WORSE than it looked last
round (+0.8580 against +0.9553) and gain-alone recovery falls from 71.0% to
49.7%. Last round's "right shape in the main" is weakened by fixing the target.

### STEP 2a wiring check

Bound-charge L1 with P_off* at the fixed DFT field: 0.00365 e against 7.56925 e
for the model's own p_off, on an int|.| of 2.0193 e. That 0.00365 is the same
number as the 3-D reconstruction's own 0.18% floor, so the algebra is right and
this proves nothing physical, as labelled.

### STEP 2b: the aggregate is MIXED, not a pass

| case | net (e) | chg L1 | bound L1 | ion L1 | cross | self | total | d total |
|---|---|---|---|---|---|---|---|---|
| DFT reference | +1.0000 | 0.00000 | 0.00000 | 0.00000 | -3.0243 | +1.5237 | -1.5005 | +0.0000 |
| baseline, model p_off | +1.0000 | 0.63739 | 0.80003 | 0.18687 | -2.6289 | +1.4718 | -1.1572 | +0.3434 |
| exact P_off*, re-solved | +1.0000 | 0.47303 | 0.63902 | 0.19199 | -3.0020 | +1.4869 | -1.5151 | **-0.0145** |

Better: charge sum L1 -25.8%, bound L1 -20.1%, cross gap +0.3953 to +0.0223 eV,
self gap -0.0520 to -0.0368 eV, total gap +0.3434 to -0.0145 eV.
Worse: ion L1 +2.7%, **potential rms 0.11093 to 0.22339 eV (+101.4%)**, and the
**mutual bound-ion term +0.0288 to +0.1261 eV from the reference (4.4x)**.

Two of the quantities the stop rule names as part of the aggregate got worse, one
of them by a factor of two, so this is not a clean improvement and should not be
reported as one. The specific tension needing a decision: the total energy gap
crosses zero and closes by 96% while the potential profile error doubles. Energy
agreement improving as field agreement degrades is the signature of compensating
errors, which is the thing this line of work exists to remove — so "the total gap
closed" cannot be taken as the verdict on its own. I am not resolving it in
either direction.

## Re-run at d602de8: the band table decides it, and against step 2

`poff_d602de8.log`. mace d602de8, pb 9b3b9ba, executed file sha256
a32e957172821355bb1963e930309729af5f31ccbdabdb9c21778d8c7c489c62, IDENTICAL to
the commit. Gate passes at 1.301e-18; both solves exit on tolerance. All STEP 1
and STEP 2b numbers reproduce to the digit.

### c_absmax, now printed: the amplitude is a learning outcome, not a limit

**c_absmax 0.007093 — 2.8% of the hard bound c_max = 0.25.** dp_rms 9.6013e-04.
The head needs 3.83x more amplitude; scaling every c_k by that factor reaches
10.9% of the bound, comfortably inside. So the missing amplitude is not a
representational limit. That bounds the amplitude question only — the
gain-alone recovery of 49.7% says the shape is a separate and unanswered matter.

### The band table fits NEITHER prepared branch

|  modes | wavelength | charge err rms, baseline | P_off* | | potential err rms, baseline | P_off* | |
|---|---|---|---|---|---|---|---|
| 1-3 | >= 15.0 A | 2.176e-05 | 6.137e-06 | better 3.5x | 6.789e-02 | 1.611e-01 | **WORSE 2.4x** |
| 4-10 | >= 4.5 A | 9.869e-05 | 6.330e-05 | better | 7.325e-02 | 8.699e-02 | WORSE |
| 11-30 | >= 1.5 A | 1.913e-04 | 1.957e-04 | worse | 2.404e-02 | 3.262e-02 | WORSE |
| 31-100 | >= 0.5 A | 1.444e-04 | 7.245e-05 | better | 8.053e-03 | 1.294e-02 | WORSE |
| 101-300 | >= 0.1 A | 1.665e-05 | 9.885e-06 | better | 6.197e-04 | 7.410e-04 | WORSE |

The two branches offered were "charge better at high k while the potential
worsens at low k" and "both worsening at low k together". Neither holds. The
charge error improves in **four of five bands including the lowest**, where it
improves by 3.5x, and the potential error worsens in **all five**. So this is not
the 1/k^2 weighting trade the branches were built around: the potential goes the
wrong way even in the bands where the charge goes the right way.

### The three potential components

| | baseline | P_off* | change |
|---|---|---|---|
| rms | 0.11093 | 0.22339 eV | +101.4% |
| mean | -0.04109 | -0.12304 eV | 3.0x |
| mean-removed rms | 0.10304 | 0.18645 eV | +81.0% |
| **longest-wavelength (45 A) amplitude** | **0.08041** | **0.20436 eV** | **+154.2%** |

The constant offset cannot explain it — the mean-removed rms still rises 81%.
And the largest relative degradation is the 45 A amplitude at +154.2%, which for
a constant-potential model IS the cross-cell drop, the quantity the model exists
to predict. By the criterion stated in advance — "if that is what doubled, the
verdict is settled against step 2 regardless of the energy" — it did more than
double, so the verdict is settled against step 2.

### Verdict

Step 2 fails on its own terms. Charge sum L1 improves 25.8%, bound L1 20.1%, and
the total energy gap closes 96% from +0.3434 to -0.0145 eV; against that, the
potential rms doubles, its mean-removed part rises 81%, the cross-cell drop
component rises 154%, the bound-ion mutual term moves 4.4x further from the
reference, and the ion L1 worsens 2.7%. Per the user's tree, the next place to
look is the other inputs and the self-consistent coupling, not training, and
step 3 does not run.

One mechanism is consistent with the whole pattern and is NOT established here,
offered only because the prepared branches do not fit and it is cheaply testable.
The potential error is `l0_inv` applied to the TOTAL charge error, which includes
a solute-side contribution that this intervention does not touch. If the model's
solvent charge error had been partially cancelling that fixed solute-side error
in the potential, then reducing the solvent charge error would remove the
cancellation and worsen the potential — in every band at once, which is what is
observed. That is a hypothesis about a compensation between the solvent and
solute channels rather than between charge and potential at different k. Testing
it needs `l0_inv` applied to each charge error separately and compared against
the potential error directly, which this run does not do.

## Re-run at 06ba920: my compensation hypothesis is REFUTED by its own gate

`poff_06ba920.log`. mace 06ba920, pb 9b3b9ba, executed file sha256
c0d5f1657da11078d9700ce148e3ce635f44fb61c23e3367018c8b206d68dc45, IDENTICAL to
the commit. Every earlier number reproduces to the digit.

The decomposition was built to test the hypothesis recorded above — that the
solvent charge error had been cancelling a fixed solute-side error in the
potential. **The gate that licenses reading it FAILED, and the hypothesis does
not survive.**

   [FAIL] the untouched bracket is identical for both cases at sign +1:
          max abs difference 1.972e-01 eV, 5.07e-01 of its own max
   sign -1 is worse: 3.162e-01 eV

The bracket was required to be identical between the two cases because the
solute side is untouched. It differs by half its own magnitude, and neither
`l0_inv` sign fixes that, so the premise is false: the bracket is not an
untouched solute-side quantity. `phi_sol` is part of the self-consistent
solution, so it moves when `p_off` changes.

### The negative correlations carry no information

| quantity | rms (eV) | |
|---|---|---|
| fixed bracket (solute side) | 0.15884 | |
| baseline: solvent-charge term | 0.07288 | |
| baseline: TOTAL potential error | 0.11093 | corr(solvent, bracket) -0.8154 |
| P_off*: solvent-charge term | 0.01939 | |
| P_off*: TOTAL potential error | 0.22339 | corr(solvent, bracket) -0.7673 |

The correlation is algebraically determined by the three rms values, because the
parts must sum to the total: from 0.15884, 0.07288 and 0.11093 alone the implied
correlation is -0.788, against the reported -0.815. Whenever the total is smaller
than one of its parts the correlation MUST be negative. So the negative sign is
forced arithmetic and is not evidence for the hypothesis it was built to test.
This is the same trap as reporting a correlation as a relative error: a quantity
already fixed by numbers on the same page being presented as an independent
measurement.

### An arithmetic proof that the split is inconsistent as printed

Independently of the gate: with a bracket rms of 0.15884 and the P_off* solvent
term at 0.01939, no correlation in [-1, 1] can produce that case's total of
0.22339 — the required value is **+3.94**. So the single printed bracket rms
cannot apply to both cases. By the triangle inequality the P_off* bracket must
lie in [0.204, 0.243] eV, i.e. **28% to 53% above the baseline's 0.15884**.

That is the substantive result of this run. The part of the potential error that
was supposed to be untouched grows by roughly a third to a half when `p_off` is
replaced. So P_off* does not merely fail to fix the potential — it makes the
non-solvent-charge part of the potential error worse as well, which is why the
degradation appears in every band.

### Consequences

- My hypothesis is refuted in its own terms. There is no fixed solute-side error
  being uncovered, so there is no 0.15884 eV floor, and the claim that the
  baseline sits "below its own floor only by cancellation" is void.
- The verdict on step 2 is unchanged and if anything firmer: the intervention
  degrades the potential both through the solvent term and through the part that
  was assumed fixed.
- Where it routes: at the self-consistent coupling, since the mechanism by which
  a change in `p_off` moves `phi_sol` is exactly that coupling. That is the same
  destination the user's tree already named, reached now by measurement rather
  than by elimination.

One wording bug to fix: the failing line's own explanatory clause reads
"-- so the split is exact and the sign is measured, not assumed", which asserts
the opposite of the FAIL it is attached to. Same class as a gate whose number
contradicts its label, inverted — here the label is right and the prose is wrong.

## Re-run at a943679: the identity closes, and it refutes the expected reading

`poff_a943679.log`. mace a943679, pb 9b3b9ba, executed file sha256
50f9e7aa342fb34f3ca7042870d6bdbc798d69385db2c38b80c36f251f15d136, IDENTICAL to
the commit.

**The identity closes at machine precision.** The measured change in the total
potential equals the dipole-feedback change plus the direct charge term, up to
the dropped G=0 constant: max abs residual **1.898e-12 eV**, 8.13e-12 of the
change itself. So the dipole-feedback path identified in the source is a complete
account of the change, and `phi_sol` moving is confirmed as the mechanism.

| term | rms (eV) | 45 A amp |
|---|---|---|
| total change in the potential | 0.13373 | 0.13003 |
| from dipole feedback (cvdip ramp) | 0.06679 | 0.07438 |
| from the charge directly (l0_inv) | 0.05925 | 0.07100 |

The two contributions are comparable in size and they reinforce rather than
oppose — the total exceeds either.

### But the dipole moves TOWARD the reference, not away

| case | dipole (e A) | error |
|---|---|---|
| DFT reference | 16.70259 | +0.00000 |
| baseline | 16.40810 | -0.29449 |
| exact P_off* | 16.65316 | **-0.04943** |

P_off* takes the dipole error from -0.29449 to -0.04943 — **5.96x closer** to the
reference. The reading written into the script was "if P_off* moves the dipole
FURTHER from the reference while improving the profile's L1, then the potential
degradation is the ramp being driven by a quantity the intervention makes worse".
That is refuted: the intervention makes the dipole substantially better.

### What the three facts together do support

The profile improves (charge sum L1 -25.8%), the dipole improves 5.96-fold, and
the potential error still doubles — with the change decomposing exactly into the
ramp and the direct charge term. So the degradation is not attributable to either
input being made worse. What it indicts is the construction of the potential
itself: at a charge profile and a dipole both closer to the reference,
`cvhar_z + cvdip` moves further from the DFT total potential. The tapered-ramp
form and `cvhar_z` are where that has to be examined, which is the "other inputs
and the self-consistent coupling" branch of the user's tree, now reached with the
mechanism named rather than by elimination.

Stated as an inference from three measured quantities, not as a tested mechanism.
What it does NOT say is why — no compensation story is offered here, after one
already failed its own gate.

### A caution on combining the three rms values

They cannot be combined. Taking `total = dipole + charge` at face value and
solving for the correlation gives **+1.2524**, which is impossible for real
arrays; and the triangle inequality caps the rms of the two-term sum at
0.06679 + 0.05925 = 0.12604 against a printed total of 0.13373, a 6.1% excess.
The gap is the dropped G=0 constant, which the residual of 1.898e-12 accounts for
but which the printed component rms values omit. So the constant is material, not
a rounding detail, and the three numbers are not a decomposition anyone should
add up.

### Step 2

Verdict unchanged. Charge and energy improve, the potential doubles, the mutual
term moves 4.4x further out, ion L1 rises. Step 3 does not run.

## Final run at 04c384b: the constant is NOT the dipole path, so there are TWO items

`poff_04c384b.log`. mace 04c384b, pb 9b3b9ba, executed file sha256
3da5565b2e93fb50e658434b4c137e23e6d13699c744d5134655e204cea9b850, IDENTICAL to
the commit. Identity still closes: max abs residual 3.545e-12 eV.

| term | rms (eV) | mean | 45 A amp |
|---|---|---|---|
| total change in the potential | 0.13373 | -0.08195 | 0.13003 |
| of it, the G=0 constant | **0.08195** | -0.08195 | 0.00000 |
| of it, the rest (mean removed) | 0.10569 | +0.00000 | 0.13003 |
| from dipole feedback (cvdip ramp) | 0.06679 | -0.00000 | 0.07438 |
| from the charge directly (l0_inv) | 0.05925 | +0.00000 | 0.07100 |

The decomposition is now internally consistent — the last three rows combine at
an implied correlation of +0.4041, and 0.06679, 0.05925 and +0.4041 reproduce
0.10569 exactly. The impossibility flagged in the previous run is gone.

### The answer to the dispatched question: 0.0%, so two items and not one

**Mean of the feedback change -0.00000 eV against the total constant
-0.08195 eV, i.e. 0.0% of it.** The criterion set in advance was "near 100% means
one open item, not two". It is zero, to within the 5e-6 eV that the printed
precision allows, so:

- The tapered `ii*cutoff` ramp does NOT produce a mean. The taper breaking the
  ramp's antisymmetry was the reason the constant might have belonged to the same
  path, and it does not — measured, not argued.
- The G=0 constant is a **separate** term, and it is the **largest single named
  part of the degradation**: 0.08195 eV rms against 0.06679 for the dipole
  feedback and 0.05925 for the direct charge term, i.e. 37.6% of the change in
  variance.
- So the user gets **two** things to look at, not one: the construction of the
  potential (the cdipol taper form and `cvhar_z`), and separately the G=0
  bookkeeping.

Cross-check that the constant is real rather than an artefact of this split: the
potential error's mean went -0.04109 to -0.12304 eV between the two cases, a
difference of -0.08195, which is exactly the constant reported here. And the
mean-removed rms still worsened 81.0%, so neither part accounts for the
degradation alone.

### Frame closed

Step 2 fails: charge sum L1 -25.8%, bound L1 -20.1% and the energy gap closing
96% from +0.3434 to -0.0145 eV do not carry it against a potential rms that
doubles, a 45 A component up 154.2%, a bound-ion mutual term 4.4x further from
the reference and ion L1 up 2.7%. Step 3 does not run — and separately would not
have been blocked by clipping, since c_absmax is 0.007093, 2.8% of the 0.25
bound, reaching 10.9% after the 3.83x amplitude shortfall.

Both candidate inputs are exonerated by measurement: the charge profile improves
and the dipole improves 5.96-fold (-0.29449 to -0.04943 e A), and the potential
still degrades. No mechanism is offered for why. In particular the story that the
baseline's wrong dipole was compensating an error in the ramp form fits but is
not stated as a finding — it is the same shape as the solute-side compensation
story that failed its own gate in this same chain, and it needs its own gate
first.

## Dispatched control: potential_rebuild (mace HEAD d40b5ec) — the SOLUTE INPUT is the priority

`potreb_d40b5ec.log`. mace d40b5ec, pb 9b3b9ba, executed file sha256
d2fede6351cd99985b2e4d4b31fc29ae8d835e99b6a4ef32947c5cb8ca700b6f, IDENTICAL to
the commit.

### Corrections to conclusions recorded ABOVE in this file

The user rejected three of the readings recorded in the previous sections. All
three are corrected here and the earlier text should be read subject to this:

1. **"Both candidate inputs are exonerated by measurement" is WRONG.** A better
   charge L1 and a better solvent dipole do not imply that every spatial
   component determining the potential improved, and a band rms does not
   establish it. Worse, the dipole measured there was the first moment of the
   SOLVENT charge, while the feedback uses
   `dip_z = val_ion_dipole_z + dsol_z - q*center_z`, which contains the model's
   SOLUTE dipole — never checked until this run. And holding `cvhar_z` fixed
   across two cases says nothing about whether `cvhar_z` is correct.
2. **The "two open items" framing is WRONG and is withdrawn.** The solver fixes
   the potential's constant by ionic electroneutrality — the residual's G=0 row
   is `mean(n_b + n_ion) + q_sol = 0` and `n_ion` depends on phi's mean through
   `n_work` — so the constant MUST move when the shape moves. A G=0 change
   therefore implies no separate bookkeeping error. The 0.0% measurement stands;
   the conclusion drawn from it does not.
3. **"Energy closing while the field degrades is the compensating-error
   signature" was too broad.** The full self-energy IMPROVED, 0.0520 to 0.0368 eV,
   and the cross energy improved 0.3953 to 0.0223; only the mutual term inside
   the self-energy worsened. The sufficient reason to pause was the potential
   alone. The pause stands, the reason given for it was wider than the evidence.

Scope limit to carry with the earlier table: its `total` column is the 1-D
solvent electrostatic energy at a FIXED bare-solute potential, containing neither
the dipole correction nor the G=0 term, so "96% closed" is not a statement about
the full DFT energy.

### The cvhar identity checks out, and its precondition holds

`cvhar_DFT = cvhar_model + l0_inv(n_e_DFT - n_e_model)` is correct. Verified in
source: `cvhar3 = phi_base - l0_inv(net_g)` with `net = neutral_v - n_e` at
pb1d_backend.py:388-390, and `neutral_v`/`phi_base` come from the baseline
tables (`self._rt_tables.fields(...)`, or `fields[0]`/`fields[1]`), so they are
independent of the electron density being substituted and cancel in the
difference. PHI genuinely never enters. The precondition is that both arms use
the same baseline row, which they do — `baseline_index.json` maps this geometry
to one row.

### GATE PASSES

Shape: max abs deviation after removing the constant **7.507e-12 eV**; the
constant itself +1.020240 eV. So cvdip, l0_inv, indmin, c_unit, center_z, the
dipole mixing and every sign are validated at once.

### RESULT: one solvent charge, two solute inputs

| solute input | L1 | max | rms | 45 A amp | dipole |
|---|---|---|---|---|---|
| DFT solute potential + dipole | 0.0167 | 0.0171 | **0.00157** | 0.00001 | -0.3653 |
| model solute potential + dipole | 8.4556 | 0.4196 | **0.20944** | 0.23008 | -0.9439 |

The all-DFT rebuild error is 0.00157 eV rms, **0.05%** of the potential's own rms
of 3.35043 eV. Swapping in the model's solute input takes it to 0.20944 eV, a
**factor of 133**.

By the branches set in advance — "all-DFT rebuilds and the model's input does not
-> the solute input is the priority" — **the solute input is the priority.** The
assembly is not at fault: it reproduces the DFT potential to 0.05% when given DFT
solute inputs, and 0.00% of cvhar_DFT's spectral energy sits above the model
grid's mode-150 cut, so grid reach is not the constraint either.

The two solute inputs differ as follows:

- electron profile: model int 660.99991 against DFT 660.99999; difference rms 6.98500
- solute potential: cvhar_model rms 3.43934, cvhar_DFT 3.48212, difference rms **0.12710 eV**
- **solute dipole: model -6.74120 against DFT -6.16258 (300 grid) and -6.16260 (native 500), so the model's error is +0.57862 e A**

That dipole error is the term never checked before, and it is about 12x the
solvent dipole error that the previous round was optimising (-0.04943 e A after
P_off*). The `dipole` column difference in the result table, -0.9439 against
-0.3653, is 0.5786 — exactly the solute dipole error, so the two are consistent.

### The constant lands correctly for the all-DFT row

| row | offset | mean after offset | DFT potential's mean |
|---|---|---|---|
| DFT solute potential + dipole | -1.062176 | -1.06218 | -1.06133 |
| model solute potential + dipole | -0.923560 | -0.92356 | -1.06133 |
| DFT total potential (stored) | -0.000849 | -1.06218 | -1.06133 |

The stored potential's own offset is -0.000849 eV, i.e. the calibration is
essentially zero as it should be. The all-DFT rebuild's constant agrees with the
DFT potential's mean to 0.85 mV; the model-solute rebuild's is off by 0.138 eV.

### What this does NOT separate

The model's solute input was swapped as a unit — `cvhar` shape and the dipole
together — so which of the two dominates is not established. Both are non-trivial
on their own: 0.12710 eV rms in `cvhar` and +0.57862 e A in the dipole. Unlike
`a1`/`p_off`, these are two separate inputs and not two halves of one identity,
so a one-at-a-time swap IS a controlled intervention here and would separate
them. Not run, not requested.

## Dispatched: lateral_bound_swap, KIT_FRAMES=n44 (mace HEAD c8560dd)

`latswap_c8560dd.log`. mace c8560dd, pb 9b3b9ba, executed file sha256
e861ef4730e84cb1a19e90e4f936578b38747824d9a6b1de6b0dc71139148659, IDENTICAL to
the commit. Peak RSS 2.33 GB, MemAvailable never below 242.2 GB, peak GPU
4811 MiB — no kill, despite four 250 MB ASCII field reads.

**SUBSET RUN: 4 of the 6 frames the user named.** The three NiN44 charged frames
(q = -0.80, -1.00, -1.32) and the neutral one. Absent: the two NiN88 frames, so
this table has **no cell-size variation**. LS6 runs all six as job 3427615; this
is the independent cross-machine reproduction of the four.

### All three gates PASS

- the model's own 3-D term reproduced through the Coulomb function: worst
  `|(cross+self) - solvent3d_energy_g|` **3.331e-16 eV**, so dE really is a
  one-term change
- the substituted lateral charge has zero plane means: worst
  `max |plane mean|` 8.743e-19 e/A^3
- the 1-D/3-D cross term stays zero after the swap: worst
  `|int rho_1d*phi[delta_new]|` 9.711e-13 eV, so `comp` and `E_bl` are genuinely
  unaffected

### TRUNCATION DOES NOT MATTER HERE — the numbers are values, not lower bounds

Power above the model grid's lateral Nyquist is **0.0%** on all four frames, and
the cross energy computed on the model grid equals the one on the native grid to
a ratio of **1.000** in every case (-2.3991, -2.1808, -2.3886, -0.4309). The
condition set in advance was "if the two cross energies agree, truncation does
not matter for the energy". They agree. So the improvements below are the actual
values and **not** lower bounds. Stating that here rather than after the table,
as instructed.

### The result: final total energy error, meV/atom

| frame | atoms | original | substituted | improvement | fraction recovered |
|---|---|---|---|---|---|
| NiN44 q=-0.80 | 207 | +12.987 | +8.991 | -3.996 | 30.8% |
| NiN44 q=-1.00 | 207 | +20.843 | +15.209 | -5.634 | 27.0% |
| NiN44 q=-1.32 | 207 | +27.309 | +20.817 | -6.492 | 23.8% |
| neutral | 207 | -6.379 | -4.499 | +1.881 | 29.5% |
| charged mean \|err\| | | 20.380 | 15.006 | -5.374 | **26.4%** |

By the user's branches this is the **partial** case, so the instruction is to
record exactly how much is recovered and how much is left: **26.4% of the charged
error is recovered, 73.6% remains — 15.006 meV/atom of an original
20.380 meV/atom.** A perfect lateral bound charge, injected with the 1-D channel,
the ionic charge, the net charge, the z-profile and the z-dipole all held fixed,
removes about a quarter of the total-energy error.

The neutral frame, where the ionic channel is identically zero and so isolates
the bound lateral effect, recovers 29.5% — consistent with the charged frames
rather than different from them.

The model's lateral charge does point the right way on every frame:
correlation with the DFT lateral charge +0.8733, +0.8632, +0.8337, +0.8396. That
is also the sign check, and it passes.

### Two trends the table shows that the summary rows do not

**The fraction recovered falls as the cell gets more charged:** 30.8%, 27.0%,
23.8% for q = -0.80, -1.00, -1.32. So the more charged the frame, the less of its
error a perfect lateral bound charge explains.

**The residual keeps the charge dependence.** After substitution the error is
still monotone in |q| — +8.991, +15.209, +20.817 meV/atom — and it grows slightly
faster with charge than the original did (factor 2.32 across the sweep against
2.10). So fixing the lateral bound charge perfectly does not remove the
charge-driven part of the total-energy error; whatever carries the remaining
three quarters is also charge-driven.

### Supporting energies (eV)

| frame | E_3d old | E_3d new | cross old | cross new | self old | self new | mutual old | mutual new |
|---|---|---|---|---|---|---|---|---|
| NiN44 q=-0.80 | -0.5297 | -1.3569 | -1.4686 | -2.3954 | +0.9388 | +1.0385 | -0.0088 | -0.0111 |
| NiN44 q=-1.00 | -0.3232 | -1.4894 | -0.9350 | -2.1792 | +0.6118 | +0.6898 | -0.0016 | -0.0129 |
| NiN44 q=-1.32 | -0.3448 | -1.6886 | -0.9654 | -2.3848 | +0.6206 | +0.6962 | -0.0077 | -0.0231 |
| neutral | -0.1582 | +0.2311 | -0.7644 | -0.4309 | +0.6061 | +0.6619 | +0.0000 | +0.0000 |

### Framing, as the user set it

This does NOT show the network can learn the reference charge. It settles whether
the line is worth the work, and the answer it gives is a quarter of the error on
the charged frames with the charge dependence of the remainder intact.

### CORRECTION to the section above: 26.4% was a subset artefact

The LS6 six-frame run (job 3427615, code 67504e7) reverses the reading. The two
NiN88 frames my machine could not run get WORSE under the swap, and they sit on
the other side of zero:

| frame | atoms | original | substituted | \|err\| old | \|err\| new | verdict | factor |
|---|---|---|---|---|---|---|---|
| NiN44 q=-0.80 | 207 | +12.987 | +8.991 | 12.987 | 8.991 | BETTER | 0.692 |
| NiN44 q=-1.00 | 207 | +20.843 | +15.209 | 20.843 | 15.209 | BETTER | 0.730 |
| NiN44 q=-1.32 | 207 | +27.309 | +20.817 | 27.309 | 20.817 | BETTER | 0.762 |
| NiN88 q=-1.00 | 339 | -5.413 | -11.515 | 5.413 | 11.515 | **WORSE** | 2.127 |
| NiN88 q=-1.31 | 339 | -2.778 | -6.514 | 2.778 | 6.514 | **WORSE** | 2.345 |
| neutral | 207 | -6.379 | -4.499 | 6.379 | 4.499 | BETTER | 0.705 |
| **all charged** | | | | **13.866** | **12.609** | BETTER | 0.909 |
| NiN44 only | | | | 20.380 | 15.006 | BETTER | 0.736 |
| NiN88 only | | | | 4.096 | 9.014 | WORSE | 2.201 |

Verified independently: 13.866 -> 12.609 is a factor 0.9094, so the aggregate
over all five charged frames recovers **9.1%, not 26.4%**. My four frames were
the three with the sign that makes the swap help, plus the neutral.

**The correct summary is per-cell, and no mean over the five is the finding.**
9.1% is the net of a 26.4% improvement on three frames and a 120% degradation on
two, and reading it as a modest uniform gain would be wrong in both directions.

### A reporting fault in the code I ran, and my share of it

The column labelled "improvement" in the log above is `err1 - err0`, a SIGNED
change, which coincides with an improvement only when the error is positive.
Fixed upstream at df8f25a with |err| before and after, the change in |err|, an
explicit verdict and a per-cell split.

My share: I computed the neutral frame's 29.5% against |err| rather than against
the signed change, which is why that figure was right — but I did not flag that
the column's convention was wrong in general. And the claim that my subset could
not have caught it is too kind: the neutral frame DID have a negative error, and
it was one sign flip away from exposing the fault. Its signed change happened to
be positive (-6.379 -> -4.499), so the label and the truth agreed by luck. All
four of my rows agreed with the label; none of them tested it.

Sharper still, and it makes my half larger than "a missed flag": the
contradiction was visible in my own pushed table. The three NiN44 rows print
gains as NEGATIVE numbers (-3.996, -5.634, -6.492) and the neutral row prints its
gain as POSITIVE (+1.881), in the same column labelled "improvement". A column in
which both signs mean "better" is self-contradicting on its face, and it was on
the page I wrote. Computing the neutral fraction against |err| is what sidestepped
the bug, and sidestepping it is what let me not look at the column beside it.

### The mechanism, and what it does and does not rest on

dE per atom is negative on every charged frame: -3.996, -5.634, -6.492, -6.102,
-3.736 meV/atom. NiN44 over-predicts, so a downward shift helps; NiN88
under-predicts, so the same-signed shift hurts. **A shift of uniform SIGN cannot
repair a bias whose sign differs between cells**, so the lateral bound charge
error is not what makes the model's total-energy error vary from cell to cell.

That conclusion rests on the sign being uniform, and only on that. It does not
need the magnitudes to be similar, and they are not: the spread is 1.74x
(3.736 to 6.492), and the |q| dependence has OPPOSITE sign in the two cells —
rising within NiN44 (-3.996, -5.634, -6.492) and falling within NiN88 (-6.102,
-3.736). So "a similar downward shift regardless of cell or charge" overstates
it; "a downward shift on every charged frame" is what is measured and is all the
argument needs.

Also worth keeping: the model was originally about 5x BETTER on NiN88
(mean |err| 4.096) than on NiN44 (20.380), and the substitution destroys that.

### What is not weakened

The swap is still a genuine like-for-like intervention, all three gates still
pass on all six frames, truncation is still 0.0% with cross ratio 1.000
everywhere, the model's lateral charge still points the right way on every frame
including NiN88 (+0.8045, +0.8242), and the lateral channel really is 71.3% of
the bound COUPLING gap. What the six frames show is that closing that coupling
gap does not close the TOTAL ENERGY error, because the error's cell-to-cell
variation lives elsewhere.

My falling-fraction trend (30.8%, 27.0%, 23.8%) survives as a WITHIN-NiN44
statement only. The two NiN88 points do not extend it — they are on the other
side of zero.

## Dispatched: charging_reconcile, steps 1-4 (mace HEAD fda2b25)

`charge_fda2b25.log`. mace fda2b25, pb 9b3b9ba, executed file sha256
9d5e94f1504ed8174503656d442239d4722b50439521c866ff34e90787057774, IDENTICAL to
the commit. The guard worked: the census ran first, reported 0 of 20 complete val
pairs with the neutral side missing for all twenty, and SKIPPED step 5 rather
than crashing. Steps 1-4 are self-contained and complete.

### STEP 4 is decisive, and the prediction held

  readout block 0: features (207, 576), max |diff| 5.551e-17, relative 1.854e-16
  readout block 1: features (207, 576), max |diff| 6.939e-17, relative 3.883e-16
  d(inter_e) across the pair: +0.000000 eV

The readout inputs are IDENTICAL at machine precision, so the energy head
contributes the same amount to both states and **cancels exactly in Delta E**.
Retraining that head cannot move the paired charging energy at all, whatever the
target — the A/B refit is invalid as a route to this error.

This is the predicted outcome and it answers the question the prediction
sharpened: charge does NOT enter the descriptor. The two frames share geometry
and species (max |dpos| 0.000e+00, max |dcell| 0.000e+00, species identical), so
a positions-and-species descriptor gives identical features by construction, and
that is what is measured.

### STEP 1: the pair, and what the label is

sid 1: label -1310.37680269 = OUTCAR sigma->0, NELECT 661.0000, E-fermi -3.8745 eV, q -1.00
sid 601: label -1304.27622534 = OUTCAR sigma->0, NELECT 660.0000, E-fermi -7.3736 eV, q +0.00
NELECT differs by exactly 1, so this is an electron-ADDITION energy, and the
label carries no reservoir term: canonical, not grand canonical.

| | charged | neutral | difference |
|---|---|---|---|
| model | -1306.062262 | -1305.596780 | **-0.465482** |
| DFT | -1310.376803 | -1304.276225 | **-6.100577** |

eps per frame +4.314540 and -1.320555 eV; eps_Delta **+5.635095 eV**, agreeing
with the +5.6350 implied by the six-frame table to 0.0001 eV.

**The model captures 7.6% of the DFT charging energy for this pair** — -0.465
against -6.101 eV, a factor of 13.

### STEP 2: the ten terms, closure PASS at 4.320e-12 eV

| term | neutral | charged | difference |
|---|---|---|---|
| e0 (atomic, per-species) | -1307.266162 | -1307.266162 | +0.000000 |
| energy head (inter_e) | +22.036824 | +22.036824 | +0.000000 |
| electron energy (optional) | +0.000000 | +0.000000 | +0.000000 **(not enabled: exact zero)** |
| solute electrostatic | -23.516097 | -20.622711 | **+2.893386** |
| 1D solvent compensation | +0.116124 | -1.553844 | **-1.669968** |
| slab dipole correction | -0.016796 | -2.626087 | **-2.609291** |
| external field . dipole | +0.000000 | +0.000000 | +0.000000 **(not enabled: exact zero)** |
| cavity energy | +3.846217 | +3.895945 | +0.049728 |
| 3D solvent energy | -0.158241 | -0.323177 | -0.164936 |
| baseline coupling E_bl | -0.638649 | +0.396949 | **+1.035598** |
| SUM of the ten | -1305.596780 | -1306.062262 | -0.465482 |

Quoting the tags rather than the bare zeros, as asked: the electron energy and
the external field . dipole are zero **because they are not enabled**, not
because they cancel. e0 and the energy head are zero **because they cancel
exactly** — e0 as a per-species constant with identical species, inter_e by the
identical features of step 4. Four exact zeros, two different reasons.

So **the entire model Delta E of -0.465 eV comes from the six live
solvent/electrostatic terms**, and the +5.635 eV error must live there too,
since the other four cannot contribute to a difference at all.

**A caution on reading which term is responsible.** The six live terms sum to
-0.465 eV but their absolute values sum to 8.423 eV — an **18.1-fold
cancellation**. So no term can be blamed by its size, and the missing 5.635 eV is
about twice the largest single live term (solute electrostatic at 2.893). This is
the same non-additivity that has caught this project repeatedly; the table
locates the error in a group of six, not in one of them.

### STEP 3: the DFT side, and two candidate correspondences

Reported as candidates only, neither folded into any total:

- **A_cav against the model's cavity energy**, the peer's candidate:
  dA_cav -0.006904 against d(cavity) +0.049728, difference +0.056632. Small, but
  **opposite in sign**.
- **A_corr against the model's slab dipole correction**, mine, and it is the
  closer of the two: dA_corr -2.636389 against d(slab dipole) -2.609291, agreeing
  to **1.03%**. If that correspondence is real, that term is essentially right
  and is not where the 5.635 eV sits. I am flagging it as a candidate for exactly
  the reason the peer flagged A_cav — two dipole-correction-like quantities
  agreeing to 1% is suggestive and is not an identification.

One internal relation the table confirms: dA_corr -2.636389 equals
dEcorr + dEcorr_band = +105.571730 - 108.208118 = -2.636388, to 1e-6. Those two
are ~100 eV terms cancelling 40-fold, which is worth knowing before anyone quotes
either alone.

Also: `Dele_z` moves +3.653562 to -6.162598, a change of -9.816160.
