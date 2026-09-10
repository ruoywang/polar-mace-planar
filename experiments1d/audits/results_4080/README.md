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
