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
of whether the port is correct. Every energy term, by contrast, was bit-identical
across the two runs.

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
