# 任务 → 代码版本台账

规则(2026-07-26 起):每个训练/评估作业启动时,作业脚本把所用代码树的
`git rev-parse HEAD` 写进 run 日志;本文件记录每个实验目录用的是哪个
commit。新实验一律登记,旧实验按已知信息回填(未知处如实标注)。

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
