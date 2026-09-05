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
