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

Dispatched to the 4090; heavier than the last two, since the cavity and the
full 3-D polarization are built on the native 168x168x500 grid, so LS6 is the
fallback if it does not fit in 24 GB. RESULT PENDING.
