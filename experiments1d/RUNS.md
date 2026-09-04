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

## 2026-08-31 NiN-mix800: 800-frame bundle + normalized density weight (train800 line)
- New data: 200 neutral SOLVATED calcs (2-NiN_single/5-44_neutral_withsolv),
  geometries bitwise == 1-44_GCE cal_N, INCAR == charged GCE minus NELECT.
  All 200 converged (cal_104/189 rerun by user after a first pass left their
  PHI right plateau at +0.95/+0.98 V; now 1e-5). Solvated conventions:
  Fermi = raw E-fermi (PHI right plateau = 0), potential_diff = right-left
  (+3.04 +/- 0.2 V), sid = 600+N, config_type NiN44neusol, solvated=1, TTF.
- Fermi solvent effect (DFT, 200 neutral pairs): +0.159 +/- 0.099 eV vs the
  vacuum twins, r=0.89, range [-0.14, +0.42] (temp artifact f91250ea).
- Bundle c-MACEsol/data/NiN-mix800 (52 GB, full copies, no symlinks):
  splits = frozen 600 splits + new 200 TWIN-ALIGNED with charged NiN44
  (same cal -> same split; 640/80/80). potential1d 800 (baselines reused
  from twins), dft_solvent1d_ref 600, grids+manifest 800 (net integral
  |max| 1e-4 e), baseline_index aliases 601-800 -> twin rows, occ cache
  800 (V1-V3 audits), cd1d cache 800.
- Closure-check note: raw periodic 1-D check gives 3.84 (new) vs 2.55 eV
  (charged twins) full-range; solvent-region gap traced to the LDIPOL
  dipole-correction / dielectric-response entanglement (twin closes to
  0.006 eV with the vacuum-gap sawtooth; neutral's 3.1 V step response is
  inside RHOB). Not a label defect; the real gate is the training loss path.
- Code @982468c: --density_3d_signal_ms normalizes the raw density MSE;
  measured 0.004684 on the 800 train windows (300k voxels/frame; old-600
  0.004693, new-200 0.004657). weight 1.0 == raw ~213.5 (w200 was 200).
- Production 2-1D_PB/10-train_800: w200 config verbatim except name,
  density weight 1.0 + signal_ms, bundle paths (occ/cd1d now in ./data).
  rhob_1d_weight stays 1.0 (w200's actual value; NOT 1-train_all's 3e5).
  Gate gate800 (dev, 2 ep, warmup 0, MACE_PB_DEBUG): job 3404091.

## 2026-08-31 gate800 + train800 production (SUBMITTED)
- Gate 3404107 (dev, 2 ep, warmup 0, MACE_PB_DEBUG, code @2b149d6): PASS.
  Loaded 640/80/80 + 4 test tables (incl. NiN44neusol); signal_ms echoed;
  0 PB errors; fresh-stage1 cache 0 files; no OOM at 3xA100 production
  layout; 175 s/epoch WITH full PB training path (500 ep ~ 24.3 h).
- Gate 3404091 (first attempt, code @982468c) caught a REAL bug: neutral
  solvated frames all fell back (758x) because layer_mean = dipole/q_ion
  blew up at q_ion ~ 0 and failed the health bounds - the PB branch never
  ran on the new frames. Fix @2b149d6: |rho_ion|-weighted center below
  |q_ion| 1e-3 (always in [0,H]) and mu = exact ionic dipole integral +
  mu_bound (identical for charged frames). After the fix the new frames
  solve at 96.5% (1812 ok / 66 early-garbage fallbacks, self-healing).
- Production: 2-1D_PB/10-train_800, job 3404125 (gpu-a100, 40 h,
  3 ranks, 500 ep, warmup 30), code @2b149d6, bundle NiN-mix800.

## 2026-08-31 solvent3d stage 1: code + labels + smoke + gate pair (branch pb-solvent3d)
- Code (independent clone claude/2-1D_PB/pmp-solvent3d, branch pb-solvent3d
  @8279346, base 982468c): equivariant Solvent3DChargeHead (zero-init, ion
  channel q-gated), backend probe (detached envelopes + 1-D baselines at
  sampled points; cavity stashed detached by the closure), solvent3d loss
  (per-channel signal-ms normalized, attach mirrors density_3d, DDP-safe per
  the de84363 pattern — every rank joins the count collective), metrics
  RMSE_solvent3d_b/_i, 5 config keys + misconfig guard. Unit tests: point
  evaluator == density_3d evaluator to 1e-17; zero-init/q-gate verified.
- Labels (job 3404111): data/NiN-mix800/solvent3d_grid, 600 frames, physics
  convention f32; integrals pass; plane averages == dft_solvent1d_ref to
  5.7e-12 (cross-validates the bundle's 1-D ref). signal_ms b=4.268e-6
  i=7.914e-8; bundle density_3d signal_ms measured 4.6934e-3.
- Smoke (job 3404120, dev, 3-GPU DDP, 2 ep, warmup 0): rc=0, no deadlock,
  0 fallbacks, solvent3d metrics live and falling from epoch 0
  (b 1.87e-3 -> 1.48e-3, i 3.7e-4 -> 1.45e-4). ~26 min/epoch on cold solves.
- GATE PAIR submitted: 3-residual_3D/gate_w{0,1}, jobs 3404171/3404172,
  34 ep, warmup 30, seed 123, 3-GPU a100; single variable = solvent3d_weight
  0 vs 1 (head enabled in both for RNG parity). Base config = w200 recipe
  with density_3d_weight 1.0 + signal_ms 4.6934e-3 (normalized w200-strength)
  on the NiN-mix800 bundle. Pass criteria: w1 hurts no legacy metric vs w0
  at equal epochs; solvent3d validation falling; memory/step-time measured.

## 2026-09-01 solvent3d label-IO saga: RESOLVED by presampled point pack
- Root cause chain (all on-node measurements): NiN-mix800 label pack (65 GB)
  must re-stream EVERY epoch on BeeGFS — read() bypasses all local caching
  (fincore 0 B after 3 full reads; 88 MB/s sustained), random mmap faults
  are latency-bound (3-5 s/graph cold), mmap sequential materialize also
  re-streams on truly cold nodes (~24 MB/s; the fast standalone timings were
  polluted by the running job's client cache — trust only the training
  process's own /proc/PID/io). Epoch times: 24 min (mmap random) ->
  14.5 min (seq read) -> 33 min (mmap materialize, cold node).
- Fix @28ca32a: solvent3d_points_npy_v1 — 300k presampled rows/frame
  (x,y,z,ref_b,ref_i f32; 3.4 GB total), each step reads ONE contiguous
  ~20 KB window; full-grid signal_ms kept for normalization; builder
  self-checks rows vs source grids; sampled ms matches full-grid to 0.3%.
- VERDICT (job 3406176, fresh node c301-004): warmup epochs 1m56s-1m58s —
  at/below the w0 baseline (2.7-4 min) and back to the production-normal
  pace. 34-epoch gate leg now fits one dev link.
- Ops lessons: crashed-session watcher loops survive as PPID=1 orphans and
  keep resubmitting ghost jobs (two caught: 3405985, 3406172; killed orphan
  PID 1896485) — new-session takeover must sweep for orphan shells; sbatch
  -> squeue visibility lag needs a submit grace + 3-empty-polls rule; a
  sentinel loop (state transitions / log staleness / epoch-time cap /
  stderr / ghost jobs) now guards all runs.

## 2026-09-01 solvent3d GATE VERDICT: PASS (w1 dev leg complete, 34 ep, 78 min)
- gate_w1_dev job 3406176 @28ca32a, points pack, rc=0, warmup ~2 min/ep,
  PB epochs ~3 min/ep. vs gate_w0 (same code, weight 0) at epoch 33 (valid):
  E 8.47 vs 9.80 meV, F 33.96 vs 33.75, potential 0.148 vs 0.183,
  fermi 0.107 vs 0.120, density_3d 0.0316 vs 0.0310, occ 0.0087 vs 0.0091,
  Phi1D 0.120 vs 0.126, rhob_1d 2.49e-4 vs 2.44e-4 -> par-or-better on
  every legacy metric (differences within the epoch-to-epoch noise band).
- solvent3d supervision IS learning through the joint path: eval
  RMSE_solvent3d_b frozen at ~2.13e-3 during warmup (head untrained by
  construction), then 1.151 -> 1.069e-3 over PB epochs 30-33 (ratio ~0.50
  of the 1-D residual after only ~1.9k labeled steps — already at the
  frozen-probe level 0.47, still falling); ion 1.28 -> 1.00e-4.
- Criteria: (1) no legacy-metric harm PASS; (2) solvent3d falling PASS;
  (3) runtime/memory PASS (3 min PB epochs, no OOM). GATE PASSED.
- a100 replication leg cancelled (redundant; frees the queue for the
  user's train800 production). Next decision: 500-ep production with
  solvent3d w=1 (recipe = this gate config) vs first extending the gate.

## 2026-09-01 s3d_prod500 SUBMITTED: 500-ep solvent3d production
- 3-residual_3D/prod500, job 3406494, code @28ca32a (pb-solvent3d), config
  == the passed gate config (w=1, points pack, density w1.0+signal_ms,
  warmup 30) with max_num_epochs 500. 3-GPU a100, 30h wall (projection:
  ~25h at gate pacing). Queued alongside the user's train800 (3404125,
  main-branch baseline on the same bundle) — the production A/B pair.

## 2026-09-04 twin-energy diagnostic (train800): energy regression root cause
- Setup: model E on both members of the 200 neutral geometry twins
  (NiN44vac sid 401-600 / NiN44neusol sid 601-800), eval mode, dev job
  3413461, ~0.7 s/pair. exp_neutralsolv_prep/twin_energy_diag.{py,npz}.
- DFT:  E_solv - E_vac = +2.944 +/- 0.088 eV/cell (+14.2 meV/atom).
- MODEL: +0.032 +/- 0.045 eV -- the twins are energetically
  indistinguishable to the model. corr(d_ml, d_dft)=0.52 (it tracks about
  half of the structure-dependent variation, none of the constant).
- The optimizer parks each pair's prediction between the labels:
  per-frame error vac +0.959 eV (=4.6 meV/atom) / neusol -1.953 eV
  (=9.4 meV/atom), summing to the 2.91 shortfall -- EXACTLY reproducing
  the final test-table energies (4.6 / 9.5). Root cause of the energy
  regression confirmed.
- Physics of the offset: cavity-formation energy. VASPsol++ A_cav prints
  3.77-3.90 per frame (consistent with ~380-390 A^2 = the two flat
  interfaces of the 190 A^2 cell); tau=9e-3 eV/A^2 -> E_cav ~ +3.5 eV,
  near-constant, plus electrostatic/ionic solvation ~ -0.5 eV
  (structure-dependent, the 0.088 spread). The model's solvent energy
  path (1-D periodic cross+self + dipole-correction delta) is PURELY
  electrostatic -- no cavity term exists, so the +3.5 eV constant is
  unrepresentable by construction.
- Candidate fixes (user to decide): (a) physical cavity term
  E_cav = tau * A_cav from the model's own cavity field (1-D shape
  factor gives the flat-interface area; tau fixed 9e-3 or learnable);
  (b) minimal learnable scalar offset * solvated flag (captures the
  constant, one parameter); (c) label-side constant shift (not preferred).

## 2026-09-04 CORRECTION: the twin offset is VASPsol++'s alignment term, NOT cavitation
- Per-frame extraction over all 200 logs: A_corr (= Ecorr + Ecorr_band,
  the potential-reference alignment correction VASPsol++ adds to TOTEN)
  = +2.828 +/- 0.130 eV with corr(E_diff, A_corr) = 0.952;
  E_diff - A_corr = +0.116 +/- 0.054 eV (net physical solvation at this
  bookkeeping level). twin_decomp.npz.
- Cavitation does NOT fit: printed cavity area 426.5 +/- 6.1 A^2 ->
  tau*area = 3.84 eV != 2.94, and it lives inside the PB functional
  A_solv (-1.01 +/- 0.17), not as a separate TOTEN add-on. The previous
  entry's tau*A_cav ~ 3.5 attribution was a numerical coincidence -
  retracted.
- Fix ranking updated: the offset is a near-constant reference-convention
  energy (cv 4.6%) -> a solvated-gated learnable energy offset (one
  scalar, plus optionally a small structure readout for the remaining
  ~0.1 eV/cell) is the CORRECT fix, not a cavity term.

## 2026-09-04 FINAL (source-verified): twin offset = A_solv + A_cav + relaxation
- solvation.F line 1690: A_corr = A_solv + A_cav. The "A_cav: 3.9044"
  print is an ENERGY in eV (= tau * area = 0.009 * 433.82, bitwise), not
  an area; the previous "alignment artifact" reading is retracted.
  Ecorr_band = int(n_val * V_corr) is standard double-counting removal
  (V_corr sits in the KS Hamiltonian); Ecorr = A_corr - Ecorr_band by
  construction -- the 4.4 mV/e "shift difference" is A_corr/660, an identity.
- 200-frame decomposition: E_diff = A_cav (+3.839 +/- 0.055, cavitation,
  near-constant) + A_solv (-1.011 +/- 0.166, PB electrostatic/ionic)
  + SCF relaxation (+0.116 +/- 0.054) = +2.944 +/- 0.088. All bitwise
  consistent with the extracted A_corr (2.828 +/- 0.130, corr 0.952).
- Fix ranking (final): mirror VASPsol in the model's solvent branch --
  E += A_solv_1d (the 1-D PB functional value, already computed by the
  solver) + tau*A_cav (flat-interface 1-D area 379 A^2 -> 3.41 eV covers
  89%; the 11% lateral-corrugation remainder is a 3-D cavity quantity,
  connecting to the residual-3D line's s_diel3). Transfers across cells,
  unlike a learned constant. Learnable solvated offset stays the cheap
  fallback.
