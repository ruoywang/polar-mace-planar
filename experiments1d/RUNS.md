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
