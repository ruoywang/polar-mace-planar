# 工作站小测试包（residual-3D 溶剂电荷诊断）

> **硬件更正（2026-09-09 由那台机器实测）**：实际是两张 **RTX 4090，各 24564 MiB**，驱动 570.153.02。本文件早先写的 4080 / 16 GB 是错的。结论不变：4090 同属 Ada，fp64 仍是 fp32 的 1/64；24 GB 也救不了 sid 201（单个 36612² fp64 Gram 就 10.7 GB，拟合同时要多个）。目录名沿用 `mini_kit_4080` 只为连续性。

一句话：**单帧诊断类测试可以在 4080 上做，而且比在 LS6 排队快；带 NiN88 的拟合、全基底检验和任何训练不行。** 决定因素不是显存也不是算力总量，而是**双精度**——PB 求解按设计必须 float64，而 4080 的 fp64 是 fp32 的 1/64，A100 是 1/2。

这个包 2.5 GB，自带两帧完整数据、模型、代码和一个冒烟测试。冒烟测试会**在你的卡上实测 fp64 速度**并逐项对照 A100 的数值，所以"能不能用、慢多少"不用估算。

---

## 1 · 这台机器能跑什么

| 测试 | 4080 | 说明 |
|---|---|---|
| `cavity_compare.py` 空腔对照（单帧） | 可以 | 主要是 168×168×500 与 100×100×300 的 fp64 网格运算，估计峰值 4–6 GB |
| `energy_reconcile.py` 统一能量对账（单帧） | 可以 | 同上 |
| `model_terms_audit.py` / `bl_check.py` / `verify_perplane.py` / `env_coverage.py` / `residual_localize.py` | 可以 | 单帧前向 + 网格统计 |
| `neutral_feasible_amp.py` 中性可行点幅度 | 可以，慢几倍 | 合成过程要对 207 原子 × 3 sigma × 9 lm 反复做 fp64 FFT |
| `joint_subspace_solve.py` 仅 sid 1 / 601 | 勉强 | A100 上三帧三基底跑了 111 分钟；这里只有两帧，但 fp64 稠密线代慢得最多 |
| `joint_*` 含 NiN88（sid 201） | **不行** | A100 40 GB 上就 OOM 过一次（36612² fp64 Gram 单矩阵 10.7 GB，且拟合同时要多个），24 GB 也无解 |
| 倒空间全基底最小自能检验 | **不行** | 内存和 fp64 稠密线代双重不够 |
| 任何训练 / 续训 | **不行** | 需要 40 GB 级显存；两张 4080 之间没有 NVLink，脚本也是单卡 |

为什么双精度是硬约束，不是我保守：`cep-dip-python-pb/pure_python/torch_pb.py` 第 20 行写着
`float64 by default to clear the cal_18 validation gate; float32 diverges on this ill-conditioned problem`。
所以不能靠"改成 fp32 提速"绕过去。

真正的收益在**排队时间**：LS6 的 `gpu-a100-dev` 队列现在经常一等就是几十分钟，一次 4 分钟的诊断也要排。本地卡是零排队，写—跑—看的迭代会明显顺。两张卡还可以同时跑两个独立诊断（比如带电帧和中性帧各一张）。

---

## 2 · 目录

```
mini_kit_4080/
  env.sh                     先 source 它（设三个路径变量 + 各诊断开关）
  README.md                  本文件
  MANIFEST.sha256            所有文件的校验和，拷完先校验
  dft/                       2.0 GB  两帧 DFT 场文件（原始 ASCII，未转换）
    1-44_GCE/cal_1/          sid 1，NiN44 q=-1.00：CHGCAR RHOB RHOION PHI POSCAR CONTCAR OUTCAR
    5-44_neutral_withsolv/cal_1/   sid 601，中性带溶剂：同上
  run/                       跑测试的工作目录
    smoke_test.py            冒烟测试（fp64 实测 + 数值逐项对照）
    smoke_job.sh             LS6 上生成参照值的作业脚本
    reference_values.json    A100 上的参照数值（由 smoke_job.sh 生成）
    config_pb1d.yaml         训练配置（这里只作参考，路径已是相对的）
    cal1_train.json          PB 求解器配置，自包含，无绝对路径
    checkpoints/             s3d_gate_bl2_run-123.model + _epoch-33.pt（107 MB）
    data/                    两帧 xyz + 各缓存（358 MB，已按 sid 切片）
    jss_1_*.npz jss_601_*.npz    已存的小矩阵（M/b/S/x/alpha/coeffs/refs），做可行性代数用
    cache/                   phi 缓存目录（空，运行时自己写）
  audits/                    24 个诊断脚本，路径已改成读环境变量
  code/
    pmp-s3denergy/           模型代码工作树，HEAD = 4b61668
    cep-dip-python-pb/       PB 求解器工作树，HEAD = 9b3b9ba
    *.bundle                 两个仓库的 git bundle（权威副本，可 git clone 出来核对）
    *.HEAD                   记录 HEAD 的 commit
```

数据切片说明（避免你以为丢了东西）：

- `data/baseline_cache` 原本是 400 行 × 2 场 × 100×100×300 的 float32（9.1 GB），这里只留了需要的 1 行（24 MB），索引 json 已重写成 `{"1": 0, "601": 0}`。**sid 1 和 601 共用同一行**——底座按几何复用，这两帧几何相同，原始索引里本来就都指向第 0 行。
- `solvent3d_grid` / `solvent3d_points` / `grid_cache_npy` 原本 64 GB / 3.4 GB / 43 GB，这里各留 sid 1 和 601 的条目，manifest 已重写成相对路径。
- 场文件保持原始 ASCII 未转二进制：这样 `read_grid` 不用改，而且和 LS6 上读到的是同一批字节。

---

## 3 · 装环境

版本要和 LS6 一致，否则数值对不上没意义（完整清单在 `code/pmp-s3denergy/requirements-freeze.txt`）：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install ase==3.22.1 e3nn==0.4.4 opt-einsum==3.3.0 opt-einsum-fx==0.1.4 \
            torch-ema==0.3 torchmetrics==1.8.2 scipy==1.13.1 pyyaml
```

torch 2.2.1 + cu121 的官方 wheel 含 sm_89，Ada 卡（4080/4090）直接支持，不用自己编译。

**解释器陷阱**：那台机器上 `python3` 会落到 anaconda base（Python 3.12.13，虽然也有 torch 2.2.1+cu121，但依赖集不同）。版本精确匹配的是 conda 环境 `pmp39`（`/home/rw28343/softwares/anaconda3/envs/pmp39/bin/python`，Python 3.9.25）。必须显式激活或用绝对路径调用；`smoke_test.py` 第二行会打印 `sys.executable` 存证。

---

## 4 · 给那台机器上的 Claude 会话看的任务书

`code/pmp-s3denergy/experiments1d/audits/TASK_4080.md`。它写明了：先验证包、再跑哪个诊断、raw 与 lateral 两种电荷绝对不能混、要回报哪些表、以及结果怎么送回来。**仓库里的那份是权威版本**，包里这份快照可能旧了，先在 `code/pmp-s3denergy` 里 `git pull`。

## 5 · 先跑冒烟测试

```bash
cd mini_kit_4080
sha256sum -c MANIFEST.sha256 --quiet     # 拷贝完整性
source env.sh
cd run
python -u smoke_test.py
```

它做三件事：

1. 实测这张卡的 fp64 DGEMM 和两个尺寸的 fp64 FFT，并打印相对 A100 慢多少倍；
2. 跑 sid 1 和 601 的前向，打印 `energy / comp_1d / E_cav / E_bl / E_3d(e_xsol,e_self) / delta` 和峰值显存；
3. 和 `reference_values.json` 里 A100 的同一批数字逐项比，容差 1e-6 相对误差，最后打印 `SMOKE PASS` 或 `SMOKE FAIL`。

**FAIL 就别用这台机器出结果**——说明版本或数据不等价，先把差在哪一项定位出来。

---

## 6 · 跑一个真诊断

```bash
source env.sh && cd run
python -u ../audits/cavity_compare.py 2>&1 | tee cav_local.log
```

脚本里的路径都走环境变量（`KIT_DFT` / `KIT_MACE_REPO` / `KIT_PB_REPO` / `KIT_PB_CONFIG`），`env.sh` 已经设好，包放在哪个目录都行。诊断开关也在 `env.sh` 里：

- `MACE_S3D_EXPORT_DELTA=1` 导出能量网格上的那个确切残差（只给诊断用，生产关闭）
- `MACE_EVAL_BL=1`、`MACE_EVAL_MODEL=...` 与 LS6 作业一致
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 16 GB 卡上更需要

如果 `cavity_compare.py` 的 `[Q1-dist]` 段报显存不足：它按 `4.0e7 / (nx*ny*nat)` 自适应分块，把那个常数调小即可，其余结果不受影响（这一段有 try 保护，失败只打印一行）。

---

## 7 · 我这边能不能直接操作那台机器

不能直连。这个会话跑在 LS6 上，只有这台机器的 shell；我刚查过，当前没有可达的其他 Claude 会话。可用的两条路：

1. **在那台机器上开一个 Claude Code 会话**，让它连到同一个账号。之后两个会话之间可以互相发消息，我可以把要跑的诊断和判据直接交给它，它把数字发回来。
2. **你手动跑**：按上面的命令跑，把 `run/*.log` 和 `run/smoke_results.json` 发我，我照常分析。

第 2 条现在就能用，不需要任何额外配置。

---

## 8 · 换机器后不成立的记账

一并记下来，免得在新机器上重复踩：

- LS6 上 `gh` 命令目前起不来（登录节点进程配额满，Go 运行时 `failed to create new OS thread`），推送要直接用 `~/.config/gh/hosts.yml` 里的 token 走 https。这条只对 LS6 成立。
- LS6 的 scratch 是 BeeGFS：`read()` 不进缓存、随机 mmap 缺页是秒级，大数组一律顺序物化。本地 SSD 上没有这个问题，所以在 4080 上跑同一脚本的 IO 行为会不一样，不能把两边的墙钟直接比较——只比数值。
- 这两帧的模型网格是 100×100×300，DFT 原生网格是 168×168×500。凡是把 DFT 场插值到模型网格的统计（`to_shape`）都会平滑掉尖锐的束缚电荷，涉及幅度的结论要在原生网格上复核。
