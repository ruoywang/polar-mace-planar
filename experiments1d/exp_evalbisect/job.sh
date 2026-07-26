#!/bin/bash
#SBATCH --job-name=p400bis
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_evalbisect/bis.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_evalbisect/bis.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=1:20:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16
cd "$BASE/exp_evalbisect"
echo "=== [A] 3-rank DDP final eval (复现训练终评) ==="
srun -n 3 env MACE_PB_DEBUG=1 PYTHONPATH="$BASE/pmp-prod400" $PY -u -m mace.cli.run_train \
  --config config_pb1d.yaml --name pb1d_prod400 --seed 123 \
  --distributed --launcher slurm \
  --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results > run3.log 2>&1
grep -A8 "Error-table on TEST" run3.log | tail -6
echo "=== [B] 单卡 final eval（同 checkpoint 同数据） ==="
srun -n 1 env MACE_PB_DEBUG=1 PYTHONPATH="$BASE/pmp-prod400" $PY -u -m mace.cli.run_train \
  --config config_pb1d.yaml --name pb1d_prod400 --seed 123 \
  --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results > run1.log 2>&1
grep -A8 "Error-table on TEST" run1.log | tail -6
echo "=== [B] 单卡 VALID 对照 ==="
grep -A7 "Error-table on TRAIN and VALID" run1.log | tail -4
echo "=== fallback 计数（3卡 vs 单卡） ==="
grep -c "PB1DDBG-FALLBACK" run3.log || echo 0
grep -c "PB1DDBG-FALLBACK" run1.log || echo 0
echo "=== fallback 样例 ==="
grep -m4 "PB1DDBG-FALLBACK" run3.log run1.log
