#!/bin/bash
#SBATCH --job-name=pb1dh1
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_h1_warmup/h1.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_h1_warmup/h1.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=1:10:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16
cd "$BASE/exp_pb1d_h1_warmup"
MACE_PB1D_WARMUP_ENC=15 PYTHONPATH="$BASE/pmp-prod" \
srun -n 3 $PY -u -m mace.cli.run_train \
  --config config_pb1d.yaml --name pb1d_h1_warmup --seed 123 \
  --distributed --launcher slurm \
  --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results > run.log 2>&1
echo "exit: $?"; grep "INFO: Epoch" run.log | cut -c1-130
