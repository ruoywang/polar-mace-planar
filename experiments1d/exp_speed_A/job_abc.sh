#!/bin/bash
#SBATCH --job-name=pb1dabc
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_speed_A/abc.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_speed_A/abc.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=1:30:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16
export MACE_PB1D_TIMING=1 MACE_PB1D_TIMING_EVERY=100
run_variant() {
  local V=$1 TREE=$2
  echo "===== 变体 $V ($TREE) ====="
  cd "$BASE/exp_speed_$V"
  srun -n 3 env PYTHONPATH="$BASE/$TREE" $PY -u -m mace.cli.run_train \
    --config config.yaml --name pb1d_prod150b --seed 123 \
    --distributed --launcher slurm \
    --work_dir . --log_dir logs --model_dir models \
    --checkpoints_dir checkpoints --results_dir results > run.log 2>&1
  echo "exit: $?"
  grep "INFO: Epoch" run.log | cut -c1-118
  grep "PB1DTIMING" run.log | tail -1
}
run_variant A pmp-prod
run_variant B pmp-prod150
run_variant C polar-mace-planar
echo "ALL VARIANTS DONE"
