#!/bin/bash
#SBATCH --job-name=pb1dt80
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_time80/t80.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_time80/t80.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --account=DMR24028
set -uo pipefail
DIR=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_time80
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export PYTHONPATH="$BASE/pmp-prod"
export MACE_PB1D_TIMING=1 MACE_PB1D_TIMING_EVERY=100
cd "$DIR"

CHAIN=$(cat chain_count 2>/dev/null || echo 0)
echo "=== chain link $CHAIN (job $SLURM_JOB_ID) ==="
if [ "$CHAIN" -ge 6 ]; then echo "chain limit reached"; exit 0; fi
echo $((CHAIN + 1)) > chain_count

timeout 6600 srun -n 3 $PY -u -m mace.cli.run_train \
  --config config_pb1d.yaml --name pb1d_time80 --seed 123 \
  --distributed --launcher slurm \
  --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results >> run.log 2>&1
rc=$?
echo "run_train exit: $rc"
if [ $rc -eq 0 ] && grep -q "INFO: Done" run.log; then
  echo "TRAINING COMPLETE"
else
  echo "not finished (rc=$rc), resubmitting"
  echo "RESUBMIT_NEEDED"  # compute nodes cannot sbatch on LS6; chained from login side
fi
