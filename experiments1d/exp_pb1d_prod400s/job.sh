#!/bin/bash
#SBATCH --job-name=pb1dp400s
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_prod400s/p400s.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_prod400s/p400s.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-a100
#SBATCH --account=DMR24028
# Single-allocation run: a 5-epoch test gate first, then the full 400-epoch
# training. No resubmission logic; the 12 h limit covers the measured ~9 h.
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16
export MACE_PB1D_TIMING=1 MACE_PB1D_TIMING_EVERY=200

# ---- phase 1: test gate (5 epochs, warmup 2, code tree pmp-prod400s) ----
cd "$BASE/exp_pb1d_prod400s/gate"
srun -n 3 env PYTHONPATH="$BASE/pmp-prod400s" $PY -u -m mace.cli.run_train \
  --config config_gate.yaml --name pb1d_gate400s --seed 123 \
  --distributed --launcher slurm \
  --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results >> gate.log 2>&1
rc=$?
echo "gate exit: $rc"
if [ $rc -ne 0 ] || ! grep -q "INFO: Done" gate.log || grep -qE "Traceback|NaN detected" gate.log; then
  echo "GATE FAILED - production not started"
  exit 1
fi
echo "GATE PASSED"

# ---- phase 2: full 400-epoch production ----
cd "$BASE/exp_pb1d_prod400s"
srun -n 3 env PYTHONPATH="$BASE/pmp-prod400s" $PY -u -m mace.cli.run_train \
  --config config_pb1d.yaml --name pb1d_prod400s --seed 123 \
  --distributed --launcher slurm \
  --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results >> run.log 2>&1
rc=$?
echo "run_train exit: $rc"
if [ $rc -eq 0 ] && grep -q "INFO: Done" run.log; then
  echo "TRAINING COMPLETE"
else
  echo "TRAINING DID NOT FINISH (exit $rc)"
fi
