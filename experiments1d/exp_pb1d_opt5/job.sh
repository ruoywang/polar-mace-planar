#!/bin/bash
#SBATCH --job-name=pb1ddiag
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_opt5/diag.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_opt5/diag.e%j
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16

echo "=== [1/2] diagnostics of the 80-epoch model (frozen training code) ==="
cd "$BASE/exp_pb1d_time80"
MODEL=$(ls pb1d_time80.model models/*.model 2>/dev/null | head -1)
echo "model: $MODEL"
MACE_PB_DEBUG= PYTHONPATH="$BASE/pmp-prod" $PY -u "$BASE/tools1d/diag_pb1d_model.py" \
  "$MODEL" data/val.xyz 40 2>&1 | grep -v Warning | tail -50

echo "=== [2/2] optimized-code 5-epoch timing (fresh, 3 GPU) ==="
cd "$BASE/exp_pb1d_opt5"
MACE_PB1D_TIMING=1 MACE_PB1D_TIMING_EVERY=100 MACE_STEP_TIMING=1 MACE_STEP_TIMING_EVERY=100 \
PYTHONPATH="$BASE/polar-mace-planar" \
srun -n 3 $PY -u -m mace.cli.run_train \
  --config config_pb1d.yaml --name pb1d_opt5 --seed 123 \
  --distributed --launcher slurm \
  --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results > run.log 2>&1
echo "opt5 exit: $?"
grep "INFO: Epoch" run.log | cut -c1-130
grep "STEPTIMING" run.log | tail -2
grep "PB1DTIMING" run.log | tail -2
