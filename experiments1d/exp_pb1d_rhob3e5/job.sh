#!/bin/bash
#SBATCH --job-name=rb3e5
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_rhob3e5/rb.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_rhob3e5/rb.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --account=DMR24028
# weight-scan link: gate on first link only; later links resume from the
# latest checkpoint (restart_latest) and exit early if training already done.
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16
export MACE_PB1D_TIMING=1 MACE_PB1D_TIMING_EVERY=200

cd "$BASE/exp_pb1d_rhob3e5"
if grep -q "INFO: Done" run.log 2>/dev/null; then echo "ALREADY COMPLETE"; exit 0; fi

if [ ! -d checkpoints ] || [ -z "$(ls -A checkpoints 2>/dev/null)" ]; then
  cd gate
  timeout 2400 srun -n 3 env PYTHONPATH="$BASE/pmp-rhob80" $PY -u -m mace.cli.run_train \
    --config config_gate.yaml --name pb1d_rhob3e5gate --seed 123 \
    --distributed --launcher slurm \
    --work_dir . --log_dir logs --model_dir models \
    --checkpoints_dir checkpoints --results_dir results >> gate.log 2>&1
  rc=$?
  echo "gate exit: $rc"
  if [ $rc -ne 0 ] || ! grep -q "INFO: Done" gate.log || grep -qE "Traceback|NaN detected" gate.log; then
    echo "GATE FAILED - training not started"; exit 1
  fi
  echo "GATE PASSED"
  cd ..
fi

timeout 6300 srun -n 3 env PYTHONPATH="$BASE/pmp-rhob80" $PY -u -m mace.cli.run_train \
  --config config_pb1d.yaml --name pb1d_rhob3e5 --seed 123 \
  --distributed --launcher slurm \
  --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results >> run.log 2>&1
rc=$?
echo "run_train exit: $rc"
if [ $rc -eq 0 ] && grep -q "INFO: Done" run.log; then echo "TRAINING COMPLETE"; else echo "LINK ENDED (exit $rc) - resume in next link"; fi
