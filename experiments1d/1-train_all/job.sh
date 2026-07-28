#!/bin/bash
#SBATCH --job-name=pb1dtrainall
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/1-train_all/train.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/1-train_all/train.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=18:00:00
#SBATCH --partition=gpu-a100
#SBATCH --account=DMR24028
# full 600-structure production training (480 train / 60 val / 60 test),
# config identical to mix400 except the data bundle (+ sid 594)
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol
CODE=$BASE/claude/2-1D_PB/pmp-trainall
cd "$BASE/1-train_all"
echo "CODE VERSION: pmp-trainall @ $(git -C $CODE rev-parse HEAD)"
if grep -q "INFO: Done" run.log 2>/dev/null; then echo "ALREADY COMPLETE"; exit 0; fi
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16
export MACE_PB1D_TIMING=1 MACE_PB1D_TIMING_EVERY=200
timeout 63000 srun -n 3 env PYTHONPATH="$CODE" $PY -u -m mace.cli.run_train \
  --config config_pb1d.yaml --name pb1d_train_all --seed 123 \
  --distributed --launcher slurm \
  --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results >> run.log 2>&1
rc=$?
echo "run_train exit: $rc"
if [ $rc -eq 0 ] && grep -q "INFO: Done" run.log; then echo "TRAINING COMPLETE"; else echo "ENDED (exit $rc)"; fi
