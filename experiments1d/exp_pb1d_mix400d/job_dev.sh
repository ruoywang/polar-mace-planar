#!/bin/bash
#SBATCH --job-name=pb1dmx400d
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_mix400d/mx400d.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_mix400d/mx400d.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --account=DMR24028
# mix 599 structures, 400 epochs, dev-queue chain link (checkpoint resume);
# each link is submitted from the login side, ~8 links expected
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
cd "$BASE/exp_pb1d_mix400d"
echo "CODE VERSION: pmp-mix @ $(git -C $BASE/pmp-mix rev-parse HEAD)"
if grep -q "INFO: Done" run.log 2>/dev/null; then echo "ALREADY COMPLETE"; exit 0; fi
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16
export MACE_PB1D_TIMING=1 MACE_PB1D_TIMING_EVERY=200
timeout 6300 srun -n 3 env PYTHONPATH="$BASE/pmp-mix" $PY -u -m mace.cli.run_train \
  --config config_pb1d.yaml --name pb1d_mix400d --seed 123 \
  --distributed --launcher slurm \
  --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results >> run.log 2>&1
rc=$?
echo "run_train exit: $rc"
if [ $rc -eq 0 ] && grep -q "INFO: Done" run.log; then echo "TRAINING COMPLETE"; else echo "LINK ENDED (exit $rc) - resume next link"; fi
