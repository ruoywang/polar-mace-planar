#!/bin/bash
#SBATCH --job-name=pb1dspd
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_speedtest/spd.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_speedtest/spd.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=0:40:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16
export MACE_PB1D_TIMING=1 MACE_PB1D_TIMING_EVERY=100 MACE_STEP_TIMING=1 MACE_STEP_TIMING_EVERY=100
cd "$BASE/exp_pb1d_speedtest"
srun -n 3 env PYTHONPATH="$BASE/polar-mace-planar" $PY -u -m mace.cli.run_train \
  --config config_pb1d.yaml --name pb1d_prod150b --seed 123 \
  --distributed --launcher slurm \
  --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results > run.log 2>&1
echo "exit: $?"
grep "INFO: Epoch" run.log | cut -c1-125
grep "STEPTIMING" run.log | tail -3
grep "PB1DTIMING" run.log | tail -2
