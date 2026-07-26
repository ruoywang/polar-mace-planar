#!/bin/bash
#SBATCH --job-name=pb1drbw
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_rhobw/rbw.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_rhobw/rbw.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --account=DMR24028
# phase 0: gradient probe; phase 1: 5-epoch gate; phase 2: 60-epoch run at weight 1e4
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16
export MACE_PB1D_TIMING=1 MACE_PB1D_TIMING_EVERY=200

cd "$BASE/exp_pb1d_rhob80"   # has post-warmup enc counters in ./cache
timeout 600 env PYTHONPATH="$BASE/pmp-rhob80" $PY -u $BASE/tools1d/probe_rhob_grad.py \
  models/pb1d_rhob60.model $BASE/train-data/val.xyz 28 \
  $BASE/train-data/dft_solvent1d_ref.npz > "$BASE/exp_pb1d_rhobw/probe.log" 2>&1
grep -E "grad norm|PATH" "$BASE/exp_pb1d_rhobw/probe.log" || true
if ! grep -q "PATH-OK" "$BASE/exp_pb1d_rhobw/probe.log"; then
  echo "PROBE FAILED - training not started"; exit 1
fi
echo "PROBE PASSED"

cd "$BASE/exp_pb1d_rhobw/gate"
timeout 2400 srun -n 3 env PYTHONPATH="$BASE/pmp-rhob80" $PY -u -m mace.cli.run_train \
  --config config_gate.yaml --name pb1d_rhobwgate --seed 123 \
  --distributed --launcher slurm \
  --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results >> gate.log 2>&1
rc=$?
echo "gate exit: $rc"
if [ $rc -ne 0 ] || ! grep -q "INFO: Done" gate.log || grep -qE "Traceback|NaN detected" gate.log; then
  echo "GATE FAILED - production not started"; exit 1
fi
echo "GATE PASSED"

cd "$BASE/exp_pb1d_rhobw"
srun -n 3 env PYTHONPATH="$BASE/pmp-rhob80" $PY -u -m mace.cli.run_train \
  --config config_pb1d.yaml --name pb1d_rhobw --seed 123 \
  --distributed --launcher slurm \
  --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results >> run.log 2>&1
rc=$?
echo "run_train exit: $rc"
if [ $rc -eq 0 ] && grep -q "INFO: Done" run.log; then echo "TRAINING COMPLETE"; else echo "TRAINING DID NOT FINISH (exit $rc)"; fi
