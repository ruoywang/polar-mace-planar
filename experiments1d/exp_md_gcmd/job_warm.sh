#!/bin/bash
#SBATCH --job-name=gcmdwarm
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_md_gcmd/warm.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_md_gcmd/warm.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --exclude=c301-001,c301-002
#SBATCH --account=DMR24028
# optimization gate: (1) rerun the runtime-baseline parity gate after the
# chunk-size change; (2) 500-step warm-start gcmd with in-flight cold checks
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
export OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MACE_PB1D_TIMING=1 MACE_PB1D_TIMING_EVERY=200
cd "$BASE/exp_runtime_baseline"
echo "CODE VERSION: polar-mace-planar @ $(git -C $BASE/polar-mace-planar rev-parse HEAD)"
echo "=== parity gate (chunk 64) ==="
env PYTHONPATH="$BASE/polar-mace-planar" $PY -u verify_runtime_baseline.py \
  "$BASE/exp_pb1d_mix150/models/pb1d_mix150.model" ./data/val.xyz 2>&1 \
  | grep -vE "cuequivariance|UserWarning|warnings.warn" | grep -E "rel-rmse|center-effect|ms/step|GATE"
cd "$BASE/exp_md_gcmd"
echo "=== warm-start gcmd 500 steps ==="
env PYTHONPATH="$BASE/polar-mace-planar:$BASE/exp_md_gcmd" $PY -u run_gcmd.py \
  "$BASE/exp_pb1d_mix400/models/pb1d_mix400.model" ./data/val.xyz 0 500 -3.36 warm 2>&1 \
  | grep -vE "cuequivariance|UserWarning|warnings.warn"
echo "warm exit: $?"
