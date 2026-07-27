#!/bin/bash
#SBATCH --job-name=jitverify
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_jit_verify/jv.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_jit_verify/jv.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0:40:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --exclude=c301-001,c301-002
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
cd "$BASE/exp_jit_verify"
echo "CODE VERSION: polar-mace-planar @ $(git -C $BASE/polar-mace-planar rev-parse HEAD)"
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
export OMP_NUM_THREADS=16
# baseline cache paths resolve relative to cwd (./data -> NiN-mix)
env PYTHONPATH="$BASE/polar-mace-planar" $PY -u verify_jit.py \
  "$BASE/exp_pb1d_mix150/models/pb1d_mix150.model" ./data/val.xyz 12 2>&1 \
  | grep -vE "cuequivariance|UserWarning|warnings.warn"
echo "verify exit: $?"
