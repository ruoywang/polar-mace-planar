#!/bin/bash
#SBATCH --job-name=gcmd
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_md_gcmd/gc.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_md_gcmd/gc.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --exclude=c301-001,c301-002
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
cd "$BASE/exp_md_gcmd"
echo "CODE VERSION: polar-mace-planar @ $(git -C $BASE/polar-mace-planar rev-parse HEAD)"
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
export OMP_NUM_THREADS=16
# 500 steps constant-potential MD at targetmu -3.36 (the example's setpoint)
env PYTHONPATH="$BASE/polar-mace-planar:$BASE/exp_md_gcmd" $PY -u run_gcmd.py \
  "$BASE/exp_pb1d_mix400/models/pb1d_mix400.model" ./data/val.xyz 0 500 -3.36 2>&1 \
  | grep -vE "cuequivariance|UserWarning|warnings.warn"
echo "gcmd exit: $?"
