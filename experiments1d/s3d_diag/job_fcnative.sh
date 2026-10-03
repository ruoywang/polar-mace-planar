#!/bin/bash
# full self-consistent solvent closure on the native grid (fullclosure_native.py); CPU fallback while the GPU dev queue is backed up.
#SBATCH -J fcnative
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH -t 02:00:00
#SBATCH -o logs/fcnative.o%j
#SBATCH -e logs/fcnative.e%j
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol; CODE=$BASE/claude/2-1D_PB/pmp-s3denergy; PY=$BASE/.venv/bin/python
cd $BASE/claude/2-1D_PB/exp_s3d_diag
echo "CODE: $(git -C $CODE log --oneline -1 | cut -c1-80)   node $(hostname)   $(date)   pairs ${KIT_PAIRS:-default} runs ${KIT_RUNS:-default} m ${KIT_M:-20} beta ${KIT_BETA:-0.5} precond ${KIT_PRECOND:-eps}"
export MACE_PB1D_LIVE_POS=1 MACE_PB1D_GRAD_PASSES=2 MACE_PB1D_AREA_EPS=1e-12 PYTHONPATH=$CODE KIT_DEVICE=cpu OMP_NUM_THREADS=64; unset MACE_PB1D_DFORCE
$PY -u fullclosure_native.py ${KIT_OUT:-fullclosure_native.json} 2>&1 | grep --line-buffered -vE "cuequivariance|UserWarning|return torch|warnings.warn"
echo "rc_fcnative=${PIPESTATUS[0]}   $(date)"
