#!/bin/bash
# Smoke test of solvent3d_full_eval.py on a CPU node: one val twin pair (+ its volume re-run), no GPU.
#SBATCH -J s3dsmoke
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH -t 01:00:00
#SBATCH -o logs/s3dsmoke.o%j
#SBATCH -e logs/s3dsmoke.e%j
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol; CODE=$BASE/claude/2-1D_PB/pmp-s3denergy; PY=$BASE/.venv/bin/python
cd $BASE/claude/2-1D_PB/exp_pair122_vsolvfix
echo "CODE: $(git -C $CODE log --oneline -1 | cut -c1-80)   node $(hostname)   $(date)"
export MACE_PB1D_LIVE_POS=1 MACE_PB1D_GRAD_PASSES=2 MACE_PB1D_AREA_EPS=1e-12 PYTHONPATH=$CODE OMP_NUM_THREADS=64; unset MACE_PB1D_DFORCE
KIT_DEVICE=cpu KIT_MAX_FRAMES=1 KIT_EXTRA_PAIRS="" $PY -u solvent3d_full_eval.py s3d_smoke 2>&1 | grep --line-buffered -vE "cuequivariance|UserWarning|return torch|warnings.warn"
echo "rc=${PIPESTATUS[0]}  $(date)"
ls -la s3d_smoke
echo "EXIT: 0"
