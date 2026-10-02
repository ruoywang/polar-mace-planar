#!/bin/bash
# production full 3-D solvent charge evaluated on the DFT native grid (production3d_native.py); CPU, diagnosis only.
#SBATCH -J prod3d
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH -t 02:00:00
#SBATCH -o logs/prod3d.o%j
#SBATCH -e logs/prod3d.e%j
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol; CODE=$BASE/claude/2-1D_PB/pmp-s3denergy; PY=$BASE/.venv/bin/python
cd $BASE/claude/2-1D_PB/exp_s3d_diag
echo "CODE: $(git -C $CODE log --oneline -1 | cut -c1-80)   node $(hostname)   $(date)   pairs ${KIT_PAIRS:-default}"
export MACE_PB1D_LIVE_POS=1 MACE_PB1D_GRAD_PASSES=2 MACE_PB1D_AREA_EPS=1e-12 PYTHONPATH=$CODE KIT_DEVICE=cpu OMP_NUM_THREADS=64; unset MACE_PB1D_DFORCE
$PY -u production3d_native.py production3d_native.json 2>&1 | grep --line-buffered -vE "cuequivariance|UserWarning|return torch|warnings.warn"
echo "rc_prod3d=${PIPESTATUS[0]}   $(date)"
