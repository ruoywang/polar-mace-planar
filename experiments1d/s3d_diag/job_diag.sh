#!/bin/bash
# Offline diagnosis of the residual-3D solvent charge (s3d_fit_diag.py); no training, no production-code change.
#   sbatch --export=ALL,KIT_MODE=smoke,KIT_FIT_PAIRS="1",OUTD=diag_smoke job_diag.sh
#SBATCH -J s3ddiag
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 01:30:00
#SBATCH -o logs/diag.o%j
#SBATCH -e logs/diag.e%j
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol; CODE=$BASE/claude/2-1D_PB/pmp-s3denergy; PY=$BASE/.venv/bin/python
cd $BASE/claude/2-1D_PB/exp_s3d_diag
echo "CODE: $(git -C $CODE log --oneline -1 | cut -c1-80)   node $(hostname)   $(date)   mode ${KIT_MODE:-smoke} out ${OUTD:?}"
export MACE_PB1D_LIVE_POS=1 MACE_PB1D_GRAD_PASSES=2 MACE_PB1D_AREA_EPS=1e-12 PYTHONPATH=$CODE OMP_NUM_THREADS=16; unset MACE_PB1D_DFORCE
$PY -u s3d_fit_diag.py $OUTD 2>&1 | grep --line-buffered -vE "cuequivariance|UserWarning|return torch|warnings.warn"
echo "rc_diag=${PIPESTATUS[0]}   $(date)"
