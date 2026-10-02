#!/bin/bash
# 3-D rigid-shift test of the ML solvent charge vs DFT (shift_scan_3d.py); diagnosis only.
#SBATCH -J shift3d
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 01:00:00
#SBATCH -o logs/shift3d.o%j
#SBATCH -e logs/shift3d.e%j
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol; CODE=$BASE/claude/2-1D_PB/pmp-s3denergy; PY=$BASE/.venv/bin/python
cd $BASE/claude/2-1D_PB/exp_s3d_diag
echo "CODE: $(git -C $CODE log --oneline -1 | cut -c1-80)   node $(hostname)   $(date)   pairs ${KIT_PAIRS:-default}"
export MACE_PB1D_LIVE_POS=1 MACE_PB1D_GRAD_PASSES=2 MACE_PB1D_AREA_EPS=1e-12 PYTHONPATH=$CODE OMP_NUM_THREADS=16; unset MACE_PB1D_DFORCE
$PY -u shift_scan_3d.py shift_scan_3d.json 2>&1 | grep --line-buffered -vE "cuequivariance|UserWarning|return torch|warnings.warn"
echo "rc_shift=${PIPESTATUS[0]}   $(date)"
