#!/bin/bash
# full self-consistent solvent closure on the native grid (fullclosure_native.py), GPU; diagnosis only.
#SBATCH -J fcgpu
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 01:30:00
#SBATCH -o logs/fcgpu.o%j
#SBATCH -e logs/fcgpu.e%j
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol; CODE=$BASE/claude/2-1D_PB/pmp-s3denergy; PY=$BASE/.venv/bin/python
cd $BASE/claude/2-1D_PB/exp_s3d_diag
echo "CODE: $(git -C $CODE log --oneline -1 | cut -c1-80)   node $(hostname)   $(date)   pairs ${KIT_PAIRS:-default} runs ${KIT_RUNS:-default} precond ${KIT_PRECOND:-none} m ${KIT_M:-12} beta ${KIT_BETA:-0.05} out ${KIT_OUT:?}"
export MACE_PB1D_LIVE_POS=1 MACE_PB1D_GRAD_PASSES=2 MACE_PB1D_AREA_EPS=1e-12 PYTHONPATH=$CODE OMP_NUM_THREADS=16 KIT_DEVICE=cuda:0; unset MACE_PB1D_DFORCE
$PY -u fullclosure_native.py $KIT_OUT 2>&1 | grep --line-buffered -vE "cuequivariance|UserWarning|return torch|warnings.warn"
echo "rc_fcgpu=${PIPESTATUS[0]}   $(date)"
