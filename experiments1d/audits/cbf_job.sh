#!/bin/bash
#SBATCH --job-name=s3dcbf
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/exp_ab_refit/cbf.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/exp_ab_refit/cbf.e%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-a100-dev,gpu-a100-small,gpu-a100
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol
CODE=$BASE/claude/2-1D_PB/pmp-s3denergy
PB=/work/08384/tg876840/ls6/repos/cep-dip-python-pb
SCRIPT=$CODE/experiments1d/audits/charge_branch_fit.py
echo "mace HEAD: $(git -C $CODE rev-parse --short HEAD)"
echo "pb   HEAD: $(git -C $PB rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "sha256:    $(sha256sum $SCRIPT | cut -d' ' -f1)"
C=$(git -C $CODE show HEAD:experiments1d/audits/charge_branch_fit.py | sha256sum | cut -d' ' -f1)
echo "committed: $C"
[ "$(sha256sum $SCRIPT | cut -d' ' -f1)" = "$C" ] && echo "PROVENANCE: IDENTICAL" || echo "PROVENANCE: DIFFERS"
cd $BASE/3-residual_3D/gate_bl
export KIT_OUT=$BASE/claude/2-1D_PB/exp_ab_refit
export KIT_FINAL_COMPARE=1
export MACE_PB1D_NO_PRELOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$BASE/.venv/bin/python -u $SCRIPT
echo "rc=$?"
echo "EXIT"
