#!/bin/bash
#SBATCH --job-name=s3dcav
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/exp_neutralsolv_prep/cav.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/exp_neutralsolv_prep/cav.e%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol
CODE=$BASE/claude/2-1D_PB/pmp-s3denergy
SCR=$BASE/claude/2-1D_PB/exp_neutralsolv_prep
echo "CODE: $(git -C $CODE log --oneline -1 | cut -c1-80)"
cd $BASE/3-residual_3D/gate_bl
export MACE_EVAL_BL=1
export MACE_EVAL_MODEL=checkpoints/s3d_gate_bl2_run-123.model
# diagnostics-only: export the energy-side residual on the energy grid
export MACE_S3D_EXPORT_DELTA=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $BASE/.venv/bin/python -u $SCR/cavity_compare.py
echo "rc=$?"
echo "EXIT"
