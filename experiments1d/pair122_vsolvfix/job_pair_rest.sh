#!/bin/bash
# Steps 4-6 of job_pair_s3d.sh (steps 1-3 done in 3479928, cancelled in step 4 at ~160 s per frame with 128 threads;
# the smoke test 3479925 on the same node ran ~4 s per frame with 64 threads -> 64 threads here).
# prod500_vsolv_fix (user 2026-09-30):
# the pair-122 page pipeline of exp_pair122_prod500 with the new model (pair dump, fully-ML CHGCARs 122/722,
# electron profiles, 200 twin-pair Fermi/potential) + the full-grid 3-D solvent charge evaluation.
#SBATCH -J pairrest
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH -t 02:00:00
#SBATCH -o logs/pair.o%j
#SBATCH -e logs/pair.e%j
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol; CODE=$BASE/claude/2-1D_PB/pmp-s3denergy; PY=$BASE/.venv/bin/python
W=$BASE/claude/2-1D_PB/exp_pair122_vsolvfix; cd $W
M=$BASE/3-residual_3D/prod500_vsolv_fix/models/prod500_vsolv_fix.model
echo "CODE: $(git -C $CODE log --oneline -1 | cut -c1-80)   model $M   node $(hostname)   $(date)"
export MACE_PB1D_LIVE_POS=1 MACE_PB1D_GRAD_PASSES=2 MACE_PB1D_AREA_EPS=1e-12 PYTHONPATH=$CODE KIT_MODEL=$M KIT_DEVICE=cpu OMP_NUM_THREADS=64; unset MACE_PB1D_DFORCE
F="cuequivariance|UserWarning|return torch|warnings.warn"
echo "===== 4. 3-D solvent charge, full grid ===== $(date +%T)"
$PY -u solvent3d_full_eval.py s3d_eval 2>&1 | grep --line-buffered -vE "$F"; echo "rc_s3d=${PIPESTATUS[0]}"
echo "===== 5. electron profiles ===== $(date +%T)"
$PY -u extract_electron_profiles_vsolvfix.py electron_profiles_vsolvfix.npz 2>&1 | grep --line-buffered -vE "$F"; echo "rc_prof=${PIPESTATUS[0]}"
echo "===== 6. fermi / potential of the 200 twin pairs ===== $(date +%T)"
$PY -u fermi_pairs_dev.py fermi_pairs_vsolvfix.npz 2>&1 | grep --line-buffered -vE "$F"; echo "rc_fermi=${PIPESTATUS[0]}"
echo "EXIT: 0   $(date)"
