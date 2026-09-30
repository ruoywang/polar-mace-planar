#!/bin/bash
# prod500_vsolv_fix (user 2026-09-30), all on one CPU node (a CPU forward is ~4 s per frame, smoke 3479925):
# the pair-122 page pipeline of exp_pair122_prod500 with the new model (pair dump, fully-ML CHGCARs 122/722,
# electron profiles, 200 twin-pair Fermi/potential) + the full-grid 3-D solvent charge evaluation.
#SBATCH -J pair122vf
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=128
#SBATCH -t 02:00:00
#SBATCH -o logs/pair.o%j
#SBATCH -e logs/pair.e%j
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol; CODE=$BASE/claude/2-1D_PB/pmp-s3denergy; PY=$BASE/.venv/bin/python
W=$BASE/claude/2-1D_PB/exp_pair122_vsolvfix; cd $W
M=$BASE/3-residual_3D/prod500_vsolv_fix/models/prod500_vsolv_fix.model
echo "CODE: $(git -C $CODE log --oneline -1 | cut -c1-80)   model $M   node $(hostname)   $(date)"
export MACE_PB1D_LIVE_POS=1 MACE_PB1D_GRAD_PASSES=2 MACE_PB1D_AREA_EPS=1e-12 PYTHONPATH=$CODE KIT_MODEL=$M KIT_DEVICE=cpu OMP_NUM_THREADS=128; unset MACE_PB1D_DFORCE
F="cuequivariance|UserWarning|return torch|warnings.warn"
echo "===== 1. pair dump ===== $(date +%T)"
$PY -u dump_structure_pair_v2_dev.py $M pair_122.xyz 0 1 structure_pair_sid122_722.npz 2>&1 | grep --line-buffered -vE "$F"; echo "rc_dump=${PIPESTATUS[0]}"
echo "===== 2. ML CHGCAR charged (sid 122) ===== $(date +%T)"
$PY -u build_chgcar_dev.py 0 /scratch/08384/tg876840/tmp/2-NiN_single/1-44_GCE/cal_122/CHGCAR $BASE/data/NiN-mix800/grid_cache_npy/density3d_net_grid_122.npy ml_sol_vsolvfix 2>&1 | grep --line-buffered -vE "$F"; echo "rc_chg_c=${PIPESTATUS[0]}"
echo "===== 3. ML CHGCAR neutral (sid 722) ===== $(date +%T)"
$PY -u build_chgcar_dev.py 1 /scratch/08384/tg876840/tmp/2-NiN_single/5-44_neutral_withsolv/cal_122/CHGCAR $BASE/data/NiN-mix800/grid_cache_npy/density3d_net_grid_722.npy ml_neutral_vsolvfix 2>&1 | grep --line-buffered -vE "$F"; echo "rc_chg_n=${PIPESTATUS[0]}"
echo "===== 4. 3-D solvent charge, full grid ===== $(date +%T)"
$PY -u solvent3d_full_eval.py s3d_eval 2>&1 | grep --line-buffered -vE "$F"; echo "rc_s3d=${PIPESTATUS[0]}"
echo "===== 5. electron profiles ===== $(date +%T)"
$PY -u extract_electron_profiles_vsolvfix.py electron_profiles_vsolvfix.npz 2>&1 | grep --line-buffered -vE "$F"; echo "rc_prof=${PIPESTATUS[0]}"
echo "===== 6. fermi / potential of the 200 twin pairs ===== $(date +%T)"
$PY -u fermi_pairs_dev.py fermi_pairs_vsolvfix.npz 2>&1 | grep --line-buffered -vE "$F"; echo "rc_fermi=${PIPESTATUS[0]}"
echo "EXIT: 0   $(date)"
