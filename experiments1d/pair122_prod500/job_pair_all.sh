#!/bin/bash
# One job for the whole prod500_w1000_ref pair-122 analysis (GPU part): pair dump (sections 1-9),
# fully-ML CHGCARs for 122 (charged) and 722 (neutral) (sections 7b/7c + 10 input), electron-density
# profiles, Fermi/potential of the 200 twin pairs (section 0b). Then pushes.
#SBATCH -J pair122p5
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -o logs/pair.o%j
#SBATCH -e logs/pair.e%j
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol; CODE=$BASE/claude/2-1D_PB/pmp-s3denergy; PY=$BASE/.venv/bin/python
W=$BASE/claude/2-1D_PB/exp_pair122_prod500; cd $W
M=$BASE/3-residual_3D/prod500_w1000_ref/models/prod500_w1000_ref.model
echo "CODE: $(git -C $CODE log --oneline -1 | cut -c1-80)   model $M   node $(hostname)   $(date)"
export MACE_PB1D_LIVE_POS=1 MACE_PB1D_GRAD_PASSES=2 MACE_PB1D_AREA_EPS=1e-12 PYTHONPATH=$CODE; unset MACE_PB1D_DFORCE
F="cuequivariance|UserWarning|return torch|warnings.warn"
echo "===== 1. pair dump ====="
$PY -u $BASE/claude/2-1D_PB/tools1d/dump_structure_pair_v2.py $M pair_122.xyz 0 1 structure_pair_sid122_722.npz 2>&1 | grep --line-buffered -vE "$F"; echo "rc_dump=${PIPESTATUS[0]}"
echo "===== 2. ML CHGCAR charged (sid 122) ====="
$PY -u build_chgcar_prod500.py 0 /scratch/08384/tg876840/tmp/2-NiN_single/1-44_GCE/cal_122/CHGCAR $BASE/data/NiN-mix800/grid_cache_npy/density3d_net_grid_122.npy ml_sol_prod500 2>&1 | grep --line-buffered -vE "$F"; echo "rc_chg_c=${PIPESTATUS[0]}"
echo "===== 3. ML CHGCAR neutral (sid 722) ====="
$PY -u build_chgcar_prod500.py 1 /scratch/08384/tg876840/tmp/2-NiN_single/5-44_neutral_withsolv/cal_122/CHGCAR $BASE/data/NiN-mix800/grid_cache_npy/density3d_net_grid_722.npy ml_neutral_prod500 2>&1 | grep --line-buffered -vE "$F"; echo "rc_chg_n=${PIPESTATUS[0]}"
echo "===== 4. electron profiles ====="
$PY -u extract_electron_profiles_prod500.py electron_profiles_prod500.npz 2>&1 | grep --line-buffered -vE "$F"; echo "rc_prof=${PIPESTATUS[0]}"
echo "===== 5. fermi / potential of the 200 twin pairs ====="
$PY -u fermi_pairs_prod500.py fermi_pairs_prod500.npz 2>&1 | grep --line-buffered -vE "$F"; echo "rc_fermi=${PIPESTATUS[0]}"
echo "===== push ====="
bash $BASE/claude/2-1D_PB/exp_term_split/job_push_dev.sh 2>&1 | grep -E "unpushed|push exit|remote now"
echo "EXIT: 0   $(date)"
