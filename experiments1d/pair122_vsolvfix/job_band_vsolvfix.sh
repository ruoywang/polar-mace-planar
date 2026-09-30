#!/bin/bash
# Non-SCF VASPsol band run (ICHARG=11) from the fully-ML CHGCAR of prod500_vsolv_fix, sid 122 solvated.
# Inputs (INCAR/KPOINTS/POTCAR/POSCAR) copied from exp_pair122_prod500/ml_sol_prod500 (= exp_band/sid122_bands/ml_sol);
# the DFT reference bands are exp_band/sid122_bands/dft_sol (unchanged). CPU build of VASP -> development partition.
#SBATCH -J b122vf
#SBATCH -p development
#SBATCH -N 2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH -o logs/band.o%j
#SBATCH -e logs/band.e%j
#SBATCH --account=DMR24028
set -uo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/exp_pair122_vsolvfix/ml_sol_vsolvfix
[ -f CHGCAR ] || { echo "no CHGCAR"; echo "EXIT: 2"; exit 2; }
echo "START ml_sol_vsolvfix $(date +%H:%M:%S) node $(hostname) CHGCAR $(stat -c %s CHGCAR) bytes"
mpirun -np $SLURM_NTASKS /work/08384/tg876840/ls6/CEP-DIP/bin/vasp_std > log.out 2>&1
rc=$?
echo "DONE rc=$rc nk=$(awk 'NR==6{print $2}' EIGENVAL 2>/dev/null) iters=$(grep -cE '^(DAV|RMM)' log.out) E-fermi: $(grep E-fermi OUTCAR | tail -n 1)"
rm -f WAVECAR CHG
echo "EXIT: $rc"
