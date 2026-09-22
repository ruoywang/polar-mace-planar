#!/bin/bash
# Non-SCF VASPsol band run (ICHARG=11) from the fully-ML CHGCAR of the prod500_w1000_ref model, sid 122
# solvated; inputs (INCAR/KPOINTS/POTCAR/POSCAR) copied from exp_band/sid122_bands/ml_sol, the DFT
# reference bands are exp_band/sid122_bands/dft_sol (unchanged). Same resources as the earlier run.
#SBATCH -J b122p5
#SBATCH -p gpu-a100-dev
#SBATCH -N 2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH -o logs/band.o%j
#SBATCH -e logs/band.e%j
#SBATCH --account=DMR24028
set -uo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/exp_pair122_prod500/ml_sol_prod500
[ -f CHGCAR ] || { echo "no CHGCAR"; exit 2; }
echo "START ml_sol_prod500 $(date +%H:%M:%S) CHGCAR $(stat -c %s CHGCAR) bytes"
mpirun -np $SLURM_NTASKS /work/08384/tg876840/ls6/CEP-DIP/bin/vasp_std > log.out 2>&1
echo "DONE rc=$? nk=$(awk 'NR==6{print $2}' EIGENVAL 2>/dev/null) iters=$(grep -cE '^(DAV|RMM)' log.out) E-fermi: $(grep E-fermi OUTCAR | tail -n 1)"
rm -f WAVECAR CHG
echo "EXIT: 0"
