#!/bin/bash
# One unconverged neutral calc, resubmit-ready. Adjust partition/account
# to the target machine (original runs: 5 nodes x 16 ranks, ~19 min each;
# a single modern node should do). Run inside cal_<N>/.
#SBATCH --job-name=neu_rerun
#SBATCH -o rerun.o%j
#SBATCH -e rerun.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=4:00:00
##SBATCH --partition=<FILL>
##SBATCH --account=<FILL>
module load vasp   # or the site-specific module / binary path
ibrun vasp_std     # TACC; use srun/mpirun elsewhere
