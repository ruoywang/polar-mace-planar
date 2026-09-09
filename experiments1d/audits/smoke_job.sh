#!/bin/bash
#SBATCH --job-name=kitref
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/mini_kit_4080/run/kitref.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/mini_kit_4080/run/kitref.e%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --account=DMR24028
set -uo pipefail
K=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/mini_kit_4080
source $K/env.sh
cd $K/run
echo "CODE: $(git -C $KIT_MACE_REPO log --oneline -1 | cut -c1-70)"
/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python -u smoke_test.py --write-ref
echo "rc=$?"
echo "EXIT"
