#!/bin/bash
#SBATCH --job-name=rhobref
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_rhob_labels/ref.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_rhob_labels/ref.e%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0:40:00
#SBATCH --partition=development
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
$PY -u $BASE/tools1d/build_solvent1d_ref.py \
  $BASE/train-data/dft_solvent1d_ref.npz \
  $BASE/train-data/train.xyz $BASE/train-data/val.xyz $BASE/train-data/test.xyz
echo "exit: $?"
