#!/bin/bash
#SBATCH --job-name=neuext
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_neutral_prep/ext.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_neutral_prep/ext.e%j
#SBATCH -N 1 -n 1 --cpus-per-task=32 --time=0:40:00 --partition=development --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
$PY -u $BASE/tools1d/extract_neutral_set.py \
  $BASE/exp_neutral_prep/neutral_draft.xyz $BASE/exp_neutral_prep/audit.tsv
echo "exit: $?"
