#!/bin/bash
#SBATCH --job-name=neu594
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_neutral_rerun/bf.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_neutral_rerun/bf.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --partition=development
#SBATCH --account=DMR24028
set -euo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
echo "CODE VERSION: polar-mace-planar @ $(git -C $BASE/polar-mace-planar rev-parse HEAD)"
$PY -u $BASE/tools1d/extract_neutral_set.py \
  $BASE/exp_neutral_prep/neutral_draft.xyz $BASE/exp_neutral_prep/audit.tsv
$PY -u $BASE/tools1d/build_neutral_caches.py \
  $BASE/exp_neutral_prep/audit.tsv $BASE/exp_neutral_prep 16
$PY -u $BASE/exp_neutral_rerun/backfill_594.py
echo "chain exit: $?"
