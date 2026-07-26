#!/bin/bash
#SBATCH --job-name=pb1dprod
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_prod/prod.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_prod/prod.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --account=DMR24028
set -uo pipefail
DIR=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_prod
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
export OMP_NUM_THREADS=16
cd "$DIR"

CHAIN=$(cat chain_count 2>/dev/null || echo 0)
echo "=== chain link $CHAIN (job $SLURM_JOB_ID) ==="
if [ "$CHAIN" -ge 12 ]; then echo "chain limit reached, stopping"; exit 0; fi
echo $((CHAIN + 1)) > chain_count

# 1h50m budget, then clean stop so the checkpoint from the last finished epoch survives
MACE_PB_DEBUG=1 PYTHONPATH="$BASE/pmp-prod" \
  timeout 6600 $PY -u -m mace.cli.run_train \
  --config config_pb1d.yaml --name pb1d_prod --seed 123 >> run.log 2>&1
rc=$?
echo "run_train exit: $rc"

if [ $rc -eq 0 ] && grep -q "INFO: Done" run.log; then
  echo "TRAINING COMPLETE after $(grep -c 'Epoch' run.log) epoch lines"
else
  echo "not finished (rc=$rc), resubmitting chain link $((CHAIN + 1))"
  sbatch "$DIR/job.sh"
fi
