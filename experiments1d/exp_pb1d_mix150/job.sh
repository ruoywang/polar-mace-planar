#!/bin/bash
#SBATCH --job-name=pb1dmix
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_mix150/mix.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_mix150/mix.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH --partition=gpu-a100
#SBATCH --account=DMR24028
# phase 0: byte-identity probe (new code, flag-less data == old code)
# phase 1: 5-epoch gate on the mixed bundle; phase 2: 150-epoch training
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16
export MACE_PB1D_TIMING=1 MACE_PB1D_TIMING_EVERY=200

cd "$BASE/exp_pb1d_rhob400a"   # old bundle + caches for the identity probe
for TREE in pmp-rhob80 pmp-mix; do
  env PYTHONPATH="$BASE/$TREE" $PY -u $BASE/tools1d/probe_solvated_identity.py \
    models/pb1d_rhob400a.model $BASE/train-data/val.xyz 0 \
    "$BASE/exp_pb1d_mix150/probe_$TREE.npz" >> "$BASE/exp_pb1d_mix150/probe.log" 2>&1
done
$PY - <<'PYEOF'
import numpy as np
a = np.load("/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_mix150/probe_pmp-rhob80.npz")
b = np.load("/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_mix150/probe_pmp-mix.npz")
bad = [k for k in a.files if not np.array_equal(a[k], b[k])]
worst = {k: float(np.abs(a[k]-b[k]).max()) for k in bad}
print("IDENTITY-OK" if not bad else f"IDENTITY-FAIL: {worst}")
PYEOF
if ! $PY -c "
import numpy as np
a=np.load('$BASE/exp_pb1d_mix150/probe_pmp-rhob80.npz'); b=np.load('$BASE/exp_pb1d_mix150/probe_pmp-mix.npz')
assert all(np.array_equal(a[k],b[k]) for k in a.files)"; then
  echo "PROBE FAILED - not byte-identical"; exit 1
fi
echo "PROBE PASSED"

cd "$BASE/exp_pb1d_mix150/gate"
timeout 2400 srun -n 3 env PYTHONPATH="$BASE/pmp-mix" $PY -u -m mace.cli.run_train \
  --config config_gate.yaml --name pb1d_mixgate --seed 123 \
  --distributed --launcher slurm \
  --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results >> gate.log 2>&1
rc=$?
echo "gate exit: $rc"
if [ $rc -ne 0 ] || ! grep -q "INFO: Done" gate.log || grep -qE "Traceback|NaN detected" gate.log; then
  echo "GATE FAILED - training not started"; exit 1
fi
echo "GATE PASSED"

cd "$BASE/exp_pb1d_mix150"
srun -n 3 env PYTHONPATH="$BASE/pmp-mix" $PY -u -m mace.cli.run_train \
  --config config_pb1d.yaml --name pb1d_mix150 --seed 123 \
  --distributed --launcher slurm \
  --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results >> run.log 2>&1
rc=$?
echo "run_train exit: $rc"
if [ $rc -eq 0 ] && grep -q "INFO: Done" run.log; then echo "TRAINING COMPLETE"; else echo "TRAINING DID NOT FINISH (exit $rc)"; fi
