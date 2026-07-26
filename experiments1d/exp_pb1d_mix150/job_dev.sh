#!/bin/bash
#SBATCH --job-name=pb1dmix
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_mix150/mix.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_mix150/mix.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --account=DMR24028
# phase 0: byte-identity probe (new code, flag-less data == old code)
# phase 1: 5-epoch gate on the mixed bundle; phase 2: 150-epoch training
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
cd "$BASE/exp_pb1d_mix150"
if grep -q "INFO: Done" run.log 2>/dev/null; then echo "ALREADY COMPLETE"; exit 0; fi
SKIP_PREP=0
[ -f probe_ok ] && grep -q "INFO: Done" gate/gate.log 2>/dev/null && SKIP_PREP=1
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16
export MACE_PB1D_TIMING=1 MACE_PB1D_TIMING_EVERY=200

if [ "$SKIP_PREP" = 0 ]; then
cd "$BASE/exp_pb1d_rhob400a"   # old bundle + caches for the identity probe
for RUN in old:pmp-rhob80 old2:pmp-rhob80 new:pmp-mix; do
  TAG="${RUN%%:*}"; TREE="${RUN##*:}"
  env PYTHONPATH="$BASE/$TREE" $PY -u $BASE/tools1d/probe_solvated_identity.py \
    models/pb1d_rhob400a.model $BASE/train-data/val.xyz 0 \
    "$BASE/exp_pb1d_mix150/probe_$TAG.npz" >> "$BASE/exp_pb1d_mix150/probe.log" 2>&1
done
if ! $PY - <<'PYEOF'
import numpy as np, sys
d = "/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_mix150"
a = np.load(f"{d}/probe_old.npz"); a2 = np.load(f"{d}/probe_old2.npz"); b = np.load(f"{d}/probe_new.npz")
jit_ = {k: float(np.abs(a[k]-a2[k]).max()) for k in a.files}   # run-to-run GPU jitter, old code
diff = {k: float(np.abs(a[k]-b[k]).max()) for k in a.files}    # old vs new code
print("gpu jitter (old vs old):", jit_)
print("old vs new:", diff)
ok = all(diff[k] <= max(10.0 * jit_[k], 1e-8) for k in a.files)
print("IDENTITY-OK (within jitter)" if ok else "IDENTITY-FAIL (beyond jitter)")
sys.exit(0 if ok else 1)
PYEOF
then
  echo "PROBE FAILED - difference beyond run-to-run jitter"; exit 1
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
touch "$BASE/exp_pb1d_mix150/probe_ok"
fi

cd "$BASE/exp_pb1d_mix150"
timeout 6300 srun -n 3 env PYTHONPATH="$BASE/pmp-mix" $PY -u -m mace.cli.run_train \
  --config config_pb1d.yaml --name pb1d_mix150 --seed 123 \
  --distributed --launcher slurm \
  --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results >> run.log 2>&1
rc=$?
echo "run_train exit: $rc"
if [ $rc -eq 0 ] && grep -q "INFO: Done" run.log; then echo "TRAINING COMPLETE"; else echo "LINK ENDED (exit $rc) - resume next link"; fi
