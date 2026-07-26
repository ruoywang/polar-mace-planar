#!/bin/bash
#SBATCH --job-name=pb1dsmk
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_smoke/smoke.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_smoke/smoke.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --account=DMR24028
set -uo pipefail
cd /scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB/exp_pb1d_smoke
BASE=/scratch/08384/tg876840/tmp/c-MACEsol/2-1D_PB
PY=/scratch/08384/tg876840/tmp/c-MACEsol/.venv/bin/python
export OMP_NUM_THREADS=16

echo "=== [1/4] planar regression: clone (pb-1d work) vs base (ad0ee85) ==="
for run in clone:polar-mace-planar base:pmp-base base2:pmp-base; do
  nm="${run%%:*}"; tree="${run##*:}"
  rm -rf "reg_$nm"; mkdir -p "reg_$nm"
  ( cd "reg_$nm" && ln -sfn ../data data && ln -sfn ../data_small data_small \
    && cp ../cal1_train.json . \
    && PYTHONPATH="$BASE/$tree" $PY -u -m mace.cli.run_train \
       --config ../config_planar.yaml --name planar_reg --seed 123 \
       > run.log 2>&1 )
  echo "run $nm ($tree) exit: $?"
done
$PY - <<'EOF'
import re
def loss_of(path):
    m = re.findall(r"loss=([\d.e+-]+)", open(path).read())
    return float(m[0]) if m else float("nan")
a, b, b2 = (loss_of(f"reg_{n}/run.log") for n in ("clone", "base", "base2"))
jitter = abs(b - b2)
diff = abs(a - b)
print(f"clone {a!r} base {b!r} base2 {b2!r}")
print(f"clone-vs-base diff {diff:.3e}; base-vs-base jitter {jitter:.3e}")
verdict = "PASS (within GPU nondeterminism)" if diff <= max(10 * jitter, 1e-6 * abs(b)) else "FAIL"
print("PLANAR REGRESSION:", verdict)
EOF

echo "=== [2/4] pb1d smoke (2 epochs, tiny subset) ==="
rm -rf run_pb1d cache; mkdir -p run_pb1d cache
( cd run_pb1d && ln -sfn ../data data && ln -sfn ../data_small data_small \
  && ln -sfn ../cache cache && cp ../cal1_train.json . \
  && MACE_PB_DEBUG=1 PYTHONPATH="$BASE/polar-mace-planar" $PY -u -m mace.cli.run_train \
     --config ../config_pb1d.yaml --name pb1d_smoke --seed 123 \
     > run.log 2>&1 )
echo "pb1d exit: $?"
tail -30 run_pb1d/run.log
echo "--- cache files:"; ls cache | head; echo "--- PB1DDBG lines:"; grep -c "PB1DDBG" run_pb1d/run.log || true
grep "PB1DDBG-FALLBACK\|PB1DDBG-ERROR" run_pb1d/run.log | head -5 || true

echo "=== [3/4] FD force check ==="
MODEL=$(ls run_pb1d/*.model 2>/dev/null | head -1)
echo "model: $MODEL"
MACE_PB1D_NO_GUARD=1 PYTHONPATH="$BASE/polar-mace-planar" $PY -u "$BASE/tools1d/test_pb1d_forces.py" \
  "$MODEL" data_small/train_small.xyz 4 2>&1 | tail -12 || echo "FD script failed"

echo "=== [4/4] done ==="
