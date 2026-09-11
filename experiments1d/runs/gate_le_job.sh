#!/bin/bash
#SBATCH --job-name=s3dgle
#SBATCH -o logs/gle.o%j
#SBATCH -e logs/gle.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --partition=gpu-a100-dev,gpu-a100-small,gpu-a100
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol
CODE=$BASE/claude/2-1D_PB/pmp-s3denergy
cd $BASE/3-residual_3D/gate_le
echo "CODE: $(git -C $CODE log --oneline -1 | cut -c1-80)"
echo "CONFIG DIFF vs the gate_bl reference run:"
diff $BASE/3-residual_3D/gate_bl/config_pb1d.yaml config_pb1d.yaml | sed 's/^/    /'
echo "REFERENCE (gate_bl epoch 33, from scratch, 34 epochs, same seed 123):"
echo "    RMSE_E_per_atom 11.75 meV, RMSE_F 32.75 meV/A, RMSE_potential 0.1227 eV"
timeout 10200 srun -n 3 env PYTHONPATH="$CODE" $BASE/.venv/bin/python -u -m mace.cli.run_train \
  --config config_pb1d.yaml --name s3d_gate_le --seed 123 \
  --distributed --launcher slurm \
  --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results > run.log 2>&1
rc=$?
echo "run_train rc=$rc"
echo "--- parameter count (the native head's size is the whole point) ---"
grep -iE "number of parameters|total number of parameters" run.log | head -4
echo "--- local_electron_energy actually constructed? ---"
grep -icE "local_electron" run.log
echo "--- first and last epochs ---"
grep -E "INFO: Epoch" run.log | head -2 | cut -c1-200
grep -E "INFO: Epoch" run.log | tail -2 | cut -c1-200
echo "rolling ckpts: $(ls checkpoints/ 2>/dev/null | tr '\n' ' ')"
echo "EXIT: $rc"
