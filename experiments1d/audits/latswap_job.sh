#!/bin/bash
#SBATCH --job-name=s3dlatsw
#SBATCH -o /scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/exp_neutralsolv_prep/latsw.o%j
#SBATCH -e /scratch/08384/tg876840/tmp/c-MACEsol/claude/2-1D_PB/exp_neutralsolv_prep/latsw.e%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-a100-dev
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol
CODE=$BASE/claude/2-1D_PB/pmp-s3denergy
PB=/work/08384/tg876840/ls6/repos/cep-dip-python-pb
SCRIPT=$CODE/experiments1d/audits/lateral_bound_swap.py
# provenance CERTIFIED, not asserted: the repo copy is executed directly, and
# its sha256 is compared against the committed blob so the log proves which
# code ran rather than claiming it
echo "mace HEAD: $(git -C $CODE rev-parse --short HEAD)"
echo "pb   HEAD: $(git -C $PB rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "executed:  $SCRIPT"
echo "sha256:    $(sha256sum $SCRIPT | cut -d' ' -f1)"
COMMITTED=$(git -C $CODE show HEAD:experiments1d/audits/lateral_bound_swap.py | sha256sum | cut -d' ' -f1)
echo "committed: $COMMITTED"
if [ "$(sha256sum $SCRIPT | cut -d' ' -f1)" = "$COMMITTED" ]; then echo "PROVENANCE: IDENTICAL"; else echo "PROVENANCE: DIFFERS"; fi
cd $BASE/3-residual_3D/gate_bl
export KIT_FRAMES=all
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( while true; do
    echo "[mem $(date +%H:%M:%S)] rss_GB=$(awk '/VmRSS/{printf "%.2f", $2/1048576}' /proc/self/status 2>/dev/null) \
avail_GB=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo) \
gpu_MiB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
    sleep 30
  done ) &
SAMP=$!
$BASE/.venv/bin/python -u $SCRIPT
rc=$?
kill $SAMP 2>/dev/null
echo "rc=$rc"
echo "EXIT"
