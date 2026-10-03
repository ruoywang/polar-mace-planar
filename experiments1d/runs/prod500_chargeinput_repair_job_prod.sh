#!/bin/bash
# PRODUCTION (reviewer plan 2026-10-03): prod500_vsolv_fix setting + charge-state inputs to the 3-D solvent head + unified full-field repair.
# Same seed / warm-up / weights / data split / EMA as prod500_vsolv_fix (seed 123, warm-up 20, energy 1000 / forces 100, 3 ranks).
# local-electron head, NO charge branch, energy_weight 1000 / forces 100, warm-up 20
# (solvent_pb1d_warmup_encounters), fast-B derivatives (LIVE_POS=1 GRAD_PASSES=2
# AREA_EPS=1e-12), density-grid RAM preload, seed 123, 3 ranks.
#
# Segmented running (2026-09-29). Each job resumes from the segment state and stops at an epoch
# boundary before its wall limit. Several jobs may wait at the same time (e.g. a gpu-a100 job and a
# chain of 2 h gpu-a100-dev jobs, user 2026-09-29): a lock in checkpoints/segments/.run_lock lets only
# one of them train; any other that starts while the holder is alive exits at once without touching
# the run, and every job exits at once when run.log already says "Training complete".
# Budget: pass WALL (the job's wall limit in s). The script reads the next epoch from the last
# "Segment state written" line and plans with the measured costs (clock start ~94-160 s after job start,
# resume to first epoch ~180 s, PB epoch 375-398 s, post-training save + evaluation 1281 s):
#   if the rest (300 s + R x 400 s + 1350 s) fits in WALL - 460 s -> BUDGET = WALL - 460 - 1350 (the job
#   finishes, and the post-training phase is guaranteed its time); otherwise BUDGET = min(WALL - 460,
#   150 + (R + 0.15) x 360 - 1), which stops at least one epoch before the end so no job is killed
#   inside the post-training phase. The stop rule in train.py is elapsed + 1.15 x last_epoch > budget.
# LS6 refuses sbatch from compute nodes, so continuations are submitted from a login node:
#   sbatch --time=02:00:00 -p gpu-a100-dev --export=ALL,RESUME=checkpoints/segments/prod500_chargeinput_repair_run-123_segment_state.pt,WALL=7200 job_prod.sh
#   sbatch --time=08:00:00 --export=ALL,RESUME=checkpoints/segments/prod500_chargeinput_repair_run-123_segment_state.pt,WALL=28800 job_prod.sh
# DRYRUN=1 runs the checks, the lock and the plan, prints them and exits before srun.
#SBATCH --job-name=p500cr
#SBATCH -o logs/prod.o%j
#SBATCH -e logs/prod.e%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-a100
#SBATCH --account=DMR24028
set -uo pipefail
BASE=/scratch/08384/tg876840/tmp/c-MACEsol; CODE=$BASE/claude/2-1D_PB/pmp-s3denergy
cd ${RUN_DIR:-$BASE/3-residual_3D/prod500_chargeinput_repair}
RESUME=${RESUME:-}; WALL=${WALL:-}; BUDGET=${BUDGET:-}; DRYRUN=${DRYRUN:-0}; JID=${SLURM_JOB_ID:-dry$$}
MAXEP=500; SEGSTATE=checkpoints/segments/prod500_chargeinput_repair_run-123_segment_state.pt
echo "CODE: $(git -C $CODE log --oneline -1 | cut -c1-80)   RESUME=${RESUME:-none}   WALL=${WALL:-unset}   job $JID   node $(hostname)   $(date)"

# 0. nothing left to do
if grep -q "Training complete" run.log 2>/dev/null; then echo "NOTHING TO DO: run.log already says Training complete"; echo "EXIT: 0"; exit 0; fi

# 1. one training job at a time
LOCK=checkpoints/segments/.run_lock; mkdir -p checkpoints/segments; got=0
for attempt in 1 2 3; do
  if mkdir "$LOCK" 2>/dev/null; then echo "$JID $(hostname) $(date +%s)" > "$LOCK/holder"; got=1; break; fi
  hid=$(awk '{print $1}' "$LOCK/holder" 2>/dev/null); hid=${hid:-unknown}
  if q=$(squeue -h -u "${USER:-tg876840}" -o "%i %T" 2>/dev/null); then
    hstate=$(echo "$q" | awk -v j="$hid" '$1==j{print $2}'); src=squeue
    alive=0; case "$hstate" in RUNNING|COMPLETING|CONFIGURING) alive=1;; esac
  else
    age=$(( $(date +%s) - $(stat -c %Y run.log 2>/dev/null || echo 0) )); src="run.log age ${age} s"
    alive=0; [ "$age" -lt 1200 ] && alive=1
  fi
  if [ $alive = 1 ]; then echo "LOCKED: job $hid holds the run ($src: ${hstate:-alive}); this job exits without training"; echo "EXIT: 0"; exit 0; fi
  echo "STALE LOCK: holder $hid is not running ($src: ${hstate:-none}); taking over"
  mv "$LOCK" "$LOCK.stale.$JID" 2>/dev/null   # rename is atomic: only one taker wins, the other retries
done
if [ $got != 1 ]; then echo "LOCKED: could not take the lock in 3 attempts; exiting without training"; echo "EXIT: 0"; exit 0; fi
trap '[ "$(awk "{print \$1}" "$LOCK/holder" 2>/dev/null)" = "$JID" ] && rm -rf "$LOCK"' EXIT
trap 'exit 143' TERM

# 2. guards
if [ -z "$RESUME" ] && [ -n "$(ls checkpoints/segments/ 2>/dev/null)" ]; then echo "REFUSING: no RESUME given but segment states exist -- this would restart from scratch"; echo "EXIT: 3"; exit 3; fi
if [ -n "$RESUME" ] && [ ! -f "$RESUME" ]; then echo "REFUSING: RESUME=$RESUME does not exist"; echo "EXIT: 3"; exit 3; fi
EXTRA=""; [ -n "$RESUME" ] && EXTRA="--resume_state $RESUME"

# 3. budget plan
NEXT=$(grep -oE "Segment state written: .*next epoch [0-9]+" run.log 2>/dev/null | grep -oE "next epoch [0-9]+" | tail -1 | awk '{print $3}'); NEXT=${NEXT:-0}
[ -z "$RESUME" ] && NEXT=0
R=$(( MAXEP - NEXT ))
if [ -n "$BUDGET" ]; then PLAN="given"
elif [ -n "$WALL" ]; then
  EMAX=$(( WALL - 160 - 300 ))
  if [ $(( 300 + R * 400 + 1350 )) -le $EMAX ]; then BUDGET=$(( EMAX - 1350 )); PLAN="finish"
  else CAP=$(awk -v r=$R 'BEGIN{printf "%d", 150 + (r + 0.15) * 360 - 1}'); BUDGET=$(( CAP < EMAX ? CAP : EMAX )); PLAN="partial"; fi
else BUDGET=169200; PLAN="default 47 h"; fi
echo "PLAN: next epoch $NEXT, remaining $R, WALL ${WALL:-unset}, budget $BUDGET s ($PLAN)"
if [ "$DRYRUN" = 1 ]; then echo "DRYRUN: lock held by $JID; would run with --segment_time_budget $BUDGET $EXTRA"; echo "EXIT: 0"; exit 0; fi

# 4. train
export MACE_PB1D_LIVE_POS=1 MACE_PB1D_GRAD_PASSES=2 MACE_PB1D_AREA_EPS=1e-12; unset MACE_PB1D_DFORCE
srun -n 3 bash -c 'echo "rank $SLURM_PROCID LIVE_POS=${MACE_PB1D_LIVE_POS:-unset} GRAD_PASSES=${MACE_PB1D_GRAD_PASSES:-unset(1)} DFORCE=${MACE_PB1D_DFORCE:-unset} AREA_EPS=${MACE_PB1D_AREA_EPS:-unset}"'
t0=$(date +%s)
srun -n 3 env PYTHONPATH="$CODE" $BASE/.venv/bin/python -u -m mace.cli.run_train \
  --config config_pb1d.yaml --name prod500_chargeinput_repair --seed 123 --max_num_epochs $MAXEP \
  --distributed --launcher slurm --work_dir . --log_dir logs --model_dir models \
  --checkpoints_dir checkpoints --results_dir results \
  --segment_time_budget $BUDGET $EXTRA >> run.log 2>&1
rc=$?
echo "run_train rc=$rc  wall $(( $(date +%s) - t0 )) s"
grep -E "Resumed segment state|Segment state written|time budget|Segment complete|Training complete|Traceback" run.log | tail -6 | cut -c1-200
grep -E "INFO: Epoch [0-9]+:" run.log | tail -2 | cut -c1-160
if [ "$rc" = 0 ] && ! grep -q "Training complete" run.log && [ -f $SEGSTATE ]; then
  echo "SEGMENT STOPPED, NOT COMPLETE: next segment must be queued from a login node (see header)"
fi
echo "EXIT: $rc"
