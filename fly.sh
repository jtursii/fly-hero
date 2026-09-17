#!/usr/bin/env bash
# Fly Brain Hero — training controls. Run from the project folder.
# Usage: ./fly.sh [check|awake|stop|resume|video|log]
set -uo pipefail

# Defaults to the newest main-lineage run dir: runs/<timestamp>_bc_full_real
# or runs/<timestamp>_bc_full_real_vN, newest by the timestamp in its name
# (not mtime, and never throwaway runs like smoke_d32/resume_test_v2).
# Override with RUN=path/to/run ./fly.sh check
RUN="${RUN:-$(ls -d runs/*/ 2>/dev/null | sed 's:/$::' | grep -E '/[0-9]{8}_[0-9]{6}_bc_full_real(_v[0-9]+)?$' | sort | tail -n1)}"
RUN="${RUN%/}"
MODEL="${MODEL:-connectome}"
# Extra train.bc flags for resume (e.g. EXTRA_ARGS="--device cpu" for tests).
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_NAME=$(basename "${RUN:-__none__}" | sed -E 's/^[0-9]{8}_[0-9]{6}_//')

# D33: the pid of the train.bc *python* process (never the uv wrapper, never
# scripts/render_gameplay_video.py) that is writing to exactly $RUN: either
# it was launched with --run-dir $RUN, or its pid is in the name of a
# TensorBoard event file inside $RUN/tb (events.out.tfevents.<t>.<host>.<pid>.N
# -- covers fresh/--init-from launches, whose command line only names the
# run, not the dir; such a launch is invisible for its first ~minute, until
# the writer is created). Name matching alone would confuse bc_full_real
# with bc_full_real_v2 or a throwaway fork.
find_pid() {
  [ -z "${RUN:-}" ] && return
  [ -d "$RUN" ] || return
  RUN_ABS=$(cd "$RUN" && pwd -P)
  ps -Ao pid=,command= | python3 -c '
import glob, os, sys
run_abs = sys.argv[1]
tb_pids = {os.path.basename(f).split(".")[-2] for f in glob.glob(os.path.join(run_abs, "tb", "events.out.tfevents.*"))}
for line in sys.stdin:
    parts = line.split()
    if len(parts) < 3:
        continue
    pid, argv = parts[0], parts[1:]
    if "python" not in os.path.basename(argv[0]):
        continue
    if not any(a == "-m" and b == "train.bc" for a, b in zip(argv, argv[1:])):
        continue
    run_dir = next((b for a, b in zip(argv, argv[1:]) if a == "--run-dir"), None)
    if run_dir is not None:
        if os.path.realpath(run_dir.rstrip("/")) == run_abs:
            print(pid); break
        continue
    if pid in tb_pids:
        print(pid); break
' "$RUN_ABS"
}

ckpt_step() {  # step number of checkpoint_latest.pt, from its target's file name
  readlink "$RUN/checkpoint_latest.pt" 2>/dev/null | sed -E 's/^checkpoint_step([0-9]+)\.pt$/\1/'
}

require_run() {
  if [ -z "${RUN:-}" ]; then
    echo "No runs/*bc_full_real* directory found (and RUN not set)."
    exit 1
  fi
}

resume_cmd() {
  # Always --resume latest --run-dir: continues $RUN in place from its own
  # newest checkpoint. Never --init-from -- for bc_full_real_v2 that would
  # re-fork from step 13000 and discard everything since (guarded below and
  # in train/bc.py). --run-name is informational once --resume is set.
  echo "uv run python -u -m train.bc --model $MODEL --run-name $RUN_NAME --resume latest --run-dir $RUN $EXTRA_ARGS"
}

case "${1:-}" in
  check)
    require_run
    PID=$(find_pid)
    if [ -n "$PID" ]; then
      ETIME=$(ps -o etime= -p "$PID" | tr -d ' ')
      echo "✅ Training is RUNNING (pid: $PID, up: $ETIME)"
    else
      echo "❌ No training process found"
    fi
    LATEST=$(ls -t "$RUN" 2>/dev/null | head -n 1)
    [ -n "$LATEST" ] && echo "last file written: $LATEST ($(( ($(date +%s) - $(stat -f %m "$RUN/$LATEST")) / 60 )) min ago)"
    scripts/status.sh "$RUN"
    echo "--- training time (active; gaps > 5 min not counted) ---"
    uv run --quiet python scripts/training_time.py "$RUN" 2>/dev/null || echo "(training time unavailable)"

    if [ -L "$RUN/checkpoint_latest.pt" ]; then
      CKPT_TARGET="$RUN/$(readlink "$RUN/checkpoint_latest.pt")"
      if [ -f "$CKPT_TARGET" ]; then
        MIN_AGO=$(( ($(date +%s) - $(stat -f %m "$CKPT_TARGET")) / 60 ))
        echo "checkpoint age: $MIN_AGO min ago ($(basename "$CKPT_TARGET"))"
      fi
    fi

    if [ -f "$RUN/metrics.jsonl" ]; then
      echo "--- last 5 eval lines ---"
      (grep '"eval_hit_rate"' "$RUN/metrics.jsonl" || true) | tail -n 5 | python3 -c "
import json, sys
lines = sys.stdin.readlines()
if not lines:
    print('(no eval yet)')
for line in lines:
    r = json.loads(line)
    print('step=%s difficulty=%s hit_rate=%.4f overstrums/min=%.1f' % (
        r.get('step'), r.get('eval_difficulty'), r.get('eval_hit_rate', 0), r.get('eval_overstrums_per_min', 0)))
"
      echo "--- watchdog (overnight safeguards, D30/D31) ---"
      tail -n 2000 "$RUN/metrics.jsonl" | python3 -c "
import json, sys
last_train, rollbacks, best_by_diff = None, [], {}
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if 'rolling_median_grad_norm' in r:
        last_train = r
    if r.get('rollback'):
        rollbacks.append(r)
    if 'eval_hit_rate' in r:
        d = r.get('eval_difficulty')
        if d is not None and r['eval_hit_rate'] > best_by_diff.get(d, (-1, None))[0]:
            best_by_diff[d] = (r['eval_hit_rate'], r.get('step'))

if last_train:
    median = last_train.get('rolling_median_grad_norm')
    print('rolling median grad_norm (last %d steps): %s' % (200, ('%.3f' % median) if median is not None else 'n/a (warming up)'))
    print('skip rate (cumulative): %.3f%%  rollback count: %s' % (
        100 * last_train.get('watchdog_skip_rate', 0), last_train.get('rollback_count', 0)))
else:
    print('no training records yet')
for d, (hr, step) in best_by_diff.items():
    print('best hit_rate @ %s: %.4f (step %s)' % (d, hr, step))
if rollbacks:
    print('rollback events:')
    for r in rollbacks:
        print('  step=%s reason=%s new_lr_brain=%.2e restored_from=%s' % (
            r.get('step'), r.get('rollback_reason'), r.get('new_lr_brain', 0), r.get('restored_from')))
"
    fi

    if [ -f "$RUN/STOPPED.txt" ]; then
      echo "🛑 STOPPED.txt present:"
      cat "$RUN/STOPPED.txt"
    fi

    FREE_GB=$(df -g . 2>/dev/null | tail -1 | awk '{print $4}')
    if [ -n "$FREE_GB" ] && [ "$FREE_GB" -lt 20 ]; then
      echo "⚠️  low disk space: ${FREE_GB}GB free (<20GB)"
    fi
    ;;
  awake)
    echo "Keeping Mac awake. Leave this window open. Ctrl+C to allow sleep again."
    caffeinate -dimsu
    ;;
  stop)
    require_run
    PID=$(find_pid)
    [ -z "$PID" ] && { echo "Nothing running."; exit 0; }
    kill -TERM "$PID" && echo "SIGTERM sent to $PID (the python process, not the uv wrapper). Waiting for checkpoint..."
    for i in $(seq 1 60); do
      [ -z "$(find_pid)" ] && { echo "✅ Stopped cleanly."; exit 0; }
      sleep 2
    done
    echo "⚠️ Still running after 2 min. Run ./fly.sh check."
    ;;
  resume)
    require_run
    if [ -n "$(find_pid)" ]; then
      echo "Already running (pid $(find_pid)). Not starting a second copy."
      exit 1
    fi
    if [ ! -L "$RUN/checkpoint_latest.pt" ]; then
      echo "No checkpoint_latest.pt in $RUN -- nothing to resume from."
      exit 1
    fi
    CMD="$(resume_cmd)"
    case "$CMD" in
      *--init-from*) echo "Refusing: resume must never use --init-from (it would re-fork and reset the step)."; exit 1 ;;
      *"--resume latest --run-dir $RUN "*) ;;
      *) echo "Refusing: resume command is not '--resume latest --run-dir $RUN': $CMD"; exit 1 ;;
    esac
    EXPECT_STEP=$(ckpt_step)
    echo "Running: $CMD"
    echo "Expecting to resume at step $EXPECT_STEP (checkpoint_latest.pt)"
    LOG_START=0; [ -f "$RUN/resume.log" ] && LOG_START=$(wc -l < "$RUN/resume.log")
    nohup $CMD >> "$RUN/resume.log" 2>&1 &
    echo "Starting... checking for up to 90s"
    GOT_STEP=""
    for i in $(seq 1 45); do
      sleep 2
      GOT_STEP=$(tail -n +"$((LOG_START + 1))" "$RUN/resume.log" 2>/dev/null | sed -nE 's/^resumed from .* at step=([0-9]+) .*/\1/p' | head -n1)
      [ -n "$GOT_STEP" ] && break
    done
    if [ -n "$GOT_STEP" ] && [ "$GOT_STEP" != "$EXPECT_STEP" ]; then
      echo "❌ Resumed at step $GOT_STEP, expected $EXPECT_STEP -- stopping it."
      P=$(find_pid); [ -n "$P" ] && kill -TERM "$P"
      exit 1
    fi
    if [ -n "$(find_pid)" ] && [ -n "$GOT_STEP" ]; then
      echo "✅ Resumed at step $GOT_STEP and running (pid $(find_pid))."
      tail -n +"$((LOG_START + 1))" "$RUN/resume.log" | grep '^\[watchdog\] restored' || true
    else
      echo "❌ Failed to start. Last log lines:"; tail -n 15 "$RUN/resume.log"
      echo "Paste this output to Claude."
    fi
    ;;
  video)
    shift
    PYTHONPATH=. uv run python scripts/render_gameplay_video.py --run "$RUN" "$@"
    ;;
  log)
    require_run
    tail -n 20 "$RUN/resume.log" 2>/dev/null || echo "No resume log yet."
    ;;
  *)
    echo "./fly.sh check           → is it running + progress + last 5 evals + checkpoint age"
    echo "./fly.sh awake           → keep Mac awake (leave window open)"
    echo "./fly.sh stop            → stop cleanly (SIGTERM, saves checkpoint)"
    echo "./fly.sh resume          → resume from the latest checkpoint (refuses if already running)"
    echo "./fly.sh video [song] [difficulty] → render a gameplay video at the current checkpoint's skill"
    echo "./fly.sh log             → last lines of the resume log"
    echo ""
    echo "RUN defaults to the newest runs/<timestamp>_bc_full_real[_vN] dir; override with RUN=path ./fly.sh ..."
    ;;
esac
