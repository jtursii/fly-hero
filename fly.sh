#!/usr/bin/env bash
# Fly Brain Hero — training controls. Run from the project folder.
# Usage: ./fly.sh [check|awake|stop|resume|video|log]
set -uo pipefail

# Defaults to the newest bc_full_real run dir (not a hardcoded timestamp --
# a fresh overnight run gets a new timestamped dir every launch). Override
# with RUN=path/to/run ./fly.sh check
RUN="${RUN:-$(ls -td runs/*bc_full_real*/ 2>/dev/null | head -n1 | sed 's:/$::')}"
MODEL="${MODEL:-connectome}"

# make_run_dir() prefixes the run dir with a timestamp, but the *process*
# command line only ever shows the plain --run-name it was launched with
# (e.g. "bc_full_real", not "20260916_123616_bc_full_real") -- strip the
# "YYYYMMDD_HHMMSS_" prefix back off $RUN's basename to match it. Matches
# only train.bc's *python* process for this run name, never the uv wrapper
# (excluded below) and never scripts/render_gameplay_video.py (a different
# module path -- "train.bc" doesn't appear in its invocation).
RUN_NAME=$(basename "${RUN:-__none__}" | sed -E 's/^[0-9]{8}_[0-9]{6}_//')
PATTERN="train\.bc.*--run-name[= ]${RUN_NAME}\b"

find_pid() {
  [ -z "${RUN:-}" ] && return
  ps -Ao pid,command | grep -E "$PATTERN" | grep -v grep | grep -v "uv run" | awk '{print $1}' | head -n1
}

require_run() {
  if [ -z "${RUN:-}" ]; then
    echo "No runs/*bc_full_real* directory found (and RUN not set)."
    exit 1
  fi
}

resume_cmd() {
  # --run-name is actually unused by bc.py's run-dir resolution once
  # --resume is set (it continues the dir --resume/--run-dir point at
  # instead) -- but it must still be the plain, timestamp-stripped name
  # ($RUN_NAME) so find_pid()'s PATTERN (built from the same value) can
  # find this process again.
  echo "uv run python -m train.bc --model $MODEL --run-name $RUN_NAME --resume latest --run-dir $RUN"
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
    echo "Running: $(resume_cmd)"
    nohup $(resume_cmd) >> "$RUN/resume.log" 2>&1 &
    echo "Starting... checking in 30s"
    sleep 30
    if [ -n "$(find_pid)" ]; then
      echo "✅ Resumed and running (pid $(find_pid))."
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
    echo "RUN defaults to the newest runs/*bc_full_real* dir; override with RUN=path ./fly.sh ..."
    ;;
esac
