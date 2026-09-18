#!/usr/bin/env bash
# Fly Brain Hero — training controls. Run from the project folder.
# Usage: ./fly.sh [check|awake|stop|resume [latest|best]|guard|video|log]
set -uo pipefail

# Defaults to the newest main-lineage run dir: runs/<timestamp>_bc_full_real,
# runs/<timestamp>_bc_full_real_vN, runs/<timestamp>_bc_trunc_bptt (D40) or
# runs/<timestamp>_bc_mid_labels (D41),
# newest by the timestamp in its name
# (not mtime, and never throwaway runs like smoke_d32/resume_test_v2).
# Override with RUN=path/to/run ./fly.sh check
RUN="${RUN:-$(ls -d runs/*/ 2>/dev/null | sed 's:/$::' | grep -E '/[0-9]{8}_[0-9]{6}_(bc_full_real(_v[0-9]+)?|bc_trunc_bptt|bc_mid_labels)$' | sort -t/ -k2 | tail -n1)}"
RUN="${RUN%/}"
MODEL="${MODEL:-connectome}"
# Extra train.bc flags for resume (e.g. EXTRA_ARGS="--device cpu" for tests,
# or a one-shot EXTRA_ARGS="--set-lr-brain 2.5e-5").
EXTRA_ARGS="${EXTRA_ARGS:-}"
# Config for resume. Unset: train.bc reuses $RUN/train_config.txt (D36).
CONFIG="${CONFIG:-}"
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

# Battery protection (guard). Prints "<source> <pct>": source is battery,
# ac, or unknown (no battery line, e.g. a desktop, or unparseable output);
# pct is empty when unknown. PMSET_OUTPUT overrides the real
# `pmset -g batt` text (tests / dry runs).
BATTERY_STOP_PCT="${BATTERY_STOP_PCT:-5}"
battery_state() {
  local out src pct
  out="${PMSET_OUTPUT-$(pmset -g batt 2>/dev/null)}"
  case "$out" in
    *"drawing from 'Battery Power'"*) src=battery ;;
    *"drawing from 'AC Power'"*) src=ac ;;
    *) src=unknown ;;
  esac
  pct=$(printf '%s\n' "$out" | grep -m1 'InternalBattery' | grep -oE '[0-9]+%' | head -n1 | tr -d '%')
  [ -z "$pct" ] && src=unknown
  echo "$src $pct"
}
# Exit 0 = stop now: on battery power at or below BATTERY_STOP_PCT. Never on
# AC, at any level, and never when the state can't be read.
battery_should_stop() {
  local src pct
  read -r src pct <<< "$(battery_state)"
  [ "$src" = battery ] && [ -n "$pct" ] && [ "$pct" -le "$BATTERY_STOP_PCT" ]
}

ckpt_step() {  # step number of checkpoint_<latest|best>.pt, from its target's file name
  readlink "$RUN/checkpoint_${1:-latest}.pt" 2>/dev/null | sed -E 's/^checkpoint_step([0-9]+)\.pt$/\1/'
}

require_run() {
  if [ -z "${RUN:-}" ]; then
    echo "No runs/*bc_full_real* directory found (and RUN not set)."
    exit 1
  fi
}

resume_cmd() {
  # Always --resume <latest|best> --run-dir: continues $RUN in place from its
  # own newest (or best) checkpoint. Never --init-from -- for bc_full_real_v2
  # that would re-fork from step 13000 and discard everything since (guarded
  # below and in train/bc.py). --run-name is informational once --resume is
  # set. Resuming from best abandons later steps; train.bc repoints
  # checkpoint_latest.pt to the resumed checkpoint (files are kept).
  local cfg=""; [ -n "$CONFIG" ] && cfg="--config $CONFIG"
  echo "uv run python -u -m train.bc --model $MODEL --run-name $RUN_NAME --resume $1 --run-dir $RUN $cfg $EXTRA_ARGS"
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
    uv run --quiet python scripts/training_time.py "$RUN" --total-only 2>/dev/null || echo "total compute: unavailable"

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
last_train, rollbacks, best_by_diff, last_eval_ma = None, [], {}, None
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
    if r.get('resumed_from_step') is not None:
        x = r['resumed_from_step']
        rollbacks = [q for q in rollbacks if q.get('step', 0) <= x]
        best_by_diff = {d: v for d, v in best_by_diff.items() if (v[1] or 0) <= x}
    if 'eval_hit_rate' in r:
        d = r.get('eval_difficulty')
        if r.get('eval_ma') is not None:
            last_eval_ma = (d, r['eval_ma'], r.get('best_eval_ma'))
        if d is not None and r['eval_hit_rate'] > best_by_diff.get(d, (-1, None))[0]:
            best_by_diff[d] = (r['eval_hit_rate'], r.get('step'))

if last_train:
    median = last_train.get('rolling_median_grad_norm')
    print('rolling median grad_norm (last %d steps): %s' % (200, ('%.3f' % median) if median is not None else 'n/a (warming up)'))
    print('skip rate (cumulative): %.3f%%  rollback count: %s' % (
        100 * last_train.get('watchdog_skip_rate', 0), last_train.get('rollback_count', 0)))
    base = last_train.get('watchdog_baseline')
    lm = last_train.get('long_median_grad_norm')
    if base is not None:
        print('watchdog baseline (frozen): %.1f  trigger: 200-step median > %.0f for 200 steps  '
              'certify: 1000-step median (now %s) < %.0f' % (base, 3 * base, ('%.0f' % lm) if lm is not None else 'n/a', 2 * base))
    else:
        print('watchdog baseline: n/a (not logged by this process)')
    if 'lr_brain' in last_train:
        print('step %s  lr_brain: %.3e  lr_readout: %.3e' % (last_train['step'], last_train['lr_brain'], last_train['lr_readout']))
else:
    print('no training records yet')
for d, (hr, step) in best_by_diff.items():
    print('best hit_rate @ %s: %.4f (step %s)' % (d, hr, step))
if last_eval_ma is not None:
    print('eval 3-MA @ %s: %.4f (best %.4f; rollback if drop > 0.08; Hard at >= 0.75)' % (
        last_eval_ma[0], last_eval_ma[1], last_eval_ma[2] or float('nan')))
if rollbacks:
    print('rollback events:')
    for r in rollbacks:
        print('  step=%s reason=%s new_lr_brain=%.2e restored_from=%s' % (
            r.get('step'), r.get('rollback_reason'), r.get('new_lr_brain', 0), r.get('restored_from')))
"
    fi

    if [ -f "$RUN/metrics.jsonl" ] && grep -q '"drift_g_rel"\|"watch_R7_finite_frac"' "$RUN/metrics.jsonl"; then
      echo "--- fork criteria vs v2 @ 42000 (D40/D41) ---"
      tail -n 3000 "$RUN/metrics.jsonl" | python3 -c "
import json, sys
W = ('R7', 'R8', 'L1', 'L5', 'AN_multi_1')
# v2 @ 42000 on the same 10 in-training val excerpts (CPU, 2026-09-17).
BASE = json.load(open('docs/d40_baseline_42000.json'))
tr, ev = None, None
for line in sys.stdin:
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if 'watch_R7_finite_frac' in r:
        tr = r
    if 'drift_g_rel' in r:
        ev = r
if tr:
    print('finite raw grad frac: ' + '  '.join('%s %.2f' % (w, tr.get('watch_%s_finite_frac' % w, 0)) for w in W))
if ev:
    print('@%s |softplus(g)-g42k|/|g42k| = %.2f%% (pass > 1%%; v2 0.26%%)  tau %.2f%%  bias %.2f%%' % (
        ev['step'], 100 * ev['drift_g_rel'], 100 * ev['drift_tau_rel'], 100 * ev['drift_bias_rel']))
    print('adam |m|: ' + '  '.join('%s %.1e' % (w, ev.get('watch_%s_adam_m' % w, 0)) for w in W))
    print('tau (init 0.05): ' + '  '.join('%s %.5f' % (w, ev.get('watch_%s_tau' % w, 0)) for w in W))
    print('singles hit %.3f (42k %.3f)  extra_fret %s (42k %s)  no_strum %s (42k %s)  wrong_lane %s (42k %s)' % (
        ev.get('eval_single_hit_rate', 0), BASE['single_hit_rate'], ev.get('eval_single_extra_fret'), BASE['extra_fret'],
        ev.get('eval_single_no_strum'), BASE['no_strum'], ev.get('eval_single_wrong_lane'), BASE['wrong_lane']))
"
      tail -n 3000 "$RUN/metrics.jsonl" | python3 -c "
import json, sys
B = json.load(open('docs/d41_baseline.json'))
b, f = B['42000']['in_training_10'], B['45000']['in_training_10']
evs = [r for r in (json.loads(l) for l in sys.stdin if '\"eval_single_fret_at_note\"' in l)]
if evs:
    print('D41 per eval (10 excerpts; ref 42000 | D40 fork @45000):')
    print('  step    hit   nothing_held  fret@note  far ovr/min  hit first3s/after')
    for r in evs[-6:]:
        print('  %-6s %.3f  %5s          %.3f      %5.1f        %.3f / %.3f' % (r['step'], r['eval_hit_rate'], r.get('eval_single_nothing_held'),
              r['eval_single_fret_at_note'], r['eval_far_overstrums_per_min'], r['eval_hit_rate_first3s'], r['eval_hit_rate_after3s']))
    print('  42000  %.3f  %5s          %.3f      %5.1f        %.3f / %.3f' % (b['hit_rate_mean'], b['single_nothing_held'], b['single_fret_at_note'], b['far_overstrums_per_min'], b['hit_rate_first3s'], b['hit_rate_after3s']))
    print('  D40@45k %.3f %5s          %.3f      %5.1f        %.3f / %.3f' % (f['hit_rate_mean'], f['single_nothing_held'], f['single_fret_at_note'], f['far_overstrums_per_min'], f['hit_rate_first3s'], f['hit_rate_after3s']))
"
      echo "paired 28-excerpt val Δ (CPU, background): ./fly.sh compare <step>"
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
    # Tells `fly.sh guard` this stop was deliberate (cleared by resume).
    echo "stopped via fly.sh stop at $(date '+%Y-%m-%dT%H:%M:%S')" > "$RUN/STOPPED_BY_USER"
    kill -TERM "$PID" && echo "SIGTERM sent to $PID (the python process, not the uv wrapper). Waiting for checkpoint..."
    for i in $(seq 1 60); do
      [ -z "$(find_pid)" ] && { echo "✅ Stopped cleanly."; exit 0; }
      sleep 2
    done
    echo "⚠️ Still running after 2 min. Run ./fly.sh check."
    ;;
  resume)
    require_run
    FROM="${2:-latest}"
    case "$FROM" in latest|best) ;; *) echo "resume from must be latest or best, got: $FROM"; exit 1 ;; esac
    if [ -n "$(find_pid)" ]; then
      echo "Already running (pid $(find_pid)). Not starting a second copy."
      exit 1
    fi
    if [ ! -L "$RUN/checkpoint_$FROM.pt" ]; then
      echo "No checkpoint_$FROM.pt in $RUN -- nothing to resume from."
      exit 1
    fi
    CMD="$(resume_cmd "$FROM")"
    case "$CMD" in
      *--init-from*) echo "Refusing: resume must never use --init-from (it would re-fork and reset the step)."; exit 1 ;;
      *"--resume $FROM --run-dir $RUN "*) ;;
      *) echo "Refusing: resume command is not '--resume $FROM --run-dir $RUN': $CMD"; exit 1 ;;
    esac
    rm -f "$RUN/STOPPED_BY_USER"
    EXPECT_STEP=$(ckpt_step "$FROM")
    echo "Running: $CMD"
    echo "Expecting to resume at step $EXPECT_STEP (checkpoint_$FROM.pt)"
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
      tail -n +"$((LOG_START + 1))" "$RUN/resume.log" | grep -E '^\[(watchdog\] restored|lr\]|resume\])' || true
    else
      echo "❌ Failed to start. Last log lines:"; tail -n 15 "$RUN/resume.log"
      echo "Paste this output to Claude."
    fi
    ;;
  guard)
    # Crash auto-resume. Run in its own terminal window: ./fly.sh guard
    # Every GUARD_INTERVAL s (default 60): if RUN's training process is gone
    # and there is no STOPPED.txt / STOPPED_BY_USER, run `fly.sh resume` and
    # log it to $RUN/guard.log. A STOPPED.txt from an exception ("crash:")
    # gets one auto-resume per night; watchdog exhaustion / user stops never. At most GUARD_MAX (default 3) auto-resumes
    # per night (logged in the last 12 h). Exits on STOPPED.txt,
    # STOPPED_BY_USER, or the cap. Battery: on battery power at <=
    # BATTERY_STOP_PCT (default 5) it runs `fly.sh stop` (checkpoint saved)
    # and exits without resuming; never on AC. --dry-run: only prints what it would do
    # (never resumes, never writes guard.log). --once: one check, then exit.
    require_run
    shift
    DRY=0; ONCE=0
    for a in "$@"; do
      case "$a" in
        --dry-run) DRY=1 ;;
        --once) ONCE=1 ;;
        *) echo "unknown guard option: $a"; exit 1 ;;
      esac
    done
    INTERVAL="${GUARD_INTERVAL:-60}"; MAX="${GUARD_MAX:-3}"
    GLOG="$RUN/guard.log"
    glog() {  # epoch first, so the per-night count can filter by time
      local line="$(date +%s) $(date '+%Y-%m-%dT%H:%M:%S') $*"
      if [ "$DRY" = 1 ]; then echo "[dry-run] would log to $GLOG: $line"; else echo "$line" >> "$GLOG"; echo "$line"; fi
    }
    echo "Guarding $RUN every ${INTERVAL}s (max $MAX auto-resumes/night)$([ "$DRY" = 1 ] && echo ' [DRY RUN]'). Ctrl+C to stop guarding."
    [ "$DRY" = 1 ] || glog "guard started (pid $$)"
    while true; do
      if [ -f "$RUN/STOPPED_BY_USER" ]; then
        glog "STOPPED_BY_USER present -- guard exiting"; exit 0
      fi
      if [ -f "$RUN/STOPPED.txt" ]; then
        REASON=$(head -n1 "$RUN/STOPPED.txt")
        # bc.py writes "stopped at step N: crash: <exception>" on an uncaught
        # exception; rollback exhaustion is never resumed. One crash
        # auto-resume per night; a second crash stays stopped.
        case "$REASON" in
          *": crash:"*)
            SINCE=$(( $(date +%s) - 43200 ))
            NC=$(awk -v s="$SINCE" '$1 >= s && /CRASH-RESUME/' "$GLOG" 2>/dev/null | wc -l | tr -d ' ')
            NA=$(awk -v s="$SINCE" '$1 >= s && /AUTO-RESUME #/' "$GLOG" 2>/dev/null | wc -l | tr -d ' ')
            if [ "$NC" -ge 1 ] || [ "$NA" -ge "$MAX" ] || [ -n "$(find_pid)" ]; then
              glog "STOPPED.txt ($REASON): crash auto-resume already used tonight (or cap reached) -- staying stopped, guard exiting"; exit 1
            fi
            glog "AUTO-RESUME #$((NA + 1)) CRASH-RESUME: STOPPED.txt ($REASON) -- moving it to STOPPED.txt.crash-$(date +%s), running fly.sh resume"
            if [ "$DRY" = 1 ]; then
              echo "[dry-run] would run: mv $RUN/STOPPED.txt $RUN/STOPPED.txt.crash-<epoch>; RUN=$RUN $0 resume"
            else
              mv "$RUN/STOPPED.txt" "$RUN/STOPPED.txt.crash-$(date +%s)"
              OUT=$(RUN="$RUN" "$0" resume 2>&1); RC=$?
              glog "resume exit=$RC: $(echo "$OUT" | grep -E '✅|❌' | head -n1)"
            fi
            [ "$ONCE" = 1 ] && exit 0
            sleep "$INTERVAL"; continue
            ;;
          *)
            glog "STOPPED.txt present ($REASON) -- not a crash, never auto-resumed -- guard exiting"; exit 0
            ;;
        esac
      fi
      # Battery protection: on battery at <= BATTERY_STOP_PCT, stop cleanly
      # (fly.sh stop: SIGTERM -> checkpoint, and STOPPED_BY_USER so nothing
      # auto-resumes) and exit without resuming. Never on AC.
      if battery_should_stop; then
        read -r BSRC BPCT <<< "$(battery_state)"
        glog "BATTERY-STOP: on battery at ${BPCT}% (<= ${BATTERY_STOP_PCT}%) -- clean stop, no auto-resume, guard exiting"
        if [ "$DRY" = 1 ]; then
          echo "[dry-run] would run: RUN=$RUN $0 stop$([ -z "$(find_pid)" ] && echo ' (nothing running)')"
        elif [ -n "$(find_pid)" ]; then
          OUT=$(RUN="$RUN" "$0" stop 2>&1); RC=$?
          glog "stop exit=$RC: $(echo "$OUT" | grep -E '✅|⚠️|Nothing' | head -n1)"
        fi
        exit 0
      fi
      PID=$(find_pid)
      if [ -n "$PID" ]; then
        echo "$(date '+%H:%M:%S') running (pid $PID)$([ "$DRY" = 1 ] && echo ' -- [dry-run] would do nothing')"
      else
        SINCE=$(( $(date +%s) - 43200 ))
        N=$(awk -v s="$SINCE" '$1 >= s && /AUTO-RESUME #/' "$GLOG" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$N" -ge "$MAX" ]; then
          glog "process gone, but $N auto-resumes already in the last 12h (max $MAX) -- guard exiting"; exit 1
        fi
        glog "AUTO-RESUME #$((N + 1)): process gone, no STOPPED.txt/STOPPED_BY_USER -- running fly.sh resume"
        if [ "$DRY" = 1 ]; then
          echo "[dry-run] would run: RUN=$RUN $0 resume"
        else
          OUT=$(RUN="$RUN" "$0" resume 2>&1); RC=$?
          glog "resume exit=$RC: $(echo "$OUT" | grep -E '✅|❌' | head -n1)"
        fi
      fi
      [ "$ONCE" = 1 ] && exit 0
      sleep "$INTERVAL"
    done
    ;;
  video)
    shift
    PYTHONPATH=. uv run python scripts/render_gameplay_video.py --run "$RUN" "$@"
    ;;
  log)
    require_run
    tail -n 20 "$RUN/resume.log" 2>/dev/null || echo "No resume log yet."
    ;;
  battery)
    # Parsed `pmset -g batt` state and the guard's decision (read-only).
    read -r BSRC BPCT <<< "$(battery_state)"
    if battery_should_stop; then D="STOP (<= ${BATTERY_STOP_PCT}% on battery)"; else D="ok (threshold ${BATTERY_STOP_PCT}%, battery only)"; fi
    echo "power: $BSRC  battery: ${BPCT:-n/a}%  guard decision: $D"
    ;;
  compare)
    # D40: paired 28-excerpt Medium val Δ of $RUN's checkpoint_step<step>.pt
    # vs v2 @ 42000, both on CPU (MPS is nondeterministic, D39), background.
    require_run
    STEP="${2:?usage: ./fly.sh compare <step>}"
    CK="$RUN/checkpoint_step${STEP}.pt"
    [ -f "$CK" ] || { echo "no $CK"; exit 1; }
    LOG="runs/compare_d40_step${STEP}.log"
    nohup uv run python -u -m eval.compare --checkpoint "$CK" \
      --baseline runs/20260916_222606_bc_full_real_v2/checkpoint_step42000.pt \
      --device cpu --run-name "compare_d40_step${STEP}" > "$LOG" 2>&1 &
    echo "started (~30 min, CPU). check: tail -n 40 $LOG"
    ;;
  *)
    echo "./fly.sh check           → is it running + progress + last 5 evals + checkpoint age"
    echo "./fly.sh awake           → keep Mac awake (leave window open)"
    echo "./fly.sh stop            → stop cleanly (SIGTERM, saves checkpoint)"
    echo "./fly.sh resume [best]   → resume from the latest (or best) checkpoint (refuses if already running)"
    echo "./fly.sh guard [--dry-run] → auto-resume after a crash (own terminal window; max 3/night)"
    echo "./fly.sh video [song] [difficulty] → render a gameplay video at the current checkpoint's skill"
    echo "./fly.sh log             → last lines of the resume log"
    echo "./fly.sh battery         → parsed pmset battery state + the guard's stop decision (BATTERY_STOP_PCT, default 5)"
    echo "./fly.sh compare <step>  → paired 28-excerpt val Δ vs v2 @ 42000, CPU, background (D40)"
    echo ""
    echo "RUN defaults to the newest runs/<timestamp>_{bc_full_real[_vN],bc_trunc_bptt} dir; override with RUN=path ./fly.sh ..."
    ;;
esac
