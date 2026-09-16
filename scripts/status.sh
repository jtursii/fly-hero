#!/usr/bin/env bash
# Print a <=15-line summary of a run directory. Usage: scripts/status.sh <run_dir>
set -euo pipefail

run_dir="${1:?usage: scripts/status.sh <run_dir>}"

if [[ ! -d "$run_dir" ]]; then
  echo "no such run dir: $run_dir"
  exit 1
fi

echo "run: $run_dir"
echo "git sha: $(cat "$run_dir/git_sha.txt" 2>/dev/null || echo unknown)"
echo "seed: $(cat "$run_dir/seed.txt" 2>/dev/null || echo unknown)  model: $(cat "$run_dir/model_kind.txt" 2>/dev/null || echo unknown)"

if [[ -f "$run_dir/metrics.jsonl" ]]; then
  # metrics.jsonl interleaves per-step training records, periodic eval
  # records, and curriculum-advance records (train/bc.py, train/
  # readout_only.py) -- python parses the last of each kind rather than
  # assuming a fixed line layout.
  tail -n 500 "$run_dir/metrics.jsonl" | python3 -c "
import json, sys
last_train, last_eval, last_advance = None, None, None
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if 'eval_hit_rate' in r:
        last_eval = r
    elif 'curriculum_advanced_to' in r:
        last_advance = r
    elif 'loss' in r:
        last_train = r

if last_train:
    print('step: %s  difficulty: %s  steps/s: %.2f' % (
        last_train.get('step'), last_train.get('difficulty'), last_train.get('steps_per_s', 0)))
    print('loss: %.4f  (fret: %.4f, strum: %.4f)  grad_norm: %.3f' % (
        last_train.get('loss', float('nan')), last_train.get('fret_loss', float('nan')),
        last_train.get('strum_loss', float('nan')), last_train.get('grad_norm', float('nan'))))
    print('max|state|: %.4f  nan_frac: %.4f  pos_weight: %.2f' % (
        last_train.get('max_abs_state', float('nan')), last_train.get('nan_frac', 0),
        last_train.get('pos_weight', float('nan'))))
else:
    print('no training records yet')

if last_eval:
    print('last eval @ step %s: difficulty=%s hit_rate=%.4f overstrums/min=%.1f n_songs=%s wall=%.1fs' % (
        last_eval.get('step'), last_eval.get('eval_difficulty'), last_eval.get('eval_hit_rate', 0),
        last_eval.get('eval_overstrums_per_min', 0), last_eval.get('eval_n_songs'), last_eval.get('eval_wall_s', 0)))
else:
    print('no eval yet')

if last_advance:
    print('curriculum: advanced to %s at step %s (pos_weight=%.2f)' % (
        last_advance.get('curriculum_advanced_to'), last_advance.get('step'), last_advance.get('new_pos_weight', 0)))
"
else
  echo "no metrics.jsonl yet"
fi

if [[ -L "$run_dir/checkpoint_latest.pt" ]]; then
  echo "latest checkpoint: $(readlink "$run_dir/checkpoint_latest.pt")"
fi

log_file=$(find "$run_dir" -maxdepth 1 -name '*.log' | head -n1)
if [[ -n "${log_file:-}" ]]; then
  echo "--- last 3 lines of $(basename "$log_file") ---"
  tail -n 3 "$log_file"
fi
