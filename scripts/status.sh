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
echo "seed: $(cat "$run_dir/seed.txt" 2>/dev/null || echo unknown)"

config_file=$(find "$run_dir" -maxdepth 1 -name '*.yaml' | head -n1)
if [[ -n "${config_file:-}" ]]; then
  echo "config: $(basename "$config_file")"
fi

log_file=$(find "$run_dir" -maxdepth 1 -name '*.log' | head -n1)
if [[ -n "${log_file:-}" ]]; then
  echo "--- last lines of $(basename "$log_file") ---"
  tail -n 8 "$log_file"
else
  echo "no log file yet"
fi
