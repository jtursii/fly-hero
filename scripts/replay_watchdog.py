"""Replays the watchdog rules (train/watchdog.py) over a run's
metrics.jsonl: which steps the grad rule would have fired at, which
checkpoints would have been certified healthy, and whether the eval-MA rule
would have fired. Usage:

  uv run python scripts/replay_watchdog.py runs/<dir> --baseline-span 18000 39000 [--from 39000 --to 42300]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from train.watchdog import baseline_from_metrics, load_metrics_lineage, replay_on_metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--baseline", type=float, default=None)
    p.add_argument("--baseline-span", type=int, nargs=2, default=None)
    p.add_argument("--from", dest="from_step", type=int, default=0)
    p.add_argument("--to", dest="to_step", type=int, default=10**12)
    args = p.parse_args()

    records = load_metrics_lineage(Path(args.run_dir) / "metrics.jsonl")
    if args.baseline is not None:
        baseline = args.baseline
    elif args.baseline_span:
        baseline = baseline_from_metrics(records, *args.baseline_span)
    else:
        p.error("give --baseline or --baseline-span")
    print(f"baseline grad_norm: {baseline:.1f} -> trigger > {3 * baseline:.0f} for 200 steps, "
          f"certify < {2 * baseline:.0f} over 1000 steps")
    out = replay_on_metrics(records, baseline)
    in_range = lambda s: args.from_step <= s <= args.to_step  # noqa: E731
    print("grad rule fires at:", [s for s in out["grad_fires"] if in_range(s)])
    print("same trigger on the run's own logged 200-step median:", [s for s in out["live_median_fires"] if in_range(s)])
    print("healthy checkpoints:", {s: h for s, h in out["healthy"].items() if in_range(s)})
    print("eval-MA rule fires:", [(s, round(m, 3), round(b, 3)) for s, m, b in out["eval_fires"] if in_range(s)] or "none")


if __name__ == "__main__":
    main()
