"""Active training time per run, along a run's lineage, and in total (D33).

Timestamps come from metrics.jsonl's `time` field when present (added
2026-09-16, D33), otherwise from the TensorBoard event wall_time of the same
logged steps (every earlier run has these). "Active" = the sum of gaps
between consecutive logged records, skipping any gap > --max-gap-s (a
stopped/crashed/paused run). Time before a process's first logged record
(graph loading, ~1 min) is not counted.

Lineage: follows init_from.txt back to the run the checkpoint came from. A
fork only counts its parent's records up to the fork step; the parent's
later steps count toward total compute only (discarded work).

Usage: uv run python scripts/training_time.py <run_dir> [--runs-root runs]
"""

from __future__ import annotations

import argparse
import filecmp
import json
import re
from pathlib import Path

RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}_(.+)$")


def load_times(run_dir: Path) -> list[tuple[float, int]]:
    """[(wall_time, step)] for every logged training/eval record, time-sorted."""
    out: list[tuple[float, int]] = []
    metrics = run_dir / "metrics.jsonl"
    timed_steps: set[int] = set()
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "time" in r and "step" in r:
                out.append((float(r["time"]), int(r["step"])))
                timed_steps.add(int(r["step"]))
    tb = run_dir / "tb"
    if tb.is_dir():
        from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

        seen: set[tuple[int, float]] = set()
        for f in sorted(tb.glob("events.out.tfevents.*")):
            for ev in EventFileLoader(str(f)).Load():
                if not ev.HasField("summary") or ev.step in timed_steps:
                    continue
                key = (ev.step, round(ev.wall_time, 1))
                if key not in seen:  # one entry per logged record, not per scalar tag
                    seen.add(key)
                    out.append((float(ev.wall_time), int(ev.step)))
    out.sort()
    return out


def active_seconds(times: list[tuple[float, int]], max_gap_s: float, max_step: int | None = None,
                   min_step: int | None = None) -> float:
    """Sum of consecutive gaps <= max_gap_s. A gap is attributed to the step
    range by its *later* record's step."""
    total = 0.0
    for (t0, _), (t1, s1) in zip(times, times[1:]):
        if max_step is not None and s1 > max_step:
            continue
        if min_step is not None and s1 <= min_step:
            continue
        if 0 <= t1 - t0 <= max_gap_s:
            total += t1 - t0
    return total


def fork_parent(run_dir: Path, runs_root: Path) -> tuple[Path, int] | None:
    """(parent_run_dir, fork_step) from init_from.txt, or None. Resolves
    backup copies by byte-comparing against each run's same-step checkpoint."""
    f = run_dir / "init_from.txt"
    if not f.exists():
        return None
    src = Path(f.read_text().strip())
    m = re.search(r"checkpoint_step(\d+)\.pt$", src.name)
    if not m:
        return None
    step = int(m.group(1))
    if RUN_DIR_RE.match(src.resolve().parent.name):
        return src.resolve().parent, step
    for cand in sorted(runs_root.glob(f"*/checkpoint_step{step}.pt")):
        if cand.parent.resolve() != run_dir.resolve() and src.exists() and filecmp.cmp(src, cand, shallow=False):
            return cand.parent, step
    return None


def fmt(s: float) -> str:
    h, m = divmod(int(round(s / 60)), 60)
    return f"{h}h{m:02d}m"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--max-gap-s", type=float, default=300.0)
    args = p.parse_args()
    runs_root, run_dir = Path(args.runs_root), Path(args.run_dir)

    # lineage: [(run_dir, max_step or None)], newest first
    lineage: list[tuple[Path, int | None]] = [(run_dir, None)]
    cur = run_dir
    while (parent := fork_parent(cur, runs_root)) is not None and len(lineage) < 20:
        lineage.append(parent)
        cur = parent[0]
    lineage_max = {d.resolve(): s for d, s in lineage}

    rows, untimed, total, lineage_total = [], [], 0.0, 0.0
    for d in sorted(x for x in runs_root.iterdir() if x.is_dir() and RUN_DIR_RE.match(x.name)):
        times = load_times(d)
        if len(times) < 2:
            if (d / "metrics.jsonl").exists():
                untimed.append(RUN_DIR_RE.match(d.name).group(1))
            continue
        secs = active_seconds(times, args.max_gap_s)
        total += secs
        name = RUN_DIR_RE.match(d.name).group(1)
        if d.resolve() in lineage_max:
            bound = lineage_max[d.resolve()]
            on = active_seconds(times, args.max_gap_s, max_step=bound)
            lineage_total += on
            note = f"lineage {fmt(on)}" + (f" (<= step {bound}; {fmt(secs - on)} discarded)" if bound is not None else "")
        else:
            note = ""
        rows.append((name, secs, note))

    chain = " -> ".join(RUN_DIR_RE.match(d.name).group(1) + (f"<={s}" if s is not None else "")
                        for d, s in reversed(lineage))
    print(f"lineage training time: {fmt(lineage_total)}  ({chain})")
    print(f"total compute:         {fmt(total)}  (all runs, incl. discarded + smoke; gaps >{args.max_gap_s / 60:.0f} min excluded)")
    for name, secs, note in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<28} {fmt(secs):>7}  {note}")
    if untimed:
        print(f"  not counted (no timestamps logged): {', '.join(untimed)}")


if __name__ == "__main__":
    main()
