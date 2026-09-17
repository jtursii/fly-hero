"""scripts/training_time.py: gap exclusion and lineage step bounds (D33)."""

from __future__ import annotations

import json

from scripts.training_time import active_seconds, fork_parent, load_times


def test_active_seconds_skips_long_gaps_and_respects_step_bounds():
    times = [(0.0, 10), (10.0, 20), (20.0, 30), (1000.0, 40), (1010.0, 50)]
    assert active_seconds(times, 300.0) == 30.0  # the 980 s gap is a stop, not training
    assert active_seconds(times, 300.0, max_step=30) == 20.0
    assert active_seconds(times, 300.0, min_step=30) == 10.0


def test_load_times_reads_metrics_time_field(tmp_path):
    recs = [dict(step=10, time=5.0, loss=1.0), dict(step=20, time=7.5, loss=1.0), dict(step=30, loss=1.0)]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    assert load_times(tmp_path) == [(5.0, 10), (7.5, 20)]


def test_fork_parent_resolves_backup_copy_by_content(tmp_path):
    runs = tmp_path / "runs"
    parent = runs / "20260101_000000_parent"
    child = runs / "20260102_000000_child"
    parent.mkdir(parents=True)
    child.mkdir()
    (parent / "checkpoint_step500.pt").write_bytes(b"abc")
    backup = tmp_path / "backups" / "checkpoint_step500.pt"
    backup.parent.mkdir()
    backup.write_bytes(b"abc")
    (child / "init_from.txt").write_text(f"{backup}\n")
    assert fork_parent(child, runs) == (parent, 500)
    backup.write_bytes(b"xyz")
    assert fork_parent(child, runs) is None
