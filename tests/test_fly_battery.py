"""fly.sh guard's battery protection: pmset parsing and the stop decision,
driven through PMSET_OUTPUT (never the real pmset) and --dry-run, against a
throwaway RUN dir, so no real training process can be touched."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

BATT = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=23003235)\t{pct}%; discharging; 0:12 remaining present: true\n"
AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=23003235)\t{pct}%; charging; 1:02 until full present: true\n"
AC_CHARGED = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=23003235)\t100%; charged; 0:00 remaining present: true\n"
DESKTOP = "Now drawing from 'AC Power'\n"


def run_fly(args: list[str], pmset: str, tmp_path: Path, **env) -> subprocess.CompletedProcess:
    run_dir = tmp_path / "20990101_000000_bc_trunc_bptt"
    run_dir.mkdir(exist_ok=True)
    e = {**os.environ, "PMSET_OUTPUT": pmset, "RUN": str(run_dir), "GUARD_INTERVAL": "1", **env}
    return subprocess.run(["bash", "fly.sh", *args], cwd=REPO, env=e, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize(
    "pmset, expect_state, expect_stop",
    [
        (BATT.format(pct=3), "battery 3%", True),
        (BATT.format(pct=5), "battery 5%", True),  # threshold is inclusive
        (BATT.format(pct=6), "battery 6%", False),
        (BATT.format(pct=100), "battery 100%", False),
        (AC.format(pct=1), "ac 1%", False),  # never on AC, at any level
        (AC_CHARGED, "ac 100%", False),
        (DESKTOP, "unknown n/a%", False),  # no battery line: never stop
        ("", "unknown n/a%", False),
    ],
)
def test_battery_parse_and_decision(tmp_path, pmset, expect_state, expect_stop):
    out = run_fly(["battery"], pmset, tmp_path).stdout
    src, pct = expect_state.split()
    assert f"power: {src}  battery: {pct}" in out
    assert ("guard decision: STOP" in out) == expect_stop


def test_threshold_env_var(tmp_path):
    out = run_fly(["battery"], BATT.format(pct=12), tmp_path, BATTERY_STOP_PCT="15").stdout
    assert "guard decision: STOP (<= 15%" in out
    out = run_fly(["battery"], BATT.format(pct=12), tmp_path).stdout
    assert "guard decision: ok" in out


def test_guard_dry_run_low_battery_stops_without_resume(tmp_path):
    r = run_fly(["guard", "--dry-run", "--once"], BATT.format(pct=4), tmp_path)
    assert r.returncode == 0
    assert "BATTERY-STOP: on battery at 4% (<= 5%)" in r.stdout
    would = [l for l in r.stdout.splitlines() if "[dry-run] would run:" in l]
    assert len(would) == 1 and would[0].rstrip().endswith("stop (nothing running)")  # a stop, never a resume
    assert not (tmp_path / "20990101_000000_bc_trunc_bptt" / "guard.log").exists()  # dry run writes nothing


def test_guard_dry_run_on_ac_low_does_not_stop(tmp_path):
    r = run_fly(["guard", "--dry-run", "--once"], AC.format(pct=2), tmp_path)
    assert "BATTERY-STOP" not in r.stdout
