"""Run-directory helper: every run writes to runs/<timestamp>_<name>/ with a
copy of its config, the seed, and the git SHA (see CLAUDE.md conventions)."""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

RUNS_ROOT = Path("runs")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def make_run_dir(name: str, config_path: str | Path, seed: int) -> Path:
    """Create runs/<timestamp>_<name>/, copy the config into it, and record
    the seed and git SHA. Returns the run directory path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    config_path = Path(config_path)
    shutil.copy(config_path, run_dir / config_path.name)

    (run_dir / "seed.txt").write_text(f"{seed}\n")
    (run_dir / "git_sha.txt").write_text(f"{git_sha()}\n")

    return run_dir
