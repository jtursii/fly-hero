"""Phase 0 smoke test: config loader, run-dir helper, and torch/MPS import."""

from __future__ import annotations

import subprocess

from flyhero.utils.config import load_config
from flyhero.utils.run import make_run_dir


def test_config_round_trip(tmp_path):
    config_path = tmp_path / "test.yaml"
    config_path.write_text("song_library: /tmp/songs\nmin_syn: 5\n")

    cfg = load_config(config_path)

    assert cfg == {"song_library": "/tmp/songs", "min_syn": 5}


def test_make_run_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "test.yaml"
    config_path.write_text("song_library: /tmp/songs\n")

    run_dir = make_run_dir("smoke", config_path, seed=0)

    assert (run_dir / "test.yaml").exists()
    assert (run_dir / "seed.txt").read_text().strip() == "0"
    assert (run_dir / "git_sha.txt").exists()


def test_torch_and_mps_import():
    import torch

    # Recorded for Gate G0; MPS availability is a machine property, not a
    # correctness assertion.
    mps_available = torch.backends.mps.is_available()
    print(f"torch.backends.mps.is_available() = {mps_available}")
    assert isinstance(mps_available, bool)
