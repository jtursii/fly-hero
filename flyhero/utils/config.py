"""YAML config loading shared by every flyhero entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    path = Path(path)
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return data or {}
