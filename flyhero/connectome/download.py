"""Download the FlyWire v783 connectome sources (Zenodo 10676866 +
flywire_annotations v3.1.0). Idempotent and resumable: a partially-downloaded
file is continued with an HTTP Range request; a complete file (matched by
size) is skipped.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Any

from flyhero.utils.config import load_config

CHUNK_SIZE = 1 << 20  # 1 MiB


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as resp:
        length = resp.headers.get("Content-Length")
        return int(length) if length is not None else None


def download_one(url: str, dest: Path, expected_size: int | None) -> None:
    """Download `url` to `dest`, resuming if `dest` already has bytes and
    isn't already the full size."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    existing = dest.stat().st_size if dest.exists() else 0
    total = expected_size if expected_size is not None else _remote_size(url)

    if total is not None and existing >= total:
        print(f"[skip] {dest.name} already complete ({existing} bytes)")
        return

    mode = "ab" if existing else "wb"
    req = urllib.request.Request(url)
    if existing:
        req.add_header("Range", f"bytes={existing}-")
        print(f"[resume] {dest.name} from byte {existing}")
    else:
        print(f"[start] {dest.name}")

    with urllib.request.urlopen(req) as resp, dest.open(mode) as f:
        downloaded = existing
        while chunk := resp.read(CHUNK_SIZE):
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded % (CHUNK_SIZE * 50) < CHUNK_SIZE:
                print(f"  {dest.name}: {downloaded / 1e6:.0f} MB", flush=True)

    final_size = dest.stat().st_size
    if total is not None and final_size != total:
        raise RuntimeError(
            f"{dest.name}: size mismatch after download "
            f"(got {final_size}, expected {total})"
        )
    print(f"[done] {dest.name}: {final_size} bytes")


def download_all(cfg: dict[str, Any]) -> None:
    raw_dir = Path(cfg["raw_dir"])
    for _, source in cfg["sources"].items():
        dest = raw_dir / source["filename"]
        download_one(source["url"], dest, source.get("size_bytes"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    download_all(cfg)


if __name__ == "__main__":
    sys.exit(main())
