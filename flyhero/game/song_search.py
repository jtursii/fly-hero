"""Fuzzy song lookup by artist/title, for scripts/render_gameplay_video.py's
song/difficulty picker. Reads only the small artist/title/duration_s arrays
out of each song's .npz (via NpzFile's lazy per-key access), not the full
notes array, so scanning the whole val/test pool is cheap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SongMatch:
    song_id: str
    split: str
    artist: str
    title: str

    @property
    def display_name(self) -> str:
        return f"{self.artist} - {self.title}"


def _load_display_fields(path: Path) -> tuple[str, str] | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return str(data["artist"]), str(data["title"])


def search_songs(
    processed_dir: Path, query: str, difficulty: str, allow_train: bool = False,
) -> list[SongMatch]:
    """Case-insensitive substring match against "artist - title" across the
    val+test splits (train included only with allow_train=True, per the
    "refuse train-split songs unless --allow-train" requirement -- a song
    that only exists in train simply isn't in the search pool otherwise, so
    it's refused by construction, not by a separate check)."""
    splits = json.loads((processed_dir / "splits.json").read_text())
    pools = ["val", "test"] + (["train"] if allow_train else [])
    query_lower = query.lower()

    matches: list[SongMatch] = []
    for split in pools:
        for sid in splits.get(split, []):
            path = processed_dir / "songs" / f"{sid}_{difficulty}.npz"
            fields = _load_display_fields(path)
            if fields is None:
                continue
            artist, title = fields
            display = f"{artist} - {title}"
            if query_lower in display.lower():
                matches.append(SongMatch(song_id=sid, split=split, artist=artist, title=title))
    return matches
