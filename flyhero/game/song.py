"""Song data model shared by the .chart/.mid parsers and ingest.py, and later
by rules.py/env.py (invariant 4: one scoring implementation reads this same
shape). A Song is one (song folder, difficulty) pair.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# lane_mask bits, frets 0-4 (green, red, yellow, blue, orange).
N_LANES = 5

# flags bits. Open notes have no bit here -- they're dropped before a note
# ever reaches this dtype (v1 rule simplification, task 5).
FLAG_FORCED = 1 << 0
FLAG_TAP = 1 << 1

NOTE_DTYPE = np.dtype(
    [
        ("time_s", np.float32),
        ("lane_mask", np.uint8),
        ("sustain_s", np.float32),
        ("flags", np.uint8),
    ]
)


@dataclass
class ParsedDifficulty:
    """What a single-format parser (chart_parser/midi_parser) returns for one
    difficulty section/track, before ingest.py attaches song.ini metadata."""

    notes: np.ndarray  # NOTE_DTYPE, sorted by time_s
    duration_s: float  # last note (incl. sustain) end time
    open_note_count: int
    total_note_count: int
    ignored_counts: dict[str, int]


@dataclass
class Song:
    song_id: str
    title: str
    artist: str
    charter: str
    difficulty: str
    notes: np.ndarray  # NOTE_DTYPE
    duration_s: float
    # Recorded per PLAN but not used for training (invariant: game rules have
    # one implementation and don't consume these).
    chart_offset_s: float = 0.0
    ini_delay_s: float = 0.0


def compute_song_id(rel_path: str) -> str:
    """Stable hash of the song folder path relative to the library root (not
    the absolute path, so IDs survive the library being moved)."""
    normalized = rel_path.replace("\\", "/")
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def build_notes_array(rows: list[tuple[float, int, float, int]]) -> np.ndarray:
    """Build a sorted NOTE_DTYPE array from (time_s, lane_mask, sustain_s,
    flags) rows. Shared by both parsers so chord-grouping/sorting behavior is
    identical regardless of source format."""
    if not rows:
        return np.zeros(0, dtype=NOTE_DTYPE)
    rows = sorted(rows, key=lambda r: r[0])
    arr = np.zeros(len(rows), dtype=NOTE_DTYPE)
    for i, (t, mask, sustain, flags) in enumerate(rows):
        arr[i] = (t, mask, sustain, flags)
    return arr


def save_song(song: Song, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        song_id=np.array(song.song_id),
        title=np.array(song.title),
        artist=np.array(song.artist),
        charter=np.array(song.charter),
        difficulty=np.array(song.difficulty),
        notes=song.notes,
        duration_s=np.array(song.duration_s, dtype=np.float32),
        chart_offset_s=np.array(song.chart_offset_s, dtype=np.float32),
        ini_delay_s=np.array(song.ini_delay_s, dtype=np.float32),
    )


def load_song(path: Path) -> Song:
    data: dict[str, Any] = np.load(path, allow_pickle=False)
    return Song(
        song_id=str(data["song_id"]),
        title=str(data["title"]),
        artist=str(data["artist"]),
        charter=str(data["charter"]),
        difficulty=str(data["difficulty"]),
        notes=data["notes"],
        duration_s=float(data["duration_s"]),
        chart_offset_s=float(data["chart_offset_s"]),
        ini_delay_s=float(data["ini_delay_s"]),
    )
