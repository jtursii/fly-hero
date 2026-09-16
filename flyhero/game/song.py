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


def merge_close_notes(notes: np.ndarray, min_gap_s: float) -> tuple[np.ndarray, int]:
    """Merge notes closer than `min_gap_s` into a chord, at load time only --
    never applied to the cache written by ingest.py. Anchor-based: each run's
    anchor is its first (earliest) note, and a subsequent note joins the run
    if it is within `min_gap_s` of the *anchor*, not the previous note, so a
    chain of sub-threshold gaps can't collapse an entire trill into one note.
    A merged note takes lane_mask=OR, sustain_s=max, flags=OR, and the
    anchor's time_s. Returns (merged_notes, n_notes_absorbed)."""
    if len(notes) == 0:
        return notes, 0

    groups: list[dict] = []
    anchor_time = float(notes["time_s"][0])
    current = {
        "time_s": anchor_time,
        "lane_mask": int(notes["lane_mask"][0]),
        "sustain_s": float(notes["sustain_s"][0]),
        "flags": int(notes["flags"][0]),
    }
    n_absorbed = 0
    for i in range(1, len(notes)):
        t = float(notes["time_s"][i])
        if t - anchor_time < min_gap_s:
            current["lane_mask"] |= int(notes["lane_mask"][i])
            current["sustain_s"] = max(current["sustain_s"], float(notes["sustain_s"][i]))
            current["flags"] |= int(notes["flags"][i])
            n_absorbed += 1
        else:
            groups.append(current)
            anchor_time = t
            current = {
                "time_s": anchor_time,
                "lane_mask": int(notes["lane_mask"][i]),
                "sustain_s": float(notes["sustain_s"][i]),
                "flags": int(notes["flags"][i]),
            }
    groups.append(current)

    merged = np.zeros(len(groups), dtype=NOTE_DTYPE)
    for i, g in enumerate(groups):
        merged[i] = (g["time_s"], g["lane_mask"], g["sustain_s"], g["flags"])
    return merged, n_absorbed


def load_song_merged(path: Path, min_gap_s: float = 1.0 / 60.0) -> Song:
    """The loader tasks 6-10 (labels/rules/render/retina/sim/env) must use --
    never raw load_song() -- so every gameplay-facing consumer sees the same
    (merged) note sequence."""
    song = load_song(path)
    song.notes, _ = merge_close_notes(song.notes, min_gap_s)
    return song


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
