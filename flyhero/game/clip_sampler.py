"""Note-aware clip sampling for bc.py, baseline_gru.py, and the readout-only
DN-rate cache (PLAN Phase 3b, user decision): 90% of sampled training clips
have >=2 notes (by note time_s) inside the 2s gradient window; 10% have 0
notes, so the model also learns not to strum on an empty highway. Burn-in
(preceding the gradient window) may contain anything -- only the gradient
window itself is classified.

Valid start times (of the *gradient window*, not the burn-in) are
precomputed once per (song_id, difficulty, burn_in_s, window_s, fps) and
cached to disk under `data/processed/clip_starts/` -- computing this per
song requires scanning a candidate start-time grid, which is fine to redo
across song loads within one process but wasteful across many training runs
of a fresh process.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flyhero.game.song import Song, load_song_merged

MIN_NOTES_WITH = 2
FRAC_WITH_NOTES = 0.9


def compute_valid_starts(
    notes_time_s: np.ndarray,
    duration_s: float,
    burn_in_s: float,
    window_s: float,
    fps: int,
    min_notes_with: int = MIN_NOTES_WITH,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (with_notes, empty): arrays of valid gradient-window start
    times (seconds, frame-aligned) where the window [start, start+window_s)
    contains >= min_notes_with notes, or exactly 0 notes, respectively.
    `start - burn_in_s >= 0` is required (there must be room for burn-in
    before the window) and `start + window_s <= duration_s`.

    Vectorized via searchsorted rather than an O(n_starts * n_notes) scan --
    real songs can have thousands of candidate start frames and hundreds to
    thousands of notes."""
    min_start = burn_in_s
    max_start = duration_s - window_s
    if max_start <= min_start:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    step = 1.0 / fps
    starts = np.arange(min_start, max_start, step)
    if len(starts) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    notes_sorted = np.sort(notes_time_s.astype(np.float64))
    lo = np.searchsorted(notes_sorted, starts, side="left")
    hi = np.searchsorted(notes_sorted, starts + window_s, side="left")
    counts = hi - lo

    with_notes = starts[counts >= min_notes_with]
    empty = starts[counts == 0]
    return with_notes, empty


def _cache_path(processed_dir: Path, difficulty: str, song_id: str, burn_in_s: float, window_s: float, fps: int) -> Path:
    tag = f"b{burn_in_s:g}_w{window_s:g}_f{fps}"
    return processed_dir / "clip_starts" / difficulty / tag / f"{song_id}.npz"


def get_valid_starts(
    song: Song, difficulty: str, burn_in_s: float, window_s: float, fps: int, processed_dir: Path,
    min_notes_with: int = MIN_NOTES_WITH,
) -> tuple[np.ndarray, np.ndarray]:
    """Cached wrapper around compute_valid_starts, keyed by
    (song_id, difficulty, burn_in_s, window_s, fps) so a config change can't
    silently reuse a stale cache."""
    path = _cache_path(processed_dir, difficulty, song.song_id, burn_in_s, window_s, fps)
    if path.exists():
        data = np.load(path)
        return data["with_notes"], data["empty"]

    with_notes, empty = compute_valid_starts(
        song.notes["time_s"], song.duration_s, burn_in_s, window_s, fps, min_notes_with,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, with_notes=with_notes, empty=empty)
    return with_notes, empty


def list_available_song_ids(processed_dir: Path, song_ids: list[str], difficulty: str) -> list[str]:
    """Not every song has every difficulty (Phase 2: ~293/652 have native
    Easy/Medium, ~568/652 have Expert) -- filters a split's song_ids down to
    those with an actual `<song_id>_<difficulty>.npz` file."""
    songs_dir = processed_dir / "songs"
    return [sid for sid in song_ids if (songs_dir / f"{sid}_{difficulty}.npz").exists()]


class ClipSampler:
    """Holds valid-start caches for one difficulty's available songs, and
    draws (song_id, start_time_s, has_notes) triples at the configured 90/10
    ratio. Songs/starts are loaded lazily and cached in memory for reuse
    across many samples within a run."""

    def __init__(
        self, processed_dir: Path, song_ids: list[str], difficulty: str, burn_in_s: float,
        window_s: float, fps: int, chord_merge_min_gap_s: float, frac_with_notes: float = FRAC_WITH_NOTES,
    ):
        self.processed_dir = processed_dir
        self.difficulty = difficulty
        self.burn_in_s = burn_in_s
        self.window_s = window_s
        self.fps = fps
        self.chord_merge_min_gap_s = chord_merge_min_gap_s
        self.frac_with_notes = frac_with_notes

        available = list_available_song_ids(processed_dir, song_ids, difficulty)
        self._song_cache: dict[str, Song] = {}
        self._starts_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # Only keep songs that have at least one valid start in *each*
        # bucket the sampler might need -- a song with no empty-window start
        # (e.g. very short/dense) simply can't supply the 10% bucket and is
        # excluded from that bucket's pool, not an error.
        self.song_ids_with_notes: list[str] = []
        self.song_ids_empty: list[str] = []
        for sid in available:
            song = self._get_song(sid)
            with_notes, empty = self._get_starts(song)
            if len(with_notes) > 0:
                self.song_ids_with_notes.append(sid)
            if len(empty) > 0:
                self.song_ids_empty.append(sid)

        if not self.song_ids_with_notes:
            raise RuntimeError(f"no {difficulty} songs with any >=2-note {window_s}s window found")

    def _get_song(self, song_id: str) -> Song:
        if song_id not in self._song_cache:
            path = self.processed_dir / "songs" / f"{song_id}_{self.difficulty}.npz"
            self._song_cache[song_id] = load_song_merged(path, self.chord_merge_min_gap_s)
        return self._song_cache[song_id]

    def _get_starts(self, song: Song) -> tuple[np.ndarray, np.ndarray]:
        if song.song_id not in self._starts_cache:
            self._starts_cache[song.song_id] = get_valid_starts(
                song, self.difficulty, self.burn_in_s, self.window_s, self.fps, self.processed_dir,
            )
        return self._starts_cache[song.song_id]

    def sample(self, rng: np.random.Generator) -> tuple[Song, float, bool]:
        """Returns (song, start_time_s, has_notes)."""
        want_notes = rng.random() < self.frac_with_notes or not self.song_ids_empty
        pool = self.song_ids_with_notes if want_notes else self.song_ids_empty
        song_id = pool[rng.integers(len(pool))]
        song = self._get_song(song_id)
        with_notes, empty = self._get_starts(song)
        starts = with_notes if want_notes else empty
        start_time_s = float(starts[rng.integers(len(starts))])
        return song, start_time_s, want_notes


def note_free_fraction_report(
    processed_dir: Path, song_ids: list[str], difficulties: list[str], burn_in_s: float,
    window_s: float, fps: int, chord_merge_min_gap_s: float,
) -> dict:
    """For each (song, difficulty), the fraction of valid gradient-window
    start times that fall in the 0-note bucket (len(empty) / (len(empty) +
    len(with_notes)), restricted to windows that are either 0-note or
    >=MIN_NOTES_WITH-note -- windows with 1 note are in neither bucket and
    excluded from this ratio, matching what the sampler actually draws
    from). Returns per-song fractions and the library-wide median."""
    per_song = []
    for difficulty in difficulties:
        available = list_available_song_ids(processed_dir, song_ids, difficulty)
        for sid in available:
            path = processed_dir / "songs" / f"{sid}_{difficulty}.npz"
            song = load_song_merged(path, chord_merge_min_gap_s)
            with_notes, empty = get_valid_starts(song, difficulty, burn_in_s, window_s, fps, processed_dir)
            total = len(with_notes) + len(empty)
            if total == 0:
                continue
            per_song.append(
                dict(song_id=sid, difficulty=difficulty, note_free_frac=len(empty) / total, n_valid_starts=total)
            )

    fracs = np.array([r["note_free_frac"] for r in per_song])
    return dict(
        per_song=per_song,
        n_song_difficulty_pairs=len(per_song),
        median_note_free_frac=float(np.median(fracs)) if len(fracs) else float("nan"),
        mean_note_free_frac=float(np.mean(fracs)) if len(fracs) else float("nan"),
    )
