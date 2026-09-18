"""Phase 2 task 6 tests: hit/miss/overstrum resolution, including the
earliest-unhit-note tie-break needed for dense charts and the real-data
densest-song stress test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flyhero.game.rules import score_playthrough, scripted_perfect_actions
from flyhero.game.song import NOTE_DTYPE, load_song_merged
from flyhero.utils.config import load_config

FPS = 60
HIT_WINDOW_S = 0.07


def _notes(rows: list[tuple[float, int, float, int]]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=NOTE_DTYPE)
    for i, r in enumerate(rows):
        arr[i] = r
    return arr


@pytest.mark.parametrize("fret_onset", ["segment", "hit_window"])
def test_scripted_perfect_player_dense_trill(fret_onset):
    rows = [(0.05 * i, 1 if i % 2 == 0 else 2, 0.0, 0) for i in range(10)]
    notes = _notes(rows)
    duration_s = 0.6

    held_seq, strum_seq = scripted_perfect_actions(notes, duration_s, FPS, HIT_WINDOW_S, fret_onset)
    metrics, events = score_playthrough(notes, held_seq, strum_seq, FPS, HIT_WINDOW_S)

    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["n_misses"] == 0
    assert metrics["n_overstrums"] == 0


def test_chord_then_overlapping_single_note_exact_match_required():
    # Chord {0,1} at t=0, single lane-0 note at t=0.1 -- close enough that
    # a naive "note in window" check (ignoring exact lane match) could
    # misattribute a strum.
    notes = _notes([(0.0, 0b00011, 0.0, 0), (0.1, 0b00001, 0.0, 0)])
    duration_s = 0.3
    n_frames = int(round(duration_s * FPS))

    held_seq = np.zeros(n_frames, dtype=np.int64)
    strum_seq = np.zeros(n_frames, dtype=bool)
    f0, f1 = round(0.0 * FPS), round(0.1 * FPS)
    held_seq[f0] = 0b00011
    held_seq[f1] = 0b00001
    strum_seq[f0] = True
    strum_seq[f1] = True

    metrics, events = score_playthrough(notes, held_seq, strum_seq, FPS, HIT_WINDOW_S)

    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["n_overstrums"] == 0
    kinds = [e.kind for e in events]
    assert kinds.count("hit") == 2

    # Holding the CHORD mask at the single note's strum frame must NOT hit
    # it (exact match only) -- it should overstrum instead.
    held_seq2 = held_seq.copy()
    held_seq2[f1] = 0b00011  # wrong -- still holding both frets
    metrics2, events2 = score_playthrough(notes, held_seq2, strum_seq, FPS, HIT_WINDOW_S)
    assert metrics2["n_overstrums"] == 1
    assert metrics2["hit_rate"] == pytest.approx(0.5)


def test_earliest_unhit_note_credited_on_overlapping_windows():
    # Two same-lane notes close enough that their +-hit_window_s windows
    # overlap; a strum inside the overlap must credit the earlier note.
    notes = _notes([(0.0, 0b00001, 0.0, 0), (0.05, 0b00001, 0.0, 0)])
    duration_s = 0.2
    n_frames = int(round(duration_s * FPS))
    held_seq = np.full(n_frames, 0b00001, dtype=np.int64)
    strum_seq = np.zeros(n_frames, dtype=bool)

    strum_frame = round(0.02 * FPS)  # inside both notes' hit windows
    strum_seq[strum_frame] = True
    second_strum_frame = round(0.06 * FPS)
    strum_seq[second_strum_frame] = True

    metrics, events = score_playthrough(notes, held_seq, strum_seq, FPS, HIT_WINDOW_S)
    hit_events = [e for e in events if e.kind == "hit"]
    assert len(hit_events) == 2
    assert hit_events[0].note_idx == 0  # earliest unhit note credited first
    assert hit_events[1].note_idx == 1
    assert metrics["n_overstrums"] == 0


def test_miss_resolved_only_when_window_closes():
    notes = _notes([(0.1, 0b00001, 0.0, 0)])
    duration_s = 0.3
    n_frames = int(round(duration_s * FPS))
    held_seq = np.zeros(n_frames, dtype=np.int64)
    strum_seq = np.zeros(n_frames, dtype=bool)  # never strummed

    from flyhero.game.rules import RuleEngine

    engine = RuleEngine(notes=notes, fps=FPS, hit_window_s=HIT_WINDOW_S)
    # First frame f with f/FPS strictly > window_end -- RuleEngine closes a
    # window on t > window_end, not t >= window_end.
    window_close_frame = int(np.floor((0.1 + HIT_WINDOW_S) * FPS)) + 1
    for f in range(window_close_frame):
        events = engine.step(f, 0, False)
        assert engine.n_misses == 0, f"missed too early at frame {f}"
        assert events == []
    events = engine.step(window_close_frame, 0, False)
    assert engine.n_misses == 1
    assert any(e.kind == "miss" for e in events)


# ---------------------------------------------------------------------------
# Real-data regression: densest Expert val song (Aquellex - Wanderflux, found
# via notes_count/duration_s over the val split -- see PROGRESS.md).
# ---------------------------------------------------------------------------

GAME_CONFIG = Path("configs/game.yaml")
PATHS_CONFIG = Path("configs/paths.yaml")


def _densest_val_song_path() -> Path | None:
    processed_dir = Path("data/processed")
    splits_path = processed_dir / "splits.json"
    if not splits_path.exists():
        return None
    splits = json.loads(splits_path.read_text())
    best_path, best_density = None, -1.0
    for song_id in splits["val"]:
        p = processed_dir / "songs" / f"{song_id}_Expert.npz"
        if not p.exists():
            continue
        song = load_song_merged(p)
        if song.duration_s <= 0:
            continue
        density = len(song.notes) / song.duration_s
        if density > best_density:
            best_density, best_path = density, p
    return best_path


@pytest.mark.parametrize("fret_onset", ["segment", "hit_window"])
def test_scripted_perfect_player_on_densest_real_val_song(fret_onset):
    """If this fails, fix labels.py, never loosen rules.py -- this is the
    direct stress test for the earliest-unhit-note tie-break and the
    max(sustain_s, hit_window_s) label hold-duration floor under real,
    heavily-overlapping-window density (measured: 90% of adjacent-note gaps
    in this song are under 2*hit_window_s)."""
    if not GAME_CONFIG.exists():
        pytest.skip("requires configs/game.yaml")
    song_path = _densest_val_song_path()
    if song_path is None:
        pytest.skip("requires data/processed/splits.json and cached songs from ingest.py")

    cfg = load_config(GAME_CONFIG)
    song = load_song_merged(song_path, cfg["chord_merge_min_gap_s"])

    held_seq, strum_seq = scripted_perfect_actions(
        song.notes, song.duration_s, cfg["fps"], cfg["hit_window_s"], fret_onset
    )
    metrics, _events = score_playthrough(
        song.notes, held_seq, strum_seq, cfg["fps"], cfg["hit_window_s"],
        max_open=cfg["max_open_notes_capacity"],
    )

    assert metrics["n_misses"] == 0
    assert metrics["n_overstrums"] == 0
    assert metrics["hit_rate"] == pytest.approx(1.0)
