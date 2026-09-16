"""Tests for flyhero/game/clip_sampler.py (PLAN Phase 3b)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flyhero.game.clip_sampler import ClipSampler, compute_valid_starts, get_valid_starts, list_available_song_ids
from flyhero.game.song import NOTE_DTYPE, Song, save_song


def test_compute_valid_starts_basic_buckets():
    # duration 10s, fps=60, burn_in=1s, window=2s -> candidate starts in [1, 8).
    # Notes at 2.0 and 2.5 -> a window starting at [0.5, 2.0] roughly covers
    # both (window [start, start+2)); place clearly-with-notes and
    # clearly-empty regions.
    notes_time_s = np.array([2.0, 2.5, 6.0])
    with_notes, empty = compute_valid_starts(
        notes_time_s, duration_s=10.0, burn_in_s=1.0, window_s=2.0, fps=60, min_notes_with=2,
    )
    assert len(with_notes) > 0
    assert len(empty) > 0
    # every with_notes start must actually contain >=2 notes in [start, start+2)
    for s in with_notes[:: max(1, len(with_notes) // 20)]:
        count = int(((notes_time_s >= s) & (notes_time_s < s + 2.0)).sum())
        assert count >= 2
    for s in empty[:: max(1, len(empty) // 20)]:
        count = int(((notes_time_s >= s) & (notes_time_s < s + 2.0)).sum())
        assert count == 0
    # starts respect burn-in and duration bounds
    assert with_notes.min() >= 1.0 if len(with_notes) else True
    assert (with_notes + 2.0 <= 10.0 + 1e-9).all()


def test_compute_valid_starts_too_short_song_returns_empty():
    with_notes, empty = compute_valid_starts(
        np.array([0.5]), duration_s=2.5, burn_in_s=1.0, window_s=2.0, fps=60,
    )
    assert len(with_notes) == 0
    assert len(empty) == 0


def test_get_valid_starts_caches_to_disk(tmp_path: Path):
    song = Song(
        song_id="testsong1", title="t", artist="a", charter="c", difficulty="Medium",
        notes=np.array([(2.0, 1, 0.0, 0), (2.3, 2, 0.0, 0), (6.0, 1, 0.0, 0)], dtype=NOTE_DTYPE),
        duration_s=10.0,
    )
    with_notes_1, empty_1 = get_valid_starts(song, "Medium", 1.0, 2.0, 60, tmp_path)
    cache_files = list((tmp_path / "clip_starts").rglob("*.npz"))
    assert len(cache_files) == 1

    with_notes_2, empty_2 = get_valid_starts(song, "Medium", 1.0, 2.0, 60, tmp_path)
    np.testing.assert_array_equal(with_notes_1, with_notes_2)
    np.testing.assert_array_equal(empty_1, empty_2)


def test_list_available_song_ids_filters_missing_difficulty(tmp_path: Path):
    songs_dir = tmp_path / "songs"
    songs_dir.mkdir(parents=True)
    song_a = Song(song_id="a", title="", artist="", charter="", difficulty="Easy", notes=np.zeros(0, dtype=NOTE_DTYPE), duration_s=5.0)
    save_song(song_a, songs_dir / "a_Easy.npz")
    # song "b" has no Easy file
    available = list_available_song_ids(tmp_path, ["a", "b"], "Easy")
    assert available == ["a"]


def _make_song(song_id: str, notes_time_s: list[float], duration_s: float) -> Song:
    notes = np.array([(t, 1, 0.0, 0) for t in notes_time_s], dtype=NOTE_DTYPE)
    return Song(song_id=song_id, title="", artist="", charter="", difficulty="Medium", notes=notes, duration_s=duration_s)


def test_clip_sampler_respects_90_10_ratio(tmp_path: Path):
    songs_dir = tmp_path / "songs"
    songs_dir.mkdir(parents=True)
    # a dense song (plenty of 2+-note windows) and a long-gap song (plenty of
    # 0-note windows), so both buckets are well populated.
    dense_notes = list(np.arange(0.0, 20.0, 0.1))
    save_song(_make_song("dense", dense_notes, 20.0), songs_dir / "dense_Medium.npz")
    save_song(_make_song("sparse", [1.0, 18.0], 20.0), songs_dir / "sparse_Medium.npz")

    sampler = ClipSampler(
        processed_dir=tmp_path, song_ids=["dense", "sparse"], difficulty="Medium",
        burn_in_s=1.0, window_s=2.0, fps=60, chord_merge_min_gap_s=1 / 60,
    )
    assert len(sampler.song_ids_with_notes) > 0
    assert len(sampler.song_ids_empty) > 0

    rng = np.random.default_rng(0)
    n = 2000
    has_notes_count = 0
    for _ in range(n):
        song, start, has_notes = sampler.sample(rng)
        assert song.duration_s == 20.0
        assert start >= 1.0
        assert start + 2.0 <= 20.0 + 1e-9
        has_notes_count += int(has_notes)

    frac = has_notes_count / n
    assert 0.85 < frac < 0.95  # ~0.9 target, some sampling noise allowed


def test_clip_sampler_falls_back_to_with_notes_if_no_empty_pool(tmp_path: Path):
    songs_dir = tmp_path / "songs"
    songs_dir.mkdir(parents=True)
    dense_notes = list(np.arange(0.0, 20.0, 0.05))  # so dense there's no 0-note window
    save_song(_make_song("alldense", dense_notes, 20.0), songs_dir / "alldense_Medium.npz")

    sampler = ClipSampler(
        processed_dir=tmp_path, song_ids=["alldense"], difficulty="Medium",
        burn_in_s=1.0, window_s=2.0, fps=60, chord_merge_min_gap_s=1 / 60,
    )
    assert sampler.song_ids_empty == []
    rng = np.random.default_rng(0)
    for _ in range(50):
        _, _, has_notes = sampler.sample(rng)
        assert has_notes is True
