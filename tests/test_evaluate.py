"""Tests for eval/evaluate.py (PLAN Phase 4 task 3): note classification,
aggregation, and that batching songs of different lengths gives the same
per-song results as running each song alone."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from eval.evaluate import aggregate, evaluate_songs, load_split_songs, note_kind, with_pre_roll
from flyhero.game.retina import build_photoreceptor_map
from flyhero.game.sim import VecRhythmEnv
from flyhero.game.song import slice_song
from flyhero.utils.config import load_config
from train.bc import build_policy

REAL_DATA = Path("data/processed/graph.npz").exists() and Path("data/processed/splits.json").exists()


def test_note_kind():
    masks = np.array([0, 1, 2, 3, 4, 5, 16, 31], dtype=np.uint8)
    assert list(note_kind(masks)) == ["open", "single", "single", "chord", "single", "chord", "single", "chord"]


def test_with_pre_roll_shifts_notes_and_extends_duration():
    from flyhero.game.song import NOTE_DTYPE, Song

    notes = np.array([(0.0, 1, 0.0, 0), (2.0, 3, 0.5, 0)], dtype=NOTE_DTYPE)
    song = Song("id", "t", "a", "c", "Medium", notes, 2.5)
    s = with_pre_roll(song, 1.0, 0.5)
    assert list(s.notes["time_s"]) == [1.0, 3.0] and s.duration_s == 4.0
    assert list(song.notes["time_s"]) == [0.0, 2.0]  # original untouched


def test_aggregate_mean_vs_pooled():
    def song(n, hits, over, minutes, single, chord):
        return dict(
            n_notes=n, n_hits=hits, hit_rate=hits / n, n_overstrums=over, overstrums_per_min=over / minutes,
            minutes=minutes,
            by_kind=dict(single=dict(n=single[0], hits=single[1]), chord=dict(n=chord[0], hits=chord[1]),
                         open=dict(n=0, hits=0)),
        )

    agg = aggregate([song(100, 80, 10, 2.0, (90, 75), (10, 5)), song(300, 150, 30, 4.0, (200, 120), (100, 30))])
    assert agg["hit_rate_mean"] == pytest.approx((0.8 + 0.5) / 2)
    assert agg["hit_rate_pooled"] == pytest.approx(230 / 400)
    assert agg["overstrums_per_min_mean"] == pytest.approx((5 + 7.5) / 2)
    assert agg["overstrums_per_min_pooled"] == pytest.approx(40 / 6)
    assert agg["single_hit_rate"] == pytest.approx(195 / 290)
    assert agg["chord_hit_rate"] == pytest.approx(35 / 110)
    assert agg["open_hit_rate"] is None and agg["n_songs"] == 2


@pytest.mark.skipif(not REAL_DATA, reason="requires real processed data (data/processed/graph.npz, splits.json)")
def test_batched_songs_match_one_at_a_time():
    cfg_game = load_config("configs/game.yaml")
    processed_dir = Path("data/processed")
    graph_np = dict(np.load(processed_dir / "graph.npz"))
    meta = json.loads((processed_dir / "graph_meta.json").read_text())
    photo_map = build_photoreceptor_map(graph_np, meta["type_names"])
    env = VecRhythmEnv(cfg_game, photo_map)
    device, dtype = torch.device("cpu"), torch.float32
    torch.manual_seed(0)
    policy = build_policy("gru", {}, cfg_game, {}, photo_map, device, dtype)

    songs = load_split_songs(processed_dir, "test", "Medium", cfg_game["chord_merge_min_gap_s"])[:3]
    # different lengths, so the batch is padded
    songs = [slice_song(s, 20.0, d) for s, d in zip(songs, (3.0, 5.0, 4.0))]
    kw = dict(pre_roll_s=1.0, post_roll_s=0.5, fps=cfg_game["fps"], hit_window_s=cfg_game["hit_window_s"],
              device=device, dtype=dtype, log=lambda *_: None)
    batched = evaluate_songs(policy, env, songs, batch_size=3, **kw)
    alone = [evaluate_songs(policy, env, [s], batch_size=1, **kw)[0] for s in songs]
    for b, a, s in zip(batched, alone, songs):
        assert b["song_id"] == s.song_id
        assert b["minutes"] == pytest.approx((s.duration_s + 0.5) / 60, abs=1 / 60 / 60)
        for k in ("n_notes", "n_hits", "n_overstrums", "by_kind"):
            assert b[k] == a[k], k


def test_aggregate_skips_songs_without_notes():
    empty = dict(song_id="x", title="Empty", n_notes=0, n_hits=0, hit_rate=0.0, n_overstrums=0,
                 overstrums_per_min=0.0, minutes=3.0,
                 by_kind={k: dict(n=0, hits=0) for k in ("single", "chord", "open")})
    full = dict(song_id="y", title="Full", n_notes=10, n_hits=7, hit_rate=0.7, n_overstrums=3,
                overstrums_per_min=1.5, minutes=2.0,
                by_kind=dict(single=dict(n=10, hits=7), chord=dict(n=0, hits=0), open=dict(n=0, hits=0)))
    agg = aggregate([empty, full])
    assert agg["n_songs"] == 1 and agg["no_note_songs"] == ["x Empty"]
    assert agg["hit_rate_mean"] == pytest.approx(0.7)
    assert agg["overstrums_per_min_pooled"] == pytest.approx(1.5)
