"""Gate G2 (PLAN Phase 2): scripted-perfect-player hit_rate on 20 random +
the 5 densest Expert val songs (by notes/duration_s), plus a random-player
baseline. Env throughput is benchmarked separately by bench/bench_env.py.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from flyhero.game.rules import score_playthrough, scripted_perfect_actions
from flyhero.game.song import load_song_merged
from flyhero.utils.config import load_config


def random_player_actions(n_notes: int, duration_s: float, fps: int, rng: np.random.Generator):
    n_frames = int(round(duration_s * fps))
    held = rng.integers(0, 32, size=n_frames).astype(np.int64)
    strum_prob = n_notes / max(n_frames, 1)
    strum = rng.random(n_frames) < strum_prob
    return held, strum


def main() -> None:
    cfg = load_config("configs/game.yaml")
    processed_dir = Path(cfg["processed_dir"])
    splits = json.loads((processed_dir / "splits.json").read_text())
    val_ids = [sid for sid in splits["val"] if (processed_dir / "songs" / f"{sid}_Expert.npz").exists()]

    rng = random.Random(0)
    random_sample_ids = rng.sample(val_ids, min(20, len(val_ids)))

    def score_song(sid: str):
        p = processed_dir / "songs" / f"{sid}_Expert.npz"
        song = load_song_merged(p, cfg["chord_merge_min_gap_s"])
        held, strum = scripted_perfect_actions(song.notes, song.duration_s, cfg["fps"], cfg["hit_window_s"])
        metrics, _ = score_playthrough(
            song.notes, held, strum, cfg["fps"], cfg["hit_window_s"], cfg["max_open_notes_capacity"]
        )
        return song, metrics

    print("=== Scripted perfect player: 20 random Expert val songs ===")
    scripted_hit_rates = []
    for sid in random_sample_ids:
        song, metrics = score_song(sid)
        scripted_hit_rates.append(metrics["hit_rate"])
        print(f"  {song.artist} - {song.title}: hit_rate={metrics['hit_rate']:.4f} overstrums={metrics['n_overstrums']} misses={metrics['n_misses']}")
    print(f"  -> min hit_rate: {min(scripted_hit_rates):.4f}, all==1.0: {all(h == 1.0 for h in scripted_hit_rates)}")

    print("\n=== Scripted perfect player: 5 densest Expert val songs ===")
    density_rows = []
    for sid in val_ids:
        song = load_song_merged(processed_dir / "songs" / f"{sid}_Expert.npz", cfg["chord_merge_min_gap_s"])
        if song.duration_s > 0:
            density_rows.append((len(song.notes) / song.duration_s, sid))
    density_rows.sort(reverse=True)
    densest_hit_rates = []
    for density, sid in density_rows[:5]:
        song, metrics = score_song(sid)
        densest_hit_rates.append(metrics["hit_rate"])
        print(f"  {density:.2f} notes/s | {song.artist} - {song.title}: hit_rate={metrics['hit_rate']:.4f} overstrums={metrics['n_overstrums']} misses={metrics['n_misses']}")
    print(f"  -> min hit_rate: {min(densest_hit_rates):.4f}, all==1.0: {all(h == 1.0 for h in densest_hit_rates)}")

    print("\n=== Random player baseline: same 20 songs ===")
    np_rng = np.random.default_rng(0)
    random_hit_rates = []
    for sid in random_sample_ids:
        song = load_song_merged(processed_dir / "songs" / f"{sid}_Expert.npz", cfg["chord_merge_min_gap_s"])
        held, strum = random_player_actions(len(song.notes), song.duration_s, cfg["fps"], np_rng)
        metrics, _ = score_playthrough(
            song.notes, held, strum, cfg["fps"], cfg["hit_window_s"], cfg["max_open_notes_capacity"]
        )
        random_hit_rates.append(metrics["hit_rate"])
    print(f"  mean hit_rate: {np.mean(random_hit_rates):.4f} (min={min(random_hit_rates):.4f}, max={max(random_hit_rates):.4f})")


if __name__ == "__main__":
    main()
