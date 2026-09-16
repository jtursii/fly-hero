"""Gate G2 throughput check: VecRhythmEnv at batch 64 must run >=10x real
time on CPU."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from flyhero.game.retina import build_photoreceptor_map
from flyhero.game.sim import VecRhythmEnv
from flyhero.game.song import load_song_merged
from flyhero.utils.config import load_config


def main() -> None:
    cfg = load_config("configs/game.yaml")
    processed_dir = Path(cfg["processed_dir"])
    graph = np.load(processed_dir / "graph.npz")
    meta = json.loads((processed_dir / "graph_meta.json").read_text())
    photo_map = build_photoreceptor_map(graph, meta["type_names"])

    splits = json.loads((processed_dir / "splits.json").read_text())
    songs = []
    for sid in splits["train"]:
        p = processed_dir / "songs" / f"{sid}_Expert.npz"
        if p.exists():
            song = load_song_merged(p, cfg["chord_merge_min_gap_s"])
            if song.duration_s * cfg["fps"] >= 300:
                songs.append(song)
        if len(songs) >= 64:
            break

    batch = len(songs)
    env = VecRhythmEnv(cfg, photo_map)
    env.reset(songs)

    n_steps = 300  # 5s of game time at 60fps
    actions = np.zeros((batch, 6), dtype=bool)

    # warm-up
    for _ in range(10):
        env.step(actions)
    env.reset(songs)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        env.step(actions)
    elapsed = time.perf_counter() - t0

    game_seconds = n_steps / cfg["fps"]
    speedup = game_seconds / elapsed
    print(f"batch={batch} steps={n_steps} wall={elapsed:.3f}s game_time={game_seconds:.1f}s")
    print(f"speedup={speedup:.1f}x real-time (gate: >=10x)")


if __name__ == "__main__":
    main()
