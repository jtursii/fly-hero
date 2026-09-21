"""One-off showcase-selection scoring (Phase 7, not part of the training/eval
gates): scores the final showcase checkpoint on every Medium chart in the
train AND test splits (never val, which is already used for training
decisions), for picking 2-3 replay candidates. This never changes or feeds
back into G4, which stays test-only (D38); it just needs more Medium songs
to pick a good-looking showcase from than the 31-song test split alone has.

Reuses eval.evaluate's full-song scoring path unchanged (load_split_songs /
evaluate_songs / score_song -> rules.score_playthrough, invariant 4) --
this script only adds: combining splits, dropping 0-note charts, dropping
near-duplicate charts (flyhero.game.ingest.normalize_key, D17 -- e.g.
"Metallica - One" / "Metallica - One (Co-op)"), and ranking by hit_rate.

Usage:
  uv run python -m scripts.showcase_score_medium --config configs/eval.yaml \\
      --checkpoint runs/20260916_222606_bc_full_real_v2/checkpoint_step42000.pt \\
      --device cpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval.evaluate import evaluate_songs, load_split_songs, resolve_run_config
from flyhero.game.ingest import normalize_key
from flyhero.game.retina import build_photoreceptor_map
from flyhero.game.sim import VecRhythmEnv
from flyhero.utils.config import load_config
from flyhero.utils.run import make_run_dir
from train.bc import build_policy

SHOWCASE_MAX_S = 330.0


def dedupe_near_duplicates(rows: list[dict]) -> list[dict]:
    """Keeps one row per normalized (artist, title) key (D17 near-dup
    groups), the one with the higher hit_rate, so the same underlying song
    (e.g. a "(Co-op)" chart) can't take two slots in the ranking."""
    groups: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (normalize_key(r["artist"]), normalize_key(r["title"]))
        best = groups.get(key)
        if best is None or r["hit_rate"] > best["hit_rate"]:
            groups[key] = r
    return list(groups.values())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--difficulty", default="Medium")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--run-name", default="showcase_score_medium")
    p.add_argument("--limit", type=int, default=None, help="debug only: cap songs per split")
    p.add_argument("--batch-size", type=int, default=None, help="override configs/eval.yaml's batch_size (inference-only, no gradient memory budget applies)")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(cfg["seed"])
    run_dir = make_run_dir(args.run_name, args.config, cfg["seed"])
    print(f"run dir: {run_dir}", flush=True)

    ckpt_file = Path(args.checkpoint).resolve()
    src_run = ckpt_file.parent
    cfg_game = load_config(resolve_run_config(src_run, "game.yaml"))
    model_kind = (src_run / "model_kind.txt").read_text().strip() if (src_run / "model_kind.txt").exists() else "connectome"
    cfg_brain = load_config(resolve_run_config(src_run, "brain.yaml")) if model_kind == "connectome" else {}
    processed_dir = Path(cfg["processed_dir"])
    graph_np = np.load(processed_dir / "graph.npz")
    meta = json.loads((processed_dir / "graph_meta.json").read_text())
    photo_map = build_photoreceptor_map(graph_np, meta["type_names"])
    env = VecRhythmEnv(cfg_game, photo_map)
    policy = build_policy(model_kind, {}, cfg_game, cfg_brain, photo_map, device, dtype)
    ckpt = torch.load(ckpt_file, map_location=device)
    policy.load_state_dict(ckpt["model_state"])
    print(f"{ckpt_file} (step {ckpt['step']}, {model_kind}, {device})", flush=True)

    all_rows = []
    for split in args.splits:
        songs = load_split_songs(processed_dir, split, args.difficulty, cfg_game["chord_merge_min_gap_s"])
        if args.limit:
            songs = songs[: args.limit]
        print(f"{split}: {len(songs)} {args.difficulty} songs", flush=True)
        t0 = time.time()
        per_song = evaluate_songs(
            policy, env, songs, cfg["pre_roll_s"], cfg["post_roll_s"], cfg_game["fps"],
            cfg_game["hit_window_s"], cfg["batch_size"], device, dtype,
            log=lambda msg, split=split: print(f"  [{split}] {msg}", flush=True),
        )
        for r in per_song:
            r["split"] = split
        all_rows.extend(per_song)
        print(f"{split}: done, wall {time.time() - t0:.0f}s", flush=True)
        (run_dir / "per_song_raw.json").write_text(json.dumps(all_rows, indent=1))

    n_before = len(all_rows)
    no_note = [r for r in all_rows if r["n_notes"] == 0]
    rows = [r for r in all_rows if r["n_notes"] > 0]
    rows = dedupe_near_duplicates(rows)
    print(f"{n_before} charts -> {len(rows)} after dropping {len(no_note)} 0-note and "
          f"{n_before - len(no_note) - len(rows)} near-duplicate charts", flush=True)

    rows.sort(key=lambda r: r["hit_rate"], reverse=True)
    top25 = rows[:25]
    for r in top25:
        r["showcase_length_ok"] = r["duration_s"] <= SHOWCASE_MAX_S

    out = dict(
        checkpoint=str(ckpt_file), step=int(ckpt["step"]), difficulty=args.difficulty,
        splits=args.splits, n_charts_scored=n_before, n_no_note=len(no_note),
        n_near_dup_dropped=n_before - len(no_note) - len(rows), top25=top25,
    )
    (run_dir / "showcase_top25.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {run_dir / 'showcase_top25.json'}", flush=True)

    print("", flush=True)
    print(f"{'#':>3} {'title':<40} {'artist':<25} {'len':>6} {'hit_rate':>9} {'ovr/min':>8} {'split':>6} {'<=330s':>7}", flush=True)
    for i, r in enumerate(top25, 1):
        mm, ss = divmod(int(round(r["duration_s"])), 60)
        length = f"{mm}:{ss:02d}"
        flag = "Y" if r["showcase_length_ok"] else "N"
        print(f"{i:>3} {r['title'][:40]:<40} {r['artist'][:25]:<25} {length:>6} "
              f"{r['hit_rate']:>9.4f} {r['overstrums_per_min']:>8.1f} {r['split']:>6} {flag:>7}", flush=True)


if __name__ == "__main__":
    main()
