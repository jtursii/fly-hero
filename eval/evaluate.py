"""Test-split evaluation (PLAN Phase 4 task 3 / Gate G4): runs a checkpoint
on every song of a split at each requested difficulty -- full songs, not
excerpts -- and writes eval.json with per-song and aggregate metrics.

Each song is played from a blank pre_roll_s lead-in (the same burn-in the
brain gets in training and in-training eval) to post_roll_s past its last
note's end, so the final notes' hit windows close before scoring stops. The
inference path is train.bc.run_clip -> decoder.decode_trace (defaults) ->
rules.score_playthrough, the same as the in-training eval; songs are batched
by length (batching doesn't change outputs beyond float noise; tested).

Usage:
  uv run python -m eval.evaluate --config configs/eval.yaml \\
      --checkpoint runs/<run>/checkpoint_best.pt [--checkpoint ...] [--difficulties Easy Medium]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from flyhero.game.decoder import decode_trace, frets_to_held_mask
from flyhero.game.retina import build_photoreceptor_map
from flyhero.game.rules import score_playthrough
from flyhero.game.sim import VecRhythmEnv
from flyhero.game.song import Song, load_song_merged
from flyhero.utils.config import load_config
from flyhero.utils.run import make_run_dir
from train.bc import build_policy, run_clip

NOTE_KINDS = ("single", "chord", "open")


def note_kind(lane_mask: np.ndarray) -> np.ndarray:
    """lane_mask[N] uint8 -> kind[N] str: 0 lanes = open, 1 = single, >=2 = chord."""
    n_lanes = np.unpackbits(lane_mask.astype(np.uint8)[:, None], axis=1).sum(axis=1)
    return np.where(n_lanes == 0, "open", np.where(n_lanes == 1, "single", "chord"))


def load_split_songs(processed_dir: Path, split: str, difficulty: str, chord_merge_min_gap_s: float) -> list[Song]:
    ids = sorted(json.loads((processed_dir / "splits.json").read_text())[split])
    songs = []
    for sid in ids:
        path = processed_dir / "songs" / f"{sid}_{difficulty}.npz"
        if path.exists():
            songs.append(load_song_merged(path, chord_merge_min_gap_s))
    return songs


def with_pre_roll(song: Song, pre_roll_s: float, post_roll_s: float) -> Song:
    """The song delayed by pre_roll_s (blank highway first), extended by post_roll_s."""
    notes = song.notes.copy()
    notes["time_s"] += pre_roll_s
    return dataclasses.replace(song, notes=notes, duration_s=pre_roll_s + song.duration_s + post_roll_s)


def score_song(song: Song, probs: np.ndarray, fps: int, hit_window_s: float) -> dict:
    """probs[F, 6] over the song's own clock (frame 0 = song time 0)."""
    frets, strum = decode_trace(probs)
    metrics, events = score_playthrough(song.notes, frets_to_held_mask(frets), strum, fps, hit_window_s)
    kinds = note_kind(song.notes["lane_mask"])
    hit = np.zeros(len(song.notes), dtype=bool)
    for e in events:
        if e.kind == "hit":
            hit[e.note_idx] = True
    by_kind = {k: dict(n=int((kinds == k).sum()), hits=int(hit[kinds == k].sum())) for k in NOTE_KINDS}
    return dict(
        song_id=song.song_id, title=song.title, artist=song.artist, duration_s=song.duration_s,
        minutes=len(probs) / fps / 60.0, **metrics, by_kind=by_kind,
    )


def evaluate_songs(
    policy, env: VecRhythmEnv, songs: list[Song], pre_roll_s: float, post_roll_s: float, fps: int,
    hit_window_s: float, batch_size: int, device, dtype, log=print,
) -> list[dict]:
    """Per-song results, in the order of `songs`."""
    pre_frames = round(pre_roll_s * fps)
    order = sorted(range(len(songs)), key=lambda i: songs[i].duration_s)
    results: dict[int, dict] = {}
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        clips = [with_pre_roll(songs[i], pre_roll_s, post_roll_s) for i in idx]
        song_frames = [round((c.duration_s - pre_roll_s) * fps) for c in clips]
        t0 = time.time()
        with torch.no_grad():
            _, logits, _ = run_clip(
                policy, env, clips, pre_frames, max(song_frames), device, dtype,
                with_grad=False, gradient_checkpointing=False,
            )
        probs = torch.sigmoid(logits).float().cpu().numpy()  # [B, T, 6]
        for b, i in enumerate(idx):
            results[i] = score_song(songs[i], probs[b, :song_frames[b]], fps, hit_window_s)
        log(f"  batch {start // batch_size + 1}: {len(idx)} songs, {max(song_frames) / fps:.0f}s, "
            f"wall {time.time() - t0:.0f}s")
    return [results[i] for i in range(len(songs))]


def aggregate(per_song: list[dict]) -> dict:
    """Song means skip charts with no notes (some ingested difficulty charts
    are empty: hit_rate is undefined there, and rules.py reports 0.0); they
    are listed in `no_note_songs`. Pooled numbers are unaffected by them."""
    empty = [r for r in per_song if r["n_notes"] == 0]
    per_song = [r for r in per_song if r["n_notes"] > 0]
    minutes = sum(r["minutes"] for r in per_song)
    n_notes = sum(r["n_notes"] for r in per_song)
    out = dict(
        n_songs=len(per_song),
        no_note_songs=[f"{r['song_id']} {r['title']}" for r in empty],
        hit_rate_mean=float(np.mean([r["hit_rate"] for r in per_song])) if per_song else 0.0,
        hit_rate_pooled=sum(r["n_hits"] for r in per_song) / n_notes if n_notes else 0.0,
        overstrums_per_min_mean=float(np.mean([r["overstrums_per_min"] for r in per_song])) if per_song else 0.0,
        overstrums_per_min_pooled=sum(r["n_overstrums"] for r in per_song) / minutes if minutes else 0.0,
        n_notes=n_notes, minutes=minutes,
    )
    for k in NOTE_KINDS:
        n = sum(r["by_kind"][k]["n"] for r in per_song)
        hits = sum(r["by_kind"][k]["hits"] for r in per_song)
        out[f"{k}_n"] = n
        out[f"{k}_hit_rate"] = hits / n if n else None
    return out


def reaggregate(eval_json: Path) -> None:
    """Recomputes every aggregate from the stored per-song results (no
    model run), keeping wall_s."""
    out = json.loads(eval_json.read_text())
    for entry in out["checkpoints"].values():
        for difficulty, v in entry["difficulties"].items():
            agg = aggregate(v["per_song"])
            agg["wall_s"] = v["aggregate"].get("wall_s")
            v["aggregate"] = agg
            print(f"step {entry['step']} {difficulty}: hit_rate mean={agg['hit_rate_mean']:.4f} "
                  f"pooled={agg['hit_rate_pooled']:.4f} (n={agg['n_songs']}, no-note: {agg['no_note_songs']})")
    eval_json.write_text(json.dumps(out, indent=1))


def resolve_run_config(run_dir: Path, name: str) -> Path:
    """The config copy the run trained with, else the repo default."""
    p = run_dir / name
    return p if p.exists() else Path("configs") / name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--checkpoint", action="append")
    p.add_argument("--reaggregate", default=None, help="recompute aggregates of an existing eval.json in place")
    p.add_argument("--difficulties", nargs="+", default=None)
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--run-name", default="eval")
    args = p.parse_args()
    if args.reaggregate:
        reaggregate(Path(args.reaggregate))
        return
    if not args.checkpoint:
        p.error("--checkpoint is required")

    cfg = load_config(args.config)
    difficulties = args.difficulties or cfg["difficulties"]
    device = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(cfg["seed"])
    run_dir = make_run_dir(args.run_name, args.config, cfg["seed"])
    print(f"run dir: {run_dir}")

    out = dict(config=cfg, split=cfg["split"], difficulties=difficulties, device=str(device), checkpoints={})
    for ckpt_path in args.checkpoint:
        ckpt_file = Path(ckpt_path).resolve()
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
        print(f"{ckpt_file} (step {ckpt['step']}, {model_kind})")

        entry = dict(path=str(ckpt_file), step=ckpt["step"], model_kind=model_kind, difficulties={})
        for difficulty in difficulties:
            songs = load_split_songs(processed_dir, cfg["split"], difficulty, cfg_game["chord_merge_min_gap_s"])
            print(f" {difficulty}: {len(songs)} {cfg['split']} songs")
            t0 = time.time()
            per_song = evaluate_songs(
                policy, env, songs, cfg["pre_roll_s"], cfg["post_roll_s"], cfg_game["fps"],
                cfg_game["hit_window_s"], cfg["batch_size"], device, dtype,
            )
            agg = aggregate(per_song)
            agg["wall_s"] = time.time() - t0
            entry["difficulties"][difficulty] = dict(aggregate=agg, per_song=per_song)
            print(f" {difficulty}: hit_rate mean={agg['hit_rate_mean']:.4f} pooled={agg['hit_rate_pooled']:.4f} "
                  f"overstrums/min={agg['overstrums_per_min_mean']:.1f} single={agg['single_hit_rate']} "
                  f"chord={agg['chord_hit_rate']} wall={agg['wall_s']:.0f}s")
            out["checkpoints"][str(ckpt_file)] = entry
            (run_dir / "eval.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {run_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
