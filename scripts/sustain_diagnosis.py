"""Sustain-note diagnosis (read-only): at a given checkpoint (default:
bc_full_real's latest), scores the fixed Easy val excerpts (train.common.
build_val_set -- same excerpts eval always uses) and splits each song's
hit/miss events by whether the underlying note is a sustain note
(sustain_s > 0) or not, reporting per-category hit_rate plus the engine's
own sustain_frac (fraction of sustain-hold frames actually held during
hits). Read-only: only ever reads from the run dir (checkpoint + its saved
configs), never writes into it. CPU-only by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flyhero.game.retina import build_photoreceptor_map
from flyhero.game.sim import VecRhythmEnv
from flyhero.utils.config import load_config
from train.bc import build_policy, score_eval_entry
from train.common import build_val_set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/20260916_123616_bc_full_real")
    p.add_argument("--ckpt", default=None, help="override checkpoint path (default: <run>/checkpoint_latest.pt)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--difficulty", default="Easy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    model_kind = (run_dir / "model_kind.txt").read_text().strip()
    cfg_bc = load_config(run_dir / "bc.yaml")
    cfg_game = load_config(run_dir / "game.yaml")
    cfg_brain = load_config(run_dir / "brain.yaml") if model_kind == "connectome" else {}

    device = torch.device(args.device)
    dtype = torch.float32

    ckpt_path = Path(args.ckpt) if args.ckpt else run_dir / "checkpoint_latest.pt"
    if ckpt_path.is_symlink():
        ckpt_path = (run_dir / ckpt_path.readlink()).resolve()
    ckpt = torch.load(ckpt_path, map_location=device)

    processed_dir = Path(cfg_bc["processed_dir"])
    graph_np = dict(np.load(processed_dir / "graph.npz"))
    graph_meta = json.loads((processed_dir / "graph_meta.json").read_text())
    photo_map = build_photoreceptor_map(graph_np, graph_meta["type_names"])
    env = VecRhythmEnv(cfg_game, photo_map)

    policy = build_policy(model_kind, cfg_bc, cfg_game, cfg_brain, photo_map, device, dtype)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()

    val_set = build_val_set(
        processed_dir, [args.difficulty], cfg_bc["eval"]["n_val_songs"], cfg_bc["eval"]["excerpt_s"],
        cfg_game["chord_merge_min_gap_s"],
    )
    entries = val_set[args.difficulty]
    print(f"checkpoint: {ckpt_path} (step {ckpt['step']})")
    print(f"{len(entries)} {args.difficulty} val excerpts\n")

    fps, hit_window_s, burn_in_s = cfg_game["fps"], cfg_game["hit_window_s"], cfg_bc["burn_in_s"]

    sus_hits = sus_misses = 0
    non_hits = non_misses = 0
    sustain_fracs = []
    per_song_rows = []
    for entry in entries:
        result = score_eval_entry(policy, env, entry, burn_in_s, fps, hit_window_s, device, dtype)
        notes = result["excerpt_notes"]
        song_sus_hits = song_sus_misses = song_non_hits = song_non_misses = 0
        for e in result["events"]:
            if e.note_idx < 0:
                continue  # overstrum, no underlying note
            is_sustain = bool(notes["sustain_s"][e.note_idx] > 0)
            if e.kind == "hit":
                if is_sustain:
                    song_sus_hits += 1
                else:
                    song_non_hits += 1
            elif e.kind == "miss":
                if is_sustain:
                    song_sus_misses += 1
                else:
                    song_non_misses += 1
        sus_hits += song_sus_hits
        sus_misses += song_sus_misses
        non_hits += song_non_hits
        non_misses += song_non_misses
        sustain_fracs.append(result["metrics"]["sustain_frac"])
        per_song_rows.append(
            (entry.song.title, song_sus_hits, song_sus_misses, song_non_hits, song_non_misses,
             result["metrics"]["sustain_frac"])
        )

    def hit_rate(hits: int, misses: int) -> float:
        return hits / (hits + misses) if (hits + misses) > 0 else float("nan")

    print(f"{'song':40s} {'sustain hit_rate':>18s} {'non-sustain hit_rate':>22s} {'sustain_frac':>13s}")
    for title, sh, sm, nh, nm, sf in per_song_rows:
        print(f"{title[:40]:40s} {hit_rate(sh, sm):18.4f} {hit_rate(nh, nm):22.4f} {sf:13.4f}")

    print()
    print(f"aggregate sustain hit_rate:     {hit_rate(sus_hits, sus_misses):.4f}  ({sus_hits} hits, {sus_misses} misses)")
    print(f"aggregate non-sustain hit_rate: {hit_rate(non_hits, non_misses):.4f}  ({non_hits} hits, {non_misses} misses)")
    print(f"mean sustain_frac (per-song engine metric): {float(np.mean(sustain_fracs)):.4f}")


if __name__ == "__main__":
    main()
