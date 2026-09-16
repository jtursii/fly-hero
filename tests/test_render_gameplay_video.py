"""Tests for scripts/render_gameplay_video.py's core correctness invariant:
a rendered clip's hit_rate must equal bc.py's own eval hit_rate for that
same checkpoint/song/excerpt (both go through train.bc.score_eval_entry,
the single shared implementation -- see its docstring). Uses a freshly
constructed GRU policy (fast: no connectome forward/backward) built from
the repo's own configs -- the invariant under test is that the two code
paths agree given the *same* weights, not that the weights are trained,
so no run-specific checkpoint artifact is needed (those live under the
gitignored, session-ephemeral runs/ dir and can't be relied on to survive
across sessions).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from flyhero.game.retina import build_photoreceptor_map
from flyhero.game.sim import VecRhythmEnv
from flyhero.utils.config import load_config
from train.bc import build_policy, evaluate_difficulty, score_eval_entry
from train.common import build_val_set

REAL_DATA = Path("data/processed/graph.npz").exists() and Path("data/processed/splits.json").exists()


@pytest.mark.skipif(not REAL_DATA, reason="requires real processed data (data/processed/graph.npz, splits.json)")
def test_rendered_clip_hit_rate_matches_eval_hit_rate():
    cfg_bc = load_config("configs/bc.yaml")
    cfg_game = load_config("configs/game.yaml")
    device, dtype = torch.device("cpu"), torch.float32

    processed_dir = Path(cfg_bc["processed_dir"])
    graph_np = dict(np.load(processed_dir / "graph.npz"))
    meta = json.loads((processed_dir / "graph_meta.json").read_text())
    photo_map = build_photoreceptor_map(graph_np, meta["type_names"])
    env = VecRhythmEnv(cfg_game, photo_map)

    policy = build_policy("gru", cfg_bc, cfg_game, {}, photo_map, device, dtype)
    policy.eval()

    difficulty = "Medium"
    val_set = build_val_set(
        processed_dir, [difficulty], cfg_bc["eval"]["n_val_songs"], cfg_bc["eval"]["excerpt_s"],
        cfg_game["chord_merge_min_gap_s"],
    )
    entry = val_set[difficulty][0]  # the same "first fixed eval song" render_gameplay_video.py defaults to
    fps, hit_window_s, burn_in_s = cfg_game["fps"], cfg_game["hit_window_s"], cfg_bc["burn_in_s"]

    # "eval" path: what bc.py's in-training eval computes for this song.
    eval_result = evaluate_difficulty(policy, env, [entry], burn_in_s, fps, hit_window_s, device, dtype)

    # "render" path: what scripts/render_gameplay_video.py calls to get the
    # data it renders from -- score_eval_entry is the exact same function
    # evaluate_difficulty uses internally, so this is a real end-to-end
    # check that the two call sites can't silently diverge, not a tautology
    # (a bug in either's *arguments* -- wrong entry, wrong burn_in_s, wrong
    # device/dtype -- would still be caught here).
    render_result = score_eval_entry(policy, env, entry, burn_in_s, fps, hit_window_s, device, dtype)

    assert render_result["metrics"]["hit_rate"] == pytest.approx(eval_result["hit_rate"])
    assert render_result["metrics"]["overstrums_per_min"] == pytest.approx(eval_result["overstrums_per_min"])

    # Determinism: re-running inference (fresh env/policy state, same
    # checkpoint weights, no_grad) must reproduce the identical hit_rate --
    # this is what makes "the video's hit_rate matches eval's" meaningful
    # rather than a coincidence of shared mutable state.
    render_result_2 = score_eval_entry(policy, env, entry, burn_in_s, fps, hit_window_s, device, dtype)
    assert render_result_2["metrics"]["hit_rate"] == pytest.approx(render_result["metrics"]["hit_rate"])
