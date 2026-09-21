"""Tests for the Phase 6 export pipeline (export/record.py, export_web.py,
validate_export.py).

The load-bearing one is test_recorded_step_matches_run_clip: record.py drives
the model through `policy.step_frame_recorded` instead of `step_frame`, so
the whole export is only trustworthy if those two produce the same logits.
That is asserted bit-for-bit rather than assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from export.export_web import choose_slots, normalize_positions, notes_json, pack_4bit
from export.record import choose_retina_channels, quantize_4bit, scope_channels
from export.validate_export import notes_from_manifest, unpack_actions
from flyhero.game.retina import build_photoreceptor_map
from flyhero.game.sim import VecRhythmEnv
from flyhero.game.song import NOTE_DTYPE, Song
from flyhero.utils.config import load_config
from train.bc import build_policy, run_clip

REAL_DATA = Path("data/processed/graph.npz").exists() and Path("data/processed/splits.json").exists()


def test_quantize_4bit_spans_levels():
    q = quantize_4bit(np.array([0.0, 1 / 15, 0.5, 0.999, 1.0], dtype=np.float32))
    assert list(q) == [0, 1, 8, 15, 15]
    assert q.dtype == np.uint8


def test_pack_4bit_round_trip():
    rng = np.random.default_rng(0)
    levels = rng.integers(0, 16, size=(7, 64)).astype(np.uint8)
    packed = pack_4bit(levels)
    assert packed.shape == (7, 32) and packed.dtype == np.uint8
    unpacked = np.empty_like(levels)
    unpacked[:, 0::2] = packed & 0x0F
    unpacked[:, 1::2] = packed >> 4
    assert np.array_equal(unpacked, levels)


def test_unpack_actions_inverts_the_packing():
    rng = np.random.default_rng(1)
    frets = rng.random((200, 5)) > 0.5
    strum = rng.random(200) > 0.7
    actions = (frets.astype(np.uint8) << np.arange(5, dtype=np.uint8)).sum(axis=1).astype(np.uint8)
    actions |= (strum.astype(np.uint8) << 5)

    held, got_strum = unpack_actions(actions)
    want_held = (frets.astype(np.int64) << np.arange(5)).sum(axis=1)
    assert np.array_equal(held, want_held)
    assert np.array_equal(got_strum, strum)


def test_notes_survive_the_manifest_round_trip():
    notes = np.array([(0.5, 1, 0.0, 0), (1.25, 6, 0.75, 0), (9.0, 0, 0.0, 0)], dtype=NOTE_DTYPE)
    back = notes_from_manifest(notes_json(notes))
    assert np.allclose(back["time_s"], notes["time_s"])
    assert np.array_equal(back["lane_mask"], notes["lane_mask"])
    assert np.allclose(back["sustain_s"], notes["sustain_s"])


def test_choose_slots_takes_every_dn_then_inputs_then_variance():
    n_nodes, k = 2000, 200
    rng = np.random.default_rng(2)
    dn_idx = np.arange(10)
    input_idx = np.arange(100, 160)
    activity = [rng.integers(6, 10, size=(40, n_nodes)).astype(np.uint8) for _ in range(2)]
    # A handful of high-variance neurons outside the DN/input blocks: they
    # swing the full 0..15 range every other frame, nothing else does.
    loud = np.array([300, 301, 302])
    for a in activity:
        a[:, loud] = np.where(np.arange(40)[:, None] % 2, 15, 0).astype(np.uint8)

    slots = choose_slots(activity, dn_idx, input_idx, k)
    idx = slots["slot_idx"]
    assert len(idx) == k and len(np.unique(idx)) == k
    assert set(dn_idx) <= set(idx.tolist())
    assert slots["n_dn"] + slots["n_input"] + slots["n_variance"] == k
    assert set(loud.tolist()) <= set(idx.tolist())  # loudest neurons make the variance fill
    assert slots["kinds"][:len(dn_idx)] == ["dn"] * len(dn_idx)


def test_choose_slots_never_double_counts_a_dn_that_is_also_an_input():
    activity = [np.zeros((5, 50), dtype=np.uint8)]
    slots = choose_slots(activity, dn_idx=np.arange(5), input_idx=np.arange(0, 10), k=20)
    assert len(np.unique(slots["slot_idx"])) == 20


def test_normalize_positions_centers_and_scales():
    pos = np.array([[0.0, 0, 0], [10, 0, 0], [0, 4, 0]], dtype=np.float32)
    out = normalize_positions(pos)
    assert out.dtype == np.float32
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(out, axis=1).max(), 1.0, atol=1e-6)


def test_scope_channels_are_named_for_every_super_class():
    names = scope_channels({"super_class_names": ["central", "optic"]})
    assert names == ["all", "dn", "retina_drive", "sc:central", "sc:optic"]


@pytest.mark.skipif(not REAL_DATA, reason="requires real processed data")
def test_choose_retina_channels_are_valid_and_spread():
    graph_np = dict(np.load("data/processed/graph.npz"))
    meta = json.loads(Path("data/processed/graph_meta.json").read_text())
    photo_map = build_photoreceptor_map(graph_np, meta["type_names"])

    idx = choose_retina_channels(photo_map, 512)
    assert len(idx) == 512 and len(np.unique(idx)) == 512
    pos = photo_map.image_pos[idx]
    assert not np.isnan(pos).any()
    assert photo_map.included_mask[idx].all()
    # image_pos is normalized to [0,1]: the sample spans the frame rather
    # than clustering in one corner.
    assert np.ptp(pos[:, 0]) > 0.5 and np.ptp(pos[:, 1]) > 0.5


@pytest.mark.skipif(not REAL_DATA, reason="requires real processed data")
def test_recorded_step_matches_run_clip():
    """step_frame_recorded must be the same computation as step_frame: the
    exported replay is only honest if the recording pass is the eval pass."""
    cfg_game = load_config("configs/game.yaml")
    cfg_brain = load_config("configs/brain.yaml")
    graph_np = dict(np.load("data/processed/graph.npz"))
    meta = json.loads(Path("data/processed/graph_meta.json").read_text())
    photo_map = build_photoreceptor_map(graph_np, meta["type_names"])
    env = VecRhythmEnv(cfg_game, photo_map)
    device, dtype = torch.device("cpu"), torch.float32
    torch.manual_seed(0)
    policy = build_policy("connectome", {}, cfg_game, cfg_brain, photo_map, device, dtype)

    notes = np.array([(0.3, 1, 0.0, 0), (0.6, 2, 0.0, 0)], dtype=NOTE_DTYPE)
    song = Song("sid", "t", "a", "c", "Medium", notes, 1.0)
    frames = 12

    _, logits_ref, _ = run_clip(policy, env, [song], 0, frames, device, dtype,
                                with_grad=False, gradient_checkpointing=False)

    env.reset([song])
    policy.reset_stream(1)
    state = policy.init_state(1, device, dtype)
    rec_logits = []
    with torch.no_grad():
        for _ in range(frames):
            obs, _, _, _ = env.step(np.zeros((1, 6), dtype=bool))
            state, logits, i_photo, rates = policy.step_frame_recorded(
                state, torch.from_numpy(obs["frame"]).to(device)
            )
            rec_logits.append(logits)
            assert rates.shape == (1, policy.brain.n_nodes)
            assert i_photo.shape == (1, policy.brain.n_input)
            assert float(rates.min()) >= 0.0 and float(rates.max()) <= 1.0

    assert torch.equal(torch.stack(rec_logits, dim=1), logits_ref)


@pytest.mark.skipif(not REAL_DATA, reason="requires real processed data")
def test_frame_step_recorded_rate_mean_matches_dn_slice():
    """The recorded whole-population mean and the DN mean the readout sees
    are the same numbers on the DN rows (one implementation, not two)."""
    cfg_brain = load_config("configs/brain.yaml")
    from flyhero.brain.rate_model import ConnectomeBrain, load_graph_and_meta

    graph, _ = load_graph_and_meta("data/processed")
    brain = ConnectomeBrain(graph, cfg_brain, device="cpu", dtype=torch.float32)
    v = brain.init_state(1)
    i_photo = torch.full((1, brain.n_input), 0.05)

    v1, dn1 = brain.frame_step(v.clone(), i_photo, n_substeps=2)
    v2, dn2, all_rates = brain.frame_step_recorded(v.clone(), i_photo, n_substeps=2)
    assert torch.equal(v1, v2) and torch.equal(dn1, dn2)
    assert torch.equal(all_rates[:, brain.dn_idx], dn2)
