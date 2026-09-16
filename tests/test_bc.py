"""Tests for train/bc.py's state-carrying between burn-in and the gradient
window (PLAN Phase 3b, user-requested regression guard against the failure
mode chaos_check.py's reliability check would otherwise mask: if the
gradient window silently re-initialized state instead of carrying the
burn-in result forward, DN/hidden-state traces would still look "reliable"
in that check for the wrong reason)."""

from __future__ import annotations

import numpy as np
import torch

from flyhero.brain.baseline_gru import GRUBaseline
from train.bc import run_clip


class FakeEnv:
    """Duck-types VecRhythmEnv's reset/step interface with deterministic,
    strictly-increasing frame content (no real song/graph data needed) --
    frame value = a per-env-element counter, so two independently-run
    FakeEnv instances (same construction args) produce byte-identical frame
    sequences."""

    def __init__(self, frame_size: int):
        self.frame_size = frame_size

    def reset(self, clips):
        self.batch = len(clips)
        self.t = 0
        return self._obs()

    def _obs(self):
        frame = np.full((self.batch, self.frame_size, self.frame_size), self.t % 256, dtype=np.uint8)
        return {"frame": frame}

    def step(self, actions):
        self.t += 1
        return self._obs(), np.zeros(self.batch), np.zeros(self.batch, dtype=bool), [{}] * self.batch


def test_run_clip_carries_burn_in_state_into_gradient_window():
    torch.manual_seed(0)
    frame_size = 8
    policy = GRUBaseline(frame_size=frame_size, cnn_out_dim=16, gru_hidden=12)
    device, dtype = torch.device("cpu"), torch.float32
    burn_in_frames, train_frames = 5, 3
    clips = [object(), object()]  # FakeEnv.reset only needs len(clips)

    # Path A: run_clip end-to-end (what bc.py actually calls).
    env_a = FakeEnv(frame_size)
    state_after_a, logits_a, _ = run_clip(
        policy, env_a, clips, burn_in_frames, train_frames, device, dtype,
        with_grad=False, gradient_checkpointing=False,
    )

    # Path B: manually replay the same burn-in, capture the state, then feed
    # it as the explicit starting point for just the gradient window's first
    # frame, on a second (identically-deterministic) FakeEnv.
    env_b = FakeEnv(frame_size)
    env_b.reset(clips)
    policy.reset_stream(len(clips))
    state_b = policy.init_state(len(clips), device, dtype)
    with torch.no_grad():
        for _ in range(burn_in_frames):
            obs, _, _, _ = env_b.step(np.zeros((len(clips), 6), dtype=bool))
            frame_t = torch.from_numpy(obs["frame"]).to(device)
            state_b, _ = policy.step_frame(state_b, frame_t)
    state_after_burn_in_b = state_b.detach().clone()

    obs, _, _, _ = env_b.step(np.zeros((len(clips), 6), dtype=bool))
    frame_t = torch.from_numpy(obs["frame"]).to(device)
    with torch.no_grad():
        _, logits_frame0_manual = policy.step_frame(state_after_burn_in_b, frame_t)

    # The bug this guards against: if run_clip's gradient window used a
    # fresh init_state() instead of the carried burn-in state, frame 0's
    # logits would differ from this manual replay (which explicitly starts
    # from the burn-in-derived state).
    torch.testing.assert_close(logits_a[:, 0, :], logits_frame0_manual)


def test_run_clip_zero_burn_in_frames_state_equals_init_state():
    """Sanity check on the same code path: with burn_in_frames=0, frame 0 of
    the gradient window must see the model's fresh init_state (nothing to
    carry), not some stale/uninitialized value."""
    torch.manual_seed(0)
    frame_size = 8
    policy = GRUBaseline(frame_size=frame_size, cnn_out_dim=16, gru_hidden=12)
    device, dtype = torch.device("cpu"), torch.float32
    clips = [object()]

    env = FakeEnv(frame_size)
    _, logits, _ = run_clip(
        policy, env, clips, burn_in_frames=0, train_frames=1, device=device, dtype=dtype,
        with_grad=False, gradient_checkpointing=False,
    )

    fresh_state = policy.init_state(1, device, dtype)
    env2 = FakeEnv(frame_size)
    env2.reset(clips)
    obs, _, _, _ = env2.step(np.zeros((1, 6), dtype=bool))
    frame_t = torch.from_numpy(obs["frame"]).to(device)
    with torch.no_grad():
        _, expected_logits = policy.step_frame(fresh_state, frame_t)

    torch.testing.assert_close(logits[:, 0, :], expected_logits)
