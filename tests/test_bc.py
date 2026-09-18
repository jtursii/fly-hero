"""Tests for train/bc.py's state-carrying between burn-in and the gradient
window (PLAN Phase 3b, user-requested regression guard against the failure
mode chaos_check.py's reliability check would otherwise mask: if the
gradient window silently re-initialized state instead of carrying the
burn-in result forward, DN/hidden-state traces would still look "reliable"
in that check for the wrong reason)."""

from __future__ import annotations

import numpy as np
import pytest
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


def test_checkpoint_round_trip_restores_full_training_state(tmp_path):
    """PLAN Phase 3b resume-verification task: a checkpoint must restore not
    just weights but optimizer moment state (so Adam doesn't restart cold),
    step/difficulty_idx/pos_weight, and both RNG streams (so resumed
    sampling/mixing is a continuation, not a reset) -- exercised end to end
    against train/common.save_checkpoint and the same loading logic
    train/bc.py's --resume path uses."""
    from train.common import save_checkpoint

    torch.manual_seed(0)
    policy = GRUBaseline(frame_size=8, cnn_out_dim=16, gru_hidden=12)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(policy.readout_parameters()), "lr": 1e-3},
            {"params": list(policy.non_readout_parameters()), "lr": 1e-4},
        ]
    )
    numpy_rng = np.random.default_rng(123)

    # Take a couple of real optimizer steps so Adam's moment buffers are
    # non-trivial (a checkpoint saved before any step would trivially
    # "round-trip" zeroed state and wouldn't catch a broken optimizer_state
    # restore).
    for _ in range(3):
        x = torch.randn(2, 12)
        loss = policy.readout(x).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    _ = numpy_rng.random(5)  # advance the RNG stream before checkpointing, like real training would

    ckpt_path = tmp_path / "checkpoint_step3.pt"
    save_checkpoint(
        ckpt_path, policy, optimizer, step=3, difficulty_idx=1, pos_weight=17.5,
        numpy_rng=numpy_rng, extra={"reason": "test"},
    )

    # What the *original* rng/torch streams would produce next, to compare
    # a restored stream against -- this is the actual continuity check.
    expected_next_numpy = numpy_rng.random(4)
    expected_next_torch = torch.rand(4)

    # Fresh model/optimizer/rng, deliberately diverged from the originals
    # (different init, different RNG position) so a no-op "restore" would
    # fail every assertion below.
    torch.manual_seed(999)
    fresh_policy = GRUBaseline(frame_size=8, cnn_out_dim=16, gru_hidden=12)
    fresh_optimizer = torch.optim.AdamW(
        [
            {"params": list(fresh_policy.readout_parameters()), "lr": 1e-3},
            {"params": list(fresh_policy.non_readout_parameters()), "lr": 1e-4},
        ]
    )
    fresh_rng = np.random.default_rng(999)
    _ = fresh_rng.random(50)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    fresh_policy.load_state_dict(ckpt["model_state"])
    fresh_optimizer.load_state_dict(ckpt["optimizer_state"])
    step = ckpt["step"]
    difficulty_idx = ckpt["difficulty_idx"]
    pos_weight = ckpt["pos_weight"]
    fresh_rng.bit_generator.state = ckpt["numpy_rng_state"]
    torch.set_rng_state(ckpt["torch_rng_state"].cpu())

    assert step == 3
    assert difficulty_idx == 1
    assert pos_weight == 17.5

    for (n1, p1), (n2, p2) in zip(policy.state_dict().items(), fresh_policy.state_dict().items()):
        assert n1 == n2
        torch.testing.assert_close(p1, p2)

    # Optimizer state restored, not reset: Adam's per-parameter exp_avg/
    # exp_avg_sq moment buffers and step counts must match exactly, not just
    # the LR config (LR config alone would "round-trip" even with a
    # completely broken optimizer_state load).
    orig_state, restored_state = optimizer.state_dict()["state"], fresh_optimizer.state_dict()["state"]
    assert set(orig_state.keys()) == set(restored_state.keys())
    for k in orig_state:
        assert orig_state[k]["step"] == restored_state[k]["step"]
        torch.testing.assert_close(orig_state[k]["exp_avg"], restored_state[k]["exp_avg"])
        torch.testing.assert_close(orig_state[k]["exp_avg_sq"], restored_state[k]["exp_avg_sq"])
    orig_lrs = [g["lr"] for g in optimizer.state_dict()["param_groups"]]
    restored_lrs = [g["lr"] for g in fresh_optimizer.state_dict()["param_groups"]]
    assert orig_lrs == restored_lrs

    # RNG streams resume exactly where they left off, not from a fresh seed.
    np.testing.assert_array_equal(fresh_rng.random(4), expected_next_numpy)
    torch.testing.assert_close(torch.rand(4), expected_next_torch)


# ---------------------------------------------------------------------------
# D36: LR schedule, rollback on base_lr, MA curriculum, --resume best
# ---------------------------------------------------------------------------


def test_lr_factor_cosine_to_quarter_over_20k():
    from train.bc import lr_factor

    sched = dict(start_step=42000, duration_steps=20000, final_frac=0.25)
    assert lr_factor(41000, sched) == 1.0
    assert lr_factor(42000, sched) == pytest.approx(1.0)
    assert lr_factor(52000, sched) == pytest.approx(0.625)
    assert lr_factor(62000, sched) == pytest.approx(0.25)
    assert lr_factor(90000, sched) == pytest.approx(0.25)
    assert lr_factor(50000, None) == 1.0


def test_schedule_scales_base_lr_and_rollback_halves_it(tmp_path):
    from train.bc import apply_lr_schedule, perform_rollback
    from train.common import save_checkpoint
    from train.watchdog import GradWatchdog

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.readout = torch.nn.Linear(2, 2)
            self.brain = torch.nn.Linear(2, 2)

        def readout_parameters(self):
            return self.readout.parameters()

        def non_readout_parameters(self):
            return self.brain.parameters()

    policy = Tiny()
    opt = torch.optim.AdamW([
        {"params": list(policy.readout_parameters()), "lr": 1e-3},
        {"params": list(policy.non_readout_parameters()), "lr": 5e-5},  # pre-D36: no base_lr
    ])
    sched = dict(start_step=0, duration_steps=100, final_frac=0.25)
    apply_lr_schedule(opt, 50, sched)
    assert opt.param_groups[0]["base_lr"] == 1e-3
    assert opt.param_groups[1]["lr"] == pytest.approx(5e-5 * 0.625)

    path = tmp_path / "healthy.pt"
    save_checkpoint(path, policy, opt, 0, 0, 1.0, np.random.default_rng(0), extra={})
    opt2 = perform_rollback(policy, opt, path, GradWatchdog(), torch.device("cpu"))
    apply_lr_schedule(opt2, 50, sched)
    assert opt2.param_groups[1]["base_lr"] == pytest.approx(2.5e-5)
    assert opt2.param_groups[1]["lr"] == pytest.approx(2.5e-5 * 0.625)
    assert opt2.param_groups[0]["base_lr"] == 1e-3
    # base_lr survives a checkpoint round trip
    save_checkpoint(path, policy, opt2, 0, 0, 1.0, np.random.default_rng(0), extra={})
    opt3 = torch.optim.AdamW([
        {"params": list(policy.readout_parameters()), "lr": 9.0},
        {"params": list(policy.non_readout_parameters()), "lr": 9.0},
    ])
    opt3.load_state_dict(torch.load(path)["optimizer_state"])
    assert opt3.param_groups[1]["base_lr"] == pytest.approx(2.5e-5)


def test_curriculum_advances_on_3_eval_moving_average_only():
    from train.bc import should_advance_curriculum
    from train.watchdog import GradWatchdog

    wd = GradWatchdog(eval_ma_window=3)
    wd.record_eval("Medium", 0.85)  # a single high eval is not enough
    assert not should_advance_curriculum(wd, "Medium", 0.75)
    wd.record_eval("Medium", 0.70)
    wd.record_eval("Medium", 0.69)  # MA 0.7467
    assert not should_advance_curriculum(wd, "Medium", 0.75)
    wd.record_eval("Medium", 0.77)  # MA (0.70+0.69+0.77)/3 = 0.72
    assert not should_advance_curriculum(wd, "Medium", 0.75)
    wd.record_eval("Medium", 0.80)
    wd.record_eval("Medium", 0.76)  # MA 0.7767
    assert should_advance_curriculum(wd, "Medium", 0.75)


def test_resume_best_resolves_and_reuses_recorded_config(tmp_path, monkeypatch):
    import sys

    from train.bc import parse_args

    (tmp_path / "train_config.txt").write_text("configs/bc_full_real_v2.yaml\n")
    monkeypatch.setattr(sys, "argv", ["bc", "--resume", "best", "--run-dir", str(tmp_path), "--set-lr-brain", "2.5e-5"])
    args = parse_args()
    assert args.resume == str(tmp_path / "checkpoint_best.pt")
    assert args.config == "configs/bc_full_real_v2.yaml"
    assert args.set_lr_brain == 2.5e-5
    monkeypatch.setattr(sys, "argv", ["bc", "--resume", "latest", "--run-dir", str(tmp_path), "--config", "x.yaml"])
    assert parse_args().config == "x.yaml"
    monkeypatch.setattr(sys, "argv", ["bc", "--set-lr-brain", "1e-5"])
    with pytest.raises(SystemExit):
        parse_args()


def test_run_clip_tbptt_same_forward_gradient_truncated():
    """D40: tbptt_frames detaches the carried state every k gradient-window
    frames. Forward logits are unchanged; the gradient of frame t's output
    equals the one from a manual replay that starts tracking at the last
    detach point before t (and differs from full BPTT)."""
    torch.manual_seed(0)
    frame_size = 8
    policy = GRUBaseline(frame_size=frame_size, cnn_out_dim=16, gru_hidden=12)
    device, dtype = torch.device("cpu"), torch.float32
    clips = [object(), object()]
    burn_in, train, k, t = 3, 6, 2, 3

    def grads(loss):
        policy.zero_grad()
        loss.backward()
        return [p.grad.clone() for p in policy.parameters()]

    _, logits_full, _ = run_clip(policy, FakeEnv(frame_size), clips, burn_in, train, device, dtype,
                                 with_grad=True, gradient_checkpointing=False)
    g_full = grads(logits_full[:, t].sum())
    _, logits_tr, _ = run_clip(policy, FakeEnv(frame_size), clips, burn_in, train, device, dtype,
                               with_grad=True, gradient_checkpointing=False, tbptt_frames=k)
    g_tr = grads(logits_tr[:, t].sum())
    torch.testing.assert_close(logits_tr, logits_full)

    # Manual: frames before the last detach point (t // k * k) without grad.
    env = FakeEnv(frame_size)
    env.reset(clips)
    policy.reset_stream(len(clips))
    state = policy.init_state(len(clips), device, dtype)
    start = t // k * k
    with torch.no_grad():
        for _ in range(burn_in + start):
            obs, _, _, _ = env.step(None)
            state, _ = policy.step_frame(state, torch.from_numpy(obs["frame"]))
    for _ in range(t - start + 1):
        obs, _, _, _ = env.step(None)
        state, logits = policy.step_frame(state, torch.from_numpy(obs["frame"]))
    g_manual = grads(logits.sum())

    for a, b in zip(g_tr, g_manual):
        torch.testing.assert_close(a, b)
    assert any(not torch.allclose(a, b) for a, b in zip(g_tr, g_full))
