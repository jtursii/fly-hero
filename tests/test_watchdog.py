"""Tests for the overnight-stability watchdog (docs/DECISIONS.md D30/D31):
GradWatchdog's pure decision logic (no model needed), plus an end-to-end
skip -> rollback -> LR-halve -> clean-stop integration test on a tiny model,
driven through the real train/bc.py functions (apply_gradient_step,
perform_rollback, write_stopped_file) rather than a reimplementation of
their logic.
"""

from __future__ import annotations

import torch

from train.bc import apply_gradient_step, perform_rollback, write_stopped_file
from train.common import save_checkpoint
from train.watchdog import GradWatchdog

# ---------------------------------------------------------------------------
# Pure logic (no model, no I/O)
# ---------------------------------------------------------------------------


def _fill_baseline(wd: GradWatchdog, value: float = 100.0) -> None:
    for _ in range(wd.window):
        should_skip, _ = wd.record_step(value, n_bad_elems=0)
        assert not should_skip
    assert wd.healthy_baseline is not None


def test_ratio_spike_is_skipped_and_excluded_from_baseline():
    wd = GradWatchdog(window=10, skip_ratio=20.0)
    _fill_baseline(wd, value=100.0)
    median_before = wd.rolling_median()

    should_skip, reason = wd.record_step(100.0 * 21.0, n_bad_elems=0)
    assert should_skip and reason == "ratio_spike"
    # the spike must not have entered the rolling window
    assert wd.rolling_median() == median_before


def test_nonfinite_grad_norm_is_skipped():
    wd = GradWatchdog(window=5)
    should_skip, reason = wd.record_step(float("nan"), n_bad_elems=0)
    assert should_skip and reason == "nonfinite_grad_norm"
    should_skip, reason = wd.record_step(float("inf"), n_bad_elems=0)
    assert should_skip and reason == "nonfinite_grad_norm"


def test_nonfinite_excess_is_skipped_even_with_normal_grad_norm():
    wd = GradWatchdog(window=10, max_expected_nonfinite=20)
    _fill_baseline(wd, value=50.0)
    should_skip, reason = wd.record_step(50.0, n_bad_elems=21)
    assert should_skip and reason == "nonfinite_excess"


def test_moderate_step_is_not_skipped():
    wd = GradWatchdog(window=10, max_expected_nonfinite=20, skip_ratio=20.0)
    _fill_baseline(wd, value=100.0)
    should_skip, reason = wd.record_step(150.0, n_bad_elems=3)
    assert not should_skip and reason is None


def test_sustained_elevation_triggers_rollback_after_consecutive_steps():
    wd = GradWatchdog(window=5, rollback_median_ratio=5.0, rollback_consecutive_steps=5)
    _fill_baseline(wd, value=10.0)

    # Too early: the rolling window is still mostly the old baseline value.
    for _ in range(2):
        wd.record_step(60.0, n_bad_elems=0)
        assert not wd.should_rollback_from_steps()

    fired = False
    for _ in range(20):
        wd.record_step(60.0, n_bad_elems=0)
        if wd.should_rollback_from_steps():
            fired = True
            break
    assert fired


def test_transient_elevation_does_not_accumulate_across_a_recovery():
    wd = GradWatchdog(window=5, rollback_median_ratio=5.0, rollback_consecutive_steps=100)
    _fill_baseline(wd, value=10.0)
    for _ in range(5):
        wd.record_step(60.0, n_bad_elems=0)
    assert wd.consecutive_bad_steps > 0
    # a full window's worth of recovery evicts every elevated entry.
    for _ in range(wd.window):
        wd.record_step(10.0, n_bad_elems=0)
    assert wd.consecutive_bad_steps == 0


def test_is_healthy_now_hysteresis():
    wd = GradWatchdog(window=5, rollback_median_ratio=5.0, healthy_ratio=2.0)
    assert wd.is_healthy_now()  # no baseline yet -- earliest checkpoints are trusted
    _fill_baseline(wd, value=10.0)
    assert wd.is_healthy_now()
    # push the window to 3x baseline: above healthy_ratio(2x), below rollback(5x)
    for _ in range(5):
        wd.record_step(30.0, n_bad_elems=0)
    assert not wd.is_healthy_now()
    assert not wd.should_rollback_from_steps()


def test_eval_drop_triggers_rollback_after_two_consecutive():
    wd = GradWatchdog(rollback_eval_drop=0.15, rollback_eval_consecutive=2)
    assert wd.record_eval("Medium", 0.80) is False  # first eval is the new best
    assert wd.record_eval("Medium", 0.90) is False  # improves further
    assert wd.record_eval("Medium", 0.70) is False  # drop 1 (0.20 > 0.15)
    assert wd.record_eval("Medium", 0.72) is True  # drop 2, consecutive -> rollback
    assert wd.best_hit_rate["Medium"] == 0.90


def test_eval_recovery_resets_consecutive_bad_count():
    wd = GradWatchdog(rollback_eval_drop=0.15, rollback_eval_consecutive=2)
    wd.record_eval("Easy", 0.90)
    wd.record_eval("Easy", 0.70)  # drop 1
    wd.record_eval("Easy", 0.85)  # recovers (within 0.15 of best) -> resets
    assert wd.record_eval("Easy", 0.70) is False  # drop 1 again, not 2 consecutive


def test_max_rollbacks_exhaustion():
    wd = GradWatchdog(max_rollbacks=3)
    for _ in range(3):
        assert not wd.exhausted()
        wd.note_rollback()
    assert wd.rollback_count == 3
    assert wd.exhausted()


def test_note_rollback_resets_baseline_and_window():
    wd = GradWatchdog(window=5)
    _fill_baseline(wd, value=10.0)
    wd.consecutive_bad_steps = 4
    wd.note_rollback()
    assert wd.healthy_baseline is None
    assert wd.consecutive_bad_steps == 0
    assert wd.rolling_median() is None


# ---------------------------------------------------------------------------
# Integration: skip -> rollback -> LR halve -> clean stop, on a tiny model,
# through the real bc.py functions. Loss is built so each parameter's
# gradient equals a constant `coef` per element (d(coef*p.sum())/dp = coef,
# independent of the current parameter value) -- this makes grad_norm exactly
# reproducible across steps regardless of how the optimizer has moved the
# weights, so the test doesn't depend on a real network's chaotic training
# dynamics to hit a specific ratio.
# ---------------------------------------------------------------------------


class _TinyPolicy(torch.nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.readout = torch.nn.Linear(2, 2)
        self.brain = torch.nn.Linear(2, 2)

    def readout_parameters(self):
        return self.readout.parameters()

    def non_readout_parameters(self):
        return list(self.brain.parameters())


def _build_tiny_policy_and_optimizer():
    policy = _TinyPolicy()
    optimizer = torch.optim.AdamW(
        [
            {"params": list(policy.readout_parameters()), "lr": 1e-3},
            {"params": list(policy.non_readout_parameters()), "lr": 1e-2},
        ]
    )
    return policy, optimizer


def _coef_loss(policy, coef: float) -> torch.Tensor:
    brain_term = sum(p.sum() for p in policy.non_readout_parameters())
    readout_term = sum(p.sum() for p in policy.readout_parameters())
    return coef * brain_term + 0.01 * readout_term


def test_end_to_end_skip_then_rollback_then_lr_halve_then_clean_stop(tmp_path):
    policy, optimizer = _build_tiny_policy_and_optimizer()
    watchdog = GradWatchdog(window=5, rollback_consecutive_steps=5, rollback_median_ratio=5.0, max_rollbacks=2)
    device = torch.device("cpu")

    # 1) establish a healthy baseline (grad_norm_brain == coef * sqrt(numel),
    # identical every step by construction) and checkpoint it.
    for _ in range(watchdog.window):
        loss = _coef_loss(policy, coef=1.0)
        result = apply_gradient_step(policy, optimizer, loss, grad_clip_norm=1e6, grad_outlier_clip=1e4, watchdog=watchdog)
        assert not result["should_skip"]
    assert watchdog.healthy_baseline is not None

    healthy_path = tmp_path / "checkpoint_last_healthy.pt"
    import numpy as np

    save_checkpoint(healthy_path, policy, optimizer, step=100, difficulty_idx=0, pos_weight=1.0,
                     numpy_rng=np.random.default_rng(0), extra={})
    healthy_weight_snapshot = {k: v.clone() for k, v in policy.state_dict().items()}

    # 2) one catastrophic step -> must be skipped, weights must not move.
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    loss = _coef_loss(policy, coef=1e20)
    result = apply_gradient_step(policy, optimizer, loss, grad_clip_norm=1e6, grad_outlier_clip=1e4, watchdog=watchdog)
    assert result["should_skip"], result
    after = policy.state_dict()
    for k in before:
        torch.testing.assert_close(before[k], after[k])

    # 3) a sustained 6x-baseline elevation (each step's grad_norm is exactly
    # 6x baseline by construction, so this reproduces the pure-logic test's
    # math above deterministically) -> eventually triggers a rollback.
    rollback_fired = False
    for _ in range(20):
        loss = _coef_loss(policy, coef=6.0)
        apply_gradient_step(policy, optimizer, loss, grad_clip_norm=1e6, grad_outlier_clip=1e4, watchdog=watchdog)
        if watchdog.should_rollback_from_steps():
            rollback_fired = True
            break
    assert rollback_fired, "expected sustained elevation to eventually trigger a rollback"

    pre_rollback_lr = optimizer.param_groups[1]["lr"]
    optimizer = perform_rollback(policy, optimizer, healthy_path, watchdog, device)

    # 4) weights restored, brain LR halved, optimizer moments reset, count bumped.
    for k, v in policy.state_dict().items():
        torch.testing.assert_close(v, healthy_weight_snapshot[k])
    assert optimizer.param_groups[1]["lr"] == pre_rollback_lr / 2.0
    assert all(len(optimizer.state[p]) == 0 for group in optimizer.param_groups for p in group["params"])
    assert watchdog.rollback_count == 1
    assert not watchdog.exhausted()

    # 5) a second sustained-elevation episode -> second rollback -> budget
    # exhausted -> the caller's clean-stop path (bc.py's main()) checkpoints
    # and writes STOPPED.txt; we test write_stopped_file directly here.
    for _ in range(watchdog.window):
        loss = _coef_loss(policy, coef=1.0)
        apply_gradient_step(policy, optimizer, loss, grad_clip_norm=1e6, grad_outlier_clip=1e4, watchdog=watchdog)
    rollback_fired = False
    for _ in range(20):
        loss = _coef_loss(policy, coef=6.0)
        apply_gradient_step(policy, optimizer, loss, grad_clip_norm=1e6, grad_outlier_clip=1e4, watchdog=watchdog)
        if watchdog.should_rollback_from_steps():
            rollback_fired = True
            break
    assert rollback_fired
    optimizer = perform_rollback(policy, optimizer, healthy_path, watchdog, device)
    assert watchdog.rollback_count == 2
    assert watchdog.exhausted()

    write_stopped_file(tmp_path, "2 rollbacks exhausted (last reason=sustained_grad_elevation)", 999)
    stopped_text = (tmp_path / "STOPPED.txt").read_text()
    assert "999" in stopped_text and "exhausted" in stopped_text


# ---------------------------------------------------------------------------
# D33: state persistence across stop/resume
# ---------------------------------------------------------------------------


def test_state_dict_round_trip_through_checkpoint(tmp_path):
    import numpy as np

    wd = GradWatchdog(window=5)
    _fill_baseline(wd, value=100.0)
    wd.record_step(3000.0, n_bad_elems=0)  # skip
    wd.record_eval("Medium", 0.52)
    wd.record_eval("Medium", 0.30)  # one bad eval pending
    wd.note_rollback()
    for v in (90.0, 110.0):
        wd.record_step(v, n_bad_elems=0)

    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW([{"params": [model.weight], "lr": 1e-3}, {"params": [model.bias], "lr": 5e-5}])
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, 7, 1, 2.0, np.random.default_rng(0), extra={}, watchdog_state=wd.state_dict())

    ckpt = torch.load(path)
    restored = GradWatchdog(window=5)
    restored.load_state_dict(ckpt["watchdog_state"])
    assert restored.state_dict() == wd.state_dict()
    assert restored.rollback_count == 1 and restored.best_hit_rate == {"Medium": 0.52}
    assert restored.rolling_median() == wd.rolling_median()
    # the halved brain LR is restored by the optimizer state, not the watchdog
    opt2 = torch.optim.AdamW([{"params": [model.weight], "lr": 1e-3}, {"params": [model.bias], "lr": 1e-4}])
    opt2.load_state_dict(ckpt["optimizer_state"])
    assert opt2.param_groups[1]["lr"] == 5e-5


def test_reconstruct_from_metrics_respects_step_bound(tmp_path):
    import json

    from train.watchdog import reconstruct_from_metrics

    lines = [
        dict(step=1000, eval_difficulty="Easy", eval_hit_rate=0.8),
        dict(step=2000, eval_difficulty="Medium", eval_hit_rate=0.45),
        dict(step=2500, rollback=True),
        dict(step=3000, eval_difficulty="Medium", eval_hit_rate=0.6),
        dict(step=3100, rollback=True),
    ]
    p = tmp_path / "metrics.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in lines) + "\nnot json\n")
    rec = reconstruct_from_metrics(p, up_to_step=2999)
    assert rec == dict(best_hit_rate={"Easy": 0.8, "Medium": 0.45}, rollback_count=1)
    assert reconstruct_from_metrics(tmp_path / "missing.jsonl", 10) == dict(best_hit_rate={}, rollback_count=0)


def test_existing_run_dirs_matches_exact_name_only(tmp_path):
    from train.bc import existing_run_dirs

    for name in ("20260916_123616_bc_full_real", "20260916_222606_bc_full_real_v2", "20260917_010101_smoke_d32"):
        (tmp_path / name).mkdir()
    assert [d.name for d in existing_run_dirs("bc_full_real_v2", tmp_path)] == ["20260916_222606_bc_full_real_v2"]
    assert [d.name for d in existing_run_dirs("bc_full_real", tmp_path)] == ["20260916_123616_bc_full_real"]
    assert existing_run_dirs("resume_test_v2", tmp_path) == []
