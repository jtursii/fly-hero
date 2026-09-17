"""Tests for the overnight-stability watchdog (docs/DECISIONS.md D30/D31):
GradWatchdog's pure decision logic (no model needed), plus an end-to-end
skip -> rollback -> LR-halve -> clean-stop integration test on a tiny model,
driven through the real train/bc.py functions (apply_gradient_step,
perform_rollback, write_stopped_file) rather than a reimplementation of
their logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from train.bc import apply_gradient_step, perform_rollback, write_stopped_file
from train.common import save_checkpoint
from train.watchdog import GradWatchdog, baseline_from_metrics, load_metrics_lineage, replay_on_metrics

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
    wd = GradWatchdog(window=5, healthy_window=10, rollback_median_ratio=5.0, healthy_ratio=2.0)
    assert wd.is_healthy_now()  # no baseline yet -- earliest checkpoints are trusted
    _fill_baseline(wd, value=10.0)
    assert not wd.is_healthy_now()  # baseline set, but the 10-step window isn't full yet
    for _ in range(5):
        wd.record_step(10.0, n_bad_elems=0)
    assert wd.is_healthy_now()
    # push the long window to 3x baseline: above healthy_ratio(2x), below rollback(5x)
    for _ in range(10):
        wd.record_step(30.0, n_bad_elems=0)
    assert not wd.is_healthy_now()
    assert not wd.should_rollback_from_steps()


def test_healthy_needs_the_long_window_not_just_the_rolling_median():
    # D35 / v2 step 42000: the 200-step median had dipped back under 2x
    # baseline, but the preceding 1000 steps were a bad stretch.
    wd = GradWatchdog(window=5, healthy_window=20, baseline_grad_norm=10.0, rollback_median_ratio=3.0)
    for _ in range(15):
        wd.record_step(80.0, n_bad_elems=0)
    for _ in range(5):
        wd.record_step(12.0, n_bad_elems=0)
    assert wd.rolling_median() < 2 * 10.0
    assert wd.long_median() >= 2 * 10.0
    assert not wd.is_healthy_now()


def test_not_healthy_while_rolling_median_is_above_trigger():
    # v2 step 40000: the escalation had just started, so the 1000-step median
    # was still low while the 200-step median was already above 3x baseline.
    wd = GradWatchdog(window=5, healthy_window=20, baseline_grad_norm=10.0, rollback_median_ratio=3.0)
    for _ in range(17):
        wd.record_step(10.0, n_bad_elems=0)
    for _ in range(3):
        wd.record_step(50.0, n_bad_elems=0)
    assert wd.long_median() < 2 * 10.0
    assert wd.rolling_median() > 3 * 10.0
    assert not wd.is_healthy_now()


def test_default_trigger_is_3x_for_200_steps():
    wd = GradWatchdog(baseline_grad_norm=100.0)
    assert (wd.rollback_median_ratio, wd.rollback_consecutive_steps, wd.healthy_window) == (3.0, 200, 1000)
    for _ in range(200):
        wd.record_step(100.0, n_bad_elems=0)
    # 2.9x never fires
    for _ in range(1000):
        wd.record_step(290.0, n_bad_elems=0)
        assert not wd.should_rollback_from_steps()
    # 3.5x: the rolling median (avg of the two middle entries) crosses 3x once
    # 100 of the 200 entries are elevated (step 100), then must stay there for
    # 200 consecutive steps (steps 100..299).
    n = 0
    while not wd.should_rollback_from_steps():
        wd.record_step(350.0, n_bad_elems=0)
        n += 1
        assert n < 1000
    assert n == 299


def test_eval_ma_drop_triggers_rollback():
    wd = GradWatchdog(eval_ma_window=3, rollback_eval_drop=0.08)
    for hr in (0.70, 0.72, 0.74):  # MA 0.72 = best
        assert wd.record_eval("Medium", hr) is False
    assert wd.best_eval_ma["Medium"] == pytest.approx(0.72)
    assert wd.record_eval("Medium", 0.60) is False  # MA 0.687, drop 0.033
    assert wd.record_eval("Medium", 0.62) is False  # MA 0.653, drop 0.067
    assert wd.record_eval("Medium", 0.55) is True  # MA 0.59, drop 0.13 > 0.08
    assert wd.best_hit_rate["Medium"] == 0.74


def test_single_bad_eval_does_not_trigger():
    # a one-off 0.2 dip (like v2's 0.635 at 41000 between 0.718 and 0.732)
    # moves a 3-eval MA by only ~0.067.
    wd = GradWatchdog(eval_ma_window=3, rollback_eval_drop=0.08)
    for hr in (0.72, 0.72, 0.72, 0.52, 0.72, 0.72):
        assert wd.record_eval("Easy", hr) is False


def test_eval_ma_waits_for_a_full_window_and_is_per_difficulty():
    wd = GradWatchdog(eval_ma_window=3)
    wd.record_eval("Easy", 0.9)
    wd.record_eval("Easy", 0.9)
    assert wd.eval_ma("Easy") is None
    wd.record_eval("Medium", 0.1)
    wd.record_eval("Easy", 0.6)
    assert wd.eval_ma("Easy") == pytest.approx(0.8)
    assert wd.eval_ma("Medium") is None


def test_max_rollbacks_exhaustion():
    wd = GradWatchdog(max_rollbacks=3)
    for _ in range(3):
        assert not wd.exhausted()
        wd.note_rollback()
    assert wd.rollback_count == 3
    assert wd.exhausted()


def test_note_rollback_keeps_frozen_baseline_and_clears_windows():
    # D35: re-measuring the baseline right after a rollback (old behavior)
    # captured a still-turbulent median in v2 (~740 vs ~340 healthy).
    wd = GradWatchdog(window=5)
    _fill_baseline(wd, value=10.0)
    for hr in (0.5, 0.6, 0.7):
        wd.record_eval("Medium", hr)
    wd.consecutive_bad_steps = 4
    wd.note_rollback()
    assert wd.healthy_baseline == 10.0
    assert wd.consecutive_bad_steps == 0
    assert wd.rolling_median() is None and wd.long_median() is None
    assert wd.eval_ma("Medium") is None
    assert wd.best_eval_ma["Medium"] == pytest.approx(0.6)
    # a turbulent post-rollback window does not move the baseline
    for _ in range(10):
        wd.record_step(74.0, n_bad_elems=0)
    assert wd.healthy_baseline == 10.0


def test_configured_baseline_is_used_immediately_and_wins_over_saved_state():
    wd = GradWatchdog(window=5, baseline_grad_norm=339.4)
    assert wd.healthy_baseline == 339.4
    for _ in range(5):
        wd.record_step(900.0, n_bad_elems=0)
    assert wd.healthy_baseline == 339.4  # first full window doesn't overwrite it
    other = GradWatchdog(window=5)
    _fill_baseline(other, value=740.0)
    wd.load_state_dict(other.state_dict())
    assert wd.healthy_baseline == 339.4
    unconfigured = GradWatchdog(window=5)
    unconfigured.load_state_dict(other.state_dict())
    assert unconfigured.healthy_baseline == 740.0


# ---------------------------------------------------------------------------
# D35: replay on bc_full_real_v2's real metrics (steps 37000-42300, trimmed
# fields; tests/fixtures/v2_metrics_37k_42k.jsonl). The 40k-41.8k escalation
# was not caught by the old rules.
# ---------------------------------------------------------------------------

V2_FIXTURE = Path(__file__).parent / "fixtures" / "v2_metrics_37k_42k.jsonl"
V2_BASELINE = 339.4  # median logged grad_norm over v2 steps 18000-39000 (configs/bc_full_real_v2.yaml)


def test_old_rules_miss_the_v2_40k_episode():
    records = load_metrics_lineage(V2_FIXTURE)
    # old: 5x a baseline re-measured after the 17385 rollback (~740), 300 steps.
    # The run's own logged 200-step median never stays above 3700 that long.
    old = replay_on_metrics(records, 740.0, rollback_median_ratio=5.0, rollback_consecutive_steps=300)
    assert old["live_median_fires"] == []


def test_fixed_rules_fire_on_the_v2_40k_episode():
    records = load_metrics_lineage(V2_FIXTURE)
    out = replay_on_metrics(records, V2_BASELINE)
    # true 200-step median (as logged live): fires ~40040, nothing earlier
    assert out["live_median_fires"] and 40000 <= out["live_median_fires"][0] <= 40100
    # the real GradWatchdog on 1-in-10 samples agrees within ~200 steps
    assert out["grad_fires"] and 39800 <= out["grad_fires"][0] <= 40100
    assert all(s >= 39800 for s in out["grad_fires"])
    # rollback target: 39500 is the last certified checkpoint; 40000-42000 are not
    assert out["healthy"][39000] and out["healthy"][39500]
    assert not any(out["healthy"][s] for s in (40000, 40500, 41000, 41500, 42000))
    # the eval-MA rule does not fire here (largest MA drop is 0.07 < 0.08)
    assert out["eval_fires"] == []


def test_baseline_from_metrics_excludes_skips_and_span_edges():
    records = [
        dict(step=10, loss=0.1, grad_norm=1.0), dict(step=20, loss=0.1, grad_norm=3.0),
        dict(step=30, loss=0.1, grad_norm=1e6, skip=True), dict(step=40, loss=0.1, grad_norm=5.0),
        dict(step=50, loss=0.1, grad_norm=1e6),
    ]
    assert baseline_from_metrics(records, 10, 40) == 3.0


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
    wd.record_eval("Medium", 0.30)
    wd.note_rollback()
    wd.record_eval("Medium", 0.40)
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
    assert rec["best_hit_rate"] == {"Easy": 0.8, "Medium": 0.45}
    assert rec["rollback_count"] == 1
    assert rec["eval_history"] == {"Easy": [], "Medium": []}  # a rollback clears every difficulty's recent evals
    empty = reconstruct_from_metrics(tmp_path / "missing.jsonl", 10)
    assert empty == dict(best_hit_rate={}, rollback_count=0, eval_history={}, best_eval_ma={})


def test_reconstruct_rebuilds_eval_ma_and_drops_abandoned_branch(tmp_path):
    import json

    from train.watchdog import reconstruct_from_metrics

    lines = [
        dict(step=1000, eval_difficulty="Medium", eval_hit_rate=0.70),
        dict(step=2000, eval_difficulty="Medium", eval_hit_rate=0.60),
        dict(step=3000, eval_difficulty="Medium", eval_hit_rate=0.80),
        dict(step=3010, loss=1.0, grad_norm=5.0),
        dict(step=4000, eval_difficulty="Medium", eval_hit_rate=0.10),  # abandoned branch
        dict(step=3000, resumed_from_step=3000),
        dict(step=4000, eval_difficulty="Medium", eval_hit_rate=0.75),
    ]
    p = tmp_path / "metrics.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    assert [r["step"] for r in load_metrics_lineage(p)] == [1000, 2000, 3000, 3000, 4000]
    rec = reconstruct_from_metrics(p, up_to_step=4000)
    assert rec["eval_history"] == {"Medium": [0.60, 0.80, 0.75]}
    assert rec["best_eval_ma"]["Medium"] == pytest.approx(0.7166666, abs=1e-6)


def test_existing_run_dirs_matches_exact_name_only(tmp_path):
    from train.bc import existing_run_dirs

    for name in ("20260916_123616_bc_full_real", "20260916_222606_bc_full_real_v2", "20260917_010101_smoke_d32"):
        (tmp_path / name).mkdir()
    assert [d.name for d in existing_run_dirs("bc_full_real_v2", tmp_path)] == ["20260916_222606_bc_full_real_v2"]
    assert [d.name for d in existing_run_dirs("bc_full_real", tmp_path)] == ["20260916_123616_bc_full_real"]
    assert existing_run_dirs("resume_test_v2", tmp_path) == []
