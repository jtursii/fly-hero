"""Overnight training-stability watchdog (user-specified overnight
safeguards; see docs/DECISIONS.md D30/D31 for the diagnosis and rationale).

Pure decision logic only -- no model, optimizer, or I/O -- so it's testable
without any connectome/GRU data (tests/test_watchdog.py). train/bc.py drives
it with real per-step grad_norm/eval numbers and does the actual
checkpoint/rollback I/O.

Two independent trigger paths, both driven only by observable training
signals (grad_norm, eval hit_rate):
  - skip:      one step's grad_norm is >> the recent rolling median, or the
               NaN-safe sanitizer (D28) had to zero far more gradient
               entries than the ~5 known-unstable cell types (R7, R8, L1,
               L5, one 2-neuron type) ever produce -- refuse to apply that
               step's update at all; don't let it into the rolling window.
  - rollback:  the rolling median has stayed elevated for a long stretch, or
               eval hit_rate has cratered for two evals running at the
               current difficulty -- something systemic, not just one bad
               clip; the caller reverts to checkpoint_last_healthy.pt,
               halves the brain LR, and resets Adam's moments.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class GradWatchdog:
    window: int = 200
    skip_ratio: float = 20.0
    # ~5 known-unstable cell types (D28) contribute at most a handful of
    # non-finite tau/bias entries plus their few gain groups per step under
    # normal operation -- comfortably under 20. More than that means
    # something new (not the known pathway) just went non-finite.
    max_expected_nonfinite: int = 20
    rollback_median_ratio: float = 5.0
    rollback_consecutive_steps: int = 300
    # Stricter-than-rollback bar for certifying a checkpoint "healthy"
    # (hysteresis): rollback fires above 5x baseline, but a checkpoint isn't
    # trusted as the rollback target unless the run is back under 2x --
    # otherwise a checkpoint taken right at the rollback boundary could
    # itself be only marginally OK.
    healthy_ratio: float = 2.0
    rollback_eval_drop: float = 0.15
    rollback_eval_consecutive: int = 2
    max_rollbacks: int = 3

    _recent: deque = field(default_factory=deque, init=False, repr=False)
    healthy_baseline: float | None = field(default=None, init=False)
    consecutive_bad_steps: int = field(default=0, init=False)
    rollback_count: int = field(default=0, init=False)
    skip_count: int = field(default=0, init=False)
    total_steps: int = field(default=0, init=False)
    best_hit_rate: dict = field(default_factory=dict, init=False)
    consecutive_bad_evals: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._recent = deque(maxlen=self.window)

    def rolling_median(self) -> float | None:
        if not self._recent:
            return None
        vals = sorted(self._recent)
        n = len(vals)
        mid = n // 2
        return float(vals[mid]) if n % 2 else float((vals[mid - 1] + vals[mid]) / 2)

    def record_step(self, grad_norm: float, n_bad_elems: int) -> tuple[bool, str | None]:
        """Call once per training step, after sanitize+clip, before
        optimizer.step(). Returns (should_skip, reason). On skip, the
        caller must not call optimizer.step() (must zero_grad() instead)
        and the step's grad_norm is excluded from the rolling window, so a
        run of bad steps can't drag the "normal" baseline up with it."""
        self.total_steps += 1
        median_before = self.rolling_median()
        is_nonfinite_grad_norm = not math.isfinite(grad_norm)
        is_nonfinite_excess = n_bad_elems > self.max_expected_nonfinite
        is_ratio_spike = (
            not is_nonfinite_grad_norm
            and median_before is not None
            and median_before > 0
            and grad_norm > self.skip_ratio * median_before
        )
        should_skip = is_nonfinite_grad_norm or is_nonfinite_excess or is_ratio_spike

        if should_skip:
            self.skip_count += 1
            reason = (
                "nonfinite_grad_norm" if is_nonfinite_grad_norm
                else "nonfinite_excess" if is_nonfinite_excess
                else "ratio_spike"
            )
        else:
            reason = None
            self._recent.append(grad_norm)
            if self.healthy_baseline is None and len(self._recent) == self.window:
                self.healthy_baseline = self.rolling_median()

        # Systemic-drift tracking is independent of this step's own skip
        # decision -- it only asks "is the rolling median elevated right
        # now", checked every step.
        median_now = self.rolling_median()
        if (
            self.healthy_baseline is not None
            and median_now is not None
            and median_now > self.rollback_median_ratio * self.healthy_baseline
        ):
            self.consecutive_bad_steps += 1
        else:
            self.consecutive_bad_steps = 0

        return should_skip, reason

    def should_rollback_from_steps(self) -> bool:
        return (
            self.healthy_baseline is not None
            and self.consecutive_bad_steps >= self.rollback_consecutive_steps
        )

    def is_healthy_now(self) -> bool:
        """Whether *this moment* is safe to certify as checkpoint_last_healthy.pt.
        True before any baseline exists yet (early in a run, the only
        checkpoints available are also the only evidence of "healthy")."""
        if self.healthy_baseline is None:
            return True
        median = self.rolling_median()
        return median is not None and median <= self.healthy_ratio * self.healthy_baseline

    def record_eval(self, difficulty: str, hit_rate: float) -> bool:
        """Returns True if this eval should trigger a rollback (2
        consecutive evals >rollback_eval_drop below this difficulty's best
        hit_rate seen so far). Also tracks the best-so-far per difficulty,
        for train/bc.py's checkpoint_best.pt."""
        best = self.best_hit_rate.get(difficulty)
        if best is None or hit_rate > best:
            self.best_hit_rate[difficulty] = hit_rate
            self.consecutive_bad_evals[difficulty] = 0
            return False
        if (best - hit_rate) > self.rollback_eval_drop:
            self.consecutive_bad_evals[difficulty] = self.consecutive_bad_evals.get(difficulty, 0) + 1
        else:
            self.consecutive_bad_evals[difficulty] = 0
        return self.consecutive_bad_evals.get(difficulty, 0) >= self.rollback_eval_consecutive

    def note_rollback(self) -> None:
        """Call after actually performing a rollback (weights restored, LR
        halved, optimizer moments reset). Baseline is re-established fresh
        under the new LR rather than carried over from the pre-rollback
        regime."""
        self.rollback_count += 1
        self.consecutive_bad_steps = 0
        self._recent.clear()
        self.healthy_baseline = None
        for d in list(self.consecutive_bad_evals):
            self.consecutive_bad_evals[d] = 0

    def exhausted(self) -> bool:
        return self.rollback_count >= self.max_rollbacks
