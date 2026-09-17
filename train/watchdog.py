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
  - rollback:  the rolling median has stayed above rollback_median_ratio x
               a *frozen* baseline for rollback_consecutive_steps, or the
               eval_ma_window-eval moving average of hit_rate has fallen
               more than rollback_eval_drop below its best at the current
               difficulty -- something systemic, not just one bad clip; the
               caller reverts to checkpoint_last_healthy.pt, halves the
               brain LR, and resets Adam's moments.

D35: the baseline is frozen -- set from a known-healthy span via
`baseline_grad_norm`, or else taken once from the first full window -- and
is never re-estimated after a rollback. (Before D35 it was re-measured from
the 200 steps right after a rollback, which were still turbulent: in
bc_full_real_v2 that froze it at ~740 instead of ~330, so the 5x trigger sat
at ~3700 and the 40k-41.8k episode, rolling median ~2000-3400, never fired.)
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
    rollback_median_ratio: float = 3.0
    rollback_consecutive_steps: int = 200
    # D35: frozen reference grad_norm (median over a known-healthy span). If
    # None, taken once from the first full rolling window; never reset.
    baseline_grad_norm: float | None = None
    # Checkpoint certification (hysteresis): a checkpoint is only trusted as
    # the rollback target if the median of the preceding healthy_window
    # steps' grad_norm is below healthy_ratio x baseline -- so a checkpoint
    # taken while the 200-step median has just dipped, inside a bad stretch,
    # isn't certified (v2's step 42000).
    healthy_ratio: float = 2.0
    healthy_window: int = 1000
    eval_ma_window: int = 3
    rollback_eval_drop: float = 0.08
    max_rollbacks: int = 3

    _recent: deque = field(default_factory=deque, init=False, repr=False)
    _long: deque = field(default_factory=deque, init=False, repr=False)
    healthy_baseline: float | None = field(default=None, init=False)
    consecutive_bad_steps: int = field(default=0, init=False)
    rollback_count: int = field(default=0, init=False)
    skip_count: int = field(default=0, init=False)
    total_steps: int = field(default=0, init=False)
    best_hit_rate: dict = field(default_factory=dict, init=False)
    eval_history: dict = field(default_factory=dict, init=False)
    best_eval_ma: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._recent = deque(maxlen=self.window)
        self._long = deque(maxlen=self.healthy_window)
        if self.baseline_grad_norm is not None:
            self.healthy_baseline = float(self.baseline_grad_norm)

    def rolling_median(self) -> float | None:
        return _median(self._recent)

    def long_median(self) -> float | None:
        """Median grad_norm over the last healthy_window finite steps
        (skipped steps included: a stretch of skipped spikes isn't healthy)."""
        return _median(self._long)

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

        if not is_nonfinite_grad_norm:
            self._long.append(grad_norm)
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
        """Whether *this moment* is safe to certify as checkpoint_last_healthy.pt:
        the preceding healthy_window steps' median grad_norm is below
        healthy_ratio x baseline, and the 200-step rolling median isn't above
        the rollback trigger right now (a stretch that has only just started
        escalating still has a low 1000-step median -- v2's step 40000).
        True before any baseline exists (early in a
        fresh run, the only checkpoints available are also the only evidence
        of "healthy"); False while the long window is still filling (e.g.
        right after a rollback or a resume without saved state)."""
        if self.healthy_baseline is None:
            return True
        if len(self._long) < self.healthy_window:
            return False
        if self.rolling_median() > self.rollback_median_ratio * self.healthy_baseline:
            return False
        return self.long_median() < self.healthy_ratio * self.healthy_baseline

    def record_eval(self, difficulty: str, hit_rate: float) -> bool:
        """Returns True if this eval should trigger a rollback: the moving
        average of the last eval_ma_window evals at this difficulty is more
        than rollback_eval_drop below the best such moving average so far.
        Also tracks the best single eval per difficulty, for train/bc.py's
        checkpoint_best.pt."""
        best = self.best_hit_rate.get(difficulty)
        if best is None or hit_rate > best:
            self.best_hit_rate[difficulty] = hit_rate
        hist = self.eval_history.setdefault(difficulty, [])
        hist.append(hit_rate)
        del hist[:-self.eval_ma_window]
        ma = self.eval_ma(difficulty)
        if ma is None:
            return False
        best_ma = self.best_eval_ma.get(difficulty)
        if best_ma is None or ma > best_ma:
            self.best_eval_ma[difficulty] = ma
            return False
        return (best_ma - ma) > self.rollback_eval_drop

    def eval_ma(self, difficulty: str) -> float | None:
        """Mean of the last eval_ma_window evals at this difficulty, or None
        until that many exist (since the last rollback)."""
        hist = self.eval_history.get(difficulty, [])
        if len(hist) < self.eval_ma_window:
            return None
        return float(sum(hist[-self.eval_ma_window:]) / self.eval_ma_window)

    def note_rollback(self) -> None:
        """Call after actually performing a rollback (weights restored, LR
        halved, optimizer moments reset). The baseline stays frozen (D35);
        the grad-norm windows and recent eval history describe the discarded
        weights, so they're cleared. Best hit_rate / best eval MA are kept."""
        self.rollback_count += 1
        self.consecutive_bad_steps = 0
        self._recent.clear()
        self._long.clear()
        self.eval_history = {d: [] for d in self.eval_history}

    def exhausted(self) -> bool:
        return self.rollback_count >= self.max_rollbacks

    def state_dict(self) -> dict:
        """Runtime state saved into every checkpoint (D33) so a stop/resume
        doesn't forget rollbacks already spent, the frozen healthy baseline,
        or the best hit_rate. (The halved brain LR needs no entry here: it
        lives in the optimizer's param_groups, which the checkpoint already
        saves and --resume already restores.)"""
        return dict(
            recent=list(self._recent), long=list(self._long), healthy_baseline=self.healthy_baseline,
            consecutive_bad_steps=self.consecutive_bad_steps, rollback_count=self.rollback_count,
            skip_count=self.skip_count, total_steps=self.total_steps,
            best_hit_rate=dict(self.best_hit_rate),
            eval_history={d: list(h) for d, h in self.eval_history.items()},
            best_eval_ma=dict(self.best_eval_ma),
        )

    def load_state_dict(self, state: dict) -> None:
        """A configured baseline_grad_norm wins over the saved one (D35: the
        baseline is chosen deliberately, from a known-healthy span)."""
        self._recent = deque(state["recent"], maxlen=self.window)
        self._long = deque(state.get("long", []), maxlen=self.healthy_window)
        if self.baseline_grad_norm is None:
            self.healthy_baseline = state["healthy_baseline"]
        self.consecutive_bad_steps = state["consecutive_bad_steps"]
        self.rollback_count = state["rollback_count"]
        self.skip_count = state["skip_count"]
        self.total_steps = state["total_steps"]
        self.best_hit_rate = dict(state["best_hit_rate"])
        self.eval_history = {d: list(h) for d, h in state.get("eval_history", {}).items()}
        self.best_eval_ma = dict(state.get("best_eval_ma", {}))


def _median(values) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    return float(vals[mid]) if n % 2 else float((vals[mid - 1] + vals[mid]) / 2)


def load_metrics_lineage(metrics_path) -> list[dict]:
    """metrics.jsonl records in order, minus abandoned branches: a
    `resumed_from_step: X` marker (written by train/bc.py when it resumes
    from a checkpoint older than the newest records, e.g. --resume best)
    drops every earlier record with step > X. Unparseable lines are skipped."""
    import json
    from pathlib import Path

    path = Path(metrics_path)
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "resumed_from_step" in r:
            x = r["resumed_from_step"]
            out = [q for q in out if q.get("step", 0) <= x]
        out.append(r)
    return out


def reconstruct_from_metrics(metrics_path, up_to_step: int, **watchdog_kwargs) -> dict:
    """Fallback for checkpoints written before D33 (no watchdog_state):
    recovers what metrics.jsonl does record -- best eval hit_rate, the
    recent eval history and best eval moving average per difficulty, and the
    rollback count -- for records at steps <= up_to_step, by replaying the
    evals through a GradWatchdog built with watchdog_kwargs. The grad-norm
    windows can't be recovered (only every 10th step is logged) and refill
    from scratch."""
    wd = GradWatchdog(**watchdog_kwargs)
    for r in load_metrics_lineage(metrics_path):
        if r.get("step", 0) > up_to_step:
            continue
        if "eval_hit_rate" in r and r.get("eval_difficulty") is not None:
            wd.record_eval(r["eval_difficulty"], r["eval_hit_rate"])
        if r.get("rollback"):
            wd.note_rollback()
    return dict(
        best_hit_rate=wd.best_hit_rate, rollback_count=wd.rollback_count,
        eval_history=wd.eval_history, best_eval_ma=wd.best_eval_ma,
    )


def baseline_from_metrics(records: list[dict], start_step: int, end_step: int) -> float:
    """Median logged (every-10th-step) grad_norm over [start_step,
    end_step], excluding skipped and non-finite steps -- how D35's frozen
    baseline is chosen from a known-healthy span."""
    vals = [
        r["grad_norm"] for r in records
        if "grad_norm" in r and start_step <= r["step"] <= end_step and not r.get("skip")
        and r["grad_norm"] is not None and math.isfinite(r["grad_norm"])
    ]
    if not vals:
        raise ValueError(f"no grad_norm records in [{start_step}, {end_step}]")
    return _median(vals)


def replay_on_metrics(
    records: list[dict], baseline: float, log_every: int = 10, checkpoint_every: int = 500,
    eval_ma_window: int = 3, rollback_eval_drop: float = 0.08, **watchdog_kwargs,
) -> dict:
    """What the watchdog rules would have done on a run's logged metrics.
    metrics.jsonl only has every log_every-th step, so the step-count
    parameters (window, rollback_consecutive_steps, healthy_window) are
    divided by log_every and the real GradWatchdog class is fed one logged
    sample per log_every steps. Each firing resets the consecutive count
    (not the windows) so later episodes are listed too; events after the
    first firing are counterfactual (the real run would have rolled back).

    Returns {"grad_fires": [step, ...], "eval_fires": [(step, ma, best_ma)],
    "healthy": {checkpoint_step: bool}, "live_median_fires": [step, ...]}
    where live_median_fires applies the same trigger to the rolling median
    the run itself logged (its true 200-step median; cross-check)."""
    kw = dict(window=200, rollback_consecutive_steps=200, healthy_window=1000)
    kw.update(watchdog_kwargs)
    wd = GradWatchdog(
        window=kw.pop("window") // log_every,
        rollback_consecutive_steps=kw.pop("rollback_consecutive_steps") // log_every,
        healthy_window=kw.pop("healthy_window") // log_every,
        baseline_grad_norm=baseline, eval_ma_window=eval_ma_window,
        rollback_eval_drop=rollback_eval_drop, **kw,
    )
    grad_fires, eval_fires, healthy, live_fires = [], [], {}, []
    live_run_start = None
    live_needed = wd.rollback_consecutive_steps * log_every
    for r in records:
        step = r.get("step", 0)
        if "grad_norm" in r and "loss" in r:
            g = r["grad_norm"]
            wd.record_step(float("nan") if g is None else float(g), int(r.get("n_bad_elems", 0)))
            if wd.should_rollback_from_steps():
                grad_fires.append(step)
                wd.consecutive_bad_steps = 0
            if step % checkpoint_every == 0:
                healthy[step] = wd.is_healthy_now()
            live = r.get("rolling_median_grad_norm")
            if live is not None and live > wd.rollback_median_ratio * baseline:
                if live_run_start is None:
                    live_run_start = step
                if step - live_run_start >= live_needed:
                    live_fires.append(step)
                    live_run_start = None
            else:
                live_run_start = None
        if "eval_hit_rate" in r and r.get("eval_difficulty") is not None:
            d = r["eval_difficulty"]
            if wd.record_eval(d, r["eval_hit_rate"]):
                eval_fires.append((step, wd.eval_ma(d), wd.best_eval_ma[d]))
    return dict(grad_fires=grad_fires, eval_fires=eval_fires, healthy=healthy, live_median_fires=live_fires)
