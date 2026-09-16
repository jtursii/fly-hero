# Real-config memory/throughput benchmark (Phase 3b, D26)

Measures the actual `bc.py` loop shape -- 1s burn-in under `no_grad`, then a
2s window with grad enabled, forward+backward through a `logits.sum()` proxy
loss (bc.py's real BCE loss doesn't exist yet at the time of this benchmark;
the proxy is enough to measure memory/time, matching `bench/bench_brain.py`'s
own justification for the same substitution) -- not Gate G3's uniform-grad
1s window, which turns out to understate real peak memory (see below).
Device/dtype fixed to Gate G3's chosen mps/float32. Every cell runs in its
own subprocess (D23: in-process peak-memory sampling is contaminated by
earlier cells in the same process).

| config | elapsed (s) | peak | samples/s | <=40GiB |
|---|---|---|---|---|
| batch=8, no checkpoint | 1.734 | 47.52 GiB | 4.612 | **False** |
| batch=8, per-frame checkpoint | 2.119 | 2.11 GiB | 3.776 | True |
| batch=4, no checkpoint | 0.925 | 26.42 GiB | **4.326** | True |
| batch=16, per-frame checkpoint | 3.960 | 3.77 GiB | 4.040 | True |

**D26: chosen config is batch=4, no gradient checkpointing** -- highest
samples/sec (4.326) among configs that clear the 40GiB peak bound, ~7% ahead
of batch=16+checkpoint (4.040) and ~15% ahead of batch=8+checkpoint (3.776).
Confirmed consistent across two independent runs of this benchmark (batch=4
fastest both times, ~4.2-4.3 samples/s).

Key finding, distinct from Gate G3's original benchmark: **batch=8 without
checkpointing, Gate G3's chosen config, does not fit under 40GB** once the
real 2s BC training window is used instead of Gate G3's 1s window (47.52GiB
vs Gate G3's measured 24.30GiB at 1s) -- checkpointing or a smaller batch is
required for the real training loop, not just optional headroom.

Trade-off noted for the record: batch=4 halves the effective per-step batch
size relative to batch=8 (fewer distinct song windows per gradient step,
which could mean noisier gradients / more steps to converge) in exchange for
higher wall-clock throughput. If gradient-noise problems show up during
training (loss not decreasing, unstable curriculum advancement), switching
to batch=8+checkpoint (3.776 samples/s, 2.11GiB peak, huge headroom) or
batch=16+checkpoint (4.040 samples/s) are the next things to try -- both are
already measured and within budget.

Repro: `PYTHONPATH=. uv run python scripts/bench_train_memory.py`.
