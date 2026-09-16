# Gate G3 benchmark (Phase 3a)

Every cell below ran in its own subprocess, so peak-memory readings are isolated per cell (see this file's module docstring for why: process peak RSS never resets within a process, and MPS's caching allocator doesn't return memory to the driver without an explicit `empty_cache()` -- both make in-process, cross-cell peak sampling unreliable after the first large cell). MPS peak = `torch.mps.driver_allocated_memory()` polled every substep; CPU peak = process peak RSS (`ru_maxrss`, bytes on macOS). Not directly comparable to each other -- see D23 in docs/DECISIONS.md.


## Brain-only grid (device x dtype x batch x min_syn, 1 game-second)

| device | dtype | batch | min_syn | fwd (s) | fwd peak | fwd+bwd (s) | fwd+bwd peak |
|---|---|---|---|---|---|---|---|
| cpu | float32 | 1 | 5 | 0.516 | 0.48 GiB | 4.976 | 5.60 GiB |
| cpu | float32 | 8 | 5 | 1.210 | 0.66 GiB | 7.558 | 23.60 GiB |
| cpu | float32 | 32 | 5 | 4.375 | 1.15 GiB | 26.128 | 51.19 GiB |
| cpu | float32 | 1 | 10 | 0.237 | 0.49 GiB | 2.300 | 2.63 GiB |
| cpu | float32 | 8 | 10 | 0.538 | 0.49 GiB | 3.134 | 10.30 GiB |
| cpu | float32 | 32 | 10 | 1.926 | 0.79 GiB | 10.247 | 36.53 GiB |
| cpu | float16 | 1 | 5 | 0.475 | 0.49 GiB | 1.727 | 3.07 GiB |
| cpu | float16 | 8 | 5 | 1.111 | 0.57 GiB | 7.692 | 12.03 GiB |
| cpu | float16 | 32 | 5 | 3.986 | 0.86 GiB | 28.278 | 43.57 GiB |
| cpu | float16 | 1 | 10 | 0.244 | 0.49 GiB | 0.786 | 1.53 GiB |
| cpu | float16 | 8 | 10 | 0.483 | 0.49 GiB | 3.157 | 5.36 GiB |
| cpu | float16 | 32 | 10 | 1.382 | 0.58 GiB | 11.176 | 18.41 GiB |
| mps | float32 | 1 | 5 | 0.055 | 1.02 GiB | 0.539 | 6.33 GiB |
| mps | float32 | 8 | 5 | 0.286 | 1.02 GiB | 0.796 | 24.27 GiB |
| mps | float32 | 32 | 5 | 1.048 | 1.02 GiB | 8.646 | 87.61 GiB |
| mps | float32 | 1 | 10 | 0.027 | 0.08 GiB | 0.246 | 2.52 GiB |
| mps | float32 | 8 | 10 | 0.156 | 1.08 GiB | 0.403 | 10.33 GiB |
| mps | float32 | 32 | 10 | 0.493 | 1.05 GiB | 1.370 | 37.24 GiB |
| mps | float16 | 1 | 5 | 0.055 | 1.02 GiB | 0.818 | 3.24 GiB |
| mps | float16 | 8 | 5 | 0.403 | 1.02 GiB | 1.558 | 12.11 GiB |
| mps | float16 | 32 | 5 | 1.743 | 1.02 GiB | 5.572 | 44.27 GiB |
| mps | float16 | 1 | 10 | 0.026 | 0.08 GiB | 0.540 | 1.20 GiB |
| mps | float16 | 8 | 10 | 0.162 | 1.08 GiB | 0.985 | 5.24 GiB |
| mps | float16 | 32 | 10 | 0.532 | 1.08 GiB | 2.865 | 19.24 GiB |

## Full pipeline (retina + brain substeps + readout), batch=8, min_syn=5, 1 game-second

| device | dtype | fwd (s) | fwd peak | fwd+bwd (s) | fwd+bwd peak | Gate G3 |
|---|---|---|---|---|---|---|
| cpu | float32 | 1.251 | 0.72 GiB | 7.382 | 23.59 GiB | FAIL |
| cpu | float16 | 1.106 | 0.62 GiB | 7.564 | 12.09 GiB | FAIL |
| mps | float32 | 0.297 | 1.05 GiB | 0.850 | 24.30 GiB | PASS |
| mps | float16 | 0.311 | 1.05 GiB | 0.726 | 12.14 GiB | PASS |

## Chosen device

**mps / float32**: fwd+bwd = 0.850s, peak = 24.30 GiB (threshold: <=3.0s, <=40GB).
