"""Per-frame behavior-cloning targets (PLAN Phase 2 task 7).

Fret target: the frame timeline is partitioned by midpoints between
consecutive (already load-time-merged, see song.py) note times, with phantom
neighbors at t=0 (before the first note) and t=duration_s (after the last
note) -- so the first note's segment starts at 0 and the last note's segment
ends at duration_s, rather than needing a real neighbor on each side.

Within note N's segment [seg_start, seg_end), the target is N's lane_mask
from seg_start until `min(seg_end, N.time_s + max(N.sustain_s,
hit_window_s))`. The `max(sustain_s, hit_window_s)` floor (not just
sustain_s) exists because the *strum* label for N lands on the frame nearest
N.time_s, which frame quantization can round to slightly after time_s; for a
zero-sustain note whose next-note midpoint is very close (dense trills),
holding only until sustain_s (0) could let the segment's target switch away
before that rounded strum frame arrives. hit_window_s (0.07s) comfortably
covers any half-frame (<=1/120s) rounding slop. For the remainder of the
segment (only reachable if the hold ends before the segment does), the
target is 0 (release).

Strum label: 1 on the frame nearest each note's time_s. Two distinct notes
must never round to the same frame -- should be impossible once notes have
gone through song.py's load-time chord merge (merged notes are already
>= 1 frame apart) -- so a collision here is raised as a bug, not merged.
"""

from __future__ import annotations

import numpy as np


class DuplicateStrumFrameError(ValueError):
    pass


def compute_frame_labels(
    notes: np.ndarray, duration_s: float, fps: int, hit_window_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (fret_target[F] uint8 lane-mask-valued, strum[F] uint8) at
    `fps` Hz over [0, duration_s)."""
    n_frames = int(round(duration_s * fps))
    fret_target = np.zeros(n_frames, dtype=np.uint8)
    strum = np.zeros(n_frames, dtype=np.uint8)

    n = len(notes)
    if n == 0:
        return fret_target, strum

    times = notes["time_s"].astype(np.float64)

    boundaries = np.empty(n + 1, dtype=np.float64)
    boundaries[0] = 0.0
    boundaries[1:n] = (times[:-1] + times[1:]) / 2.0
    boundaries[n] = duration_s

    strum_frames = np.round(times * fps).astype(np.int64)
    seen: dict[int, int] = {}
    for i, f in enumerate(strum_frames.tolist()):
        if f in seen:
            j = seen[f]
            raise DuplicateStrumFrameError(
                f"notes {j} (t={times[j]:.4f}s) and {i} (t={times[i]:.4f}s) both "
                f"round to strum frame {f} -- should be impossible after "
                "song.py's load-time chord merge; check the merge threshold."
            )
        seen[f] = i
        if 0 <= f < n_frames:
            strum[f] = 1

    eps = 1e-9
    for i in range(n):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        hold_end = min(seg_end, times[i] + max(float(notes["sustain_s"][i]), hit_window_s))

        f_start = max(int(np.ceil(seg_start * fps - eps)), 0)
        f_hold_end = min(int(np.ceil(hold_end * fps - eps)), n_frames)
        if f_hold_end > f_start:
            fret_target[f_start:f_hold_end] = notes["lane_mask"][i]

    return fret_target, strum
