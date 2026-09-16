"""Grayscale note-highway renderer (PLAN Phase 2 task 8).

Top-down highway: 5 vertical lanes, notes scroll downward at a constant
apparent speed set by `lookahead_s` (constant regardless of song tempo -- a
note enters the top of frame `lookahead_s` before its time_s and reaches the
strikeline exactly at time_s), to a strikeline near the bottom.
"""

from __future__ import annotations

import numpy as np

N_LANES = 5
BACKGROUND = 20
LANE_LINE = 60
STRIKELINE = 130
NOTE_TAIL = 160
NOTE_HEAD = 255
STRIKELINE_ROW_FRAC = 0.85

# The background/lane-lines/strikeline never change for a fixed frame_size --
# render_frame is called once per env per step (a Python loop over the
# batch, see sim.py), so recomputing linspace and redrawing static lines on
# every call was a real, measured cost (Gate G2's throughput benchmark).
_LANE_EDGES_CACHE: dict[int, list[int]] = {}
_BASE_FRAME_CACHE: dict[int, np.ndarray] = {}


def _lane_edges(frame_size: int) -> list[int]:
    if frame_size not in _LANE_EDGES_CACHE:
        _LANE_EDGES_CACHE[frame_size] = [
            int(round(x)) for x in np.linspace(0, frame_size, N_LANES + 1)
        ]
    return _LANE_EDGES_CACHE[frame_size]


def _base_frame(frame_size: int) -> np.ndarray:
    if frame_size not in _BASE_FRAME_CACHE:
        frame = np.full((frame_size, frame_size), BACKGROUND, dtype=np.uint8)
        edges = _lane_edges(frame_size)
        for edge in edges[1:-1]:
            if 0 <= edge < frame_size:
                frame[:, edge] = LANE_LINE
        strikeline_row = int(round(STRIKELINE_ROW_FRAC * (frame_size - 1)))
        frame[strikeline_row, :] = STRIKELINE
        _BASE_FRAME_CACHE[frame_size] = frame
    return _BASE_FRAME_CACHE[frame_size]


def render_frame(
    notes: np.ndarray, t: float, lookahead_s: float, frame_size: int = 64
) -> np.ndarray:
    frame = _base_frame(frame_size).copy()
    lane_edges = _lane_edges(frame_size)
    strikeline_row = int(round(STRIKELINE_ROW_FRAC * (frame_size - 1)))

    def row_for_time(note_time: float) -> float:
        frac = 1.0 - (note_time - t) / lookahead_s  # 0 at top, 1 at strikeline
        return frac * strikeline_row

    visible = notes[(notes["time_s"] >= t - 0.5) & (notes["time_s"] <= t + lookahead_s)]
    for note in visible:
        note_time = float(note["time_s"])
        sustain = float(note["sustain_s"])
        lane_mask = int(note["lane_mask"])
        head_row = row_for_time(note_time)
        tail_row = row_for_time(note_time + sustain) if sustain > 0 else head_row

        for lane in range(N_LANES):
            if not (lane_mask & (1 << lane)):
                continue
            col_start = max(int(round(lane_edges[lane])), 0)
            col_end = min(int(round(lane_edges[lane + 1])), frame_size)

            r0 = max(0, min(int(round(min(tail_row, head_row))), frame_size - 1))
            r1 = max(0, min(int(round(max(tail_row, head_row))), frame_size - 1))
            if sustain > 0 and r1 > r0:
                frame[r0:r1, col_start:col_end] = np.maximum(
                    frame[r0:r1, col_start:col_end], NOTE_TAIL
                )
            head_r = int(round(head_row))
            if 0 <= head_r < frame_size:
                lo, hi = max(head_r - 1, 0), min(head_r + 1, frame_size)
                frame[lo:hi, col_start:col_end] = NOTE_HEAD

    return frame
