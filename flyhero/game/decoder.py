"""Converts model output probabilities to game actions (PLAN Phase 3b).
Used by eval (in-training and evaluate.py) and export -- invariant 1: inputs
are model outputs only (fret/strum probabilities), never chart data.

frets: fret_prob > 0.5, independently per lane.

strum: a strict local maximum of strum_prob above 0.5 -- strict on the left
(p[t] > p[t-1]), non-strict on the right (p[t] >= p[t+1]), which selects the
*first* frame of a plateau (a run of equal, locally-maximal values) as the
strum frame, per the user's explicit choice. Frame 0/F-1 are compared
against -inf sentinels (no wraparound).

This only needs the full trace at once (not causal/streaming): sim.py's
env.step()'s rendered frame does not depend on the actions passed in (_obs()
is called after advancing frame_idx and is a pure function of song time), so
eval/export can run the whole clip open-loop first (collect probabilities),
decode the whole trace here, and only then score the decoded actions against
notes with rules.score_playthrough() -- no real-time causal decoding is
needed anywhere in this project.
"""

from __future__ import annotations

import numpy as np


def decode_frets(fret_prob: np.ndarray) -> np.ndarray:
    """fret_prob: [F, 5] float in [0,1]. Returns bool[F, 5]."""
    return fret_prob > 0.5


def decode_strum(strum_prob: np.ndarray) -> np.ndarray:
    """strum_prob: [F] float in [0,1]. Returns bool[F]: a strict local
    maximum (see module docstring for the plateau-tiebreak rule) with
    value > 0.5."""
    n = len(strum_prob)
    if n == 0:
        return np.zeros(0, dtype=bool)
    left = np.empty(n, dtype=np.float64)
    left[0] = -np.inf
    left[1:] = strum_prob[:-1]
    right = np.empty(n, dtype=np.float64)
    right[-1] = -np.inf
    right[:-1] = strum_prob[1:]

    is_peak = (strum_prob > left) & (strum_prob >= right)
    return is_peak & (strum_prob > 0.5)


def decode_trace(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """probs: [F, 6] (5 fret probabilities + 1 strum probability), already
    sigmoid-applied. Returns (frets[F,5] bool, strum[F] bool)."""
    frets = decode_frets(probs[:, :5])
    strum = decode_strum(probs[:, 5])
    return frets, strum


def frets_to_held_mask(frets: np.ndarray) -> np.ndarray:
    """frets: bool[F, 5] -> held_mask[F] int64 lane bitmask, matching
    rules.py's RuleEngine/score_playthrough convention (bit i = fret i)."""
    n_frames = frets.shape[0]
    held_mask = np.zeros(n_frames, dtype=np.int64)
    for lane in range(5):
        held_mask |= frets[:, lane].astype(np.int64) << lane
    return held_mask
