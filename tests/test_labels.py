"""Phase 2 task 7 tests: fret-target/strum label generation."""

from __future__ import annotations

import numpy as np
import pytest

from flyhero.game.labels import DuplicateStrumFrameError, compute_frame_labels
from flyhero.game.song import NOTE_DTYPE

FPS = 60
HIT_WINDOW_S = 0.07


def _notes(rows: list[tuple[float, int, float, int]]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=NOTE_DTYPE)
    for i, r in enumerate(rows):
        arr[i] = r
    return arr


def test_dense_trill_alternating_lanes_no_collision():
    # 10 notes, 50ms apart, alternating lane 0 / lane 1, no sustain.
    rows = [(0.05 * i, 1 if i % 2 == 0 else 2, 0.0, 0) for i in range(10)]
    notes = _notes(rows)
    duration_s = 0.5

    fret_target, strum = compute_frame_labels(notes, duration_s, FPS, HIT_WINDOW_S)

    # Each note's own strum frame (0.05s * i * 60 = 3i, exact) must be
    # active with that note's own lane_mask.
    for i, (t, lane_mask, _sustain, _flags) in enumerate(rows):
        f = round(t * FPS)
        assert strum[f] == 1
        assert fret_target[f] == lane_mask

    assert strum.sum() == len(rows)  # no two notes collided onto one frame


def test_chord_then_overlapping_single_note_segments_dont_leak():
    # Chord {0,1} at t=0, then a single lane-0 note shortly after.
    notes = _notes([(0.0, 0b00011, 0.0, 0), (0.05, 0b00001, 0.0, 0)])
    fret_target, strum = compute_frame_labels(notes, 0.2, FPS, HIT_WINDOW_S)

    f0 = round(0.0 * FPS)
    f1 = round(0.05 * FPS)
    assert fret_target[f0] == 0b00011
    assert fret_target[f1] == 0b00001
    assert strum[f0] == 1 and strum[f1] == 1


def test_sustain_cut_short_by_midpoint_then_next_note_starts_clean():
    # Note A: sustain long enough to naturally outlast the midpoint with B.
    # Midpoint(A,B) = 0.05, well before A's naive sustain end (0.3s) -- the
    # midpoint must win, and B's own segment must start exactly there with
    # B's lane_mask, not a leftover release gap belonging to nobody. (Exact
    # frame index of the handoff is float32-rounding-sensitive since
    # NOTE_DTYPE stores time_s as float32 -- assert the qualitative
    # no-gap/no-overlap/early-handoff properties, not a hand-computed frame
    # index.)
    notes = _notes([(0.0, 0b00001, 0.3, 0), (0.1, 0b00010, 0.0, 0)])
    duration_s = 0.3
    fret_target, strum = compute_frame_labels(notes, duration_s, FPS, HIT_WINDOW_S)

    handoff = int(np.argmax(fret_target == 0b00010))  # first frame with B's mask
    assert handoff > 0
    assert np.all(fret_target[:handoff] == 0b00001)  # solid A up to the handoff, no gaps
    assert handoff < round(0.1 * FPS)  # midpoint won well before B's own note time

    f_a, f_b = round(0.0 * FPS), round(0.1 * FPS)
    assert strum[f_a] == 1 and fret_target[f_a] == 0b00001
    assert strum[f_b] == 1 and fret_target[f_b] == 0b00010

    # B's own hold (time_s=0.1, sustain=0) covers at least through
    # hit_window_s past its own time, then releases (no third note to claim
    # it).
    still_b_at = round((0.1 + HIT_WINDOW_S - 0.01) * FPS)
    assert fret_target[still_b_at] == 0b00010
    tail_start = round((0.1 + HIT_WINDOW_S + 0.02) * FPS)
    assert np.all(fret_target[tail_start:] == 0)


def test_no_sustain_note_rounds_forward_still_covered_by_hit_window_floor():
    """A zero-sustain note whose time_s doesn't land on a frame boundary can
    round UP to the next frame for its strum label. Without the
    max(sustain_s, hit_window_s) hold-duration floor, the hold would end at
    exactly time_s (since sustain_s=0), excluding that rounded-forward
    strum frame from the note's own target -- this is exactly the bug the
    floor fixes."""
    time_s = 0.0126  # 0.0126*60 = 0.756 -> rounds UP to frame 1 (frame time 1/60=0.01667)
    notes = _notes([(time_s, 0b00100, 0.0, 0)])
    fret_target, strum = compute_frame_labels(notes, 1.0, FPS, HIT_WINDOW_S)

    strum_frame = round(time_s * FPS)
    assert strum_frame == 1
    assert strum[strum_frame] == 1
    assert fret_target[strum_frame] == 0b00100


def test_duplicate_strum_frame_raises():
    # Two notes closer than one frame -- should be impossible after song.py's
    # load-time chord merge, so this is a bug-detection safety net.
    notes = _notes([(0.0, 0b00001, 0.0, 0), (0.005, 0b00010, 0.0, 0)])
    with pytest.raises(DuplicateStrumFrameError):
        compute_frame_labels(notes, 0.1, FPS, HIT_WINDOW_S)


def test_empty_notes():
    notes = _notes([])
    fret_target, strum = compute_frame_labels(notes, 1.0, FPS, HIT_WINDOW_S)
    assert len(fret_target) == 60
    assert fret_target.sum() == 0
    assert strum.sum() == 0


def test_hit_window_onset_starts_at_hit_window_not_midpoint():
    # Sparse notes (1s apart): "segment" holds lane 0 from the 0.5s midpoint
    # (and note 0 from frame 0); "hit_window" only from time_s - hit_window_s,
    # the earliest frame rules.py can reward. Hold end is identical.
    notes = _notes([(0.5, 0b00001, 0.0, 0), (1.5, 0b00010, 0.0, 0)])
    seg, strum_seg = compute_frame_labels(notes, 2.0, FPS, HIT_WINDOW_S, "segment")
    hw, strum_hw = compute_frame_labels(notes, 2.0, FPS, HIT_WINDOW_S, "hit_window")

    assert np.array_equal(strum_seg, strum_hw)
    assert seg[0] == 0b00001 and hw[0] == 0
    assert seg[round(1.0 * FPS)] == 0b00010
    on = np.flatnonzero(hw == 0b00010)
    assert on[0] == int(np.ceil((1.5 - HIT_WINDOW_S) * FPS - 1e-9))
    assert on[-1] == np.flatnonzero(seg == 0b00010)[-1]
    # Never positive where "segment" isn't, and same lane wherever positive.
    assert np.all((hw == 0) | (hw == seg))


def test_hit_window_onset_dense_notes_fall_back_to_midpoint():
    # 100ms gaps < 2*hit_window_s: midpoint is later than time_s - hit_window_s,
    # so both onsets agree.
    rows = [(0.1 * i + 0.2, 1 << (i % 3), 0.0, 0) for i in range(6)]
    notes = _notes(rows)
    seg, _ = compute_frame_labels(notes, 1.0, FPS, HIT_WINDOW_S, "segment")
    hw, _ = compute_frame_labels(notes, 1.0, FPS, HIT_WINDOW_S, "hit_window")
    first = round((0.2 - HIT_WINDOW_S) * FPS)
    assert np.array_equal(seg[first:], hw[first:])


def test_unknown_fret_onset_raises():
    with pytest.raises(ValueError):
        compute_frame_labels(_notes([(0.1, 1, 0.0, 0)]), 0.5, FPS, HIT_WINDOW_S, "bogus")
