"""Tests for flyhero/game/decoder.py (PLAN Phase 3b)."""

from __future__ import annotations

import numpy as np

from flyhero.game.decoder import decode_frets, decode_strum, decode_trace, frets_to_held_mask


def test_decode_frets_thresholds_independently_per_lane():
    fret_prob = np.array(
        [
            [0.51, 0.49, 0.0, 1.0, 0.5],
            [0.6, 0.6, 0.6, 0.6, 0.6],
        ]
    )
    frets = decode_frets(fret_prob)
    np.testing.assert_array_equal(frets, [[True, False, False, True, False], [True] * 5])


def test_decode_strum_single_isolated_spike():
    strum_prob = np.array([0.1, 0.2, 0.9, 0.3, 0.1])
    strum = decode_strum(strum_prob)
    np.testing.assert_array_equal(strum, [False, False, True, False, False])


def test_decode_strum_below_threshold_never_fires():
    strum_prob = np.array([0.1, 0.49, 0.3, 0.1])
    strum = decode_strum(strum_prob)
    assert not strum.any()


def test_decode_strum_plateau_takes_first_frame():
    strum_prob = np.array([0.1, 0.9, 0.9, 0.9, 0.2])
    strum = decode_strum(strum_prob)
    np.testing.assert_array_equal(strum, [False, True, False, False, False])


def test_decode_strum_monotonic_increase_to_boundary_fires_last_frame():
    # No frame after the plateau to be non-strictly-less-than, so the last
    # frame is a valid peak (right sentinel is -inf).
    strum_prob = np.array([0.1, 0.6, 0.9])
    strum = decode_strum(strum_prob)
    np.testing.assert_array_equal(strum, [False, False, True])


def test_decode_strum_first_frame_is_peak():
    strum_prob = np.array([0.9, 0.2, 0.1])
    strum = decode_strum(strum_prob)
    np.testing.assert_array_equal(strum, [True, False, False])


def test_decode_strum_two_separate_spikes_both_fire():
    strum_prob = np.array([0.1, 0.9, 0.1, 0.1, 0.8, 0.1])
    strum = decode_strum(strum_prob)
    np.testing.assert_array_equal(strum, [False, True, False, False, True, False])


def test_decode_strum_empty():
    assert decode_strum(np.zeros(0)).shape == (0,)


def test_decode_strum_dense_trill_39ms_spacing():
    """39ms spacing at 60fps is ~2.34 frames apart -- alternating 2-and-3-
    frame gaps (mean 2.34) between spikes. Each spike must resolve to its own
    strum frame, distinct from its neighbors, even though they're close
    enough that a wider local-max window would merge them."""
    fps = 60
    spacing_s = 0.039
    spike_times_s = np.arange(0, 1.0, spacing_s)
    spike_frames = np.unique(np.round(spike_times_s * fps).astype(int))
    assert len(spike_frames) >= 2  # sanity: the trill actually has >=2 distinct frames

    n_frames = int(spike_frames.max()) + 3
    strum_prob = np.full(n_frames, 0.1)
    strum_prob[spike_frames] = 0.95

    strum = decode_strum(strum_prob)
    fired_frames = np.nonzero(strum)[0]
    np.testing.assert_array_equal(fired_frames, spike_frames)


def test_decode_trace_combines_frets_and_strum():
    probs = np.array(
        [
            [0.9, 0.1, 0.1, 0.1, 0.1, 0.2],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ]
    )
    frets, strum = decode_trace(probs)
    assert frets.shape == (3, 5)
    assert strum.shape == (3,)
    np.testing.assert_array_equal(frets[0], [True, False, False, False, False])
    np.testing.assert_array_equal(strum, [False, True, False])


def test_frets_to_held_mask_bit_packing():
    frets = np.array(
        [
            [True, False, False, False, False],  # bit 0 -> 1
            [False, False, False, False, True],  # bit 4 -> 16
            [True, True, True, True, True],  # 31
            [False, False, False, False, False],  # 0
        ]
    )
    held_mask = frets_to_held_mask(frets)
    np.testing.assert_array_equal(held_mask, [1, 16, 31, 0])
