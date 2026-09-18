"""Tests for eval/compare.py (the paired eval instrument): note
categorization, failure-mode classification against hand-built probability
traces, paired-difference statistics, and the invariant that the default
thresholds decode identically to the in-training eval's decoder.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.compare import (
    CATEGORIES,
    MISS_MODES,
    aggregate_condition,
    analyze_excerpt,
    pool_diag,
    check_reproduces_in_training_eval,
    classify_miss,
    note_category,
    paired_diff,
)
from flyhero.game.decoder import decode_trace
from flyhero.game.song import NOTE_DTYPE

FPS = 60
HIT_WINDOW_S = 0.07


def make_notes(spec: list[tuple[float, int]]) -> np.ndarray:
    """[(time_s, lane_mask)] -> NOTE_DTYPE array, no sustains."""
    return np.array([(t, m, 0.0, 0) for t, m in spec], dtype=NOTE_DTYPE)


def probs_from(n_frames: int, strum_frames: list[int], held: dict[int, int]) -> np.ndarray:
    """A probability trace [F, 6] that decodes (at the default 0.5/0.5
    thresholds) to exactly `strum_frames` as strums and `held` as the lane
    bitmask on each listed frame. Strum frames get an isolated 0.9 spike so
    each is a strict local maximum."""
    probs = np.zeros((n_frames, 6), dtype=np.float32)
    for f in strum_frames:
        probs[f, 5] = 0.9
    for f, mask in held.items():
        for lane in range(5):
            if (mask >> lane) & 1:
                probs[f, lane] = 0.9
    return probs


def test_note_category_partitions_every_mask():
    masks = np.array([0, 1, 4, 3, 6, 5, 9, 17, 7, 31], dtype=np.uint8)
    assert list(note_category(masks)) == [
        "open",           # 0b00000
        "single",         # 0b00001
        "single",         # 0b00100
        "chord2_adj",     # lanes 0,1
        "chord2_adj",     # lanes 1,2
        "chord2_nonadj",  # lanes 0,2
        "chord2_nonadj",  # lanes 0,3
        "chord2_nonadj",  # lanes 0,4
        "chord3plus",     # lanes 0,1,2
        "chord3plus",     # all five
    ]


def test_note_category_handles_empty():
    assert len(note_category(np.zeros(0, dtype=np.uint8))) == 0


def test_classify_miss_each_mode():
    # note at 1.0s (frame 60), lanes 0+1 (mask 3); hit window +-0.07s = +-4.2 frames
    note_t, lane = 1.0, 0b00011

    def mode(strum_frames, held_mask_at_strum):
        strums = np.array(strum_frames, dtype=np.int64)
        held = np.zeros(200, dtype=np.int64)
        for f in strum_frames:
            held[f] = held_mask_at_strum
        return classify_miss(lane, note_t, strums, held, FPS, HIT_WINDOW_S)

    assert mode([], 0) == "no_strum"
    assert mode([20], 0b00011) == "no_strum"        # strum far outside the window
    assert mode([60], 0) == "nothing_held"
    assert mode([60], 0b00001) == "missing_fret"    # holding one of the two lanes
    assert mode([60], 0b00111) == "extra_fret"      # both lanes plus one more
    assert mode([60], 0b00110) == "wrong_lane"      # two lanes, wrong ones
    assert mode([60], 0b00011) == "strum_consumed"  # exactly right -> an earlier note took it
    assert mode([60], 0b11100) == "other"           # 3 held vs 2 wanted, no overlap


def test_classify_miss_picks_nearest_strum_in_window():
    # frames 57 and 62 are both inside +-4.2 frames of frame 60; 62 is nearer
    held = np.zeros(200, dtype=np.int64)
    held[57] = 0b00001  # would classify as missing_fret
    held[62] = 0
    assert classify_miss(0b00011, 1.0, np.array([57, 62]), held, FPS, HIT_WINDOW_S) == "nothing_held"


def test_analyze_excerpt_hit_and_miss_breakdown():
    # 4 notes, one per failure mode under test (at 60fps, time_s * 60 = frame):
    #   0.5s frame  30 single 0b00001 -- strum + right fret     -> hit
    #   1.5s frame  90 single 0b00010 -- no strum in the window  -> no_strum
    #   2.5s frame 150 chord  0b00011 -- only lane 1 held        -> missing_fret
    #   3.5s frame 210 chord  0b00101 -- strum with nothing held -> nothing_held
    notes = make_notes([(0.5, 0b00001), (1.5, 0b00010), (2.5, 0b00011), (3.5, 0b00101)])
    probs = probs_from(
        n_frames=300,
        strum_frames=[30, 150, 210],
        held={30: 0b00001, 150: 0b00010},
    )

    r = analyze_excerpt(probs, notes, FPS, HIT_WINDOW_S, 0.5, 0.5)
    assert r["n_notes"] == 4
    assert r["n_hits"] == 1  # only the first single
    assert r["n_overstrums"] == 2  # the strums at 150 and 210 match no note
    cat = r["by_category"]
    assert cat["single"]["n"] == 2 and cat["single"]["hits"] == 1
    assert cat["single"]["miss_modes"]["no_strum"] == 1
    assert cat["chord2_adj"]["n"] == 1 and cat["chord2_adj"]["miss_modes"]["missing_fret"] == 1
    assert cat["chord2_nonadj"]["n"] == 1 and cat["chord2_nonadj"]["miss_modes"]["nothing_held"] == 1
    # every note is accounted for: hits + all miss modes == n, per category
    for c in CATEGORIES:
        e = cat[c]
        assert e["hits"] + sum(e["miss_modes"][m] for m in MISS_MODES) == e["n"]


def test_analyze_excerpt_score_matches_reward_weights():
    notes = make_notes([(0.5, 0b00001), (1.5, 0b00010)])
    probs = probs_from(300, strum_frames=[30, 200], held={30: 0b00001, 200: 0b10000})
    r = analyze_excerpt(probs, notes, FPS, HIT_WINDOW_S, 0.5, 0.5)
    assert r["n_hits"] == 1 and r["n_misses"] == 1 and r["n_overstrums"] == 1
    assert r["score"] == 1 - 1 - 0.3


def test_analyze_excerpt_thresholds_change_decoding():
    # strum probability 0.6: decoded at threshold 0.5, dropped at 0.7.
    notes = make_notes([(0.5, 0b00001)])
    probs = np.zeros((100, 6), dtype=np.float32)
    probs[30, 5] = 0.6
    probs[30, 0] = 0.6
    assert analyze_excerpt(probs, notes, FPS, HIT_WINDOW_S, 0.5, 0.5)["n_hits"] == 1
    assert analyze_excerpt(probs, notes, FPS, HIT_WINDOW_S, 0.7, 0.5)["n_hits"] == 0
    # fret threshold above the fret probability: strum fires, fret isn't held
    r = analyze_excerpt(probs, notes, FPS, HIT_WINDOW_S, 0.5, 0.7)
    assert r["n_hits"] == 0 and r["n_overstrums"] == 1


def test_analyze_excerpt_defaults_match_in_training_decoder():
    """analyze_excerpt at its defaults must decode exactly what the
    in-training eval's decode_trace() produces -- this is what makes the
    default-threshold numbers comparable to the logged eval history."""
    rng = np.random.default_rng(0)
    probs = rng.random((600, 6)).astype(np.float32)
    frets_ref, strum_ref = decode_trace(probs)
    frets_new, strum_new = decode_trace(probs, 0.5, 0.5)
    assert np.array_equal(frets_ref, frets_new) and np.array_equal(strum_ref, strum_new)


def test_reproduction_check_passes_on_agreement_and_raises_on_drift():
    per_excerpt = [dict(hit_rate=0.7), dict(hit_rate=0.5), dict(hit_rate=0.9)]
    out = check_reproduces_in_training_eval(per_excerpt, [0.7, 0.5, 0.9], 2, log=lambda *_: None)
    assert out["max_abs_diff_vs_score_eval_entry"] == 0.0
    assert out["n_in_training_eval"] == 2
    assert abs(out["in_training_eval_subset_hit_rate"] - 0.6) < 1e-12  # mean of the first two

    with pytest.raises(AssertionError, match="drifted"):
        check_reproduces_in_training_eval(per_excerpt, [0.7, 0.5, 0.8], 2, log=lambda *_: None)


def test_paired_diff_mean_and_se():
    d = paired_diff([1.0, 2.0, 3.0], [0.5, 0.5, 0.5])
    assert d["n"] == 3
    assert abs(d["mean"] - 1.5) < 1e-12
    assert abs(d["se"] - float(np.std([0.5, 1.5, 2.5], ddof=1) / np.sqrt(3))) < 1e-12


def test_paired_diff_is_zero_against_itself():
    vals = [0.1, 0.7, 0.4]
    d = paired_diff(vals, vals)
    assert d["mean"] == 0.0 and d["se"] == 0.0


def test_paired_diff_beats_unpaired_se_on_correlated_data():
    """The reason this script is paired: per-excerpt song difficulty
    dominates the absolute spread, so a small shared shift is invisible in
    absolute SEs but clear in the paired one."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.7, 0.15, 28)  # song-to-song spread
    cond = base + 0.02  # a uniform +0.02 improvement
    unpaired_se = float(np.std(cond, ddof=1) / np.sqrt(28))
    d = paired_diff(list(cond), list(base))
    assert abs(d["mean"] - 0.02) < 1e-9
    assert d["se"] < unpaired_se / 100  # paired SE is ~0 here, unpaired is ~0.03


def test_aggregate_condition_pooled_and_paired():
    def excerpt(n, hits, over, minutes, cat_counts):
        by_cat = {
            c: dict(n=cat_counts.get(c, (0, 0))[0], hits=cat_counts.get(c, (0, 0))[1],
                    miss_modes=dict.fromkeys(MISS_MODES, 0))
            for c in CATEGORIES
        }
        return dict(
            hit_rate=hits / n, score=float(hits - (n - hits) - 0.3 * over), minutes=minutes,
            n_notes=n, n_hits=hits, n_misses=n - hits, n_overstrums=over,
            overstrums_per_min=over / minutes, sustain_frac=1.0, by_category=by_cat,
        )

    # excerpt 2 has no non-adjacent chords, so the paired category diff must
    # use only excerpt 1 and report n=1.
    a = [excerpt(100, 80, 10, 1.0, {"single": (90, 75), "chord2_nonadj": (10, 5)}),
         excerpt(100, 40, 20, 1.0, {"single": (100, 40)})]
    b = [excerpt(100, 70, 10, 1.0, {"single": (90, 70), "chord2_nonadj": (10, 0)}),
         excerpt(100, 40, 20, 1.0, {"single": (100, 40)})]

    agg = aggregate_condition(a, b)
    assert agg["n_excerpts"] == 2
    assert abs(agg["hit_rate_mean"] - 0.6) < 1e-12          # mean of 0.8, 0.4
    assert abs(agg["hit_rate_pooled"] - 0.6) < 1e-12        # 120 / 200
    assert abs(agg["hit_rate_diff"]["mean"] - 0.05) < 1e-12  # (+0.10, 0.0) / 2
    assert agg["excerpts_better"] == 1
    nonadj = agg["by_category"]["chord2_nonadj"]
    assert nonadj["n"] == 10 and nonadj["hit_rate_pooled"] == 0.5
    assert nonadj["diff"]["n"] == 1 and abs(nonadj["diff"]["mean"] - 0.5) < 1e-12
    assert agg["by_category"]["chord3plus"]["n"] == 0


def test_aggregate_condition_without_baseline_has_no_diffs():
    e = dict(hit_rate=0.5, score=0.0, minutes=1.0, n_notes=10, n_hits=5, n_misses=5,
             n_overstrums=0, overstrums_per_min=0.0, sustain_frac=1.0,
             by_category={c: dict(n=0, hits=0, miss_modes=dict.fromkeys(MISS_MODES, 0)) for c in CATEGORIES})
    agg = aggregate_condition([e], None)
    assert "hit_rate_diff" not in agg and "diff" not in agg["by_category"]["single"]


def test_analyze_excerpt_d41_diagnostics():
    # singles at 1.0s (f60, lane 0, held + strummed -> hit), 2.0s (f120, lane 1,
    # nothing held, strummed), 5.0s (f300, lane 2, held + strummed -> hit);
    # plus a strum at 4.0s (f240) far from any note.
    notes = make_notes([(1.0, 0b00001), (2.0, 0b00010), (5.0, 0b00100)])
    probs = probs_from(n_frames=400, strum_frames=[60, 120, 240, 300], held={60: 0b00001, 300: 0b00100})
    d = analyze_excerpt(probs, notes, FPS, HIT_WINDOW_S, 0.5, 0.5)["diag"]
    assert d["single_n"] == 3 and d["single_fret_at_note"] == 2
    assert d["far_overstrums"] == 1  # the f120 overstrum is next to a note (wrong frets), not far
    assert (d["early_n"], d["early_hits"], d["late_n"], d["late_hits"]) == (2, 1, 1, 1)
    p = pool_diag([d, d], minutes=2 * 400 / FPS / 60)
    assert p["single_fret_at_note"] == pytest.approx(2 / 3)
    assert p["hit_rate_first3s"] == pytest.approx(0.5) and p["hit_rate_after3s"] == pytest.approx(1.0)
    assert p["far_overstrums_per_min"] == pytest.approx(2 / (800 / FPS / 60))
