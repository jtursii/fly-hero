"""Phase 2 task 9 tests: EMA startup-transient fix, mirrored left/right
placement, and (real-data) the R1-6/R8-share-a-Mi1 retinotopy check."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flyhero.game.retina import (
    PhotoreceptorMap,
    build_photoreceptor_map,
    compute_retina_currents,
    place_on_screen,
)

REAL_GRAPH_PATH = Path("data/processed/graph.npz")
REAL_GRAPH_META_PATH = Path("data/processed/graph_meta.json")


def test_ema_initialized_with_first_frame_no_startup_transient():
    photo_map = PhotoreceptorMap(
        photoreceptor_idx=np.array([0, 1]),
        included_mask=np.array([True, True]),
        image_pos=np.array([[0.5, 0.5], [0.5, 0.5]]),
        medulla_flip=False,
        calibration_report={},
    )
    frames = np.zeros((3, 8, 8), dtype=np.uint8)
    frames[0] = 100
    frames[1] = 200  # a genuine change on frame 1
    frames[2] = 200

    currents = compute_retina_currents(frames, photo_map, gain=4.0, ema_alpha=0.5, blur_sigma=0.001)

    np.testing.assert_allclose(currents[0], 0.0, atol=1e-6)
    assert np.all(currents[1] > 0)  # the real onset is still detected


def test_place_on_screen_mirrors_left_and_right():
    h_raw = np.array([0.0, 0.5, 1.0])
    right = place_on_screen(h_raw, side=1)
    left = place_on_screen(h_raw, side=0)
    np.testing.assert_allclose(right, [0.5, 0.75, 1.0])
    np.testing.assert_allclose(left, [0.5, 0.25, 0.0])


@pytest.mark.skipif(
    not (REAL_GRAPH_PATH.exists() and REAL_GRAPH_META_PATH.exists()),
    reason="requires the real graph.npz/graph_meta.json built from FlyWire v783 data",
)
def test_real_photoreceptor_map_retinotopy_and_exclusions():
    graph = np.load(REAL_GRAPH_PATH)
    meta = json.loads(REAL_GRAPH_META_PATH.read_text())

    photo_map = build_photoreceptor_map(graph, meta["type_names"])

    # The 10 soma-backed photoreceptors (found during the Phase-1 NaN-
    # position work) must be excluded from retina input entirely.
    assert photo_map.n_excluded == 10

    report = photo_map.calibration_report
    assert report["column_ids_available"] is False
    assert report["n_mi1_groups_tested"] > 0
    # Required threshold: R1-6 (via L1) / R8 pairs sharing a Mi1 column must
    # map to nearby image positions, median < 2 lattice spacings.
    assert report["chosen_median_distance_lattice_spacings"] < 2.0

    included = photo_map.included_mask
    assert not np.isnan(photo_map.image_pos[included]).any()
    assert np.isnan(photo_map.image_pos[~included]).all()
