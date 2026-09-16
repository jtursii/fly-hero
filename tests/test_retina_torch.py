"""Phase 3a test: the torch retina port matches the numpy reference
(compute_retina_currents) on the same frames.

compute_retina_currents computes internally in float64, while TorchRetina
(matching sim.py::_obs()'s actual training-time hot path) uses float32 --
a small precision gap is expected and tolerated with a looser tolerance on
realistic (random) frames; a second test with only exactly-representable
frame values (0/255) uses a tight tolerance to catch algorithmic, not just
precision, divergence.
"""

from __future__ import annotations

import numpy as np
import torch

from flyhero.game.retina import PhotoreceptorMap, compute_retina_currents
from flyhero.game.retina_torch import TorchRetina


def _tiny_photo_map() -> PhotoreceptorMap:
    return PhotoreceptorMap(
        photoreceptor_idx=np.array([0, 1, 2]),
        included_mask=np.array([True, True, False]),
        image_pos=np.array([[0.2, 0.3], [0.7, 0.8], [np.nan, np.nan]]),
        medulla_flip=False,
        calibration_report={},
    )


def _run_torch_retina(photo_map: PhotoreceptorMap, frames: np.ndarray, **kwargs) -> np.ndarray:
    retina = TorchRetina.build(photo_map, frame_size=frames.shape[1], device="cpu", **kwargs)
    frames_t = torch.from_numpy(frames)
    out = torch.stack([retina.step(frames_t[f : f + 1])[0] for f in range(len(frames))])
    return out.numpy()


def test_torch_retina_matches_numpy_reference_loose_tolerance():
    photo_map = _tiny_photo_map()
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(6, 16, 16)).astype(np.uint8)

    np_currents = compute_retina_currents(frames, photo_map, gain=4.0, ema_alpha=0.2, blur_sigma=1.0)
    torch_currents = _run_torch_retina(photo_map, frames, gain=4.0, ema_alpha=0.2, blur_sigma=1.0)

    np.testing.assert_allclose(torch_currents, np_currents, atol=1e-4, rtol=1e-3)
    # excluded photoreceptor (index 2) must be exactly zero in both
    np.testing.assert_allclose(torch_currents[:, 2], 0.0, atol=0)
    np.testing.assert_allclose(np_currents[:, 2], 0.0, atol=0)


def test_torch_retina_matches_numpy_reference_tight_tolerance_binary_frames():
    photo_map = _tiny_photo_map()
    frames = np.zeros((4, 16, 16), dtype=np.uint8)
    frames[1:, 4:12, 4:12] = 255  # a block turns on after frame 0

    np_currents = compute_retina_currents(frames, photo_map, gain=4.0, ema_alpha=0.2, blur_sigma=1.0)
    torch_currents = _run_torch_retina(photo_map, frames, gain=4.0, ema_alpha=0.2, blur_sigma=1.0)

    np.testing.assert_allclose(torch_currents, np_currents, atol=1e-5, rtol=1e-4)


def test_torch_retina_ema_initialized_with_first_frame_no_startup_transient():
    photo_map = PhotoreceptorMap(
        photoreceptor_idx=np.array([0, 1]),
        included_mask=np.array([True, True]),
        image_pos=np.array([[0.5, 0.5], [0.5, 0.5]]),
        medulla_flip=False,
        calibration_report={},
    )
    frames = np.zeros((3, 8, 8), dtype=np.uint8)
    frames[0] = 100
    frames[1] = 200
    frames[2] = 200

    currents = _run_torch_retina(photo_map, frames, gain=4.0, ema_alpha=0.5, blur_sigma=0.001)

    np.testing.assert_allclose(currents[0], 0.0, atol=1e-6)
    assert np.all(currents[1] > 0)
