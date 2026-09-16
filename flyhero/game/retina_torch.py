"""Torch port of retina.py's per-frame sampling path (PLAN Phase 3 task 1's
retina-on-the-brain's-device requirement), so the retina encoder can run on
whatever device the brain runs on without a CPU/numpy round-trip every
frame. This module is self-contained -- it doesn't modify retina.py, sim.py,
or env.py (Phase 2, already gated at G2) -- and only ports the per-frame
sampling math (blur, nearest-pixel gather, EMA), not the one-time
PCA/calibration code in retina.py, which stays numpy since it only runs once
at startup to build a PhotoreceptorMap.

step()'s formula mirrors flyhero/game/sim.py::VecRhythmEnv._obs()'s inlined
hot path exactly (float32 throughout, precomputed nearest-pixel indices,
dense gather-then-mask), independently re-derived here rather than imported
from sim.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from flyhero.game.retina import PhotoreceptorMap, _nearest_rc


def gaussian_blur_batch_torch(
    frames: torch.Tensor, sigma: float = 1.0, kernel_size: int = 5
) -> torch.Tensor:
    """frames: [B,H,W] float, any device. Mirrors retina.py's
    gaussian_blur_batch (hand-rolled separable Gaussian blur, edge-replicate
    padding) exactly, preserving dtype."""
    device, dtype = frames.device, frames.dtype
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    kernel = torch.exp(-(ax**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    half = kernel_size // 2
    b, h, w = frames.shape

    frames_4d = frames.unsqueeze(1)  # [B,1,H,W] -- replicate padding needs 4D
    padded_w = F.pad(frames_4d, (half, half, 0, 0), mode="replicate").squeeze(1)
    row_conv = torch.zeros((b, h, w), dtype=dtype, device=device)
    for i in range(kernel_size):
        row_conv = row_conv + kernel[i] * padded_w[:, :, i : i + w]

    padded_h = F.pad(row_conv.unsqueeze(1), (0, 0, half, half), mode="replicate").squeeze(1)
    col_conv = torch.zeros((b, h, w), dtype=dtype, device=device)
    for i in range(kernel_size):
        col_conv = col_conv + kernel[i] * padded_h[:, i : i + h, :]

    return col_conv


def build_torch_sampling_tables(
    photo_map: PhotoreceptorMap, frame_size: int, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirrors sim.py::VecRhythmEnv.__init__'s precompute: nearest-pixel
    (row, col) per photoreceptor plus a dense 0/1 mask (excluded/unresolved
    photoreceptors point at pixel (0,0) and are zeroed by the mask)."""
    valid = photo_map.included_mask & ~np.isnan(photo_map.image_pos[:, 0])
    n_photo = len(photo_map.photoreceptor_idx)
    sample_r = np.zeros(n_photo, dtype=np.int64)
    sample_c = np.zeros(n_photo, dtype=np.int64)
    r_valid, c_valid = _nearest_rc(photo_map.image_pos[valid], frame_size, frame_size)
    sample_r[valid] = r_valid
    sample_c[valid] = c_valid
    sample_mask = valid.astype(np.float32)

    return (
        torch.as_tensor(sample_r, dtype=torch.long, device=device),
        torch.as_tensor(sample_c, dtype=torch.long, device=device),
        torch.as_tensor(sample_mask, dtype=torch.float32, device=device),
    )


@dataclass
class TorchRetina:
    sample_r: torch.Tensor  # long [n_photo]
    sample_c: torch.Tensor  # long [n_photo]
    sample_mask: torch.Tensor  # float32 [n_photo]
    included_mask: torch.Tensor  # bool [n_photo]
    gain: float
    ema_alpha: float
    blur_sigma: float
    ema: torch.Tensor | None = field(default=None, init=False, repr=False)

    @classmethod
    def build(
        cls,
        photo_map: PhotoreceptorMap,
        frame_size: int,
        gain: float,
        ema_alpha: float,
        blur_sigma: float,
        device: torch.device | str,
    ) -> "TorchRetina":
        sample_r, sample_c, sample_mask = build_torch_sampling_tables(photo_map, frame_size, device)
        included_mask = torch.as_tensor(photo_map.included_mask, dtype=torch.bool, device=device)
        return cls(sample_r, sample_c, sample_mask, included_mask, gain, ema_alpha, blur_sigma)

    def reset(self) -> None:
        self.ema = None

    def step(self, frames_uint8: torch.Tensor) -> torch.Tensor:
        """frames_uint8: [B,H,W] (uint8 or any dtype convertible to float).
        Returns current [B, n_photo] float32, matching sim.py::_obs()'s
        formula exactly (gain*(intensity-ema), EMA initialized with the
        first frame -- no startup transient, same as compute_retina_currents)."""
        frames = frames_uint8.to(dtype=torch.float32) / 255.0
        blurred = gaussian_blur_batch_torch(frames, sigma=self.blur_sigma)
        intensity = blurred[:, self.sample_r, self.sample_c] * self.sample_mask
        if self.ema is None:
            self.ema = intensity.clone()
        current = self.gain * (intensity - self.ema)
        self.ema = self.ema_alpha * intensity + (1 - self.ema_alpha) * self.ema
        return current * self.included_mask.to(current.dtype)
