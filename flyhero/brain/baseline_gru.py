"""CNN -> GRU baseline (PLAN Phase 3 task 5): validates labels, the loss, the
decoder, and the training loop -- not the fly. Consumes the rendered frame
directly, not the retina encoder (invariant 1 constrains ConnectomeBrain's
inputs specifically; PLAN says this baseline exists precisely to check the
rest of the pipeline independent of the connectome).

Exposes the same (init_state, reset_stream, step_frame) interface as
flyhero.brain.policy.ConnectomeBrainPolicy so train/bc.py can drive either
model without knowing which one it has.
"""

from __future__ import annotations

import torch
from torch import nn

from flyhero.brain.readout import Readout


class FrameCNN(nn.Module):
    """3 stride-2 conv layers (64x64 -> 8x8 at 64 channels, for the default
    frame_size=64), flattened and projected to out_dim. Flatten dim is
    measured with a dummy forward pass at construction time (not
    nn.LazyLinear) so all parameters exist before the optimizer is built --
    LazyLinear's weight doesn't exist until the first real forward call,
    which would silently drop it from a hyperparameter group set up before
    that."""

    def __init__(self, frame_size: int, in_channels: int = 1, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, frame_size, frame_size)
            flat_dim = int(self.conv(dummy).flatten(1).shape[1])
        self.proj = nn.Linear(flat_dim, out_dim)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """frame: [B, in_channels, H, W] float. Returns [B, out_dim]."""
        feat = self.conv(frame).flatten(1)
        return torch.relu(self.proj(feat))


class GRUBaseline(nn.Module):
    def __init__(self, frame_size: int, n_actions: int = 6, cnn_out_dim: int = 128, gru_hidden: int = 256):
        super().__init__()
        self.cnn = FrameCNN(frame_size, out_dim=cnn_out_dim)
        self.gru_hidden = gru_hidden
        self.gru_cell = nn.GRUCell(cnn_out_dim, gru_hidden)
        self.readout = Readout(gru_hidden, n_actions)

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch, self.gru_hidden, device=device, dtype=dtype)

    def reset_stream(self, batch: int) -> None:
        """No separate visual-preprocessing state (unlike the retina's EMA)
        -- present only so train/bc.py can call it uniformly on both models."""

    def step_frame(self, state: torch.Tensor, frame_uint8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """frame_uint8: [B,H,W]. Returns (new_state[B,hidden], logits[B,6])."""
        frame = frame_uint8.to(dtype=state.dtype) / 255.0
        feat = self.cnn(frame.unsqueeze(1))
        new_state = self.gru_cell(feat, state)
        logits = self.readout(new_state)
        return new_state, logits

    def readout_parameters(self):
        return self.readout.parameters()

    def non_readout_parameters(self):
        return list(self.cnn.parameters()) + list(self.gru_cell.parameters())
