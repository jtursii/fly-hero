"""Wraps ConnectomeBrain + the retina encoder + Readout behind the same
(init_state, reset_stream, step_frame) interface as
flyhero.brain.baseline_gru.GRUBaseline, so train/bc.py can drive either
model without a branch. Retina -> brain -> readout is invariant 1/2's exact
data path (game observations enter only through the retina into designated
input neurons; actions come only from a linear readout of DN activity) --
this wrapper doesn't change that path, it only presents it uniformly.
"""

from __future__ import annotations

import torch
from torch import nn

from flyhero.brain.rate_model import ConnectomeBrain
from flyhero.brain.readout import Readout
from flyhero.game.retina_torch import TorchRetina


class ConnectomeBrainPolicy(nn.Module):
    def __init__(self, brain: ConnectomeBrain, retina: TorchRetina, readout: Readout, substeps_per_frame: int):
        super().__init__()
        self.brain = brain
        self.retina = retina
        self.readout = readout
        self.substeps_per_frame = substeps_per_frame

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.brain.init_state(batch)

    def reset_stream(self, batch: int) -> None:
        self.retina.reset()

    def step_frame(self, state: torch.Tensor, frame_uint8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        i_photo = self.retina.step(frame_uint8).to(dtype=self.brain.dtype)
        new_state, dn_rate_mean = self.brain.frame_step(state, i_photo, n_substeps=self.substeps_per_frame)
        logits = self.readout(dn_rate_mean)
        return new_state, logits

    def step_frame_recorded(
        self, state: torch.Tensor, frame_uint8: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """step_frame plus the intermediates export/record.py needs:
        (new_state, logits, i_photo[B, n_input], all_rate_mean[B, N]). The
        computation is identical to step_frame's -- a test asserts the two
        produce bit-identical logits on the same clip."""
        i_photo = self.retina.step(frame_uint8).to(dtype=self.brain.dtype)
        new_state, dn_rate_mean, all_rate_mean = self.brain.frame_step_recorded(
            state, i_photo, n_substeps=self.substeps_per_frame
        )
        logits = self.readout(dn_rate_mean)
        return new_state, logits, i_photo, all_rate_mean

    def readout_parameters(self):
        return self.readout.parameters()

    def non_readout_parameters(self):
        return list(self.brain.parameters())
