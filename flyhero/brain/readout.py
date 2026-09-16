"""Linear readout from descending-neuron (DN) rates only (PLAN Phase 3 task
2). Invariant 2: actions come only from a linear readout of DN activity --
any change to its inputs needs the user's approval and a DECISIONS.md entry.
"""

from __future__ import annotations

import torch
from torch import nn


class Readout(nn.Module):
    def __init__(self, n_dn: int, n_actions: int = 6):
        super().__init__()
        self.linear = nn.Linear(n_dn, n_actions)

    def forward(self, dn_rate: torch.Tensor) -> torch.Tensor:
        """dn_rate: [B, n_dn], the mean DN rate over a frame's substeps
        (ConnectomeBrain.frame_step's second return value). Returns
        [B, n_actions] logits (5 frets + strum)."""
        return self.linear(dn_rate)
