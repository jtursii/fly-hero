from __future__ import annotations

import torch

from flyhero.brain.readout import Readout


def test_readout_shape_and_dtype():
    n_dn = 1303
    readout = Readout(n_dn, n_actions=6)
    dn_rate = torch.rand(8, n_dn)
    logits = readout(dn_rate)
    assert logits.shape == (8, 6)
    assert logits.dtype == torch.float32
