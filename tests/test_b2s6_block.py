from __future__ import annotations

import torch

from src.models.b2s6_block import B2S6Block, B2S6BlockConfig


def test_b2s6_delta_shape_and_positivity() -> None:
    torch.manual_seed(0)
    cfg = B2S6BlockConfig(d_model=12, num_channel_blocks=3, state_dim=6)
    block = B2S6Block(cfg)

    x = torch.randn(2, 9, 12)
    delta = block.compute_delta(x)

    assert delta.shape == (2, 9, 3, 4)
    assert torch.all(delta > 0)

    y = block(x)
    assert y.shape == x.shape
