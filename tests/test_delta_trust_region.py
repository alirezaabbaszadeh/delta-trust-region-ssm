from __future__ import annotations

import torch

from src.models.seq_classifier import SequenceClassifier, SequenceClassifierConfig
from src.optim.delta_trust_region import apply_delta_trust_region, snapshot_delta_params


def test_delta_tr_projection_enforces_eps_bound() -> None:
    torch.manual_seed(0)

    model = SequenceClassifier(
        SequenceClassifierConfig(
            vocab_size=32,
            num_classes=2,
            d_model=16,
            num_layers=1,
            num_channel_blocks=4,
            state_dim=4,
            dropout=0.0,
        )
    )

    input_ids = torch.randint(1, 31, (3, 32))
    attention_mask = torch.ones_like(input_ids)

    old_snapshots = snapshot_delta_params(model)
    delta_old = model.collect_delta_values(input_ids, attention_mask).detach()

    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "w_delta") and hasattr(module, "b_delta"):
                module.w_delta.add_(0.8)
                module.b_delta.add_(0.8)

    eps = 0.01
    metrics = apply_delta_trust_region(
        model=model,
        old_snapshots=old_snapshots,
        delta_old=delta_old,
        delta_fn=lambda: model.collect_delta_values(input_ids, attention_mask),
        eps=eps,
        search_steps=12,
        sample_stride=1,
        max_positions=4096,
    )

    assert metrics["drift_pre_max"] > eps
    assert metrics["drift_post_max"] <= eps + 1e-4
    assert 0.0 <= metrics["trust_region_scale"] <= 1.0
