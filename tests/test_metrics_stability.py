from __future__ import annotations

import torch

from src.utils.metrics import CollapseTracker, StabilityMetricConfig, drift_stats, spike_from_history


def test_spike_from_median_window() -> None:
    cfg = StabilityMetricConfig(spike_ratio=2.0, spike_window=5)
    history = [1.0, 1.1, 1.0, 1.2, 1.0]

    assert not spike_from_history(history, 1.8, cfg)
    assert spike_from_history(history, 2.5, cfg)


def test_collapse_tracker_patience() -> None:
    cfg = StabilityMetricConfig(collapse_margin=0.05, collapse_patience=3)
    tracker = CollapseTracker(cfg)

    assert not tracker.update(current_acc=0.6, num_classes=2)
    assert not tracker.update(current_acc=0.51, num_classes=2)
    assert not tracker.update(current_acc=0.5, num_classes=2)
    assert tracker.update(current_acc=0.49, num_classes=2)


def test_drift_stats_reports_max_and_p99() -> None:
    old = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float32)
    new = torch.tensor([0.0, 0.2, 0.4, 0.35], dtype=torch.float32)

    stats = drift_stats(old, new)
    assert stats["drift_max"] >= stats["drift_p99"]
    assert stats["drift_mean"] > 0.0
