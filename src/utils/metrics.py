from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StabilityMetricConfig:
    spike_ratio: float = 2.0
    spike_window: int = 10
    collapse_margin: float = 0.05
    collapse_patience: int = 3


def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.ndim != 2:
        raise ValueError(f"Expected logits shape [B, C], got {tuple(logits.shape)}")
    preds = logits.argmax(dim=-1)
    return float((preds == labels).float().mean().detach().cpu())


def grad_l2_norm(params: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum().cpu())
    return float(total**0.5)


def drift_stats(delta_old: torch.Tensor, delta_new: torch.Tensor) -> dict[str, float]:
    diff = (delta_new - delta_old).abs().reshape(-1)
    return {
        "drift_max": float(diff.max().detach().cpu()),
        "drift_p99": float(torch.quantile(diff.detach(), 0.99).cpu()),
        "drift_mean": float(diff.mean().detach().cpu()),
    }


def spike_from_history(loss_history: list[float], current_loss: float, cfg: StabilityMetricConfig) -> bool:
    if cfg.spike_window <= 0:
        return False
    if len(loss_history) < cfg.spike_window:
        return False
    window = loss_history[-cfg.spike_window :]
    baseline = sorted(window)[len(window) // 2]
    if baseline <= 0:
        return False
    return bool((current_loss / baseline) >= cfg.spike_ratio)


class CollapseTracker:
    def __init__(self, cfg: StabilityMetricConfig):
        self.cfg = cfg
        self._streak = 0

    @property
    def streak(self) -> int:
        return self._streak

    def update(self, current_acc: float, num_classes: int) -> bool:
        chance = 1.0 / max(1, int(num_classes))
        threshold = chance + self.cfg.collapse_margin
        if current_acc <= threshold:
            self._streak += 1
        else:
            self._streak = 0
        return bool(self._streak >= self.cfg.collapse_patience)


# Compatibility wrappers kept for legacy callsites.
def is_spike(current_loss: float, previous_loss: float | None, ratio: float = 2.0) -> bool:
    if previous_loss is None:
        return False
    if previous_loss <= 0:
        return False
    return bool(current_loss / previous_loss >= ratio)


def is_collapse(current_acc: float, num_classes: int, margin: float = 0.05) -> bool:
    chance = 1.0 / max(1, num_classes)
    return bool(current_acc <= chance + margin)
