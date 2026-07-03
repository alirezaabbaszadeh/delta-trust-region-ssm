from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LadoConfig:
    enabled: bool = True
    l_ref: int = 256
    alpha: float = 0.5
    warmup_frac: float = 0.05
    clip_delta_norm: float | None = 0.1
    delta_weight_decay: float = 0.0


def compute_lr_delta(lr_base: float, seq_len: int, l_ref: int, alpha: float) -> float:
    scale = (l_ref / float(seq_len)) ** float(alpha)
    if scale > 1.0:
        scale = 1.0
    return lr_base * scale


def build_two_group_adam(
    model,
    *,
    lr_other: float,
    lr_delta: float,
    weight_decay: float = 0.0,
    delta_weight_decay: float = 0.0,
) -> tuple[torch.optim.Optimizer, int]:
    delta_params = list(model.delta_parameters())
    delta_param_ids = {id(p) for p in delta_params}
    other_params = [p for p in model.parameters() if id(p) not in delta_param_ids]

    optimizer = torch.optim.Adam(
        [
            {"params": other_params, "lr": lr_other, "weight_decay": weight_decay},
            {"params": delta_params, "lr": lr_delta, "weight_decay": delta_weight_decay},
        ]
    )
    delta_group_index = 1
    return optimizer, delta_group_index


def set_group_lr(optimizer: torch.optim.Optimizer, group_index: int, lr: float) -> None:
    optimizer.param_groups[group_index]["lr"] = float(lr)


def get_group_lr(optimizer: torch.optim.Optimizer, group_index: int) -> float:
    return float(optimizer.param_groups[group_index]["lr"])


def clip_delta_grads(delta_params: list[torch.nn.Parameter], max_norm: float) -> float:
    return float(torch.nn.utils.clip_grad_norm_(delta_params, max_norm=max_norm))


def grad_l2_norm(params: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum().cpu())
    return float(total**0.5)

