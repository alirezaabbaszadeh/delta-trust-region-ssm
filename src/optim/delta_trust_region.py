from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeltaTrustRegionConfig:
    enabled: bool = False
    l_ref: int = 256
    alpha: float = 0.5
    eps_ref: float = 0.05
    search_steps: int = 12
    sample_stride: int = 1
    max_positions: int = 1024


def compute_eps(seq_len: int, *, eps_ref: float, l_ref: int, alpha: float) -> float:
    scale = (l_ref / float(seq_len)) ** float(alpha)
    if scale > 1.0:
        scale = 1.0
    return float(eps_ref) * float(scale)


def _collect_delta_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    modules: list[torch.nn.Module] = []
    for module in model.modules():
        if hasattr(module, "w_delta") and hasattr(module, "b_delta"):
            modules.append(module)
    if not modules:
        raise ValueError("No modules with w_delta/b_delta found in model.")
    return modules


def snapshot_delta_params(model: torch.nn.Module) -> list[tuple[torch.Tensor, torch.Tensor]]:
    snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
    for module in _collect_delta_modules(model):
        w = getattr(module, "w_delta").detach().clone()
        b = getattr(module, "b_delta").detach().clone()
        snapshots.append((w, b))
    return snapshots


def restore_delta_params(model: torch.nn.Module, snapshots: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    modules = _collect_delta_modules(model)
    if len(modules) != len(snapshots):
        raise ValueError("Snapshot size does not match model delta module count.")
    with torch.no_grad():
        for module, (w_old, b_old) in zip(modules, snapshots):
            getattr(module, "w_delta").copy_(w_old)
            getattr(module, "b_delta").copy_(b_old)


def _interpolate_delta_params(
    model: torch.nn.Module,
    *,
    old_snapshots: list[tuple[torch.Tensor, torch.Tensor]],
    step_snapshots: list[tuple[torch.Tensor, torch.Tensor]],
    scale: float,
) -> None:
    modules = _collect_delta_modules(model)
    with torch.no_grad():
        for module, (w_old, b_old), (w_step, b_step) in zip(modules, old_snapshots, step_snapshots):
            w_proj = w_old + float(scale) * (w_step - w_old)
            b_proj = b_old + float(scale) * (b_step - b_old)
            getattr(module, "w_delta").copy_(w_proj)
            getattr(module, "b_delta").copy_(b_proj)


def _drift_stats(
    delta_old: torch.Tensor,
    delta_new: torch.Tensor,
    *,
    sample_stride: int,
    max_positions: int,
) -> tuple[float, float]:
    diff = (delta_new - delta_old).abs()
    if sample_stride > 1 and diff.ndim >= 2:
        diff = diff[:, ::sample_stride, ...]

    flat = diff.reshape(-1)
    if max_positions > 0 and flat.numel() > max_positions:
        idx = torch.linspace(0, flat.numel() - 1, steps=max_positions, device=flat.device)
        flat = flat.index_select(0, idx.long())

    drift_max = float(flat.max().detach().cpu())
    drift_p99 = float(torch.quantile(flat.detach(), q=0.99).cpu())
    return drift_max, drift_p99


@torch.no_grad()
def apply_delta_trust_region(
    *,
    model: torch.nn.Module,
    old_snapshots: list[tuple[torch.Tensor, torch.Tensor]],
    delta_old: torch.Tensor,
    delta_fn: callable,
    eps: float,
    search_steps: int,
    sample_stride: int = 1,
    max_positions: int = 1024,
) -> dict[str, float]:
    """Project Δ-parameter updates so drift stays <= eps on the same batch."""
    step_snapshots = snapshot_delta_params(model)
    delta_new = delta_fn().detach()
    drift_pre_max, drift_pre_p99 = _drift_stats(
        delta_old,
        delta_new,
        sample_stride=sample_stride,
        max_positions=max_positions,
    )

    if drift_pre_max <= float(eps):
        return {
            "drift_pre_max": drift_pre_max,
            "drift_pre_p99": drift_pre_p99,
            "drift_post_max": drift_pre_max,
            "drift_post_p99": drift_pre_p99,
            "trust_region_scale": 1.0,
        }

    lo = 0.0
    hi = 1.0
    for _ in range(int(search_steps)):
        mid = 0.5 * (lo + hi)
        _interpolate_delta_params(
            model,
            old_snapshots=old_snapshots,
            step_snapshots=step_snapshots,
            scale=mid,
        )
        delta_mid = delta_fn().detach()
        drift_mid_max, _ = _drift_stats(
            delta_old,
            delta_mid,
            sample_stride=sample_stride,
            max_positions=max_positions,
        )
        if drift_mid_max <= float(eps):
            lo = mid
        else:
            hi = mid

    _interpolate_delta_params(
        model,
        old_snapshots=old_snapshots,
        step_snapshots=step_snapshots,
        scale=lo,
    )
    delta_post = delta_fn().detach()
    drift_post_max, drift_post_p99 = _drift_stats(
        delta_old,
        delta_post,
        sample_stride=sample_stride,
        max_positions=max_positions,
    )

    return {
        "drift_pre_max": drift_pre_max,
        "drift_pre_p99": drift_pre_p99,
        "drift_post_max": drift_post_max,
        "drift_post_p99": drift_post_p99,
        "trust_region_scale": float(lo),
    }
