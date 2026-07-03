from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeltaTrustRegionConfig:
    """
    Length-aware trust region on the *outputs* of Δ (input-dependent sampling intervals).

    For the toy S6Regressor, Δ_t = softplus(w * u_t + b). After an optimizer step proposes
    (w_new, b_new), we project the update back towards (w_old, b_old) so that:

        max_{t,batch} |Δ_t(new) - Δ_t(old)| <= eps(L)

    where eps(L) shrinks with the sequence length.
    """

    enabled: bool = True
    l_ref: int = 256
    alpha: float = 0.5
    eps_ref: float = 0.05
    search_steps: int = 12


def compute_eps(seq_len: int, *, eps_ref: float, l_ref: int, alpha: float) -> float:
    scale = (l_ref / float(seq_len)) ** float(alpha)
    if scale > 1.0:
        scale = 1.0
    return float(eps_ref) * float(scale)


@torch.no_grad()
def apply_softplus_affine_delta_trust_region(
    *,
    u: torch.Tensor,
    w: torch.nn.Parameter,
    b: torch.nn.Parameter,
    w_old: torch.Tensor,
    b_old: torch.Tensor,
    eps: float,
    search_steps: int,
) -> dict[str, float]:
    """
    Project (w, b) update so that max change in Δ(u) is bounded by eps.

    Returns metrics:
      - delta_drift_max_pre
      - delta_drift_max_post
      - trust_region_scale (in [0,1])
    """
    if u.ndim != 2:
        raise ValueError(f"Expected u to have shape [batch, seq_len], got {tuple(u.shape)}")

    def delta_from_params(wv: torch.Tensor, bv: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(wv * u + bv)

    delta_old = delta_from_params(w_old, b_old)

    w_new = w.detach()
    b_new = b.detach()
    delta_new = delta_from_params(w_new, b_new)

    drift_pre = float((delta_new - delta_old).abs().max().cpu())
    if not (drift_pre > eps):
        return {"delta_drift_max_pre": drift_pre, "delta_drift_max_post": drift_pre, "trust_region_scale": 1.0}

    # Binary search over interpolation scale s in [0, 1].
    lo = 0.0
    hi = 1.0
    for _ in range(int(search_steps)):
        mid = 0.5 * (lo + hi)
        w_mid = w_old + mid * (w_new - w_old)
        b_mid = b_old + mid * (b_new - b_old)
        delta_mid = delta_from_params(w_mid, b_mid)
        drift_mid = float((delta_mid - delta_old).abs().max().cpu())
        if drift_mid <= eps:
            lo = mid
        else:
            hi = mid

    scale = float(lo)
    w_proj = w_old + scale * (w_new - w_old)
    b_proj = b_old + scale * (b_new - b_old)
    w.copy_(w_proj)
    b.copy_(b_proj)

    delta_post = delta_from_params(w.detach(), b.detach())
    drift_post = float((delta_post - delta_old).abs().max().cpu())
    return {"delta_drift_max_pre": drift_pre, "delta_drift_max_post": drift_post, "trust_region_scale": scale}

