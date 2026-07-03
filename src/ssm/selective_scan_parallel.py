from __future__ import annotations

import torch


def _validate_inputs(
    A: torch.Tensor,
    Bdisc: torch.Tensor,
    C: torch.Tensor,
    Dskip: torch.Tensor | None,
    u: torch.Tensor | None,
) -> None:
    if A.ndim != 4:
        raise ValueError(f"Expected A to have shape [B, L, P, N], got {tuple(A.shape)}")
    if Bdisc.shape != A.shape:
        raise ValueError(f"Expected Bdisc to match A shape {tuple(A.shape)}, got {tuple(Bdisc.shape)}")
    if C.ndim != 3:
        raise ValueError(f"Expected C to have shape [B, L, N], got {tuple(C.shape)}")

    b, l, p, n = A.shape
    if C.shape != (b, l, n):
        raise ValueError(f"Expected C shape {(b, l, n)}, got {tuple(C.shape)}")

    if Dskip is not None:
        if Dskip.ndim != 1 or Dskip.shape[0] != p:
            raise ValueError(f"Expected Dskip shape [{p}], got {tuple(Dskip.shape)}")
        if u is None:
            raise ValueError("u is required when Dskip is provided.")

    if u is not None and u.shape != (b, l, p):
        raise ValueError(f"Expected u shape {(b, l, p)}, got {tuple(u.shape)}")


def selective_scan_naive(
    A: torch.Tensor,
    Bdisc: torch.Tensor,
    C: torch.Tensor,
    Dskip: torch.Tensor | None = None,
    u: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference recurrence implementation.

    Recurrence per channel/state:
      X_t = A_t * X_{t-1} + Bdisc_t
      y_t = sum_n(X_t * C_t) + Dskip * u_t
    """
    _validate_inputs(A, Bdisc, C, Dskip, u)

    b, l, p, n = A.shape
    state = torch.zeros((b, p, n), device=A.device, dtype=A.dtype)
    outputs: list[torch.Tensor] = []
    for t in range(l):
        state = A[:, t, :, :] * state + Bdisc[:, t, :, :]
        y_t = (state * C[:, t, None, :]).sum(dim=-1)
        if Dskip is not None and u is not None:
            y_t = y_t + Dskip[None, :] * u[:, t, :]
        outputs.append(y_t)
    return torch.stack(outputs, dim=1)


def selective_scan_parallel(
    A: torch.Tensor,
    Bdisc: torch.Tensor,
    C: torch.Tensor,
    Dskip: torch.Tensor | None = None,
    u: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Parallel selective scan using cumulative products/sums over time.

    This computes the same recurrence as ``selective_scan_naive`` in O(L) time with vectorized
    operations by using:
      P_t = prod_{k<=t} A_k
      X_t = P_t * sum_{k<=t}(Bdisc_k / P_k)
    """
    _validate_inputs(A, Bdisc, C, Dskip, u)

    # A should be positive in discretized SSMs (exp(delta * A_cont)); clamp for numerical safety.
    A_safe = A.clamp_min(float(eps))
    prefix = torch.cumprod(A_safe, dim=1)
    prefix_safe = prefix.clamp_min(float(eps))

    state = prefix * torch.cumsum(Bdisc / prefix_safe, dim=1)
    y = (state * C[:, :, None, :]).sum(dim=-1)

    if Dskip is not None and u is not None:
        y = y + Dskip[None, None, :] * u
    return y
