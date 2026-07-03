from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ToyTaskConfig:
    seq_len: int
    state_dim: int = 16


def _ensure_last_one(u: torch.Tensor) -> torch.Tensor:
    """Avoid degenerate outputs when using input-dependent C (as in S6-style units)."""
    if u.ndim != 2:
        raise ValueError(f"Expected u to have shape [batch, seq_len], got {tuple(u.shape)}")
    u = u.clone()
    u[:, -1] = 1.0
    return u


class S4DRegressor(nn.Module):
    """
    A minimal diagonal SSM (S4D-like) regressor for a univariate sequence.

    This is a deliberately small, research-friendly implementation intended for toy stability
    experiments (not a performance-optimized kernel).
    """

    def __init__(self, cfg: ToyTaskConfig):
        super().__init__()
        self.cfg = cfg

        # Continuous-time diagonal A with negative entries.
        self.log_A = nn.Parameter(torch.zeros(cfg.state_dim))
        self.B = nn.Parameter(torch.randn(cfg.state_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(cfg.state_dim) * 0.1)

        # Δ = exp(b) (positive)
        self.b = nn.Parameter(torch.tensor(-3.0))

    def delta_parameters(self) -> list[nn.Parameter]:
        return [self.b]

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u = _ensure_last_one(u)
        batch_size, seq_len = u.shape
        if seq_len != self.cfg.seq_len:
            raise ValueError(f"Expected seq_len={self.cfg.seq_len}, got {seq_len}")

        A_c = -torch.exp(self.log_A)  # [n]
        delta = torch.exp(self.b)  # scalar
        A_d = torch.exp(delta * A_c)  # [n]
        # (A_d - 1) / A_c is well-defined since A_c < 0.
        B_d = ((A_d - 1.0) / A_c) * self.B  # [n]

        x = torch.zeros(batch_size, self.cfg.state_dim, device=u.device, dtype=u.dtype)
        for t in range(seq_len):
            u_t = u[:, t].unsqueeze(-1)  # [b, 1]
            x = A_d.unsqueeze(0) * x + B_d.unsqueeze(0) * u_t

        y = (self.C.unsqueeze(0) * x).sum(dim=-1)  # [b]
        return y


class S6Regressor(nn.Module):
    """
    A minimal S6/Mamba-like selective diagonal SSM regressor for a univariate sequence.

    Key idea: input-dependent Δ (and input-dependent B and C scaling), which can make training
    unstable on long sequences if Δ-parameters are optimized too aggressively.
    """

    def __init__(self, cfg: ToyTaskConfig):
        super().__init__()
        self.cfg = cfg

        self.log_A = nn.Parameter(torch.zeros(cfg.state_dim))
        self.B_weight = nn.Parameter(torch.randn(cfg.state_dim) * 0.1)
        self.B_bias = nn.Parameter(torch.zeros(cfg.state_dim))
        self.C = nn.Parameter(torch.randn(cfg.state_dim) * 0.1)

        # Δ_t = softplus(w * u_t + b)
        self.w = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(-3.0))

    def delta_parameters(self) -> list[nn.Parameter]:
        return [self.w, self.b]

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u = _ensure_last_one(u)
        batch_size, seq_len = u.shape
        if seq_len != self.cfg.seq_len:
            raise ValueError(f"Expected seq_len={self.cfg.seq_len}, got {seq_len}")

        A_c = -torch.exp(self.log_A)  # [n]
        A_c_b = A_c.unsqueeze(0)  # [1, n]

        x = torch.zeros(batch_size, self.cfg.state_dim, device=u.device, dtype=u.dtype)
        for t in range(seq_len):
            u_t = u[:, t]  # [b]
            delta_t = torch.nn.functional.softplus(self.w * u_t + self.b)  # [b]

            A_d_t = torch.exp(delta_t.unsqueeze(-1) * A_c_b)  # [b, n]
            B_in_t = self.B_weight.unsqueeze(0) * u_t.unsqueeze(-1) + self.B_bias.unsqueeze(0)  # [b, n]
            B_d_t = ((A_d_t - 1.0) / A_c_b) * B_in_t  # [b, n]

            x = A_d_t * x + B_d_t

        # C_t = u_t * C  -> y = (u_last * C)·x = u_last * (C·x)
        u_last = u[:, -1]  # always 1.0 by construction
        y = u_last * (self.C.unsqueeze(0) * x).sum(dim=-1)
        return y

