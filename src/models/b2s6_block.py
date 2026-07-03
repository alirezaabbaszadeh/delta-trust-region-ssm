from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.ssm.selective_scan_parallel import selective_scan_parallel


@dataclass(frozen=True)
class B2S6BlockConfig:
    d_model: int
    num_channel_blocks: int
    state_dim: int
    use_dskip: bool = True


class B2S6Block(nn.Module):
    """Block-biased selective SSM layer (B2S6-style channel block decomposition)."""

    def __init__(self, cfg: B2S6BlockConfig):
        super().__init__()
        if cfg.d_model % cfg.num_channel_blocks != 0:
            raise ValueError(
                f"d_model ({cfg.d_model}) must be divisible by num_channel_blocks ({cfg.num_channel_blocks})."
            )

        self.cfg = cfg
        self.num_blocks = cfg.num_channel_blocks
        self.block_width = cfg.d_model // cfg.num_channel_blocks
        self.state_dim = cfg.state_dim

        # Continuous-time diagonal A entries are constrained negative via -exp(log_A).
        self.log_A = nn.Parameter(torch.zeros(cfg.state_dim))

        h = self.num_blocks
        p = self.block_width
        n = self.state_dim

        self.B_weight = nn.Parameter(torch.randn(h, n, p) * 0.02)
        self.B_bias = nn.Parameter(torch.zeros(h, p, n))
        self.C = nn.Parameter(torch.randn(h, p, n) * 0.02)

        self.w_delta = nn.Parameter(torch.zeros(h, p))
        self.b_delta = nn.Parameter(torch.full((h, p), -2.0))

        if cfg.use_dskip:
            self.Dskip = nn.Parameter(torch.ones(h, p))
        else:
            self.register_parameter("Dskip", None)

    def delta_parameters(self) -> list[nn.Parameter]:
        return [self.w_delta, self.b_delta]

    def continuous_A(self) -> torch.Tensor:
        return -torch.exp(self.log_A)

    def compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Return blockwise Δ with shape [B, L, H, P]."""
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B, L, D], got {tuple(x.shape)}")
        b, l, d = x.shape
        if d != self.cfg.d_model:
            raise ValueError(f"Expected d_model={self.cfg.d_model}, got {d}")

        deltas: list[torch.Tensor] = []
        p = self.block_width
        for j in range(self.num_blocks):
            x_j = x[:, :, j * p : (j + 1) * p]
            s = torch.einsum("blp,p->bl", x_j, self.w_delta[j])
            delta = F.softplus(s[:, :, None] + self.b_delta[j][None, None, :])
            deltas.append(delta)
        return torch.stack(deltas, dim=2)

    def forward(self, x: torch.Tensor, return_delta: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B, L, D], got {tuple(x.shape)}")
        b, l, d = x.shape
        if d != self.cfg.d_model:
            raise ValueError(f"Expected d_model={self.cfg.d_model}, got {d}")

        p = self.block_width
        n = self.state_dim
        a_cont = self.continuous_A()  # [N], negative
        a_cont_safe = torch.where(a_cont.abs() < 1e-8, torch.full_like(a_cont, -1e-8), a_cont)

        outputs: list[torch.Tensor] = []
        deltas: list[torch.Tensor] = []

        for j in range(self.num_blocks):
            x_j = x[:, :, j * p : (j + 1) * p]  # [B, L, P]

            s = torch.einsum("blp,p->bl", x_j, self.w_delta[j])
            delta = F.softplus(s[:, :, None] + self.b_delta[j][None, None, :])  # [B, L, P]
            deltas.append(delta)

            A_d = torch.exp(delta[:, :, :, None] * a_cont[None, None, None, :])  # [B, L, P, N]

            # Input-dependent B plus channel-specific bias.
            b_in_linear = torch.einsum("blp,np->bln", x_j, self.B_weight[j])  # [B, L, N]
            B_in = b_in_linear[:, :, None, :] + self.B_bias[j][None, None, :, :]  # [B, L, P, N]

            B_disc = ((A_d - 1.0) / a_cont_safe[None, None, None, :]) * B_in
            C_t = torch.einsum("blp,pn->bln", x_j, self.C[j])  # [B, L, N]

            Dskip_j = None if self.Dskip is None else self.Dskip[j]
            y_j = selective_scan_parallel(A_d, B_disc, C_t, Dskip=Dskip_j, u=x_j)
            outputs.append(y_j)

        y = torch.cat(outputs, dim=-1)
        if return_delta:
            return y, torch.stack(deltas, dim=2)
        return y
