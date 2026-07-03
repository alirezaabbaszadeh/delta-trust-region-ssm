from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class S4DLiteConfig:
    vocab_size: int
    num_classes: int
    d_model: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    pad_token_id: int = 0


class S4DLiteMixer(nn.Module):
    """Diagonal SSM-style mixer used as a lightweight S4D proxy baseline."""

    def __init__(self, d_model: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(d_model))
        self.B = nn.Parameter(torch.randn(d_model) * 0.02)
        self.C = nn.Parameter(torch.randn(d_model) * 0.02)
        self.log_delta = nn.Parameter(torch.full((d_model,), -2.0))
        self.skip = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _bsz, _seq_len, d_model = x.shape

        A_c = -torch.exp(self.log_A)  # negative
        delta = torch.exp(self.log_delta).clamp_min(1e-6)
        A_d = torch.exp(delta * A_c)
        A_c_safe = torch.where(A_c.abs() < 1e-8, torch.full_like(A_c, -1e-8), A_c)
        B_d = ((A_d - 1.0) / A_c_safe) * self.B

        # Vectorized recurrence with masking:
        #   state_t = A_t * state_{t-1} + B_t
        # where A_t=1,B_t=0 on padded positions to keep state unchanged.
        m = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
        A_t = m * A_d.view(1, 1, d_model) + (1.0 - m)
        B_t = m * (x * B_d.view(1, 1, d_model))

        A_safe = A_t.clamp_min(1e-12)
        prefix = torch.cumprod(A_safe, dim=1)
        state = prefix * torch.cumsum(B_t / prefix.clamp_min(1e-12), dim=1)

        return state * self.C.view(1, 1, d_model) + x * self.skip.view(1, 1, d_model)


class S4DLiteClassifier(nn.Module):
    def __init__(self, cfg: S4DLiteConfig):
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.layers = nn.ModuleList([S4DLiteMixer(cfg.d_model) for _ in range(cfg.num_layers)])
        self.drop = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

    def delta_parameters(self) -> list[nn.Parameter]:
        return []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids [B, L], got {tuple(input_ids.shape)}")

        if attention_mask is None:
            attention_mask = (input_ids != self.cfg.pad_token_id).long()

        x = self.embedding(input_ids)
        for layer in self.layers:
            y = layer(x, attention_mask)
            x = x + self.drop(y)

        x = self.norm(x)
        weights = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)
