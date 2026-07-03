from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TransformerLiteConfig:
    vocab_size: int
    num_classes: int
    d_model: int = 128
    num_layers: int = 2
    nhead: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1
    max_seq_len: int = 4096
    pad_token_id: int = 0


class TransformerLiteClassifier(nn.Module):
    def __init__(self, cfg: TransformerLiteConfig):
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

    def delta_parameters(self) -> list[nn.Parameter]:
        return []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids [B, L], got {tuple(input_ids.shape)}")

        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.cfg.max_seq_len}")

        if attention_mask is None:
            attention_mask = (input_ids != self.cfg.pad_token_id).long()

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        weights = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)
