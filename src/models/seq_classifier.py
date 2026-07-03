from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.b2s6_block import B2S6Block, B2S6BlockConfig


@dataclass(frozen=True)
class SequenceClassifierConfig:
    vocab_size: int
    num_classes: int
    d_model: int = 128
    num_layers: int = 2
    num_channel_blocks: int = 4
    state_dim: int = 16
    dropout: float = 0.0
    pad_token_id: int = 0


class SequenceClassifier(nn.Module):
    def __init__(self, cfg: SequenceClassifierConfig):
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.layers = nn.ModuleList(
            [
                B2S6Block(
                    B2S6BlockConfig(
                        d_model=cfg.d_model,
                        num_channel_blocks=cfg.num_channel_blocks,
                        state_dim=cfg.state_dim,
                        use_dskip=True,
                    )
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

    def delta_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for layer in self.layers:
            params.extend(layer.delta_parameters())
        return params

    def non_delta_parameters(self) -> list[nn.Parameter]:
        delta_param_ids = {id(p) for p in self.delta_parameters()}
        return [p for p in self.parameters() if id(p) not in delta_param_ids]

    def _resolve_attention_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is not None:
            return attention_mask
        return (input_ids != self.cfg.pad_token_id).to(dtype=input_ids.dtype)

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        collect_deltas: bool,
        apply_dropout: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids with shape [B, L], got {tuple(input_ids.shape)}")

        x = self.embedding(input_ids)
        delta_values: list[torch.Tensor] = []

        for layer in self.layers:
            if collect_deltas:
                y, delta = layer(x, return_delta=True)
                delta_values.append(delta)
            else:
                y = layer(x)

            if apply_dropout:
                y = self.dropout(y)
            x = x + y

        x = self.norm(x)
        all_deltas = None
        if collect_deltas:
            all_deltas = torch.stack(delta_values, dim=2)  # [B, L, num_layers, H, P]
        return x, all_deltas

    def collect_delta_values(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Drift/projection calculations should be deterministic; disable dropout path.
        _, all_deltas = self._encode(
            input_ids,
            attention_mask,
            collect_deltas=True,
            apply_dropout=False,
        )
        if all_deltas is None:
            raise RuntimeError("Failed to collect delta values.")
        return all_deltas

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        attention_mask = self._resolve_attention_mask(input_ids, attention_mask)
        x, _ = self._encode(input_ids, attention_mask, collect_deltas=False, apply_dropout=self.training)

        weights = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)
