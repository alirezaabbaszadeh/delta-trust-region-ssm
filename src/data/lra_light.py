from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class LraLightDataConfig:
    task: str
    seq_len: int
    batch_size: int
    vocab_size: int
    num_classes: int
    train_size: int
    val_size: int
    test_size: int
    pad_token_id: int = 0
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2


_DEFAULT_TASK_SPECS = {
    "listops": {"vocab_size": 64, "num_classes": 10},
    "text": {"vocab_size": 256, "num_classes": 2},
    "pathfinder": {"vocab_size": 8, "num_classes": 2},
}


class SyntheticLraDataset(Dataset):
    def __init__(
        self,
        *,
        task: str,
        split: str,
        size: int,
        seq_len: int,
        vocab_size: int,
        num_classes: int,
        pad_token_id: int,
        seed: int,
    ):
        self.task = task
        self.split = split
        self.size = int(size)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.num_classes = int(num_classes)
        self.pad_token_id = int(pad_token_id)

        g = torch.Generator().manual_seed(int(seed))
        self.input_ids, self.attention_mask, self.labels = self._generate(g)

    def _sample_lengths(self, g: torch.Generator) -> torch.Tensor:
        min_len = max(8, self.seq_len // 2)
        return torch.randint(min_len, self.seq_len + 1, (self.size,), generator=g)

    def _apply_padding(self, tokens: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # tokens: [N, L]
        idx = torch.arange(self.seq_len).unsqueeze(0)
        mask = idx < lengths.unsqueeze(1)
        padded = tokens.clone()
        padded[~mask] = self.pad_token_id
        return padded, mask.long()

    def _generate_listops(self, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.randint(1, self.vocab_size, (self.size, self.seq_len), generator=g)
        lengths = self._sample_lengths(g)
        tokens, mask = self._apply_padding(tokens, lengths)
        masked = tokens * mask
        labels = (masked.sum(dim=1) % self.num_classes).long()
        return tokens, mask, labels

    def _generate_text(self, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.randint(1, self.vocab_size, (self.size, self.seq_len), generator=g)
        lengths = self._sample_lengths(g)
        tokens, mask = self._apply_padding(tokens, lengths)
        masked = tokens * mask
        token_mean = masked.float().sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        labels = (token_mean > (self.vocab_size / 2.0)).long()
        if self.num_classes > 2:
            labels = labels % self.num_classes
        return tokens, mask, labels

    def _generate_pathfinder(self, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.randint(1, self.vocab_size, (self.size, self.seq_len), generator=g)
        lengths = self._sample_lengths(g)
        tokens, mask = self._apply_padding(tokens, lengths)

        # Proxy signal: detect dense local patterns (as a cheap stand-in for path existence).
        binary = (tokens % 2 == 0).long() * mask
        windows = binary.unfold(dimension=1, size=min(8, self.seq_len), step=1)
        if windows.numel() == 0:
            dense = torch.zeros(self.size)
        else:
            dense = (windows.sum(dim=-1) >= 6).any(dim=-1).float()
        labels = dense.long()
        if self.num_classes > 2:
            labels = labels % self.num_classes
        return tokens, mask, labels

    def _generate(self, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.task == "listops":
            return self._generate_listops(g)
        if self.task == "text":
            return self._generate_text(g)
        if self.task == "pathfinder":
            return self._generate_pathfinder(g)
        raise ValueError(f"Unsupported task: {self.task}")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def resolve_task_defaults(task: str) -> dict[str, int]:
    task_key = task.lower()
    if task_key not in _DEFAULT_TASK_SPECS:
        raise ValueError(f"Unknown LRA-light task: {task}")
    return dict(_DEFAULT_TASK_SPECS[task_key])


def build_lra_light_dataloaders(cfg: LraLightDataConfig, *, seed: int) -> dict[str, DataLoader]:
    task = cfg.task.lower()
    train_ds = SyntheticLraDataset(
        task=task,
        split="train",
        size=cfg.train_size,
        seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        num_classes=cfg.num_classes,
        pad_token_id=cfg.pad_token_id,
        seed=seed,
    )
    val_ds = SyntheticLraDataset(
        task=task,
        split="val",
        size=cfg.val_size,
        seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        num_classes=cfg.num_classes,
        pad_token_id=cfg.pad_token_id,
        seed=seed + 1,
    )
    test_ds = SyntheticLraDataset(
        task=task,
        split="test",
        size=cfg.test_size,
        seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size,
        num_classes=cfg.num_classes,
        pad_token_id=cfg.pad_token_id,
        seed=seed + 2,
    )

    loader_kwargs = {
        "num_workers": int(cfg.num_workers),
        "pin_memory": bool(cfg.pin_memory),
    }
    if int(cfg.num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        loader_kwargs["prefetch_factor"] = int(max(2, int(cfg.prefetch_factor)))

    return {
        "train": DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs),
    }
