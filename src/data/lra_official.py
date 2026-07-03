from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class LraOfficialDataConfig:
    task: str
    seq_len: int
    batch_size: int
    manifest_path: str = "configs/datasets/lra_official_manifest.yaml"
    raw_root: str = "data/raw/lra_official"
    processed_root: str = "data/processed/lra_official"
    pad_token_id: int = 0
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    strict_parity: bool = True
    enforce_no_overlap: bool = True
    max_cross_split_overlap_ratio: float = 0.01


class TensorDictDataset(Dataset):
    def __init__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_split_name(name: str) -> str:
    lower = name.lower()
    if lower in {"valid", "validation", "dev"}:
        return "val"
    if lower in {"train", "val", "test"}:
        return lower
    return lower


def load_lra_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a mapping: {path}")

    required = ["manifest_version", "preprocess_version", "tasks"]
    for key in required:
        if key not in manifest:
            raise ValueError(f"Missing required manifest key: {key}")
    if not isinstance(manifest["tasks"], dict):
        raise ValueError("Manifest 'tasks' must be a mapping.")

    return manifest


def _download_to_path(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, out_path.open("wb") as out_file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)


def _resolve_source_file(spec: dict[str, Any], *, raw_dir: Path, split_name: str) -> Path:
    local_path = spec.get("path")
    if local_path:
        path = Path(str(local_path))
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    url = str(spec.get("url", "")).strip()
    if not url:
        raise FileNotFoundError(
            f"Split '{split_name}' has no 'path' or 'url' in manifest. "
            "Please set official source in configs/datasets/lra_official_manifest.yaml"
        )

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        return Path(urllib.request.url2pathname(parsed.path))

    filename = spec.get("filename")
    if not filename:
        filename = Path(parsed.path).name
        if not filename:
            filename = f"{split_name}.bin"

    out_path = raw_dir / filename
    if not out_path.exists():
        _download_to_path(url, out_path)
    return out_path


def _verify_sha256(path: Path, expected_sha256: str | None) -> str:
    actual_sha = sha256_file(path)
    if expected_sha256:
        expected = expected_sha256.strip().lower()
        if expected and expected != actual_sha:
            raise ValueError(
                f"SHA256 mismatch for {path}. expected={expected}, actual={actual_sha}. "
                "Update manifest or source file."
            )
    return actual_sha


def _load_records(path: Path, file_format: str) -> tuple[torch.Tensor | list[list[int]], list[int]]:
    fmt = file_format.lower().strip()
    if fmt == "pt":
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            tokens = payload.get("input_ids")
            if tokens is None:
                tokens = payload.get("tokens")
            labels = payload.get("labels")
        elif isinstance(payload, list):
            tokens = [row["input_ids"] for row in payload]
            labels = [row["label"] for row in payload]
        else:
            raise ValueError(f"Unsupported .pt payload type: {type(payload)!r}")

        if tokens is None or labels is None:
            raise ValueError(f"Expected keys input_ids/tokens and labels in {path}")

        token_data: torch.Tensor | list[list[int]]
        if isinstance(tokens, torch.Tensor):
            if tokens.ndim != 2:
                raise ValueError(f"Expected 2D input_ids tensor in {path}, got shape={tuple(tokens.shape)}")
            token_data = tokens.long()
        else:
            token_data = [[int(v) for v in row] for row in tokens]

        if isinstance(labels, torch.Tensor):
            label_list = [int(v) for v in labels.tolist()]
        else:
            label_list = [int(v) for v in labels]

        n_rows = int(token_data.shape[0]) if isinstance(token_data, torch.Tensor) else len(token_data)
        if n_rows != len(label_list):
            raise ValueError(f"Token/label size mismatch in {path}")
        return token_data, label_list

    if fmt == "jsonl":
        token_lists: list[list[int]] = []
        label_list: list[int] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                tokens = row.get("input_ids")
                if tokens is None:
                    tokens = row.get("tokens")
                label = row.get("label")
                if tokens is None or label is None:
                    raise ValueError(f"JSONL row missing input_ids/tokens or label in {path}")
                token_lists.append([int(v) for v in tokens])
                label_list.append(int(label))
        return token_lists, label_list

    raise ValueError(f"Unsupported split format: {file_format}")


def _pad_or_truncate(
    tokens: list[int],
    seq_len: int,
    *,
    pad_token_id: int,
    truncation_side: str,
) -> tuple[list[int], list[int], int]:
    true_len = len(tokens)
    if true_len >= seq_len:
        if truncation_side == "left":
            clipped = tokens[true_len - seq_len : true_len]
        else:
            clipped = tokens[:seq_len]
        mask = [1] * seq_len
        return clipped, mask, min(true_len, seq_len)

    padded = tokens + [pad_token_id] * (seq_len - true_len)
    mask = [1] * true_len + [0] * (seq_len - true_len)
    return padded, mask, true_len


def _length_stats(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}

    tensor = torch.tensor(lengths, dtype=torch.float32)
    return {
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
    }


def _label_hist(labels: list[int]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for label in labels:
        key = str(int(label))
        hist[key] = hist.get(key, 0) + 1
    return hist


def _save_processed_split(
    *,
    source_path: Path,
    out_path: Path,
    split_name: str,
    preprocess_version: str,
    pad_token_id: int,
    truncation_side: str,
    seq_len: int,
    source_format: str,
    source_sha256: str,
) -> dict[str, Any]:
    token_data, labels = _load_records(source_path, source_format)

    if isinstance(token_data, torch.Tensor):
        tokens = token_data
        n, cur_len = int(tokens.shape[0]), int(tokens.shape[1])
        if cur_len >= seq_len:
            if truncation_side == "left":
                input_ids = tokens[:, cur_len - seq_len : cur_len]
            else:
                input_ids = tokens[:, :seq_len]
            attention_mask = torch.ones((n, seq_len), dtype=torch.long)
            true_lengths = [int(seq_len)] * n
        else:
            pad_width = int(seq_len - cur_len)
            pad = torch.full((n, pad_width), int(pad_token_id), dtype=torch.long)
            input_ids = torch.cat([tokens, pad], dim=1)
            attention_mask = torch.cat(
                [
                    torch.ones((n, cur_len), dtype=torch.long),
                    torch.zeros((n, pad_width), dtype=torch.long),
                ],
                dim=1,
            )
            true_lengths = [int(cur_len)] * n
    else:
        rows: list[list[int]] = []
        masks: list[list[int]] = []
        true_lengths: list[int] = []
        for tokens in token_data:
            clipped, mask, true_len = _pad_or_truncate(
                tokens,
                seq_len,
                pad_token_id=pad_token_id,
                truncation_side=truncation_side,
            )
            rows.append(clipped)
            masks.append(mask)
            true_lengths.append(true_len)
        input_ids = torch.tensor(rows, dtype=torch.long)
        attention_mask = torch.tensor(masks, dtype=torch.long)

    payload = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    processed_sha = sha256_file(out_path)
    return {
        "split": split_name,
        "num_samples": int(len(labels)),
        "seq_len": int(seq_len),
        "seq_len_stats": _length_stats(true_lengths),
        "label_hist": _label_hist(labels),
        "sha256": processed_sha,
        "source_sha256": source_sha256,
        "source_format": source_format,
        "preprocess_version": preprocess_version,
        "pad_token_id": int(pad_token_id),
        "truncation_side": truncation_side,
    }


def _validate_parity(task_name: str, expected: dict[str, Any], split_meta: dict[str, Any], strict: bool) -> None:
    if not expected:
        return

    def _handle(msg: str) -> None:
        if strict:
            raise ValueError(msg)
        print(f"[WARN] {msg}")

    expected_n = expected.get("num_samples")
    if expected_n is not None:
        actual_n = int(split_meta.get("num_samples", 0))
        if int(expected_n) != actual_n:
            _handle(
                f"Parity mismatch for task={task_name}, split={split_meta.get('split')}: "
                f"expected num_samples={expected_n}, actual={actual_n}"
            )

    expected_hist = expected.get("label_hist")
    if expected_hist is not None:
        actual_hist = split_meta.get("label_hist", {})
        if {str(k): int(v) for k, v in expected_hist.items()} != {str(k): int(v) for k, v in actual_hist.items()}:
            _handle(
                f"Parity mismatch for task={task_name}, split={split_meta.get('split')}: "
                f"expected label_hist={expected_hist}, actual={actual_hist}"
            )


def _sequence_signatures(input_ids: torch.Tensor) -> np.ndarray:
    if input_ids.ndim != 2:
        raise ValueError(f"Expected 2D input_ids tensor, got shape={tuple(input_ids.shape)}")

    arr = input_ids.detach().cpu().contiguous().numpy()
    if np.amin(arr) >= 0 and np.amax(arr) <= np.iinfo(np.uint16).max:
        arr = np.ascontiguousarray(arr, dtype=np.uint16)
    else:
        arr = np.ascontiguousarray(arr, dtype=np.uint32)

    n_rows = int(arr.shape[0])
    signatures = np.empty(n_rows, dtype="|V16")
    for idx in range(n_rows):
        digest = hashlib.blake2b(arr[idx].tobytes(), digest_size=16).digest()
        signatures[idx] = np.frombuffer(digest, dtype="|V16")[0]
    return signatures


def _compute_integrity_report(processed_dir: Path) -> dict[str, Any]:
    split_signatures: dict[str, np.ndarray] = {}
    split_sizes: dict[str, int] = {}
    split_report: dict[str, dict[str, Any]] = {}

    for split in ("train", "val", "test"):
        split_path = processed_dir / f"{split}.pt"
        input_ids, _, _ = _load_processed_split(split_path)
        signatures = _sequence_signatures(input_ids)
        unique = np.unique(signatures)

        total = int(signatures.shape[0])
        uniq_n = int(unique.shape[0])
        dup_n = int(total - uniq_n)

        split_signatures[split] = unique
        split_sizes[split] = total
        split_report[split] = {
            "num_sequences": total,
            "unique_sequences": uniq_n,
            "duplicate_sequences": dup_n,
            "duplicate_ratio": float(dup_n / max(total, 1)),
        }

    overlaps: dict[str, Any] = {}
    total_overlap = 0
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = int(np.intersect1d(split_signatures[a], split_signatures[b], assume_unique=True).shape[0])
        total_overlap += inter
        denom = max(min(split_sizes[a], split_sizes[b]), 1)
        overlaps[f"{a}_{b}"] = {
            "overlap_count": inter,
            "overlap_ratio_min_split": float(inter / denom),
        }

    return {
        "hash_method": "blake2b_128_per_sequence",
        "splits": split_report,
        "cross_split_overlap": overlaps,
        "cross_split_overlap_total": int(total_overlap),
    }


def prepare_lra_official_task(
    *,
    manifest_path: str | Path,
    task: str,
    seq_len: int,
    raw_root: str | Path = "data/raw/lra_official",
    processed_root: str | Path = "data/processed/lra_official",
    pad_token_id: int | None = None,
    strict_parity: bool = True,
    enforce_no_overlap: bool = True,
    max_cross_split_overlap_ratio: float = 0.01,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = load_lra_manifest(manifest_file)

    task_key = task.lower()
    task_spec = manifest["tasks"].get(task_key)
    if task_spec is None:
        raise ValueError(f"Task '{task}' not found in manifest {manifest_file}")

    defaults = manifest.get("defaults", {})
    preprocess_version = str(manifest.get("preprocess_version", "v1"))
    truncation_side = str(defaults.get("truncation_side", "right")).lower()
    if truncation_side not in {"left", "right"}:
        raise ValueError(f"Unsupported truncation_side: {truncation_side}")

    eff_pad_token_id = int(defaults.get("pad_token_id", 0) if pad_token_id is None else pad_token_id)
    raw_dir = Path(raw_root) / task_key
    out_dir = Path(processed_root) / task_key
    out_dir.mkdir(parents=True, exist_ok=True)

    split_specs = task_spec.get("splits", {})
    if not split_specs:
        raise ValueError(f"Task '{task_key}' has no splits in manifest")

    task_meta: dict[str, Any] = {
        "task": task_key,
        "manifest_path": str(manifest_file),
        "manifest_sha256": sha256_file(manifest_file),
        "preprocess_version": preprocess_version,
        "seq_len": int(seq_len),
        "pad_token_id": eff_pad_token_id,
        "truncation_side": truncation_side,
        "splits": {},
    }

    expected_splits = task_spec.get("expected_splits", {})

    for raw_split_name, split_spec in split_specs.items():
        split_name = _normalize_split_name(raw_split_name)
        source_path = _resolve_source_file(split_spec, raw_dir=raw_dir, split_name=split_name)
        if not source_path.exists():
            raise FileNotFoundError(f"Raw split file does not exist: {source_path}")

        source_sha = _verify_sha256(source_path, split_spec.get("sha256"))

        out_path = out_dir / f"{split_name}.pt"
        meta_path = out_dir / f"{split_name}.meta.json"

        if out_path.exists() and meta_path.exists() and not overwrite:
            with meta_path.open("r", encoding="utf-8") as f:
                split_meta = json.load(f)
        else:
            split_meta = _save_processed_split(
                source_path=source_path,
                out_path=out_path,
                split_name=split_name,
                preprocess_version=preprocess_version,
                pad_token_id=eff_pad_token_id,
                truncation_side=truncation_side,
                seq_len=int(seq_len),
                source_format=str(split_spec.get("format", "pt")),
                source_sha256=source_sha,
            )
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(split_meta, f, ensure_ascii=False, indent=2)

        expected_for_split = expected_splits.get(raw_split_name) or expected_splits.get(split_name) or {}
        _validate_parity(task_key, expected_for_split, split_meta, strict=strict_parity)
        task_meta["splits"][split_name] = split_meta

    integrity = _compute_integrity_report(out_dir)
    task_meta["integrity"] = integrity
    with (out_dir / "leakage_report.json").open("w", encoding="utf-8") as f:
        json.dump(integrity, f, ensure_ascii=False, indent=2)

    if enforce_no_overlap:
        overlaps = integrity.get("cross_split_overlap", {})
        max_pair_ratio = 0.0
        for info in overlaps.values():
            if isinstance(info, dict):
                max_pair_ratio = max(max_pair_ratio, float(info.get("overlap_ratio_min_split", 0.0)))
        integrity["max_pair_overlap_ratio"] = float(max_pair_ratio)
        if max_pair_ratio > float(max_cross_split_overlap_ratio):
            raise ValueError(
                f"Data leakage threshold exceeded for task={task_key}: "
                f"max_pair_overlap_ratio={max_pair_ratio:.6f} > "
                f"max_cross_split_overlap_ratio={float(max_cross_split_overlap_ratio):.6f}"
            )

    task_meta["fingerprint"] = sha256_json(task_meta)
    with (out_dir / "fingerprint.json").open("w", encoding="utf-8") as f:
        json.dump(task_meta, f, ensure_ascii=False, indent=2)

    return task_meta


def _load_processed_split(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Processed split must be dict, got {type(payload)!r}: {path}")
    try:
        input_ids = payload["input_ids"].long()
        attention_mask = payload["attention_mask"].long()
        labels = payload["labels"].long()
    except KeyError as exc:
        raise ValueError(f"Processed split missing key {exc} in {path}") from exc
    return input_ids, attention_mask, labels


def _ensure_task_prepared(cfg: LraOfficialDataConfig, *, overwrite: bool = False) -> dict[str, Any]:
    out_dir = Path(cfg.processed_root) / cfg.task.lower()
    fingerprint_path = out_dir / "fingerprint.json"
    required = [out_dir / "train.pt", out_dir / "val.pt", out_dir / "test.pt", fingerprint_path]

    if all(path.exists() for path in required) and not overwrite:
        with fingerprint_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    return prepare_lra_official_task(
        manifest_path=cfg.manifest_path,
        task=cfg.task,
        seq_len=cfg.seq_len,
        raw_root=cfg.raw_root,
        processed_root=cfg.processed_root,
        pad_token_id=cfg.pad_token_id,
        strict_parity=cfg.strict_parity,
        enforce_no_overlap=cfg.enforce_no_overlap,
        max_cross_split_overlap_ratio=cfg.max_cross_split_overlap_ratio,
        overwrite=overwrite,
    )


def build_lra_official_dataloaders(
    cfg: LraOfficialDataConfig,
    *,
    seed: int,
    ensure_prepared: bool = False,
    overwrite: bool = False,
) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    if ensure_prepared:
        fingerprint = _ensure_task_prepared(cfg, overwrite=overwrite)
    else:
        out_dir = Path(cfg.processed_root) / cfg.task.lower()
        fingerprint_path = out_dir / "fingerprint.json"
        if not fingerprint_path.exists():
            raise FileNotFoundError(
                f"Missing prepared official dataset for task={cfg.task}. "
                f"Run scripts/prepare_lra_official.py first (manifest={cfg.manifest_path})."
            )
        with fingerprint_path.open("r", encoding="utf-8") as f:
            fingerprint = json.load(f)

    base = Path(cfg.processed_root) / cfg.task.lower()
    train_ids, train_mask, train_labels = _load_processed_split(base / "train.pt")
    val_ids, val_mask, val_labels = _load_processed_split(base / "val.pt")
    test_ids, test_mask, test_labels = _load_processed_split(base / "test.pt")

    train_ds = TensorDictDataset(input_ids=train_ids, attention_mask=train_mask, labels=train_labels)
    val_ds = TensorDictDataset(input_ids=val_ids, attention_mask=val_mask, labels=val_labels)
    test_ds = TensorDictDataset(input_ids=test_ids, attention_mask=test_mask, labels=test_labels)

    generator = torch.Generator().manual_seed(int(seed))
    loader_kwargs: dict[str, Any] = {
        "num_workers": int(cfg.num_workers),
        "pin_memory": bool(cfg.pin_memory),
    }
    if int(cfg.num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        loader_kwargs["prefetch_factor"] = int(max(2, int(cfg.prefetch_factor)))

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=generator,
            **loader_kwargs,
        ),
        "val": DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs),
    }
    return loaders, fingerprint
