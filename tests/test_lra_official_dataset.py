from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from src.data.lra_official import (
    LraOfficialDataConfig,
    build_lra_official_dataloaders,
    prepare_lra_official_task,
    sha256_file,
)


def _write_raw_split(path: Path, samples: list[list[int]], labels: list[int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": samples, "labels": labels}, path)
    return sha256_file(path)


def _write_manifest(path: Path, train_path: Path, val_path: Path, test_path: Path, wrong_parity: bool = False) -> None:
    manifest = {
        "manifest_version": 1,
        "preprocess_version": "unit-test-v1",
        "defaults": {"pad_token_id": 0, "truncation_side": "right"},
        "tasks": {
            "listops": {
                "num_classes": 3,
                "vocab_size": 100,
                "expected_splits": {
                    "train": {
                        "num_samples": 99 if wrong_parity else 3,
                        "label_hist": {"0": 1, "1": 1, "2": 1},
                    },
                    "val": {
                        "num_samples": 2,
                        "label_hist": {"0": 1, "1": 1},
                    },
                    "test": {
                        "num_samples": 2,
                        "label_hist": {"1": 2},
                    },
                },
                "splits": {
                    "train": {
                        "path": str(train_path),
                        "format": "pt",
                        "sha256": sha256_file(train_path),
                    },
                    "val": {
                        "path": str(val_path),
                        "format": "pt",
                        "sha256": sha256_file(val_path),
                    },
                    "test": {
                        "path": str(test_path),
                        "format": "pt",
                        "sha256": sha256_file(test_path),
                    },
                },
            }
        },
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)


def test_prepare_lra_official_task_parity_and_fingerprint(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    train = raw_dir / "listops_train.pt"
    val = raw_dir / "listops_val.pt"
    test = raw_dir / "listops_test.pt"

    _write_raw_split(train, [[1, 2, 3], [4, 5], [6, 7, 8, 9]], [0, 1, 2])
    _write_raw_split(val, [[1], [2, 3, 4, 5]], [0, 1])
    _write_raw_split(test, [[9, 9], [8]], [1, 1])

    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, train, val, test)

    meta = prepare_lra_official_task(
        manifest_path=manifest_path,
        task="listops",
        seq_len=4,
        raw_root=raw_dir,
        processed_root=processed_dir,
        strict_parity=True,
        overwrite=True,
    )

    assert "fingerprint" in meta and len(meta["fingerprint"]) == 64
    assert meta["splits"]["train"]["num_samples"] == 3
    assert meta["splits"]["val"]["num_samples"] == 2
    assert meta["splits"]["test"]["num_samples"] == 2

    train_pt = processed_dir / "listops" / "train.pt"
    assert train_pt.exists()

    payload = torch.load(train_pt, map_location="cpu")
    assert payload["input_ids"].shape == (3, 4)
    assert payload["attention_mask"].shape == (3, 4)

    fingerprint_json = processed_dir / "listops" / "fingerprint.json"
    assert fingerprint_json.exists()
    with fingerprint_json.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["splits"]["train"]["num_samples"] == 3


def test_prepare_lra_official_task_raises_on_parity_mismatch(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    train = raw_dir / "listops_train.pt"
    val = raw_dir / "listops_val.pt"
    test = raw_dir / "listops_test.pt"

    _write_raw_split(train, [[1, 2, 3], [4, 5], [6, 7, 8, 9]], [0, 1, 2])
    _write_raw_split(val, [[1], [2, 3, 4, 5]], [0, 1])
    _write_raw_split(test, [[9, 9], [8]], [1, 1])

    manifest_path = tmp_path / "manifest_bad.yaml"
    _write_manifest(manifest_path, train, val, test, wrong_parity=True)

    try:
        prepare_lra_official_task(
            manifest_path=manifest_path,
            task="listops",
            seq_len=4,
            raw_root=raw_dir,
            processed_root=processed_dir,
            strict_parity=True,
            overwrite=True,
        )
    except ValueError as exc:
        assert "Parity mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for parity mismatch")


def test_build_lra_official_dataloaders_reads_prepared_data(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    train = raw_dir / "listops_train.pt"
    val = raw_dir / "listops_val.pt"
    test = raw_dir / "listops_test.pt"

    _write_raw_split(train, [[1, 2, 3], [4, 5], [6, 7, 8, 9]], [0, 1, 2])
    _write_raw_split(val, [[1], [2, 3, 4, 5]], [0, 1])
    _write_raw_split(test, [[9, 9], [8]], [1, 1])

    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, train, val, test)

    prepare_lra_official_task(
        manifest_path=manifest_path,
        task="listops",
        seq_len=4,
        raw_root=raw_dir,
        processed_root=processed_dir,
        strict_parity=True,
        overwrite=True,
    )

    cfg = LraOfficialDataConfig(
        task="listops",
        seq_len=4,
        batch_size=2,
        manifest_path=str(manifest_path),
        raw_root=str(raw_dir),
        processed_root=str(processed_dir),
        strict_parity=True,
    )
    loaders, fingerprint = build_lra_official_dataloaders(cfg, seed=0, ensure_prepared=False)

    batch = next(iter(loaders["train"]))
    assert batch["input_ids"].shape[1] == 4
    assert "fingerprint" in fingerprint
