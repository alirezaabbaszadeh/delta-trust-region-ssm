from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

TASKS = ("listops", "text", "pathfinder")
SPLITS = ("train", "val", "test")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sample_split_payload(seed: int) -> dict:
    rows = [[seed + 1, seed + 2], [seed + 3], [seed + 4, seed + 5, seed + 6]]
    labels = [seed % 2, (seed + 1) % 2, seed % 2]
    return {"input_ids": rows, "labels": labels}


def _run_importer(input_path: Path, out_root: Path, manifest: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/import_official_lra_download.py",
        "--input",
        str(input_path),
        "--out-root",
        str(out_root),
        "--manifest",
        str(manifest),
        "--task-map",
        "auto",
        "--strict",
        "--overwrite",
    ]
    proc = subprocess.run(cmd, cwd=_repo_root(), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout


def _assert_manifest_and_splits(manifest_path: Path, out_root: Path) -> None:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)

    src = manifest.get("source", {})
    assert "official release import" in str(src.get("name", "")).lower()
    assert str(src.get("reference_url", "")).startswith("https://storage.googleapis.com/long-range-arena/")

    for task in TASKS:
        for split in SPLITS:
            split_path = out_root / task / f"{split}.pt"
            assert split_path.exists(), f"missing split: {split_path}"

            spec = manifest["tasks"][task]["splits"][split]
            assert spec["sha256"] == _sha256_file(split_path)

            payload = torch.load(split_path, map_location="cpu")
            labels = payload["labels"]
            n = len(labels)
            assert manifest["tasks"][task]["expected_splits"][split]["num_samples"] == n


def test_import_split_ready_directory(tmp_path: Path) -> None:
    downloads = tmp_path / "downloads_split_ready"
    out_root = tmp_path / "raw"
    manifest = tmp_path / "manifest.yaml"

    for ti, task in enumerate(TASKS):
        for si, split in enumerate(SPLITS):
            payload = _sample_split_payload(seed=ti * 10 + si)
            p = downloads / task / f"{split}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, p)

    _run_importer(downloads, out_root, manifest)
    _assert_manifest_and_splits(manifest, out_root)


def test_import_merged_task_files_directory(tmp_path: Path) -> None:
    downloads = tmp_path / "downloads_merged"
    out_root = tmp_path / "raw"
    manifest = tmp_path / "manifest.yaml"
    downloads.mkdir(parents=True, exist_ok=True)

    merged_names = {
        "listops": "listops.pt",
        "text": "imdb_lra.pt",
        "pathfinder": "pathfinder_lra.pt",
    }

    for ti, task in enumerate(TASKS):
        obj = {}
        for si, split in enumerate(SPLITS):
            obj[split] = _sample_split_payload(seed=ti * 10 + si)
        torch.save(obj, downloads / merged_names[task])

    _run_importer(downloads, out_root, manifest)
    _assert_manifest_and_splits(manifest, out_root)


def test_import_bundle_gz_file(tmp_path: Path) -> None:
    downloads = tmp_path / "downloads_bundle"
    out_root = tmp_path / "raw"
    manifest = tmp_path / "manifest.yaml"
    downloads.mkdir(parents=True, exist_ok=True)

    combined = {}
    for ti, task in enumerate(TASKS):
        combined[task] = {}
        for si, split in enumerate(SPLITS):
            combined[task][split] = _sample_split_payload(seed=ti * 10 + si)

    bundle_raw = downloads / "lra_release"
    torch.save(combined, bundle_raw)

    bundle_gz = downloads / "lra_release.gz"
    with bundle_raw.open("rb") as src, gzip.open(bundle_gz, "wb") as dst:
        dst.write(src.read())

    _run_importer(bundle_gz, out_root, manifest)
    _assert_manifest_and_splits(manifest, out_root)

    report = json.loads((out_root / "import_report.json").read_text(encoding="utf-8"))
    assert report["import_mode"] in {"combined_bundle", "split_ready", "merged_task_files"}
