from __future__ import annotations

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


def _run(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout


def _write_split_ready_downloads(download_dir: Path) -> None:
    for ti, task in enumerate(TASKS):
        for si, split in enumerate(SPLITS):
            rows = [[1 + ti + si, 2 + ti + si], [3 + ti + si], [4 + ti + si, 5 + ti + si]]
            labels = [0, 1, 0]
            p = download_dir / task / f"{split}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"input_ids": rows, "labels": labels}, p)


def test_end_to_end_smoke_import_prepare_stageb_summarize(tmp_path: Path) -> None:
    repo = _repo_root()

    downloads = tmp_path / "downloads"
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    manifest = tmp_path / "manifest.yaml"
    runs_out = tmp_path / "stage_b"
    summary_out = tmp_path / "summary"
    cfg_dir = tmp_path / "configs"

    _write_split_ready_downloads(downloads)

    _run(
        [
            sys.executable,
            "scripts/import_official_lra_download.py",
            "--input",
            str(downloads),
            "--out-root",
            str(raw_root),
            "--manifest",
            str(manifest),
            "--strict",
            "--overwrite",
        ],
        cwd=repo,
    )

    cfg_dir.mkdir(parents=True, exist_ok=True)
    listops_cfg = {
        "task": "listops",
        "seq_len": 16,
        "batch_size": 2,
        "epochs": 1,
        "lr": 0.0003,
        "weight_decay": 0.01,
        "vocab_size": 64,
        "num_classes": 2,
        "pad_token_id": 0,
        "num_workers": 0,
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "strict_parity": True,
        "enforce_no_overlap": True,
        "max_cross_split_overlap_ratio": 1.0,
        "enforce_param_budget": True,
        "param_budget_tolerance": 0.10,
        "spike_ratio": 2.0,
        "spike_window": 3,
        "collapse_margin": 0.05,
        "collapse_patience": 2,
    }
    (cfg_dir / "lra_listops_1660ti.yaml").write_text(
        yaml.safe_dump(listops_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    _run(
        [
            sys.executable,
            "scripts/prepare_lra_official.py",
            "--manifest",
            str(manifest),
            "--tasks",
            "listops",
            "--task-config-dir",
            str(cfg_dir),
            "--raw-root",
            str(raw_root),
            "--processed-root",
            str(processed_root),
            "--overwrite",
        ],
        cwd=repo,
    )

    _run(
        [
            sys.executable,
            "scripts/run_stage_b.py",
            "--tasks",
            "listops",
            "--variants",
            "base",
            "--seeds",
            "0",
            "--out",
            str(runs_out),
            "--config-dir",
            str(cfg_dir),
            "--dataset-manifest",
            str(manifest),
            "--data-source",
            "official_lra",
            "--device",
            "cpu",
            "--max-train-steps",
            "2",
        ],
        cwd=repo,
    )

    _run(
        [
            sys.executable,
            "scripts/summarize_runs.py",
            "--runs-dir",
            str(runs_out / "runs"),
            "--out",
            str(summary_out),
            "--data-source",
            "official_lra",
            "--expected-tasks",
            "listops",
            "--expected-variants",
            "base",
            "--expected-seeds",
            "0",
            "--min-runs-per-group",
            "1",
            "--strict-matrix",
            "--strict-fingerprint",
            "--strict-artifacts",
            "--strict-logging",
            "--strict-manifest",
            "--strict-single-task",
            "--no-plots",
        ],
        cwd=repo,
    )

    audit = json.loads((summary_out / "readiness_audit.json").read_text(encoding="utf-8"))
    assert audit["matrix"]["ok"] is True
    assert audit["fingerprint_consistency"]["ok"] is True
    assert audit["manifest_consistency"]["ok"] is True
    assert audit["artifact_integrity"]["ok"] is True
    assert audit["stage_b_logging"]["ok"] is True
