from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


TRAIN_COLUMNS = [
    "epoch",
    "step",
    "loss",
    "acc",
    "grad_delta_l2",
    "lr",
    "lr_delta",
    "spike",
    "collapse",
    "collapse_streak",
    "eps_delta",
    "drift_pre_max",
    "drift_pre_p99",
    "drift_post_max",
    "drift_post_p99",
    "trust_region_scale",
    "time_sec",
    "batch_tokens",
    "tokens_per_sec",
    "gpu_mem_alloc_mb",
    "gpu_mem_reserved_mb",
]

EVAL_COLUMNS = ["epoch", "split", "loss", "acc"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_csv(path: Path, columns: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)


def _create_stage_b_status(stage_root: Path, log_path: Path) -> None:
    status_file = stage_root / "stage_b_runs_20260218_000000.jsonl"
    records = [
        {"event": "stage_b_start", "utc": "2026-02-18T00:00:00Z"},
        {
            "event": "run_attempt",
            "task": "listops",
            "variant": "base",
            "seed": 0,
            "attempt": 1,
            "max_attempts": 1,
            "status": "ok",
            "returncode": 0,
            "log_path": str(log_path),
        },
        {"event": "stage_b_end", "utc": "2026-02-18T00:01:00Z", "failed_runs": 0},
    ]
    status_file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    (stage_root / "stage_b_runs.latest.txt").write_text(str(status_file), encoding="utf-8")


def _create_complete_run(stage_root: Path, *, training_regime: str = "single_task", device: str = "cuda") -> Path:
    run_dir = stage_root / "runs" / "listops" / "base" / "seed_0"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    cfg = {
        "task": "listops",
        "variant": "base",
        "seed": 0,
        "data_source": "official_lra",
        "device": device,
        "dataset_manifest": "configs/datasets/lra_official_manifest.yaml",
        "training_regime": training_regime,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    fingerprint = {
        "fingerprint": "abc123",
        "manifest_sha256": "deadbeef",
    }
    (run_dir / "dataset_fingerprint.json").write_text(json.dumps(fingerprint), encoding="utf-8")

    system = {"cuda_available": device.startswith("cuda")}
    (run_dir / "system.json").write_text(json.dumps(system), encoding="utf-8")

    summary = {
        "task": "listops",
        "variant": "base",
        "seed": 0,
        "device": device,
        "global_step": 1,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    (run_dir / "rng_state.pt").write_bytes(b"rng")
    (run_dir / "checkpoints" / "latest.pt").write_bytes(b"latest")
    (run_dir / "checkpoints" / "best.pt").write_bytes(b"best")

    _write_csv(
        run_dir / "metrics_train.csv",
        TRAIN_COLUMNS,
        {k: 1 for k in TRAIN_COLUMNS},
    )
    _write_csv(
        run_dir / "metrics_eval.csv",
        EVAL_COLUMNS,
        {"epoch": 1, "split": "test", "loss": 0.1, "acc": 0.9},
    )

    return run_dir


def _run_summarize(stage_root: Path, out_dir: Path, extra: list[str]) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        "scripts/summarize_runs.py",
        "--runs-dir",
        str(stage_root / "runs"),
        "--out",
        str(out_dir),
        "--expected-tasks",
        "listops",
        "--expected-variants",
        "base",
        "--expected-seeds",
        "0",
        "--min-runs-per-group",
        "1",
        "--data-source",
        "official_lra",
        "--no-plots",
    ] + extra
    return subprocess.run(cmd, cwd=_repo_root(), capture_output=True, text=True)


def test_summarize_strict_audit_passes_for_complete_run(tmp_path: Path) -> None:
    stage_root = tmp_path / "stage_b"
    log_path = stage_root / "logs" / "listops" / "base" / "seed_0_attempt_1.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("ok\n", encoding="utf-8")

    _create_complete_run(stage_root, training_regime="single_task", device="cuda")
    _create_stage_b_status(stage_root, log_path)

    out_dir = tmp_path / "summary"
    proc = _run_summarize(
        stage_root,
        out_dir,
        [
            "--strict-matrix",
            "--strict-fingerprint",
            "--strict-artifacts",
            "--strict-logging",
            "--strict-manifest",
            "--strict-single-task",
            "--require-cuda",
        ],
    )
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout

    audit = json.loads((out_dir / "readiness_audit.json").read_text(encoding="utf-8"))
    assert audit["matrix"]["ok"] is True
    assert audit["fingerprint_consistency"]["ok"] is True
    assert audit["manifest_consistency"]["ok"] is True
    assert audit["artifact_integrity"]["ok"] is True
    assert audit["stage_b_logging"]["ok"] is True


def test_summarize_strict_artifacts_fails_on_policy_and_missing_files(tmp_path: Path) -> None:
    stage_root = tmp_path / "stage_b"
    log_path = stage_root / "logs" / "listops" / "base" / "seed_0_attempt_1.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("ok\n", encoding="utf-8")

    run_dir = _create_complete_run(stage_root, training_regime="multi_task", device="cpu")
    (run_dir / "metrics_eval.csv").unlink()  # force incomplete artifact
    _create_stage_b_status(stage_root, log_path)

    out_dir = tmp_path / "summary"
    proc = _run_summarize(
        stage_root,
        out_dir,
        ["--strict-artifacts", "--strict-single-task", "--require-cuda"],
    )
    assert proc.returncode != 0

    audit = json.loads((out_dir / "readiness_audit.json").read_text(encoding="utf-8"))
    assert audit["artifact_integrity"]["ok"] is False
    issues = audit["artifact_integrity"]["issue_runs"][0]["issues"]
    joined = "\n".join(issues)
    assert "missing required files" in joined
    assert "training_regime must be single_task" in joined
    assert "config.device must be cuda" in joined
