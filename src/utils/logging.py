from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch


class CsvLogger:
    def __init__(self, path: Path, fieldnames: list[str], append: bool = False):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append and self.path.exists() else "w"
        self._file = self.path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if mode == "w":
            self._writer.writeheader()

    def log(self, row: dict) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "CsvLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _safe_git_hash(cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def gather_system_info(cwd: Path | None = None) -> dict:
    cwd = cwd or Path.cwd()
    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "git_hash": _safe_git_hash(cwd),
    }

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        info["gpu_name"] = torch.cuda.get_device_name(device_index)
        props = torch.cuda.get_device_properties(device_index)
        info["gpu_total_vram_gb"] = round(props.total_memory / (1024**3), 3)
    else:
        info["gpu_name"] = None
        info["gpu_total_vram_gb"] = None

    return info


def make_run_dir(base_out: Path, task: str, variant: str, seed: int) -> Path:
    run_dir = base_out / "runs" / task / variant / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir
