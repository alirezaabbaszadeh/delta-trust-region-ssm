from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _run(cmd: list[str], *, cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return int(proc.returncode), proc.stdout, proc.stderr


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fail-fast prelaunch gate for journal-grade costly runs.")
    p.add_argument("--manifest", default="configs/datasets/lra_official_manifest.yaml")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--task-config-pattern", default="configs/lra_{task}_1660ti.yaml")
    p.add_argument("--tasks", default="listops,text,pathfinder")
    p.add_argument("--variants", default="base,fixed_delta,lr_only,lado,dtr,dtrl,transformer_lite,s4d_lite")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--method-config", default="configs/method_variants.yaml")
    p.add_argument("--processed-root", default="data/processed/lra_official")

    p.add_argument("--runs-out", default="output/official_stage_b_cloud")
    p.add_argument("--summary-out", default="output/summary_official_cloud")
    p.add_argument("--out", default="output/prelaunch_gate.json")

    p.add_argument("--require-official-source", action="store_true", default=False)
    p.add_argument("--require-cuda", action="store_true", default=False)
    p.add_argument("--allow-nonempty-runs-out", action="store_true", default=False)

    p.add_argument("--runtime-smoke", action="store_true", default=False)
    p.add_argument("--smoke-out", default="output/prelaunch_smoke")
    p.add_argument("--smoke-variants", default="base,dtrl,transformer_lite,s4d_lite")
    p.add_argument("--smoke-seeds", default="0")
    p.add_argument("--smoke-max-train-steps", type=int, default=2)
    p.add_argument("--smoke-no-plots", action="store_true", default=True)
    p.add_argument("--smoke-amp", action="store_true", default=True)
    p.add_argument("--strict", action="store_true", default=True)
    return p.parse_args()


def _count_existing_runs(runs_out: Path) -> int:
    runs_dir = runs_out / "runs"
    if not runs_dir.exists():
        return 0
    n = 0
    for p in runs_dir.glob("*/*/seed_*"):
        if p.is_dir():
            n += 1
    return n


def main() -> None:
    args = parse_args()

    report: dict[str, Any] = {
        "utc": _utc_now(),
        "ok": True,
        "checks": {},
        "commands": {},
    }

    runs_out = ROOT / args.runs_out
    summary_out = ROOT / args.summary_out

    # 1) Filesystem safety gate (avoid silent mixed experiments unless explicitly allowed).
    existing_runs = _count_existing_runs(runs_out)
    fs_ok = bool(args.allow_nonempty_runs_out or existing_runs == 0)
    report["checks"]["runs_out_clean"] = {
        "ok": fs_ok,
        "runs_out": str(runs_out),
        "existing_run_dirs": int(existing_runs),
        "allow_nonempty_runs_out": bool(args.allow_nonempty_runs_out),
    }
    if not fs_ok:
        report["ok"] = False

    # 2) Disk availability gate.
    total, used, free = shutil.disk_usage(str(ROOT))
    min_free_gb = 30.0
    disk_ok = float(free / (1024**3)) >= min_free_gb
    report["checks"]["disk_free"] = {
        "ok": disk_ok,
        "free_gb": float(free / (1024**3)),
        "min_required_gb": min_free_gb,
    }
    if not disk_ok:
        report["ok"] = False

    # 3) CUDA availability gate (when required).
    cuda_ok = True
    if bool(args.require_cuda):
        cuda_ok = bool(torch.cuda.is_available())
    report["checks"]["cuda_available"] = {
        "ok": cuda_ok,
        "require_cuda": bool(args.require_cuda),
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }
    if not cuda_ok:
        report["ok"] = False

    # 4) Static journal audit gate.
    audit_cmd = [
        sys.executable,
        "scripts/audit_journal_readiness.py",
        "--manifest",
        args.manifest,
        "--config-dir",
        args.config_dir,
        "--task-config-pattern",
        args.task_config_pattern,
        "--tasks",
        args.tasks,
        "--method-config",
        args.method_config,
        "--processed-root",
        args.processed_root,
        "--out",
        "output/readiness_audit.prelaunch.json",
        "--strict",
    ]
    if bool(args.require_official_source):
        audit_cmd.append("--require-official-source")
    if bool(args.require_cuda):
        audit_cmd.append("--require-cuda")

    rc, out, err = _run(audit_cmd, cwd=ROOT)
    report["commands"]["audit"] = {"cmd": audit_cmd, "returncode": rc}
    report["checks"]["audit_journal_readiness"] = {
        "ok": rc == 0,
        "stdout_tail": out[-2000:],
        "stderr_tail": err[-2000:],
    }
    if rc != 0:
        report["ok"] = False

    # 5) Optional runtime smoke gate (to catch server/runtime/env errors before full spend).
    if bool(args.runtime_smoke):
        smoke_out = str(args.smoke_out)
        smoke_run_cmd = [
            sys.executable,
            "scripts/run_stage_b.py",
            "--tasks",
            args.tasks,
            "--variants",
            args.smoke_variants,
            "--seeds",
            args.smoke_seeds,
            "--out",
            smoke_out,
            "--config-dir",
            args.config_dir,
            "--model-config",
            "configs/model_b2s6_1660ti.yaml",
            "--method-config",
            args.method_config,
            "--device",
            "cuda" if bool(args.require_cuda) else "auto",
            "--data-source",
            "official_lra",
            "--dataset-manifest",
            args.manifest,
            "--max-train-steps",
            str(int(args.smoke_max_train_steps)),
            "--retry-failed",
            "0",
            "--skip-completed",
            "--deterministic",
            "--no-skip-completed",
        ]
        if bool(args.require_cuda):
            smoke_run_cmd.append("--require-cuda")
        if bool(args.smoke_amp):
            smoke_run_cmd.append("--amp")
        if bool(args.smoke_no_plots):
            smoke_run_cmd.append("--no-plots")

        rc_s, out_s, err_s = _run(smoke_run_cmd, cwd=ROOT)
        report["commands"]["runtime_smoke_run"] = {"cmd": smoke_run_cmd, "returncode": rc_s}

        smoke_sum_cmd = [
            sys.executable,
            "scripts/summarize_runs.py",
            "--runs-dir",
            str((ROOT / smoke_out / "runs")),
            "--out",
            str(ROOT / smoke_out / "summary"),
            "--data-source",
            "official_lra",
            "--expected-tasks",
            args.tasks,
            "--expected-variants",
            args.smoke_variants,
            "--expected-seeds",
            args.smoke_seeds,
            "--min-runs-per-group",
            "1",
            "--strict-matrix",
            "--strict-fingerprint",
            "--strict-artifacts",
            "--strict-logging",
            "--strict-manifest",
            "--strict-single-task",
            "--no-plots",
        ]
        if bool(args.require_cuda):
            smoke_sum_cmd.append("--require-cuda")

        rc_ss, out_ss, err_ss = _run(smoke_sum_cmd, cwd=ROOT)
        report["commands"]["runtime_smoke_summarize"] = {"cmd": smoke_sum_cmd, "returncode": rc_ss}

        smoke_ok = (rc_s == 0) and (rc_ss == 0)
        report["checks"]["runtime_smoke"] = {
            "ok": smoke_ok,
            "run_returncode": rc_s,
            "summarize_returncode": rc_ss,
            "run_stdout_tail": out_s[-1500:],
            "run_stderr_tail": err_s[-1500:],
            "summarize_stdout_tail": out_ss[-1500:],
            "summarize_stderr_tail": err_ss[-1500:],
            "smoke_out": str(ROOT / smoke_out),
        }
        if not smoke_ok:
            report["ok"] = False

    # 6) Contract snapshot.
    report["contract"] = {
        "tasks": _parse_csv(args.tasks),
        "variants": _parse_csv(args.variants),
        "seeds": [int(x) for x in _parse_csv(args.seeds)],
        "runs_out": str(runs_out),
        "summary_out": str(summary_out),
        "task_config_pattern": args.task_config_pattern,
        "manifest": args.manifest,
        "require_cuda": bool(args.require_cuda),
        "require_official_source": bool(args.require_official_source),
    }

    out_path = ROOT / args.out
    _write_json(out_path, report)

    print(
        json.dumps(
            {
                "ok": bool(report["ok"]),
                "out": str(out_path),
                "checks": {k: bool(v.get("ok", False)) for k, v in report["checks"].items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if bool(args.strict) and not bool(report["ok"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
