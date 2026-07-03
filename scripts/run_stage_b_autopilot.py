from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_RUN_FILES = [
    "config.json",
    "system.json",
    "dataset_fingerprint.json",
    "metrics_train.csv",
    "metrics_eval.csv",
    "rng_state.pt",
    "run_summary.json",
    "checkpoints/latest.pt",
    "checkpoints/best.pt",
]


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _pause_requested(pause_file: Path | None) -> bool:
    return bool(pause_file is not None and pause_file.exists())


def _read_json_dict(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing"
    except Exception as exc:
        return None, f"parse_error: {exc}"

    if not isinstance(obj, dict):
        return None, "invalid_json_type"
    return obj, None


def _csv_has_rows(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            n = sum(1 for _ in f)
        return n >= 2
    except Exception:
        return False


def _is_run_complete(run_dir: Path, *, require_cuda: bool) -> bool:
    for rel in REQUIRED_RUN_FILES:
        if not (run_dir / rel).exists():
            return False

    run_summary, err = _read_json_dict(run_dir / "run_summary.json")
    if run_summary is None:
        return False

    try:
        steps = int(run_summary.get("global_step"))
    except Exception:
        return False
    if steps <= 0:
        return False

    if require_cuda and not str(run_summary.get("device", "")).lower().startswith("cuda"):
        return False

    if not _csv_has_rows(run_dir / "metrics_train.csv"):
        return False
    if not _csv_has_rows(run_dir / "metrics_eval.csv"):
        return False

    return True


def _count_completed(*, out_root: Path, tasks: list[str], variants: list[str], seeds: list[int], require_cuda: bool) -> int:
    done = 0
    for task in tasks:
        for variant in variants:
            for seed in seeds:
                run_dir = out_root / "runs" / task / variant / f"seed_{seed}"
                if _is_run_complete(run_dir, require_cuda=require_cuda):
                    done += 1
    return done


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autopilot Stage-B until full task x variant x seed matrix is complete.")
    p.add_argument("--tasks", default="listops,text,pathfinder")
    p.add_argument("--variants", default="base,fixed_delta,lr_only,lado,dtr,dtrl,transformer_lite,s4d_lite")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--out", default="output/official_stage_b_v3")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--task-config-pattern", default="")
    p.add_argument("--model-config", default="configs/model_b2s6_1660ti.yaml")
    p.add_argument("--method-config", default="configs/method_variants.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--require-cuda", action="store_true", default=False)
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--allow-tf32", action="store_true", default=False)
    p.add_argument("--cudnn-benchmark", action="store_true", default=False)
    p.add_argument("--torch-compile", action="store_true", default=False)
    p.add_argument(
        "--torch-compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
    )
    p.add_argument("--no-plots", action="store_true", default=False)
    p.add_argument("--max-train-steps", type=int, default=0)
    p.add_argument("--param-budget-tolerance", type=float, default=0.10)
    p.add_argument("--allow-param-budget-mismatch", action="store_true", default=False)
    p.add_argument("--data-source", choices=["official_lra", "synthetic"], default="official_lra")
    p.add_argument("--dataset-manifest", default="configs/datasets/lra_official_manifest.yaml")
    p.add_argument("--retry-failed", type=int, default=2)
    p.add_argument("--deterministic", action="store_true", default=False)
    p.add_argument("--deterministic-strict", action="store_true", default=False)
    p.add_argument("--pause-file", default="")
    p.add_argument("--chunk-hours", type=float, default=6.0)
    p.add_argument("--max-cycles", type=int, default=200)
    p.add_argument("--sleep-seconds", type=float, default=2.0)
    p.add_argument("--max-no-progress-cycles", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tasks = parse_csv_list(args.tasks)
    variants = parse_csv_list(args.variants)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    pause_file = Path(args.pause_file) if str(args.pause_file).strip() else (out_root / "PAUSE_REQUESTED")

    total = len(tasks) * len(variants) * len(seeds)
    hist_path = out_root / "stage_b_autopilot_history.jsonl"

    append_jsonl(
        hist_path,
        {
            "event": "autopilot_start",
            "utc": utc_now_iso(),
            "total_runs": total,
            "tasks": tasks,
            "variants": variants,
            "seeds": seeds,
            "out": str(out_root),
            "task_config_pattern": str(args.task_config_pattern),
            "chunk_hours": float(args.chunk_hours),
            "max_cycles": int(args.max_cycles),
            "max_no_progress_cycles": int(args.max_no_progress_cycles),
            "pause_file": str(pause_file),
            "require_cuda": bool(args.require_cuda),
            "amp": bool(args.amp),
            "amp_dtype": str(args.amp_dtype),
            "allow_tf32": bool(args.allow_tf32),
            "cudnn_benchmark": bool(args.cudnn_benchmark),
            "torch_compile": bool(args.torch_compile),
            "torch_compile_mode": str(args.torch_compile_mode),
            "no_plots": bool(args.no_plots),
        },
    )

    no_progress_cycles = 0
    for cycle in range(1, int(args.max_cycles) + 1):
        if _pause_requested(pause_file):
            done_now = _count_completed(
                out_root=out_root,
                tasks=tasks,
                variants=variants,
                seeds=seeds,
                require_cuda=bool(args.require_cuda),
            )
            append_jsonl(
                hist_path,
                {
                    "event": "autopilot_pause_requested",
                    "utc": utc_now_iso(),
                    "cycle": cycle,
                    "completed": done_now,
                    "total": total,
                    "pause_file": str(pause_file),
                },
            )
            print(f"Autopilot paused by marker {pause_file}: {done_now}/{total}")
            return

        done_before = _count_completed(
            out_root=out_root,
            tasks=tasks,
            variants=variants,
            seeds=seeds,
            require_cuda=bool(args.require_cuda),
        )
        if done_before >= total:
            append_jsonl(
                hist_path,
                {
                    "event": "autopilot_complete",
                    "utc": utc_now_iso(),
                    "cycle": cycle,
                    "completed": done_before,
                    "total": total,
                },
            )
            print(f"Autopilot complete: {done_before}/{total}")
            return

        cmd = [
            sys.executable,
            "scripts/run_stage_b.py",
            "--tasks",
            ",".join(tasks),
            "--variants",
            ",".join(variants),
            "--seeds",
            ",".join(str(s) for s in seeds),
            "--out",
            str(out_root),
            "--config-dir",
            args.config_dir,
            "--task-config-pattern",
            args.task_config_pattern,
            "--model-config",
            args.model_config,
            "--method-config",
            args.method_config,
            "--device",
            args.device,
            "--data-source",
            args.data_source,
            "--dataset-manifest",
            args.dataset_manifest,
            "--resume",
            "--skip-completed",
            "--retry-failed",
            str(int(args.retry_failed)),
            "--max-runtime-hours",
            str(float(args.chunk_hours)),
            "--pause-file",
            str(pause_file),
            "--param-budget-tolerance",
            str(float(args.param_budget_tolerance)),
        ]
        if bool(args.require_cuda):
            cmd.append("--require-cuda")
        if bool(args.amp):
            cmd.append("--amp")
            cmd.extend(["--amp-dtype", str(args.amp_dtype)])
        if bool(args.allow_tf32):
            cmd.append("--allow-tf32")
        if bool(args.cudnn_benchmark):
            cmd.append("--cudnn-benchmark")
        if bool(args.torch_compile):
            cmd.append("--torch-compile")
            cmd.extend(["--torch-compile-mode", str(args.torch_compile_mode)])
        if bool(args.no_plots):
            cmd.append("--no-plots")
        if bool(args.deterministic) or bool(args.deterministic_strict):
            cmd.append("--deterministic")
        if bool(args.deterministic_strict):
            cmd.append("--deterministic-strict")
        if int(args.max_train_steps) > 0:
            cmd.extend(["--max-train-steps", str(int(args.max_train_steps))])
        if bool(args.allow_param_budget_mismatch):
            cmd.append("--allow-param-budget-mismatch")

        cycle_start = time.time()
        append_jsonl(
            hist_path,
            {
                "event": "cycle_start",
                "utc": utc_now_iso(),
                "cycle": cycle,
                "done_before": done_before,
                "total": total,
                "cmd": cmd,
            },
        )

        proc = subprocess.run(cmd)

        done_after = _count_completed(
            out_root=out_root,
            tasks=tasks,
            variants=variants,
            seeds=seeds,
            require_cuda=bool(args.require_cuda),
        )
        gained = int(done_after - done_before)

        append_jsonl(
            hist_path,
            {
                "event": "cycle_end",
                "utc": utc_now_iso(),
                "cycle": cycle,
                "returncode": int(proc.returncode),
                "duration_sec": float(time.time() - cycle_start),
                "done_before": done_before,
                "done_after": done_after,
                "gained": gained,
                "remaining": int(total - done_after),
            },
        )

        if done_after >= total:
            append_jsonl(
                hist_path,
                {
                    "event": "autopilot_complete",
                    "utc": utc_now_iso(),
                    "cycle": cycle,
                    "completed": done_after,
                    "total": total,
                },
            )
            print(f"Autopilot complete: {done_after}/{total}")
            return

        if _pause_requested(pause_file):
            append_jsonl(
                hist_path,
                {
                    "event": "autopilot_pause_requested",
                    "utc": utc_now_iso(),
                    "cycle": cycle,
                    "completed": done_after,
                    "total": total,
                    "pause_file": str(pause_file),
                },
            )
            print(f"Autopilot paused by marker {pause_file}: {done_after}/{total}")
            return

        if gained <= 0:
            no_progress_cycles += 1
        else:
            no_progress_cycles = 0

        if no_progress_cycles >= int(args.max_no_progress_cycles):
            append_jsonl(
                hist_path,
                {
                    "event": "autopilot_stop_no_progress",
                    "utc": utc_now_iso(),
                    "cycle": cycle,
                    "done_after": done_after,
                    "total": total,
                    "no_progress_cycles": no_progress_cycles,
                },
            )
            raise SystemExit(
                f"Autopilot stopped: no progress for {no_progress_cycles} consecutive cycles "
                f"({done_after}/{total} completed)."
            )

        if float(args.sleep_seconds) > 0:
            time.sleep(float(args.sleep_seconds))

    append_jsonl(
        hist_path,
        {
            "event": "autopilot_stop_max_cycles",
            "utc": utc_now_iso(),
            "max_cycles": int(args.max_cycles),
            "total": total,
        },
    )
    raise SystemExit(f"Autopilot reached max cycles={int(args.max_cycles)} without full completion.")


if __name__ == "__main__":
    main()
