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
            # Header + at least one data row.
            n = sum(1 for _ in f)
        return n >= 2
    except Exception:
        return False


def _is_run_complete(run_dir: Path, *, require_cuda: bool) -> tuple[bool, str]:
    missing = [rel for rel in REQUIRED_RUN_FILES if not (run_dir / rel).exists()]
    if missing:
        return False, f"missing_files={missing}"

    run_summary, err = _read_json_dict(run_dir / "run_summary.json")
    if run_summary is None:
        return False, f"run_summary.{err}"

    steps_raw = run_summary.get("global_step")
    try:
        steps = int(steps_raw)
    except Exception:
        return False, f"run_summary.global_step_invalid={steps_raw!r}"
    if steps <= 0:
        return False, f"run_summary.global_step_nonpositive={steps}"

    if require_cuda:
        dev = str(run_summary.get("device", "")).lower()
        if not dev.startswith("cuda"):
            return False, f"run_summary.device_non_cuda={dev!r}"

    if not _csv_has_rows(run_dir / "metrics_train.csv"):
        return False, "metrics_train_empty"
    if not _csv_has_rows(run_dir / "metrics_eval.csv"):
        return False, "metrics_eval_empty"

    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage B sweeps over task x variant x seed.")
    parser.add_argument("--tasks", default="listops,text,pathfinder")
    parser.add_argument(
        "--variants",
        default="base,fixed_delta,lr_only,lado,dtr,dtrl,transformer_lite,s4d_lite",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out", default="output/official_stage_b")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--task-config-pattern", default="")
    parser.add_argument("--model-config", default="configs/model_b2s6_1660ti.yaml")
    parser.add_argument("--method-config", default="configs/method_variants.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--allow-tf32", action="store_true", default=False)
    parser.add_argument("--cudnn-benchmark", action="store_true", default=False)
    parser.add_argument("--torch-compile", action="store_true", default=False)
    parser.add_argument(
        "--torch-compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
    )
    parser.add_argument("--no-plots", action="store_true", default=False)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--param-budget-tolerance", type=float, default=0.10)
    parser.add_argument("--allow-param-budget-mismatch", action="store_true", default=False)
    parser.add_argument("--data-source", choices=["official_lra", "synthetic"], default="official_lra")
    parser.add_argument("--dataset-manifest", default="configs/datasets/lra_official_manifest.yaml")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--retry-failed", type=int, default=0)
    parser.add_argument("--max-runtime-hours", type=float, default=0.0)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--deterministic-strict", action="store_true", default=False)
    parser.add_argument("--logs-dir", default="")
    parser.add_argument("--status-file", default="")
    parser.add_argument("--pause-file", default="")
    parser.add_argument("--skip-completed", dest="skip_completed", action="store_true", default=True)
    parser.add_argument("--no-skip-completed", dest="skip_completed", action="store_false")
    args = parser.parse_args()

    tasks = parse_csv_list(args.tasks)
    variants = parse_csv_list(args.variants)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]

    out_root = Path(args.out)
    logs_dir = Path(args.logs_dir) if args.logs_dir else out_root / "logs"
    pause_file = Path(args.pause_file) if str(args.pause_file).strip() else (out_root / "PAUSE_REQUESTED")
    if args.status_file:
        status_file = Path(args.status_file)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        status_file = out_root / f"stage_b_runs_{stamp}.jsonl"

    status_file = status_file.resolve()

    latest_ptr = out_root / "stage_b_runs.latest.txt"
    latest_ptr.parent.mkdir(parents=True, exist_ok=True)
    latest_ptr.write_text(str(status_file), encoding="utf-8")

    script = Path(__file__).resolve().parent / "train_lra_light.py"
    config_dir = Path(args.config_dir)

    total = len(tasks) * len(variants) * len(seeds)
    idx = 0
    failures: list[tuple[str, str, int, str]] = []
    start_time = time.time()

    run_header = {
        "event": "stage_b_start",
        "utc": utc_now_iso(),
        "total_runs": total,
        "tasks": tasks,
        "variants": variants,
        "seeds": seeds,
        "out": str(out_root),
        "logs_dir": str(logs_dir),
        "status_file": str(status_file),
        "data_source": args.data_source,
        "dataset_manifest": args.dataset_manifest,
        "task_config_pattern": args.task_config_pattern,
        "device": args.device,
        "require_cuda": bool(args.require_cuda),
        "amp": bool(args.amp),
        "amp_dtype": str(args.amp_dtype),
        "allow_tf32": bool(args.allow_tf32),
        "cudnn_benchmark": bool(args.cudnn_benchmark),
        "torch_compile": bool(args.torch_compile),
        "torch_compile_mode": str(args.torch_compile_mode),
        "no_plots": bool(args.no_plots),
        "skip_completed": bool(args.skip_completed),
        "pause_file": str(pause_file),
    }
    append_jsonl(status_file, run_header)

    stop_requested = False
    stop_reason = ""
    for task in tasks:
        if stop_requested:
            break
        pattern = str(args.task_config_pattern).strip()
        if pattern:
            if "{task}" in pattern:
                task_cfg = Path(pattern.format(task=task))
            else:
                pattern_path = Path(pattern)
                if pattern_path.is_dir():
                    task_cfg = pattern_path / f"lra_{task}_1660ti.yaml"
                else:
                    task_cfg = pattern_path
        else:
            task_cfg = config_dir / f"lra_{task}_1660ti.yaml"

        if not task_cfg.exists():
            fallback = config_dir / f"lra_{task}_1660ti.yaml"
            if fallback.exists() and fallback != task_cfg:
                task_cfg = fallback
        if not task_cfg.exists():
            raise SystemExit(f"Missing task config: {task_cfg}")

        for variant in variants:
            if stop_requested:
                break

            for seed in seeds:
                if _pause_requested(pause_file):
                    print(f"Pause requested via {pause_file}. Stopping after current completed runs.")
                    stop_requested = True
                    stop_reason = "pause_requested"
                    break

                elapsed_hours = (time.time() - start_time) / 3600.0
                if args.max_runtime_hours > 0 and elapsed_hours > args.max_runtime_hours:
                    print(f"Reached --max-runtime-hours={args.max_runtime_hours}. Stopping early.")
                    stop_requested = True
                    stop_reason = "max_runtime_hours"
                    break

                run_dir = out_root / "runs" / task / variant / f"seed_{seed}"
                if bool(args.skip_completed):
                    complete, reason = _is_run_complete(run_dir, require_cuda=bool(args.require_cuda))
                    if complete:
                        idx += 1
                        print(f"[{idx}/{total}] Skip completed task={task} variant={variant} seed={seed}")
                        append_jsonl(
                            status_file,
                            {
                                "event": "run_skip",
                                "utc": utc_now_iso(),
                                "task": task,
                                "variant": variant,
                                "seed": seed,
                                "reason": reason,
                                "run_dir": str(run_dir),
                            },
                        )
                        continue

                idx += 1
                cmd = [
                    sys.executable,
                    str(script),
                    "--task",
                    task,
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
                    "--config",
                    str(task_cfg),
                    "--model-config",
                    args.model_config,
                    "--method-config",
                    args.method_config,
                    "--out",
                    args.out,
                    "--device",
                    args.device,
                    "--data-source",
                    args.data_source,
                    "--dataset-manifest",
                    args.dataset_manifest,
                ]
                resume_ckpt_path = run_dir / "checkpoints" / "latest.pt"
                resume_enabled = bool(args.resume and resume_ckpt_path.exists())

                if args.amp:
                    cmd.append("--amp")
                    cmd.extend(["--amp-dtype", str(args.amp_dtype)])
                if args.allow_tf32:
                    cmd.append("--allow-tf32")
                if args.cudnn_benchmark:
                    cmd.append("--cudnn-benchmark")
                if args.torch_compile:
                    cmd.append("--torch-compile")
                    cmd.extend(["--torch-compile-mode", str(args.torch_compile_mode)])
                if args.no_plots:
                    cmd.append("--no-plots")
                if args.require_cuda:
                    cmd.append("--require-cuda")
                if resume_enabled:
                    cmd.append("--resume")
                if args.deterministic or args.deterministic_strict:
                    cmd.append("--deterministic")
                if args.deterministic_strict:
                    cmd.append("--deterministic-strict")
                if args.max_train_steps > 0:
                    cmd.extend(["--max-train-steps", str(args.max_train_steps)])
                cmd.extend(["--param-budget-tolerance", str(args.param_budget_tolerance)])
                if args.allow_param_budget_mismatch:
                    cmd.append("--allow-param-budget-mismatch")

                print(f"[{idx}/{total}] Running task={task} variant={variant} seed={seed}")

                attempts = 0
                max_attempts = 1 + max(0, int(args.retry_failed))
                while attempts < max_attempts:
                    if _pause_requested(pause_file):
                        print(f"Pause requested via {pause_file}. Not launching new attempts.")
                        stop_requested = True
                        stop_reason = "pause_requested"
                        break

                    attempts += 1
                    log_path = logs_dir / task / variant / f"seed_{seed}_attempt_{attempts}.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)

                    t0 = time.time()
                    start_iso = utc_now_iso()
                    with log_path.open("w", encoding="utf-8") as lf:
                        lf.write("# stage_b_run\n")
                        lf.write(f"# start_utc: {start_iso}\n")
                        lf.write(f"# task: {task}\n")
                        lf.write(f"# variant: {variant}\n")
                        lf.write(f"# seed: {seed}\n")
                        lf.write(f"# attempt: {attempts}/{max_attempts}\n")
                        lf.write(f"# resume_enabled: {resume_enabled}\n")
                        lf.write(f"# resume_ckpt_path: {resume_ckpt_path}\n")
                        lf.write(f"# cmd: {' '.join(cmd)}\n\n")
                        lf.flush()

                        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

                    end_iso = utc_now_iso()
                    duration = time.time() - t0
                    ok = proc.returncode == 0
                    record = {
                        "event": "run_attempt",
                        "start_utc": start_iso,
                        "end_utc": end_iso,
                        "duration_sec": duration,
                        "task": task,
                        "variant": variant,
                        "seed": seed,
                        "attempt": attempts,
                        "max_attempts": max_attempts,
                        "resume_enabled": resume_enabled,
                        "resume_ckpt_path": str(resume_ckpt_path),
                        "status": "ok" if ok else "fail",
                        "returncode": int(proc.returncode),
                        "log_path": str(log_path),
                        "cmd": cmd,
                    }
                    append_jsonl(status_file, record)

                    if ok:
                        print(f"  OK task={task} variant={variant} seed={seed} attempt={attempts} log={log_path}")
                        break

                    if _pause_requested(pause_file):
                        print(
                            f"Pause requested via {pause_file}. "
                            f"Stopping after interrupted attempt task={task} variant={variant} seed={seed}."
                        )
                        stop_requested = True
                        stop_reason = "pause_requested"
                        break

                    if attempts >= max_attempts:
                        failures.append((task, variant, seed, f"returncode={proc.returncode}"))
                        print(f"FAILED task={task} variant={variant} seed={seed} attempts={attempts} log={log_path}")
                    else:
                        print(f"Retry {attempts}/{max_attempts - 1} for task={task} variant={variant} seed={seed}")

                if stop_requested:
                    break

    footer = {
        "event": "stage_b_end",
        "utc": utc_now_iso(),
        "failed_runs": len(failures),
        "stopped_early": stop_requested,
        "stop_reason": stop_reason,
        "elapsed_hours": (time.time() - start_time) / 3600.0,
    }
    append_jsonl(status_file, footer)

    if failures:
        fail_path = out_root / "stage_b_failures.txt"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        with fail_path.open("w", encoding="utf-8") as f:
            for task, variant, seed, message in failures:
                f.write(f"task={task} variant={variant} seed={seed} error={message}\n")
        if stop_requested and stop_reason == "pause_requested":
            print(f"Stage B paused with pending failures logged at: {fail_path}")
        else:
            print(f"Stage B finished with failures. See: {fail_path}")
            raise SystemExit(1)

    print("Stage B sweep complete.")


if __name__ == "__main__":
    main()
