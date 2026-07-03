from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be mapping: {path}")
    return data


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(value)))


def _round_batch(value: int, *, quantum: int = 8) -> int:
    q = max(1, int(quantum))
    v = max(1, int(value))
    return int(max(q, int(round(v / q) * q)))


def _build_train_cmd(
    *,
    task: str,
    variant: str,
    seed: int,
    config_path: Path,
    model_config: str,
    method_config: str,
    out_dir: Path,
    device: str,
    data_source: str,
    dataset_manifest: str,
    max_train_steps: int,
    require_cuda: bool,
    amp: bool,
    amp_dtype: str,
    allow_tf32: bool,
    cudnn_benchmark: bool,
    torch_compile: bool,
    torch_compile_mode: str,
    deterministic: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/train_lra_light.py",
        "--task",
        task,
        "--variant",
        variant,
        "--seed",
        str(int(seed)),
        "--config",
        str(config_path),
        "--model-config",
        str(model_config),
        "--method-config",
        str(method_config),
        "--out",
        str(out_dir),
        "--device",
        str(device),
        "--data-source",
        str(data_source),
        "--dataset-manifest",
        str(dataset_manifest),
        "--max-train-steps",
        str(int(max_train_steps)),
        "--param-budget-tolerance",
        "0.1",
        "--no-plots",
    ]
    if require_cuda:
        cmd.append("--require-cuda")
    if amp:
        cmd.append("--amp")
        cmd.extend(["--amp-dtype", str(amp_dtype)])
    if allow_tf32:
        cmd.append("--allow-tf32")
    if cudnn_benchmark:
        cmd.append("--cudnn-benchmark")
    if torch_compile:
        cmd.append("--torch-compile")
        cmd.extend(["--torch-compile-mode", str(torch_compile_mode)])
    if deterministic:
        cmd.append("--deterministic")
    return cmd


def _trial_run(
    *,
    task: str,
    variant: str,
    seed: int,
    base_cfg: dict[str, Any],
    batch_size: int,
    model_config: str,
    method_config: str,
    out_root: Path,
    device: str,
    data_source: str,
    dataset_manifest: str,
    max_train_steps: int,
    require_cuda: bool,
    amp: bool,
    amp_dtype: str,
    allow_tf32: bool,
    cudnn_benchmark: bool,
    torch_compile: bool,
    torch_compile_mode: str,
    deterministic: bool,
    timeout_seconds: int,
    num_workers_override: int,
    prefetch_factor_override: int,
) -> dict[str, Any]:
    trial_dir = out_root / task / variant / f"bs_{int(batch_size)}"
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(base_cfg)
    cfg["batch_size"] = int(batch_size)
    if num_workers_override >= 0:
        cfg["num_workers"] = int(num_workers_override)
    if prefetch_factor_override > 0:
        cfg["prefetch_factor"] = int(prefetch_factor_override)

    cfg_path = trial_dir / "task_config.yaml"
    _save_yaml(cfg_path, cfg)

    log_path = trial_dir / "trial.log"
    cmd = _build_train_cmd(
        task=task,
        variant=variant,
        seed=seed,
        config_path=cfg_path,
        model_config=model_config,
        method_config=method_config,
        out_dir=trial_dir,
        device=device,
        data_source=data_source,
        dataset_manifest=dataset_manifest,
        max_train_steps=max_train_steps,
        require_cuda=require_cuda,
        amp=amp,
        amp_dtype=amp_dtype,
        allow_tf32=allow_tf32,
        cudnn_benchmark=cudnn_benchmark,
        torch_compile=torch_compile,
        torch_compile_mode=torch_compile_mode,
        deterministic=deterministic,
    )

    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write("# batch_tune_trial\n")
        lf.write(f"# utc_start: {_utc_now()}\n")
        lf.write(f"# task: {task}\n")
        lf.write(f"# variant: {variant}\n")
        lf.write(f"# batch_size: {batch_size}\n")
        lf.write(f"# cmd: {' '.join(cmd)}\n\n")
        lf.flush()

        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=max(1, int(timeout_seconds)),
            )
            rc = int(proc.returncode)
            timed_out = False
        except subprocess.TimeoutExpired:
            rc = 124
            timed_out = True

    duration = time.time() - t0

    run_dir = trial_dir / "runs" / task / variant / f"seed_{int(seed)}"
    run_summary_path = run_dir / "run_summary.json"
    ok_artifact = False
    device_value = ""
    global_step = 0
    peak_gpu_mem_alloc_mb = 0.0
    peak_gpu_mem_reserved_mb = 0.0
    if run_summary_path.exists():
        try:
            run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
            if isinstance(run_summary, dict):
                device_value = str(run_summary.get("device", ""))
                global_step = int(run_summary.get("global_step", 0) or 0)
                peak_gpu_mem_alloc_mb = float(run_summary.get("peak_gpu_mem_alloc_mb", 0.0) or 0.0)
                peak_gpu_mem_reserved_mb = float(run_summary.get("peak_gpu_mem_reserved_mb", 0.0) or 0.0)
                ok_artifact = global_step > 0
        except Exception:
            ok_artifact = False

    log_tail = ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        log_tail = "\n".join(lines[-60:])
    except Exception:
        pass

    oom = "out of memory" in log_tail.lower()
    success = (rc == 0) and ok_artifact and ((not require_cuda) or device_value.lower().startswith("cuda"))

    return {
        "task": task,
        "variant": variant,
        "batch_size": int(batch_size),
        "success": bool(success),
        "returncode": int(rc),
        "timed_out": bool(timed_out),
        "oom": bool(oom),
        "duration_sec": float(duration),
        "global_step": int(global_step),
        "device": device_value,
        "peak_gpu_mem_alloc_mb": float(peak_gpu_mem_alloc_mb),
        "peak_gpu_mem_reserved_mb": float(peak_gpu_mem_reserved_mb),
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "log_tail": log_tail,
    }


def _run_trial_with_args(
    *,
    task: str,
    variant: str,
    base_cfg: dict[str, Any],
    batch_size: int,
    args: argparse.Namespace,
    out_root: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    return _trial_run(
        task=task,
        variant=variant,
        seed=args.seed,
        base_cfg=base_cfg,
        batch_size=int(batch_size),
        model_config=args.model_config,
        method_config=args.method_config,
        out_root=out_root,
        device=args.device,
        data_source=args.data_source,
        dataset_manifest=args.dataset_manifest,
        max_train_steps=args.max_train_steps,
        require_cuda=args.require_cuda,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        allow_tf32=args.allow_tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        torch_compile=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode,
        deterministic=args.deterministic,
        timeout_seconds=max(1, int(timeout_seconds)),
        num_workers_override=args.num_workers_override,
        prefetch_factor_override=args.prefetch_factor_override,
    )


def _search_best_batch(
    *,
    task: str,
    base_cfg: dict[str, Any],
    args: argparse.Namespace,
    out_root: Path,
) -> dict[str, Any]:
    min_batch = int(args.min_batch)
    max_batch = int(args.max_batch)
    growth = max(2, int(args.growth_factor))

    trials: list[dict[str, Any]] = []

    first = _run_trial_with_args(
        task=task,
        variant=args.variant,
        base_cfg=base_cfg,
        batch_size=min_batch,
        args=args,
        out_root=out_root,
        timeout_seconds=int(args.timeout_seconds),
    )
    trials.append(first)

    if not first["success"]:
        return {
            "task": task,
            "variant": args.variant,
            "ok": False,
            "reason": f"minimum batch failed: bs={min_batch}",
            "best_batch": 0,
            "trials": trials,
        }

    best = min_batch
    hi_fail: int | None = None
    probe = min_batch

    while True:
        nxt = min(max_batch, int(probe * growth))
        if nxt <= probe:
            break

        trial = _run_trial_with_args(
            task=task,
            variant=args.variant,
            base_cfg=base_cfg,
            batch_size=nxt,
            args=args,
            out_root=out_root,
            timeout_seconds=int(args.timeout_seconds),
        )
        trials.append(trial)
        if trial["success"]:
            best = nxt
            probe = nxt
            if nxt >= max_batch:
                break
            continue

        hi_fail = nxt
        break

    if hi_fail is not None and hi_fail - best > 1:
        lo = best
        hi = hi_fail
        while hi - lo > 1:
            mid = (lo + hi) // 2
            trial = _run_trial_with_args(
                task=task,
                variant=args.variant,
                base_cfg=base_cfg,
                batch_size=mid,
                args=args,
                out_root=out_root,
                timeout_seconds=int(args.timeout_seconds),
            )
            trials.append(trial)
            if trial["success"]:
                lo = mid
                best = mid
            else:
                hi = mid

    return {
        "task": task,
        "variant": args.variant,
        "ok": True,
        "best_batch": int(best),
        "trials": trials,
    }


def _build_descending_probes(
    *,
    start_batch: int,
    min_batch: int,
    max_batch: int,
    divisor: float,
    max_probes: int,
) -> list[int]:
    div = max(1.1, float(divisor))
    probes: list[int] = []
    seen: set[int] = set()
    cur = _clamp_int(_round_batch(int(start_batch)), min_batch, max_batch)

    while len(probes) < max(1, int(max_probes)):
        if cur not in seen:
            probes.append(cur)
            seen.add(cur)
        if cur <= min_batch:
            break

        nxt = _round_batch(int(math.floor(cur / div)))
        if nxt >= cur:
            nxt = cur - 8
        cur = _clamp_int(nxt, min_batch, max_batch)

    if min_batch not in seen:
        probes.append(min_batch)

    out: list[int] = []
    seen2: set[int] = set()
    for bs in probes:
        if bs not in seen2:
            out.append(bs)
            seen2.add(bs)
    return out


def _search_best_batch_descend(
    *,
    task: str,
    variant: str,
    base_cfg: dict[str, Any],
    args: argparse.Namespace,
    out_root: Path,
    predicted_start: int,
) -> dict[str, Any]:
    min_batch = int(args.min_batch)
    max_batch = int(args.max_batch)

    probes = _build_descending_probes(
        start_batch=int(predicted_start),
        min_batch=min_batch,
        max_batch=max_batch,
        divisor=float(args.descend_divisor),
        max_probes=int(args.descend_max_probes),
    )

    trials: list[dict[str, Any]] = []
    first_success: int | None = None
    first_fail_above: int | None = None
    fast_timeout = max(1, min(int(args.timeout_seconds), int(args.fast_timeout_seconds)))

    for bs in probes:
        trial = _run_trial_with_args(
            task=task,
            variant=variant,
            base_cfg=base_cfg,
            batch_size=bs,
            args=args,
            out_root=out_root,
            timeout_seconds=fast_timeout,
        )
        trials.append(trial)

        if trial["success"]:
            first_success = int(bs)
            break
        first_fail_above = int(bs)

    if first_success is None:
        return {
            "task": task,
            "variant": variant,
            "ok": False,
            "reason": "no feasible batch found in descending probes",
            "predicted_start": int(predicted_start),
            "probes": probes,
            "best_batch": 0,
            "trials": trials,
        }

    best = int(first_success)

    refine_steps = max(0, int(args.descend_refine_steps))
    if refine_steps > 0 and first_fail_above is not None and int(first_fail_above) > int(best):
        lo = int(best)
        hi = int(first_fail_above)
        for _ in range(refine_steps):
            if hi - lo <= 8:
                break
            mid = _round_batch((lo + hi) // 2)
            if mid <= lo or mid >= hi:
                break

            trial = _run_trial_with_args(
                task=task,
                variant=variant,
                base_cfg=base_cfg,
                batch_size=mid,
                args=args,
                out_root=out_root,
                timeout_seconds=fast_timeout,
            )
            trials.append(trial)
            if trial["success"]:
                lo = int(mid)
                best = int(mid)
            else:
                hi = int(mid)

    return {
        "task": task,
        "variant": variant,
        "ok": True,
        "predicted_start": int(predicted_start),
        "probes": probes,
        "best_batch": int(best),
        "trials": trials,
    }


def _load_task_cfgs(tasks: list[str], args: argparse.Namespace) -> tuple[dict[str, tuple[Path, dict[str, Any]]], dict[str, str]]:
    cfgs: dict[str, tuple[Path, dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    for task in tasks:
        src_cfg_path = (ROOT / str(args.task_config_pattern).format(task=task)).resolve()
        if not src_cfg_path.exists():
            errors[task] = f"missing task config: {src_cfg_path}"
            continue
        cfgs[task] = (src_cfg_path, _load_yaml(src_cfg_path))
    return cfgs, errors


def _write_tuned_cfg(
    *,
    task: str,
    cfg: dict[str, Any],
    tuned_batch: int,
    args: argparse.Namespace,
) -> Path:
    tuned_cfg = dict(cfg)
    tuned_cfg["batch_size"] = int(tuned_batch)
    if int(args.num_workers_override) >= 0:
        tuned_cfg["num_workers"] = int(args.num_workers_override)
    if int(args.prefetch_factor_override) > 0:
        tuned_cfg["prefetch_factor"] = int(args.prefetch_factor_override)

    dst_cfg_path = (ROOT / str(args.out_config_pattern).format(task=task)).resolve()
    _save_yaml(dst_cfg_path, tuned_cfg)
    return dst_cfg_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-tune max stable batch size per task/GPU profile.")
    p.add_argument("--tasks", default="listops,text,pathfinder")
    p.add_argument("--task-config-pattern", default="configs/lra_{task}_4090_fast.yaml")
    p.add_argument("--out-config-pattern", default="configs/lra_{task}_4090_tuned.yaml")

    p.add_argument("--variant", default="base")
    p.add_argument("--seed", type=int, default=0)
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
    p.add_argument("--deterministic", action="store_true", default=False)

    p.add_argument("--data-source", choices=["official_lra", "synthetic"], default="official_lra")
    p.add_argument("--dataset-manifest", default="configs/datasets/lra_official_manifest.yaml")

    p.add_argument("--max-train-steps", type=int, default=2)
    p.add_argument("--timeout-seconds", type=int, default=900)

    p.add_argument("--min-batch", type=int, default=16)
    p.add_argument("--max-batch", type=int, default=1024)
    p.add_argument("--growth-factor", type=int, default=2)

    p.add_argument("--num-workers-override", type=int, default=-1)
    p.add_argument("--prefetch-factor-override", type=int, default=0)

    p.add_argument(
        "--strategy",
        choices=["per_task_exponential", "global_heavy_descend"],
        default="per_task_exponential",
    )
    p.add_argument("--heavy-task", default="pathfinder")
    p.add_argument("--heavy-variant", default="transformer_lite")
    p.add_argument("--predicted-start-batch", type=int, default=0)
    p.add_argument("--predicted-start-multiplier", type=float, default=8.0)
    p.add_argument("--descend-divisor", type=float, default=2.0)
    p.add_argument("--descend-max-probes", type=int, default=6)
    p.add_argument("--descend-refine-steps", type=int, default=1)
    p.add_argument("--propagate-ratio", type=float, default=0.90)
    p.add_argument("--fast-timeout-seconds", type=int, default=240)

    p.add_argument("--trial-out", default="output/batch_tuning")
    p.add_argument("--out-report", default="output/batch_tuning/report.json")
    p.add_argument("--strict", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = _parse_csv(args.tasks)
    if not tasks:
        raise SystemExit("No tasks selected")

    out_root = (ROOT / args.trial_out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "utc": _utc_now(),
        "ok": True,
        "strategy": str(args.strategy),
        "tasks": tasks,
        "task_config_pattern": args.task_config_pattern,
        "out_config_pattern": args.out_config_pattern,
        "variant": args.variant,
        "seed": int(args.seed),
        "max_train_steps": int(args.max_train_steps),
        "timeout_seconds": int(args.timeout_seconds),
        "search": {
            "min_batch": int(args.min_batch),
            "max_batch": int(args.max_batch),
            "growth_factor": int(args.growth_factor),
            "predicted_start_batch": int(args.predicted_start_batch),
            "predicted_start_multiplier": float(args.predicted_start_multiplier),
            "descend_divisor": float(args.descend_divisor),
            "descend_max_probes": int(args.descend_max_probes),
            "descend_refine_steps": int(args.descend_refine_steps),
            "propagate_ratio": float(args.propagate_ratio),
            "fast_timeout_seconds": int(args.fast_timeout_seconds),
        },
        "results": {},
        "written_configs": {},
    }

    task_cfgs, cfg_errors = _load_task_cfgs(tasks, args)
    for task, msg in cfg_errors.items():
        report["results"][task] = {
            "task": task,
            "ok": False,
            "reason": msg,
            "best_batch": 0,
            "trials": [],
        }
        report["ok"] = False

    if args.strategy == "per_task_exponential":
        for task in tasks:
            if task not in task_cfgs:
                continue
            _, base_cfg = task_cfgs[task]
            result = _search_best_batch(task=task, base_cfg=base_cfg, args=args, out_root=out_root)
            report["results"][task] = result

            if result.get("ok") and int(result.get("best_batch", 0)) > 0:
                dst_cfg_path = _write_tuned_cfg(
                    task=task,
                    cfg=base_cfg,
                    tuned_batch=int(result["best_batch"]),
                    args=args,
                )
                report["written_configs"][task] = str(dst_cfg_path)
            else:
                report["ok"] = False
    else:
        if not task_cfgs:
            report["ok"] = False
        else:
            heavy_task = str(args.heavy_task).strip().lower()
            if heavy_task not in task_cfgs:
                heavy_task = max(
                    task_cfgs.keys(),
                    key=lambda t: int(task_cfgs[t][1].get("seq_len", 0)),
                )

            _, heavy_cfg = task_cfgs[heavy_task]
            predicted_start = int(args.predicted_start_batch)
            if predicted_start <= 0:
                base_bs = int(heavy_cfg.get("batch_size", args.min_batch) or args.min_batch)
                predicted_start = int(round(base_bs * float(args.predicted_start_multiplier)))
            predicted_start = _clamp_int(
                _round_batch(predicted_start),
                int(args.min_batch),
                int(args.max_batch),
            )

            heavy_result = _search_best_batch_descend(
                task=heavy_task,
                variant=str(args.heavy_variant),
                base_cfg=heavy_cfg,
                args=args,
                out_root=out_root,
                predicted_start=predicted_start,
            )

            report["heavy"] = {
                "task": heavy_task,
                "variant": str(args.heavy_variant),
                "predicted_start": int(predicted_start),
                "result_ok": bool(heavy_result.get("ok")),
                "best_batch": int(heavy_result.get("best_batch", 0) or 0),
            }

            if not bool(heavy_result.get("ok")):
                report["results"][heavy_task] = heavy_result
                report["ok"] = False
            else:
                heavy_best = int(heavy_result.get("best_batch", 0) or 0)
                propagated_batch = _clamp_int(
                    _round_batch(int(math.floor(heavy_best * float(args.propagate_ratio)))),
                    int(args.min_batch),
                    int(args.max_batch),
                )
                if propagated_batch <= 0:
                    propagated_batch = int(args.min_batch)

                report["global_propagated_batch"] = int(propagated_batch)
                report["results"][heavy_task] = heavy_result

                for task in tasks:
                    if task not in task_cfgs:
                        continue
                    _, cfg = task_cfgs[task]
                    task_result = {
                        "task": task,
                        "ok": True,
                        "mode": "propagated_from_heavy",
                        "best_batch": int(propagated_batch),
                        "heavy_task": heavy_task,
                        "heavy_variant": str(args.heavy_variant),
                        "heavy_best_batch": int(heavy_best),
                        "propagate_ratio": float(args.propagate_ratio),
                        "trials": heavy_result.get("trials", []) if task == heavy_task else [],
                    }
                    report["results"][task] = task_result

                    dst_cfg_path = _write_tuned_cfg(
                        task=task,
                        cfg=cfg,
                        tuned_batch=int(propagated_batch),
                        args=args,
                    )
                    report["written_configs"][task] = str(dst_cfg_path)

    out_report = (ROOT / args.out_report).resolve()
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": bool(report["ok"]),
                "out_report": str(out_report),
                "strategy": str(args.strategy),
                "written_configs": report["written_configs"],
                "global_propagated_batch": report.get("global_propagated_batch"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if bool(args.strict) and not bool(report["ok"]):
        raise SystemExit("Batch tuning failed for at least one task.")


if __name__ == "__main__":
    main()
