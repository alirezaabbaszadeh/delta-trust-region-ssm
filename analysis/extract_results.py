#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from common import load_json, sha256_file, write_json, write_text
from stats import bootstrap_ci_percentile, holm_adjust, wilcoxon_signed_rank_exact


TASKS = ["listops", "text", "pathfinder"]
CORE_VARIANTS = ["base", "fixed_delta", "dtr", "dtrl"]
ALL_VARIANTS = ["base", "fixed_delta", "lr_only", "lado", "dtr", "dtrl", "transformer_lite", "s4d_lite"]
SEEDS = [0, 1, 2, 3, 4]
COMPARE_VARIANTS = ["fixed_delta", "dtr", "dtrl"]


def _cell_id(task: str, variant: str, seed: int) -> str:
    return f"{task}/{variant}/seed_{seed}"


def _finite(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def summarize(values: list[float | None], *, n_boot: int, seed: int) -> dict[str, Any]:
    clean = _finite(values)
    if not clean:
        return {"values": values, "n": 0, "mean": None, "sd": None, "median": None, "min": None, "max": None, "ci95_boot": None}
    return {
        "values": values,
        "n": len(clean),
        "mean": statistics.mean(clean),
        "sd": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
        "ci95_boot": list(bootstrap_ci_percentile(clean, n_boot=n_boot, seed=seed)),
    }


def dedup_by_step_keep_last(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    info = {"raw_rows": int(len(df)), "dedup_applied": False}
    if "step" not in df.columns:
        info.update({"dedup_reason": "missing_step_column", "dedup_rows": int(len(df)), "unique_steps": None})
        return df.copy(), info
    dedup = df.drop_duplicates(subset=["step"], keep="last").copy()
    info.update(
        {
            "dedup_applied": len(dedup) != len(df),
            "dedup_reason": "step_duplicates" if len(dedup) != len(df) else "not_required",
            "dedup_rows": int(len(dedup)),
            "unique_steps": int(dedup["step"].nunique()),
        }
    )
    return dedup, info


def reconstruct_wall_time_sec(df: pd.DataFrame) -> tuple[float | None, list[float]]:
    if "time_sec" not in df.columns:
        return None, []
    values = [float(value) for value in df["time_sec"].dropna().to_list()]
    if not values:
        return None, []
    segments: list[float] = []
    previous = values[0]
    for value in values[1:]:
        if value < previous:
            segments.append(previous)
        previous = value
    segments.append(previous)
    return sum(segments), segments


def _quantiles(series: pd.Series) -> dict[str, float] | None:
    values = series.replace([float("inf"), -float("inf")], pd.NA).dropna().astype(float)
    if values.empty:
        return None
    return {
        "median": float(values.quantile(0.50)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def _compute_metrics(raw: pd.DataFrame, final_path: pd.DataFrame) -> dict[str, Any]:
    wall, segments = reconstruct_wall_time_sec(raw)
    tokens = None
    if "batch_tokens" in final_path.columns and final_path["batch_tokens"].notna().any():
        tokens = int(final_path["batch_tokens"].fillna(0).sum())
    throughput = float(tokens / wall) if tokens is not None and wall and wall > 0 else None

    def peak(column: str) -> float | None:
        return float(raw[column].dropna().max()) if column in raw.columns and raw[column].notna().any() else None

    return {
        "wall_time_sec": wall,
        "wall_time_segments_sec": segments,
        "total_final_path_tokens": tokens,
        "tokens_per_sec": throughput,
        "peak_gpu_mem_alloc_mb": peak("gpu_mem_alloc_mb"),
        "peak_gpu_mem_reserved_mb": peak("gpu_mem_reserved_mb"),
    }


def _mechanism_metrics(df: pd.DataFrame) -> dict[str, Any] | None:
    required = {"eps_delta", "drift_pre_max", "drift_post_max", "trust_region_scale"}
    if not required.issubset(df.columns) or not df["eps_delta"].notna().any():
        return None
    work = df.dropna(subset=list(required)).copy()
    if work.empty:
        return None
    eps = work["eps_delta"].astype(float)
    pre = work["drift_pre_max"].astype(float)
    post = work["drift_post_max"].astype(float)
    scale = work["trust_region_scale"].astype(float)
    tolerance = eps.abs() * 1e-6 + 1e-12
    active = scale < 1.0 - 1e-6
    pre_violation = pre > eps + tolerance
    post_violation = post > eps + tolerance
    conditional_success = pre_violation & ~post_violation
    active_reduction = (post[active] / pre[active].where(pre[active] != 0)).dropna()
    zero_scale_residual = post_violation & (scale <= 1e-12)
    active_scales = scale[active]
    return {
        "num_logged_steps": int(len(work)),
        "active_steps": int(active.sum()),
        "activation_rate": float(active.mean()),
        "pre_violation_steps": int(pre_violation.sum()),
        "pre_violation_rate": float(pre_violation.mean()),
        "post_violation_steps": int(post_violation.sum()),
        "post_violation_rate": float(post_violation.mean()),
        "conditional_projection_success_count": int(conditional_success.sum()),
        "conditional_projection_success_rate": float(conditional_success.sum() / pre_violation.sum()) if pre_violation.any() else None,
        "active_drift_post_over_pre": _quantiles(active_reduction),
        "drift_pre_over_epsilon": _quantiles(pre / eps),
        "drift_post_over_epsilon": _quantiles(post / eps),
        "accepted_scale_active": _quantiles(active_scales),
        "mean_trust_region_scale": float(scale.mean()),
        "zero_scale_residual_violation_count": int(zero_scale_residual.sum()),
        "zero_scale_residual_violation_rate": float(zero_scale_residual.sum() / pre_violation.sum()) if pre_violation.any() else None,
    }


def _extract_cell(run_dir: Path) -> dict[str, Any]:
    summary = load_json(run_dir / "run_summary.json")
    raw = pd.read_csv(run_dir / "metrics_train.csv")
    final_path, dedup = dedup_by_step_keep_last(raw)
    compute = _compute_metrics(raw, final_path)
    test = summary.get("test_metrics") or {}
    collapse = final_path["collapse"].astype(float) if "collapse" in final_path.columns else pd.Series(dtype=float)
    spike = final_path["spike"].astype(float) if "spike" in final_path.columns else pd.Series(dtype=float)
    streak = final_path["collapse_streak"].astype(float) if "collapse_streak" in final_path.columns else pd.Series(dtype=float)
    return {
        "test_acc": float(test["acc"]) if test.get("acc") is not None else None,
        "test_loss": float(test["loss"]) if test.get("loss") is not None else None,
        "num_trainable_params": summary.get("num_trainable_params"),
        "num_steps": int(len(final_path)),
        "collapse_count": int(collapse.sum()) if not collapse.empty else 0,
        "collapse_rate": float(collapse.mean()) if not collapse.empty else None,
        "spike_count": int(spike.sum()) if not spike.empty else 0,
        "spike_rate": float(spike.mean()) if not spike.empty else None,
        "collapse_streak": {
            "max": int(streak.max()) if not streak.empty else 0,
            "positive_step_count": int((streak > 0).sum()) if not streak.empty else 0,
            "quantiles": _quantiles(streak),
        },
        "compute": compute,
        "compute_reconstruction": {**dedup, "source": "metrics_train.csv", "wall_time_policy": "sum_monotonic_segment_endpoints"},
        "mechanism": _mechanism_metrics(final_path),
    }


def _paired_comparison(task: str, variant: str, cells: dict[str, Any], *, n_boot: int, seed: int) -> dict[str, Any]:
    base = [float(cells[_cell_id(task, "base", s)]["test_acc"]) for s in SEEDS]
    other = [float(cells[_cell_id(task, variant, s)]["test_acc"]) for s in SEEDS]
    diffs = [value - reference for value, reference in zip(other, base)]
    wil = wilcoxon_signed_rank_exact(diffs)
    return {
        "task": task,
        "variant": variant,
        "reference": "base",
        "seeds": SEEDS,
        "paired_differences": diffs,
        "difference_summary": summarize(diffs, n_boot=n_boot, seed=seed),
        "direction_counts": {
            "positive": sum(value > 0 for value in diffs),
            "zero": sum(value == 0 for value in diffs),
            "negative": sum(value < 0 for value in diffs),
        },
        "wilcoxon_exact_two_sided": {
            "effective_nonzero_n": wil.n,
            "w_plus": wil.w_plus,
            "p_raw": wil.p_two_sided,
            "p_holm_global_9": None,
        },
        "rank_biserial": wil.rank_biserial,
    }


def _aggregate_mechanism(cells: dict[str, Any], *, n_boot: int, seed_start: int) -> tuple[dict[str, Any], int]:
    metrics = [
        "activation_rate",
        "active_steps",
        "pre_violation_rate",
        "post_violation_rate",
        "conditional_projection_success_rate",
        "mean_trust_region_scale",
        "zero_scale_residual_violation_rate",
    ]
    output: dict[str, Any] = {}
    cursor = seed_start
    for task in TASKS:
        output[task] = {}
        for variant in ["dtr", "dtrl"]:
            run_rows = [cells[_cell_id(task, variant, seed)]["mechanism"] for seed in SEEDS]
            aggregate: dict[str, Any] = {"runs": run_rows}
            for metric in metrics:
                aggregate[metric] = summarize([row.get(metric) if row else None for row in run_rows], n_boot=n_boot, seed=cursor)
                cursor += 1
            for family, key in [
                ("active_drift_post_over_pre", "median"),
                ("drift_pre_over_epsilon", "p99"),
                ("drift_post_over_epsilon", "p99"),
                ("accepted_scale_active", "median"),
            ]:
                aggregate[f"{family}_{key}"] = summarize(
                    [((row.get(family) or {}).get(key) if row else None) for row in run_rows],
                    n_boot=n_boot,
                    seed=cursor,
                )
                cursor += 1
            output[task][variant] = aggregate
    return output, cursor


def _aggregate_compute(cells: dict[str, Any], *, n_boot: int, seed_start: int) -> tuple[dict[str, Any], int]:
    output: dict[str, Any] = {"primary_seeds": [1, 2, 3, 4], "seed0_sensitivity_separate": True, "tasks": {}}
    cursor = seed_start
    for task in TASKS:
        output["tasks"][task] = {}
        for variant in CORE_VARIANTS:
            runs = [cells[_cell_id(task, variant, seed)]["compute"] for seed in [1, 2, 3, 4]]
            output["tasks"][task][variant] = {
                metric: summarize([run.get(metric) for run in runs], n_boot=n_boot, seed=cursor + index)
                for index, metric in enumerate(["wall_time_sec", "tokens_per_sec", "peak_gpu_mem_alloc_mb", "peak_gpu_mem_reserved_mb"])
            }
            cursor += 4
        for variant in ["fixed_delta", "dtr", "dtrl"]:
            wall_overhead: list[float | None] = []
            throughput_change: list[float | None] = []
            for seed in [1, 2, 3, 4]:
                base = cells[_cell_id(task, "base", seed)]["compute"]
                other = cells[_cell_id(task, variant, seed)]["compute"]
                wall_overhead.append(other["wall_time_sec"] / base["wall_time_sec"] - 1 if base["wall_time_sec"] else None)
                throughput_change.append(other["tokens_per_sec"] / base["tokens_per_sec"] - 1 if base["tokens_per_sec"] else None)
            output["tasks"][task][variant]["paired_wall_time_overhead_fraction"] = summarize(wall_overhead, n_boot=n_boot, seed=cursor)
            output["tasks"][task][variant]["paired_throughput_change_fraction"] = summarize(throughput_change, n_boot=n_boot, seed=cursor + 1)
            cursor += 2
    return output, cursor


def _format_number(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the paper's single numerical source of truth.")
    parser.add_argument("--snapshot", type=Path, default=Path("evidence_snapshot"))
    parser.add_argument("--out-results", type=Path, default=Path("results/results.json"))
    parser.add_argument("--out-macros", type=Path, default=Path("results/results_macros.tex"))
    parser.add_argument("--n-boot", type=int, default=20_000)
    parser.add_argument("--analysis-seed", type=int, default=20260703)
    args = parser.parse_args()

    manifest = load_json(args.snapshot / "manifest.json")
    cells: dict[str, Any] = {}
    for row in manifest["cells"]:
        key = _cell_id(row["task"], row["variant"], int(row["seed"]))
        run_dir = args.snapshot / "runs" / row["task"] / row["variant"] / f"seed_{row['seed']}"
        metrics = _extract_cell(run_dir)
        cells[key] = {**metrics, "task": row["task"], "variant": row["variant"], "seed": row["seed"], "status": "complete", "role": row["role"]}
    for row in manifest["oom_cells"]:
        key = _cell_id(row["task"], row["variant"], int(row["seed"]))
        cells[key] = {"task": row["task"], "variant": row["variant"], "seed": row["seed"], "status": "oom", "role": row["role"], "evidence": row}

    cursor = args.analysis_seed
    task_groups: dict[str, Any] = {}
    for task in TASKS:
        variants: dict[str, Any] = {}
        for variant in CORE_VARIANTS:
            values = [cells[_cell_id(task, variant, seed)]["test_acc"] for seed in SEEDS]
            variants[variant] = summarize(values, n_boot=args.n_boot, seed=cursor)
            cursor += 1
        task_groups[f"{task}_core_5seed"] = {"task": task, "metric": "test_accuracy", "seeds": SEEDS, "variants": variants}

    comparisons: list[dict[str, Any]] = []
    for task in TASKS:
        for variant in COMPARE_VARIANTS:
            comparisons.append(_paired_comparison(task, variant, cells, n_boot=args.n_boot, seed=cursor))
            cursor += 1
    adjusted = holm_adjust([row["wilcoxon_exact_two_sided"]["p_raw"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["wilcoxon_exact_two_sided"]["p_holm_global_9"] = value

    mechanism, cursor = _aggregate_mechanism(cells, n_boot=args.n_boot, seed_start=cursor)
    compute, cursor = _aggregate_compute(cells, n_boot=args.n_boot, seed_start=cursor)
    mechanism_gate_runs = []
    for task in TASKS:
        for variant in ["dtr", "dtrl"]:
            for seed, run in zip(SEEDS, mechanism[task][variant]["runs"]):
                if run and run["active_steps"] > 0:
                    mechanism_gate_runs.append(
                        {
                            "task": task,
                            "variant": variant,
                            "seed": seed,
                            "active_steps": run["active_steps"],
                            "conditional_success_rate": run["conditional_projection_success_rate"],
                            "median_post_over_pre": (run["active_drift_post_over_pre"] or {}).get("median"),
                        }
                    )
    success_rates = [row["conditional_success_rate"] for row in mechanism_gate_runs if row["conditional_success_rate"] is not None]
    success_majority_fraction = (
        sum(rate > 0.5 for rate in success_rates) / len(success_rates) if success_rates else 0.0
    )
    mechanism_gate = {
        "active_runs": mechanism_gate_runs,
        "multiple_seeds": len({row["seed"] for row in mechanism_gate_runs}) >= 2,
        "all_active_runs_reduce_median_drift": all((row["median_post_over_pre"] or float("inf")) < 1 for row in mechanism_gate_runs),
        "median_conditional_success_exceeds_failure": statistics.median(success_rates) > 0.5 if success_rates else False,
        "active_run_success_majority_fraction": success_majority_fraction,
        "success_not_driven_by_one_run": success_majority_fraction >= 0.75,
    }
    mechanism_gate["passed"] = all(
        [
            mechanism_gate["multiple_seeds"],
            mechanism_gate["all_active_runs_reduce_median_drift"],
            mechanism_gate["median_conditional_success_exceeds_failure"],
            mechanism_gate["success_not_driven_by_one_run"],
        ]
    )

    representative_protocol = {
        task: next(
            row["protocol"]
            for row in manifest["cells"]
            if row["role"] == "core" and row["task"] == task and row["variant"] == "base" and row["seed"] == 0
        )
        for task in TASKS
    }
    results = {
        "version": 3,
        "evidence_snapshot": {
            "content_sha256": manifest["content_sha256"],
            "manifest_sha256": sha256_file(args.snapshot / "manifest.json"),
            "sha256sums_sha256": sha256_file(args.snapshot / "SHA256SUMS.txt"),
            "data_quality_report_sha256": sha256_file(args.snapshot / "data_quality_report.json"),
            "protocol_parity_report_sha256": sha256_file(args.snapshot / "protocol_parity_report.json"),
        },
        "protocol": {
            "tasks": TASKS,
            "core_variants": CORE_VARIANTS,
            "seeds": SEEDS,
            "bootstrap": {"method": "percentile", "resamples": args.n_boot, "analysis_seed": args.analysis_seed, "confidence": 0.95},
            "multiple_comparisons": {"method": "Holm", "family": "3 tasks x 3 paired comparisons versus base", "count": 9},
            "representative_by_task": representative_protocol,
        },
        "cells": dict(sorted(cells.items())),
        "task_groups": task_groups,
        "comparisons": comparisons,
        "mechanism": {"tasks": mechanism, "claim_gate": mechanism_gate},
        "compute": compute,
        "exploratory": {"seed0_variants": ALL_VARIANTS, "oom_cells": [[row["task"], row["variant"], row["seed"]] for row in manifest["oom_cells"]]},
        "claim_inputs": {
            "mechanism_gate_passed": mechanism_gate["passed"],
            "accuracy_advantage_established": False,
            "core_complete_cells": sum(row["role"] == "core" for row in manifest["cells"]),
        },
    }
    write_json(args.out_results, results)

    data_quality = load_json(args.snapshot / "data_quality_report.json")
    macro_lines = [
        "%% AUTO-GENERATED: tools/extract_results.py",
        f"%% evidence_snapshot_content_sha256={manifest['content_sha256']}",
        "\\newcommand{\\EvidenceSnapshotIDRaw}{" + manifest["content_sha256"] + "}",
        "\\DeclareRobustCommand{\\EvidenceSnapshotID}{\\texttt{\\seqsplit{" + manifest["content_sha256"] + "}}}",
        "\\newcommand{\\EvidenceSnapshotShortID}{\\texttt{" + manifest["content_sha256"][:12] + "}}",
        "\\newcommand{\\CoreRunCount}{60}",
    ]
    for task in TASKS:
        label = {"listops": "ListOps", "text": "Text", "pathfinder": "Pathfinder"}[task]
        for variant in CORE_VARIANTS:
            name = {"base": "Base", "fixed_delta": "FixedDelta", "dtr": "DTR", "dtrl": "DTRL"}[variant]
            stats = task_groups[f"{task}_core_5seed"]["variants"][variant]
            macro_lines.append(f"\\newcommand{{\\{label}{name}AccMean}}{{\\num{{{_format_number(stats['mean'])}}}}}")
    for task in TASKS:
        task_label = {"listops": "ListOps", "text": "Text", "pathfinder": "Pathfinder"}[task]
        for variant in ["dtr", "dtrl"]:
            variant_label = variant.upper()
            mech = mechanism[task][variant]
            macro_lines.append(
                f"\\newcommand{{\\{task_label}{variant_label}ActivationPct}}{{\\num{{{_format_number(100 * mech['activation_rate']['mean'], 2)}}}\\%}}"
            )
            macro_lines.append(
                f"\\newcommand{{\\{task_label}{variant_label}SuccessPct}}{{\\num{{{_format_number(100 * mech['conditional_projection_success_rate']['mean'], 2)}}}\\%}}"
            )
            overhead = compute["tasks"][task][variant]["paired_wall_time_overhead_fraction"]["mean"]
            macro_lines.append(
                f"\\newcommand{{\\{task_label}{variant_label}TimeOverheadPct}}{{\\num{{{_format_number(100 * overhead, 1)}}}\\%}}"
            )
    dtr_overheads = [compute["tasks"][task]["dtr"]["paired_wall_time_overhead_fraction"]["mean"] for task in TASKS]
    dtrl_overheads = [compute["tasks"][task]["dtrl"]["paired_wall_time_overhead_fraction"]["mean"] for task in TASKS]
    macro_lines.extend(
        [
            f"\\newcommand{{\\DTRTimeOverheadRange}}{{\\num{{{100 * min(dtr_overheads):.0f}}}--\\num{{{100 * max(dtr_overheads):.0f}}}\\%}}",
            f"\\newcommand{{\\DTRLTimeOverheadRange}}{{\\num{{{100 * min(dtrl_overheads):.0f}}}--\\num{{{100 * max(dtrl_overheads):.0f}}}\\%}}",
            f"\\newcommand{{\\FivePairWilcoxonMinimumP}}{{\\num{{{comparisons[0]['wilcoxon_exact_two_sided']['p_raw']:.4f}}}}}",
        ]
    )
    for pair, macro in [("train_val", "TextTrainValOverlapPct"), ("train_test", "TextTrainTestOverlapPct"), ("val_test", "TextValTestOverlapPct")]:
        rate = data_quality["tasks"]["text"]["cross_split_overlap"][pair]["overlap_ratio_min_split"]
        macro_lines.append(f"\\newcommand{{\\{macro}}}{{\\num{{{100 * rate:.3f}}}\\%}}")
    write_text(args.out_macros, "\n".join(macro_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
