from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.statistics import bootstrap_mean_ci, holm_bonferroni_correction, wilcoxon_signed_rank


DEFAULT_TASKS = "listops,text,pathfinder"
DEFAULT_VARIANTS = "base,fixed_delta,lr_only,lado,dtr,dtrl,transformer_lite,s4d_lite"
DEFAULT_SEEDS = "0,1,2,3,4"
METRICS = ["val_acc", "test_acc", "spike_rate", "collapse_rate", "drift_post_max_mean", "time_sec"]
REQUIRED_RUN_FILES = [
    "config.json",
    "dataset_fingerprint.json",
    "metrics_train.csv",
    "metrics_eval.csv",
    "system.json",
    "rng_state.pt",
    "run_summary.json",
    "checkpoints/latest.pt",
    "checkpoints/best.pt",
]
REQUIRED_TRAIN_COLUMNS = [
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
REQUIRED_EVAL_COLUMNS = ["epoch", "split", "loss", "acc"]


def parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage B runs into CSV/plots/LaTeX table.")
    parser.add_argument("--runs-dir", default="output/official_stage_b/runs")
    parser.add_argument("--out", default="output/summary_official")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)

    parser.add_argument("--data-source", choices=["official_lra", "synthetic", "all"], default="official_lra")
    parser.add_argument("--expected-tasks", default=DEFAULT_TASKS)
    parser.add_argument("--expected-variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--expected-seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--min-runs-per-group", type=int, default=5)
    parser.add_argument("--strict-matrix", action="store_true", default=False)
    parser.add_argument("--strict-fingerprint", action="store_true", default=False)
    parser.add_argument("--strict-artifacts", action="store_true", default=False)
    parser.add_argument("--strict-logging", action="store_true", default=False)
    parser.add_argument("--strict-single-task", action="store_true", default=False)
    parser.add_argument("--strict-manifest", action="store_true", default=False)
    parser.add_argument("--require-cuda", action="store_true", default=False)
    parser.add_argument("--plot", dest="plot", action="store_true", default=True)
    parser.add_argument("--no-plots", dest="plot", action="store_false")
    return parser.parse_args()


def _hash_json(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_run_config(seed_dir: Path) -> dict[str, Any]:
    path = seed_dir / "config.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _read_json_dict(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, "missing"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, f"parse_error: {exc}"
    if not isinstance(raw, dict):
        return {}, "not_a_json_object"
    return raw, None


def _read_fingerprint(seed_dir: Path) -> tuple[str, dict[str, Any]]:
    path = seed_dir / "dataset_fingerprint.json"
    if not path.exists():
        return "", {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "", {}
    if not isinstance(raw, dict):
        return "", {}

    fp = raw.get("fingerprint")
    if isinstance(fp, str) and fp:
        return fp, raw
    return _hash_json(raw), raw


def collect_runs(runs_dir: Path, *, data_source_filter: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for seed_dir in runs_dir.glob("*/*/seed_*"):
        if not seed_dir.is_dir():
            continue

        try:
            task = seed_dir.parents[1].name
            variant = seed_dir.parents[0].name
            seed = int(seed_dir.name.replace("seed_", ""))
        except Exception:
            continue

        cfg = _read_run_config(seed_dir)
        run_source = str(cfg.get("data_source", ""))
        if data_source_filter != "all" and run_source != data_source_filter:
            continue

        eval_path = seed_dir / "metrics_eval.csv"
        train_path = seed_dir / "metrics_train.csv"
        if not eval_path.exists() or not train_path.exists():
            continue

        eval_df = pd.read_csv(eval_path)
        train_df = pd.read_csv(train_path)
        if eval_df.empty or train_df.empty:
            continue

        val_df = eval_df[eval_df["split"] == "val"]
        test_df = eval_df[eval_df["split"] == "test"]
        val_last = val_df.sort_values("epoch").tail(1)
        test_last = test_df.sort_values("epoch").tail(1)

        drift_mean = None
        if "drift_post_max" in train_df.columns and train_df["drift_post_max"].notna().any():
            drift_mean = float(train_df["drift_post_max"].dropna().mean())

        fingerprint, fingerprint_obj = _read_fingerprint(seed_dir)
        rows.append(
            {
                "task": task,
                "variant": variant,
                "seed": seed,
                "run_dir": str(seed_dir),
                "data_source": run_source,
                "device": cfg.get("device", ""),
                "training_regime": cfg.get("training_regime", ""),
                "deterministic": cfg.get("deterministic", None),
                "deterministic_strict": cfg.get("deterministic_strict", None),
                "dataset_manifest": cfg.get("dataset_manifest", ""),
                "dataset_fingerprint": fingerprint,
                "dataset_manifest_sha256": fingerprint_obj.get("manifest_sha256", ""),
                "val_loss": float(val_last["loss"].iloc[0]) if not val_last.empty else None,
                "val_acc": float(val_last["acc"].iloc[0]) if not val_last.empty else None,
                "test_loss": float(test_last["loss"].iloc[0]) if not test_last.empty else None,
                "test_acc": float(test_last["acc"].iloc[0]) if not test_last.empty else None,
                "spike_rate": float(train_df["spike"].mean()) if "spike" in train_df else None,
                "collapse_rate": float(train_df["collapse"].mean()) if "collapse" in train_df else None,
                "drift_post_max_mean": drift_mean,
                "time_sec": float(train_df["time_sec"].max()) if "time_sec" in train_df else None,
                "steps": int(train_df["step"].max() + 1) if "step" in train_df else len(train_df),
            }
        )

    return pd.DataFrame(rows)


def validate_matrix(
    df: pd.DataFrame,
    *,
    expected_tasks: list[str],
    expected_variants: list[str],
    expected_seeds: list[int],
    min_runs_per_group: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "expected": {
            "tasks": expected_tasks,
            "variants": expected_variants,
            "seeds": expected_seeds,
            "min_runs_per_group": int(min_runs_per_group),
        },
        "counts": {},
        "missing_pairs": [],
        "missing_seeds": [],
        "low_run_groups": [],
        "ok": True,
    }

    if df.empty:
        report["ok"] = False
        report["missing_pairs"] = [f"{t}::{v}" for t in expected_tasks for v in expected_variants]
        return report

    grouped = df.groupby(["task", "variant"])
    for (task, variant), sub in grouped:
        seeds = sorted(int(x) for x in set(sub["seed"].tolist()))
        report["counts"][f"{task}::{variant}"] = {
            "n_runs": int(len(sub)),
            "n_unique_seeds": int(len(seeds)),
            "seeds": seeds,
        }

    for task in expected_tasks:
        for variant in expected_variants:
            key = f"{task}::{variant}"
            info = report["counts"].get(key)
            if info is None:
                report["missing_pairs"].append(key)
                report["ok"] = False
                continue

            if int(info["n_unique_seeds"]) < int(min_runs_per_group):
                report["low_run_groups"].append(
                    {
                        "task": task,
                        "variant": variant,
                        "n_unique_seeds": int(info["n_unique_seeds"]),
                        "required": int(min_runs_per_group),
                    }
                )
                report["ok"] = False

            have_seeds = set(int(s) for s in info["seeds"])
            for seed in expected_seeds:
                if int(seed) not in have_seeds:
                    report["missing_seeds"].append({"task": task, "variant": variant, "seed": int(seed)})
                    report["ok"] = False

    return report


def validate_fingerprints(df: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {"tasks": {}, "ok": True}
    if df.empty:
        report["ok"] = False
        return report

    for task, sub in df.groupby("task"):
        fps = sorted({str(x) for x in sub["dataset_fingerprint"].dropna().tolist() if str(x)})
        report["tasks"][task] = {
            "n_unique_fingerprints": int(len(fps)),
            "fingerprints": fps,
        }
        if len(fps) != 1:
            report["ok"] = False

    return report


def validate_manifest_consistency(df: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {"tasks": {}, "ok": True}
    if df.empty:
        report["ok"] = False
        return report

    for task, sub in df.groupby("task"):
        manifests = sorted({str(x) for x in sub["dataset_manifest"].dropna().tolist() if str(x)})
        sha_values = sorted({str(x) for x in sub["dataset_manifest_sha256"].dropna().tolist() if str(x)})
        report["tasks"][task] = {
            "n_unique_manifest_paths": int(len(manifests)),
            "manifest_paths": manifests,
            "n_unique_manifest_sha256": int(len(sha_values)),
            "manifest_sha256": sha_values,
        }
        if len(manifests) != 1 or len(sha_values) != 1:
            report["ok"] = False

    return report


def _missing_columns(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    return [c for c in required_cols if c not in df.columns]


def audit_run_artifacts(
    runs_dir: Path,
    *,
    data_source_filter: str,
    require_cuda: bool,
    strict_single_task: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "ok": True,
        "n_run_dirs": 0,
        "n_complete_run_dirs": 0,
        "n_issue_runs": 0,
        "issue_runs": [],
    }

    for seed_dir in sorted(runs_dir.glob("*/*/seed_*")):
        if not seed_dir.is_dir():
            continue

        try:
            task = seed_dir.parents[1].name
            variant = seed_dir.parents[0].name
            seed = int(seed_dir.name.replace("seed_", ""))
        except Exception:
            continue

        run_id = f"{task}/{variant}/seed_{seed}"
        report["n_run_dirs"] += 1
        issues: list[str] = []

        missing_files = [rel for rel in REQUIRED_RUN_FILES if not (seed_dir / rel).exists()]
        if missing_files:
            issues.append(f"missing required files: {missing_files}")

        cfg, cfg_err = _read_json_dict(seed_dir / "config.json")
        sys_info, sys_err = _read_json_dict(seed_dir / "system.json")
        run_summary, summary_err = _read_json_dict(seed_dir / "run_summary.json")

        if cfg_err and cfg_err != "missing":
            issues.append(f"config.json {cfg_err}")
        if sys_err and sys_err != "missing":
            issues.append(f"system.json {sys_err}")
        if summary_err and summary_err != "missing":
            issues.append(f"run_summary.json {summary_err}")

        if cfg:
            if str(cfg.get("task", task)) != task:
                issues.append(f"config.task mismatch: cfg={cfg.get('task')} path={task}")
            if str(cfg.get("variant", variant)) != variant:
                issues.append(f"config.variant mismatch: cfg={cfg.get('variant')} path={variant}")
            cfg_seed = cfg.get("seed", seed)
            try:
                if int(cfg_seed) != int(seed):
                    issues.append(f"config.seed mismatch: cfg={cfg_seed} path={seed}")
            except Exception:
                issues.append(f"config.seed is non-integer: {cfg_seed}")

            run_source = str(cfg.get("data_source", ""))
            if data_source_filter != "all" and run_source != data_source_filter:
                issues.append(f"data_source mismatch: cfg={run_source} expected={data_source_filter}")

            if strict_single_task and str(cfg.get("training_regime", "")).strip() != "single_task":
                issues.append(f"training_regime must be single_task, got={cfg.get('training_regime')}")

            if require_cuda and not str(cfg.get("device", "")).lower().startswith("cuda"):
                issues.append(f"config.device must be cuda*, got={cfg.get('device')}")

        if require_cuda and sys_info and not bool(sys_info.get("cuda_available", False)):
            issues.append("system.cuda_available is false")

        train_path = seed_dir / "metrics_train.csv"
        if train_path.exists():
            try:
                train_df = pd.read_csv(train_path)
                missing_cols = _missing_columns(train_df, REQUIRED_TRAIN_COLUMNS)
                if missing_cols:
                    issues.append(f"metrics_train.csv missing required columns: {missing_cols}")
            except Exception as exc:
                issues.append(f"metrics_train.csv parse_error: {exc}")

        eval_path = seed_dir / "metrics_eval.csv"
        if eval_path.exists():
            try:
                eval_df = pd.read_csv(eval_path)
                missing_cols = _missing_columns(eval_df, REQUIRED_EVAL_COLUMNS)
                if missing_cols:
                    issues.append(f"metrics_eval.csv missing required columns: {missing_cols}")
            except Exception as exc:
                issues.append(f"metrics_eval.csv parse_error: {exc}")

        if run_summary:
            steps = run_summary.get("global_step")
            if steps is not None:
                try:
                    if int(steps) <= 0:
                        issues.append(f"run_summary.global_step must be >0, got={steps}")
                except Exception:
                    issues.append(f"run_summary.global_step is non-integer: {steps}")

            if require_cuda and not str(run_summary.get("device", "")).lower().startswith("cuda"):
                issues.append(f"run_summary.device must be cuda*, got={run_summary.get('device')}")

        if issues:
            report["ok"] = False
            report["n_issue_runs"] += 1
            report["issue_runs"].append({"run": run_id, "issues": issues})
        else:
            report["n_complete_run_dirs"] += 1

    return report


def audit_stage_b_logging(runs_dir: Path) -> dict[str, Any]:
    stage_b_root = runs_dir.parent
    report: dict[str, Any] = {
        "ok": True,
        "stage_b_root": str(stage_b_root),
        "latest_pointer": "",
        "status_file": "",
        "events": {"stage_b_start": 0, "stage_b_end": 0, "run_attempt": 0},
        "missing_log_files": 0,
        "issues": [],
    }

    latest_ptr = stage_b_root / "stage_b_runs.latest.txt"
    if not latest_ptr.exists():
        report["ok"] = False
        report["issues"].append(f"missing latest pointer: {latest_ptr}")
        return report

    try:
        raw_ptr = latest_ptr.read_text(encoding="utf-8").strip()
    except Exception as exc:
        report["ok"] = False
        report["issues"].append(f"failed to read latest pointer: {exc}")
        return report

    report["latest_pointer"] = raw_ptr
    status_file = Path(raw_ptr)
    if not status_file.is_absolute():
        candidate_cwd = (Path.cwd() / status_file).resolve()
        candidate_stage_root = (stage_b_root / status_file).resolve()
        if candidate_cwd.exists():
            status_file = candidate_cwd
        elif candidate_stage_root.exists():
            status_file = candidate_stage_root
        else:
            status_file = candidate_cwd
    report["status_file"] = str(status_file)

    if not status_file.exists():
        report["ok"] = False
        report["issues"].append(f"status file does not exist: {status_file}")
        return report

    try:
        with status_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                event = str(obj.get("event", ""))
                if event in report["events"]:
                    report["events"][event] += 1
                if event == "run_attempt":
                    log_path = Path(str(obj.get("log_path", "")))
                    if not log_path.exists():
                        report["missing_log_files"] += 1
    except Exception as exc:
        report["ok"] = False
        report["issues"].append(f"failed to parse status file: {exc}")
        return report

    if report["events"]["stage_b_start"] == 0:
        report["ok"] = False
        report["issues"].append("stage_b_start event missing in status file")
    if report["events"]["stage_b_end"] == 0:
        report["ok"] = False
        report["issues"].append("stage_b_end event missing in status file")
    if report["events"]["run_attempt"] == 0:
        report["ok"] = False
        report["issues"].append("run_attempt events missing in status file")
    if int(report["missing_log_files"]) > 0:
        report["ok"] = False
        report["issues"].append(f"{report['missing_log_files']} run_attempt log files are missing")

    return report


def aggregate_runs(df: pd.DataFrame, *, n_bootstrap: int) -> pd.DataFrame:
    if df.empty:
        return df

    rows = []
    grouped = df.groupby(["task", "variant"], as_index=False)
    for _, group in grouped:
        row = {
            "task": group["task"].iloc[0],
            "variant": group["variant"].iloc[0],
            "n_runs": int(group["seed"].nunique()),
        }
        for metric in METRICS:
            values = [float(v) for v in group[metric].dropna().tolist()]
            if not values:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
                row[f"{metric}_ci_lower"] = float("nan")
                row[f"{metric}_ci_upper"] = float("nan")
                continue

            ci = bootstrap_mean_ci(values, n_resamples=n_bootstrap, ci=0.95, seed=0)
            series = pd.Series(values)
            row[f"{metric}_mean"] = float(ci.mean)
            row[f"{metric}_std"] = float(series.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci_lower"] = float(ci.lower)
            row[f"{metric}_ci_upper"] = float(ci.upper)

        rows.append(row)

    agg = pd.DataFrame(rows).sort_values(["task", "variant"]).reset_index(drop=True)

    base = agg[agg["variant"] == "base"][["task", "time_sec_mean"]].rename(columns={"time_sec_mean": "base_time_sec_mean"})
    agg = agg.merge(base, on="task", how="left")
    agg["time_overhead_vs_base"] = agg["time_sec_mean"] / agg["base_time_sec_mean"]
    return agg


def compute_significance(df: pd.DataFrame, *, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        empty = pd.DataFrame()
        return empty, empty

    tests: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    correction_items: list[tuple[str, float]] = []

    metrics = ["test_acc", "spike_rate", "collapse_rate", "time_sec"]
    for task, task_df in df.groupby("task"):
        base_df = task_df[task_df["variant"] == "base"]
        if base_df.empty:
            continue

        for variant, var_df in task_df.groupby("variant"):
            if variant == "base":
                continue

            merged = var_df.merge(base_df, on="seed", suffixes=("_var", "_base"))
            if merged.empty:
                continue

            for metric in metrics:
                x = merged[f"{metric}_var"].tolist()
                y = merged[f"{metric}_base"].tolist()
                if not x or not y:
                    continue

                result = wilcoxon_signed_rank(x, y)
                test_name = f"{task}::{variant}::{metric}"
                correction_items.append((test_name, float(result.p_value)))
                tests.append(
                    {
                        "test_name": test_name,
                        "task": task,
                        "variant": variant,
                        "metric": metric,
                        "n_pairs": int(result.n),
                        "w_stat": float(result.statistic),
                        "p_raw": float(result.p_value),
                        "rank_biserial": float(result.rank_biserial),
                    }
                )
                effects.append(
                    {
                        "task": task,
                        "variant": variant,
                        "metric": metric,
                        "effect_size": float(result.rank_biserial),
                        "method": "rank_biserial",
                    }
                )

    if not tests:
        return pd.DataFrame(), pd.DataFrame()

    corrected = holm_bonferroni_correction(correction_items, alpha=alpha)
    for row in tests:
        info = corrected[row["test_name"]]
        row["p_holm"] = float(info["p_holm"])
        row["reject_h0"] = bool(info["reject_h0"])

    stats_df = pd.DataFrame(tests).sort_values(["task", "metric", "variant"]).reset_index(drop=True)
    effects_df = pd.DataFrame(effects).sort_values(["task", "metric", "variant"]).reset_index(drop=True)
    return stats_df, effects_df


def write_latex_table(agg_df: pd.DataFrame, stats_df: pd.DataFrame, out_path: Path) -> None:
    if agg_df.empty:
        out_path.write_text("% No runs found\n", encoding="utf-8")
        return

    p_lookup = {}
    if not stats_df.empty:
        for _, row in stats_df[stats_df["metric"] == "test_acc"].iterrows():
            p_lookup[(row["task"], row["variant"])] = row["p_holm"]

    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Task & Variant & TestAcc (95\\% CI) & SpikeRate (95\\% CI) & $p_{holm}$ " + r"\\",
        "\\midrule",
    ]
    for _, row in agg_df.iterrows():
        task = row["task"]
        variant = row["variant"]
        test_mean = row["test_acc_mean"]
        test_lo = row["test_acc_ci_lower"]
        test_hi = row["test_acc_ci_upper"]

        spike_mean = row["spike_rate_mean"]
        spike_lo = row["spike_rate_ci_lower"]
        spike_hi = row["spike_rate_ci_upper"]

        pval = p_lookup.get((task, variant), float("nan"))
        ptxt = "-" if pd.isna(pval) else f"{pval:.4f}"

        lines.append(
            f"{task} & {variant} & {test_mean:.4f} [{test_lo:.4f}, {test_hi:.4f}] & "
            f"{spike_mean:.4f} [{spike_lo:.4f}, {spike_hi:.4f}] & {ptxt} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(agg_df: pd.DataFrame, out_dir: Path, *, enabled: bool = True) -> None:
    if not enabled or agg_df.empty:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    for task, sub in agg_df.groupby("task"):
        sub = sub.sort_values("variant")

        plt.figure(figsize=(10, 4.5))
        plt.bar(sub["variant"], sub["test_acc_mean"])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Mean Test Accuracy")
        plt.title(f"{task}: Test Accuracy by Variant")
        plt.tight_layout()
        plt.savefig(out_dir / f"{task}_test_accuracy.png", dpi=180)
        plt.close()

        plt.figure(figsize=(10, 4.5))
        plt.bar(sub["variant"], sub["spike_rate_mean"])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Mean Spike Rate")
        plt.title(f"{task}: Spike Rate by Variant")
        plt.tight_layout()
        plt.savefig(out_dir / f"{task}_spike_rate.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_df = collect_runs(runs_dir, data_source_filter=str(args.data_source))
    run_df.to_csv(out_dir / "runs_flat.csv", index=False)

    expected_tasks = parse_csv(args.expected_tasks)
    expected_variants = parse_csv(args.expected_variants)
    expected_seeds = [int(x) for x in parse_csv(args.expected_seeds)]

    matrix_report = validate_matrix(
        run_df,
        expected_tasks=expected_tasks,
        expected_variants=expected_variants,
        expected_seeds=expected_seeds,
        min_runs_per_group=int(args.min_runs_per_group),
    )
    fp_report = validate_fingerprints(run_df)
    manifest_report = validate_manifest_consistency(run_df)
    artifact_report = audit_run_artifacts(
        runs_dir,
        data_source_filter=str(args.data_source),
        require_cuda=bool(args.require_cuda),
        strict_single_task=bool(args.strict_single_task),
    )
    logging_report = audit_stage_b_logging(runs_dir)

    audit = {
        "runs_dir": str(runs_dir),
        "data_source": str(args.data_source),
        "n_runs": int(len(run_df)),
        "matrix": matrix_report,
        "fingerprint_consistency": fp_report,
        "manifest_consistency": manifest_report,
        "artifact_integrity": artifact_report,
        "stage_b_logging": logging_report,
    }
    (out_dir / "readiness_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    if bool(args.strict_matrix) and not bool(matrix_report.get("ok", False)):
        raise SystemExit(f"Matrix readiness check failed. See {out_dir / 'readiness_audit.json'}")
    if bool(args.strict_fingerprint) and not bool(fp_report.get("ok", False)):
        raise SystemExit(f"Fingerprint consistency check failed. See {out_dir / 'readiness_audit.json'}")
    if bool(args.strict_artifacts) and not bool(artifact_report.get("ok", False)):
        raise SystemExit(f"Run artifact integrity check failed. See {out_dir / 'readiness_audit.json'}")
    if bool(args.strict_logging) and not bool(logging_report.get("ok", False)):
        raise SystemExit(f"Stage-B logging check failed. See {out_dir / 'readiness_audit.json'}")
    if bool(args.strict_manifest) and not bool(manifest_report.get("ok", False)):
        raise SystemExit(f"Manifest consistency check failed. See {out_dir / 'readiness_audit.json'}")

    agg_df = aggregate_runs(run_df, n_bootstrap=int(args.bootstrap_resamples))
    agg_df.to_csv(out_dir / "summary.csv", index=False)

    stats_df, effects_df = compute_significance(run_df, alpha=float(args.alpha))
    stats_df.to_csv(out_dir / "stats_significance.csv", index=False)
    effects_df.to_csv(out_dir / "effect_sizes.csv", index=False)

    write_latex_table(agg_df, stats_df, out_dir / "ablation_table_camera_ready.tex")
    write_latex_table(agg_df, stats_df, out_dir / "ablation_table.tex")
    plot_summary(agg_df, out_dir, enabled=bool(args.plot))

    print(f"Saved summary to: {out_dir}")
    if not agg_df.empty:
        print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()
