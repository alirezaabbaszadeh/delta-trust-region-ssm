from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit journal-readiness before long Stage-B runs.")
    p.add_argument("--manifest", default="configs/datasets/lra_official_manifest.yaml")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--task-config-pattern", default="configs/lra_{task}_1660ti.yaml")
    p.add_argument("--tasks", default="", help="Comma list. If empty, all manifest tasks are audited.")
    p.add_argument("--method-config", default="configs/method_variants.yaml")
    p.add_argument("--processed-root", default="data/processed/lra_official")
    p.add_argument("--out", default="output/readiness_audit.json")
    p.add_argument("--strict", action="store_true", default=False)
    p.add_argument("--require-official-source", action="store_true", default=False)
    p.add_argument("--require-cuda", action="store_true", default=False)
    return p.parse_args()


def resolve_model_cfg(variant_cfg: dict[str, Any], task: str) -> Path:
    by_task = variant_cfg.get("model_config_by_task", {})
    if isinstance(by_task, dict) and task in by_task:
        return ROOT / str(by_task[task])
    return ROOT / str(variant_cfg.get("model_config", "configs/model_b2s6_1660ti.yaml"))


def resolve_task_cfg_path(config_dir: Path, pattern: str, task: str) -> Path:
    p = str(pattern)
    if "{task}" in p:
        p = p.format(task=task)
    candidate = Path(p)
    if candidate.is_absolute():
        return candidate

    # If pattern already includes config dir prefix, honor it.
    if str(candidate).startswith(str(config_dir.name) + "/"):
        return ROOT / candidate

    return config_dir / candidate.name if "/" not in str(candidate) else ROOT / candidate


def _max_pair_overlap_ratio(leak_obj: dict[str, Any]) -> float:
    explicit = leak_obj.get("max_pair_overlap_ratio")
    if explicit is not None:
        try:
            return float(explicit)
        except (TypeError, ValueError):
            pass

    cross = leak_obj.get("cross_split_overlap", {})
    if not isinstance(cross, dict):
        return 0.0
    ratios = []
    for info in cross.values():
        if isinstance(info, dict):
            try:
                ratios.append(float(info.get("overlap_ratio_min_split", 0.0)))
            except (TypeError, ValueError):
                ratios.append(0.0)
    return max(ratios) if ratios else 0.0


def _load_task_cfg_or_issue(
    *,
    task: str,
    config_dir: Path,
    pattern: str,
    report: dict[str, Any],
) -> tuple[dict[str, Any] | None, Path]:
    path = resolve_task_cfg_path(config_dir, pattern, task)
    if not path.exists():
        report["config_compatibility"]["ok"] = False
        report["config_compatibility"]["issues"].append(
            {"task": task, "issue": f"missing task config: {path}"}
        )
        report["methodology_policy"]["ok"] = False
        report["methodology_policy"]["issues"].append(
            {"task": task, "issue": f"missing task config: {path}"}
        )
        return None, path

    try:
        cfg = load_yaml(path)
    except Exception as exc:
        report["config_compatibility"]["ok"] = False
        report["config_compatibility"]["issues"].append(
            {"task": task, "issue": f"task config parse error: {exc}", "path": str(path)}
        )
        report["methodology_policy"]["ok"] = False
        report["methodology_policy"]["issues"].append(
            {"task": task, "issue": f"task config parse error: {exc}", "path": str(path)}
        )
        return None, path

    return cfg, path


def _audit_methodology_policy(
    *,
    task: str,
    cfg: dict[str, Any],
    cfg_path: Path,
    require_cuda: bool,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    def _bool_true(key: str) -> None:
        if bool(cfg.get(key, False)) is not True:
            issues.append({"task": task, "path": str(cfg_path), "issue": f"{key} must be true"})

    _bool_true("strict_parity")
    _bool_true("enforce_no_overlap")
    _bool_true("enforce_param_budget")

    overlap_thr = float(cfg.get("max_cross_split_overlap_ratio", 0.01))
    if overlap_thr > 0.01:
        issues.append(
            {
                "task": task,
                "path": str(cfg_path),
                "issue": "max_cross_split_overlap_ratio must be <= 0.01",
                "value": overlap_thr,
            }
        )

    for key in ("seq_len", "batch_size", "epochs"):
        value = int(cfg.get(key, 0))
        if value <= 0:
            issues.append(
                {
                    "task": task,
                    "path": str(cfg_path),
                    "issue": f"{key} must be positive integer",
                    "value": value,
                }
            )

    lr = float(cfg.get("lr", 0.0))
    if lr <= 0:
        issues.append(
            {
                "task": task,
                "path": str(cfg_path),
                "issue": "lr must be positive",
                "value": lr,
            }
        )

    if int(cfg.get("pad_token_id", 0)) != 0:
        issues.append(
            {
                "task": task,
                "path": str(cfg_path),
                "issue": "pad_token_id must be 0 for deterministic official parity",
                "value": cfg.get("pad_token_id"),
            }
        )

    if require_cuda and bool(cfg.get("pin_memory", False)) is not True:
        issues.append(
            {
                "task": task,
                "path": str(cfg_path),
                "issue": "pin_memory should be true when require-cuda is enabled",
                "value": cfg.get("pin_memory"),
            }
        )

    num_workers = int(cfg.get("num_workers", 0))
    if num_workers < 0:
        issues.append(
            {
                "task": task,
                "path": str(cfg_path),
                "issue": "num_workers must be >= 0",
                "value": num_workers,
            }
        )

    return issues


def main() -> None:
    args = parse_args()

    manifest_path = ROOT / args.manifest
    method_cfg_path = ROOT / args.method_config
    config_dir = ROOT / args.config_dir
    processed_root = ROOT / args.processed_root

    manifest = load_yaml(manifest_path)
    method_cfg = load_yaml(method_cfg_path)
    variants = method_cfg.get("variants", {})

    manifest_tasks = [t for t in manifest.get("tasks", {}).keys()]
    selected_tasks = parse_csv(args.tasks) if str(args.tasks).strip() else manifest_tasks

    report: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "method_config_path": str(method_cfg_path),
        "task_config_pattern": str(args.task_config_pattern),
        "selected_tasks": selected_tasks,
        "source": manifest.get("source", {}),
        "source_policy": {"ok": True, "issues": []},
        "dataset": {"ok": True, "tasks": {}, "errors": []},
        "config_compatibility": {"ok": True, "issues": []},
        "param_fairness": {"ok": True, "issues": []},
        "methodology_policy": {"ok": True, "issues": []},
    }

    if args.require_official_source:
        src = manifest.get("source", {})
        src_text = " ".join(str(src.get(k, "")).lower() for k in ("name", "reference_url", "official_reference"))
        mirror_markers = ("mirror", "huggingface", "monteirot")
        if any(marker in src_text for marker in mirror_markers):
            msg = "manifest source appears to be a mirror/bootstrap copy, not an official LRA release source"
            report["source_policy"]["ok"] = False
            report["source_policy"]["issues"].append(msg)
            report["dataset"]["ok"] = False
            report["dataset"]["errors"].append(msg)

    token_label_ranges: dict[str, dict[str, int]] = {}
    task_cfg_cache: dict[str, dict[str, Any]] = {}
    task_cfg_path_cache: dict[str, Path] = {}

    for task in selected_tasks:
        if task not in manifest.get("tasks", {}):
            report["dataset"]["ok"] = False
            report["dataset"]["errors"].append(f"task not in manifest: {task}")
            continue

        task_cfg, task_cfg_path = _load_task_cfg_or_issue(
            task=task,
            config_dir=config_dir,
            pattern=str(args.task_config_pattern),
            report=report,
        )
        task_cfg_path_cache[task] = task_cfg_path
        if task_cfg is not None:
            task_cfg_cache[task] = task_cfg
            policy_issues = _audit_methodology_policy(
                task=task,
                cfg=task_cfg,
                cfg_path=task_cfg_path,
                require_cuda=bool(args.require_cuda),
            )
            if policy_issues:
                report["methodology_policy"]["ok"] = False
                report["methodology_policy"]["issues"].extend(policy_issues)

        task_info: dict[str, Any] = {"ok": True, "splits": {}}
        task_spec = manifest["tasks"][task]

        for split, spec in task_spec.get("splits", {}).items():
            split_info: dict[str, Any] = {"ok": True}
            raw_path = ROOT / str(spec.get("path", ""))
            if not raw_path.exists():
                split_info["ok"] = False
                split_info["error"] = f"missing raw split: {raw_path}"
                report["dataset"]["ok"] = False
                report["dataset"]["errors"].append(f"{task}/{split}: {split_info['error']}")
            else:
                expected_sha = str(spec.get("sha256", "")).strip().lower()
                actual_sha = sha256_file(raw_path)
                split_info["raw_sha_ok"] = bool((not expected_sha) or (actual_sha == expected_sha))
                split_info["raw_sha_expected"] = expected_sha
                split_info["raw_sha_actual"] = actual_sha
                if expected_sha and actual_sha != expected_sha:
                    split_info["ok"] = False
                    report["dataset"]["ok"] = False
                    report["dataset"]["errors"].append(f"{task}/{split}: raw sha mismatch")
            task_info["splits"][split] = split_info

        proc_dir = processed_root / task
        for split in ["train", "val", "test"]:
            pt = proc_dir / f"{split}.pt"
            meta_path = proc_dir / f"{split}.meta.json"
            if not pt.exists() or not meta_path.exists():
                task_info["ok"] = False
                report["dataset"]["ok"] = False
                report["dataset"]["errors"].append(f"{task}/{split}: missing processed pt/meta")
                continue

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            proc_sha = sha256_file(pt)
            if str(meta.get("sha256", "")) != proc_sha:
                task_info["ok"] = False
                report["dataset"]["ok"] = False
                report["dataset"]["errors"].append(f"{task}/{split}: processed sha mismatch")

        fp = proc_dir / "fingerprint.json"
        leak = proc_dir / "leakage_report.json"
        if not fp.exists() or not leak.exists():
            task_info["ok"] = False
            report["dataset"]["ok"] = False
            report["dataset"]["errors"].append(f"{task}: missing fingerprint/leakage_report")
        else:
            leak_obj = json.loads(leak.read_text(encoding="utf-8"))
            overlap = int(leak_obj.get("cross_split_overlap_total", 0))
            max_pair_ratio = _max_pair_overlap_ratio(leak_obj)
            overlap_thr = 0.01
            if task_cfg is not None:
                overlap_thr = float(task_cfg.get("max_cross_split_overlap_ratio", overlap_thr))

            task_info["cross_split_overlap_total"] = overlap
            task_info["max_pair_overlap_ratio"] = max_pair_ratio
            task_info["max_cross_split_overlap_ratio_threshold"] = float(overlap_thr)
            if max_pair_ratio > float(overlap_thr):
                task_info["ok"] = False
                report["dataset"]["ok"] = False
                report["dataset"]["errors"].append(
                    f"{task}: max_pair_overlap_ratio={max_pair_ratio:.6f} > threshold={float(overlap_thr):.6f}"
                )

        train_pt = proc_dir / "train.pt"
        if train_pt.exists():
            payload = torch.load(train_pt, map_location="cpu")
            x = payload["input_ids"]
            y = payload["labels"]
            token_label_ranges[task] = {
                "token_min": int(x.min().item()),
                "token_max": int(x.max().item()),
                "label_min": int(y.min().item()),
                "label_max": int(y.max().item()),
            }
            task_info["token_label_ranges"] = token_label_ranges[task]

        report["dataset"]["tasks"][task] = task_info

    for task, rng in token_label_ranges.items():
        task_cfg = task_cfg_cache.get(task)
        cfg_path = task_cfg_path_cache.get(task)
        if task_cfg is None:
            continue

        vocab_size = int(task_cfg.get("vocab_size", -1))
        num_classes = int(task_cfg.get("num_classes", -1))
        if rng["token_max"] >= vocab_size:
            report["config_compatibility"]["ok"] = False
            report["config_compatibility"]["issues"].append(
                {
                    "task": task,
                    "path": str(cfg_path),
                    "issue": "vocab_size too small",
                    "token_max": rng["token_max"],
                    "vocab_size": vocab_size,
                }
            )
        if rng["label_max"] >= num_classes:
            report["config_compatibility"]["ok"] = False
            report["config_compatibility"]["issues"].append(
                {
                    "task": task,
                    "path": str(cfg_path),
                    "issue": "num_classes too small",
                    "label_max": rng["label_max"],
                    "num_classes": num_classes,
                }
            )

    try:
        from scripts.train_lra_light import build_model, count_trainable_params
    except ModuleNotFoundError:
        scripts_dir = ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from train_lra_light import build_model, count_trainable_params

    base_variant_cfg = variants.get("base", {})
    for task in selected_tasks:
        task_cfg = task_cfg_cache.get(task)
        if task_cfg is None:
            continue

        vocab_size = int(task_cfg.get("vocab_size"))
        num_classes = int(task_cfg.get("num_classes"))
        tol = float(task_cfg.get("param_budget_tolerance", 0.10))

        base_model_cfg = load_yaml(resolve_model_cfg(base_variant_cfg, task))
        base_family = str(base_variant_cfg.get("model_family", "b2s6"))
        base_model = build_model(
            family=base_family,
            model_cfg=base_model_cfg,
            vocab_size=vocab_size,
            num_classes=num_classes,
            pad_token_id=0,
        )
        base_params = count_trainable_params(base_model)

        for variant_name, variant_cfg in variants.items():
            model_cfg = load_yaml(resolve_model_cfg(variant_cfg, task))
            family = str(variant_cfg.get("model_family", "b2s6"))
            model = build_model(
                family=family,
                model_cfg=model_cfg,
                vocab_size=vocab_size,
                num_classes=num_classes,
                pad_token_id=0,
            )
            n_params = count_trainable_params(model)
            diff = abs(float(n_params - base_params)) / float(max(1, base_params))
            if diff > tol:
                report["param_fairness"]["ok"] = False
                report["param_fairness"]["issues"].append(
                    {
                        "task": task,
                        "variant": variant_name,
                        "family": family,
                        "num_params": n_params,
                        "base_num_params": base_params,
                        "diff_vs_base": diff,
                        "tolerance": tol,
                    }
                )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "source_policy_ok": report["source_policy"]["ok"],
        "dataset_ok": report["dataset"]["ok"],
        "config_compatibility_ok": report["config_compatibility"]["ok"],
        "param_fairness_ok": report["param_fairness"]["ok"],
        "methodology_policy_ok": report["methodology_policy"]["ok"],
        "out": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.strict and (
        not report["source_policy"]["ok"]
        or not report["dataset"]["ok"]
        or not report["config_compatibility"]["ok"]
        or not report["param_fairness"]["ok"]
        or not report["methodology_policy"]["ok"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
