from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.lra_official import load_lra_manifest, prepare_lra_official_task


def parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def load_task_cfg(task: str, config_dir: Path) -> dict:
    cfg_path = config_dir / f"lra_{task}_1660ti.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official LRA splits with checksum + deterministic preprocessing.")
    parser.add_argument("--manifest", default="configs/datasets/lra_official_manifest.yaml")
    parser.add_argument("--tasks", default="listops,text,pathfinder")
    parser.add_argument("--seq-len", type=int, default=0, help="Override sequence length for all tasks (0 = use per-task config).")
    parser.add_argument("--task-config-dir", default="configs")
    parser.add_argument("--raw-root", default="data/raw/lra_official")
    parser.add_argument("--processed-root", default="data/processed/lra_official")
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--strict-parity", dest="strict_parity", action="store_true", default=True)
    parser.add_argument("--no-strict-parity", dest="strict_parity", action="store_false")
    parser.add_argument("--enforce-no-overlap", dest="enforce_no_overlap", action="store_true", default=True)
    parser.add_argument("--allow-overlap", dest="enforce_no_overlap", action="store_false")
    parser.add_argument(
        "--max-cross-split-overlap-ratio",
        type=float,
        default=-1.0,
        help="If >=0, override per-task threshold; else read from task config (default 0.01).",
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = load_lra_manifest(manifest_path)

    tasks = parse_csv(args.tasks)
    config_dir = Path(args.task_config_dir)

    report: dict[str, dict] = {}
    for task in tasks:
        if task not in manifest.get("tasks", {}):
            raise SystemExit(f"Task '{task}' is not declared in manifest {manifest_path}")

        task_cfg = load_task_cfg(task, config_dir)
        seq_len = int(args.seq_len) if int(args.seq_len) > 0 else int(task_cfg.get("seq_len", 1024))
        overlap_thr = (
            float(args.max_cross_split_overlap_ratio)
            if float(args.max_cross_split_overlap_ratio) >= 0
            else float(task_cfg.get("max_cross_split_overlap_ratio", 0.01))
        )

        meta = prepare_lra_official_task(
            manifest_path=manifest_path,
            task=task,
            seq_len=seq_len,
            raw_root=args.raw_root,
            processed_root=args.processed_root,
            pad_token_id=args.pad_token_id,
            strict_parity=bool(args.strict_parity),
            enforce_no_overlap=bool(args.enforce_no_overlap),
            max_cross_split_overlap_ratio=overlap_thr,
            overwrite=bool(args.overwrite),
        )
        integrity = meta.get("integrity", {})
        report[task] = {
            "seq_len": seq_len,
            "fingerprint": meta.get("fingerprint"),
            "num_samples": {k: int(v.get("num_samples", 0)) for k, v in meta.get("splits", {}).items()},
            "cross_split_overlap_total": int(integrity.get("cross_split_overlap_total", 0)),
            "max_pair_overlap_ratio": float(integrity.get("max_pair_overlap_ratio", 0.0)),
            "max_cross_split_overlap_ratio_threshold": float(overlap_thr),
        }
        print(
            f"Prepared task={task} seq_len={seq_len} "
            f"fingerprint={meta.get('fingerprint')} "
            f"overlap_total={integrity.get('cross_split_overlap_total', 0)} "
            f"max_pair_overlap_ratio={integrity.get('max_pair_overlap_ratio', 0.0):.6f}"
        )

    out_path = Path(args.processed_root) / "prepare_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
