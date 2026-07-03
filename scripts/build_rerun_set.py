from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build deterministic rerun set from readiness_audit.json")
    p.add_argument("--audit", required=True, help="Path to readiness_audit.json")
    p.add_argument("--out", required=True, help="Output rerun_set.json path")
    p.add_argument("--print-commands", action="store_true", default=False)
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return raw


def _run_tuple(task: str, variant: str, seed: int) -> tuple[str, str, int]:
    return (str(task), str(variant), int(seed))


def main() -> None:
    args = parse_args()
    audit_path = Path(args.audit)
    if not audit_path.is_absolute():
        audit_path = (Path.cwd() / audit_path).resolve()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()

    audit = _load_json(audit_path)

    expected = audit.get("matrix", {}).get("expected", {})
    tasks = [str(x) for x in expected.get("tasks", [])]
    variants = [str(x) for x in expected.get("variants", [])]
    seeds = [int(x) for x in expected.get("seeds", [])]

    required = {_run_tuple(t, v, s) for t in tasks for v in variants for s in seeds}

    present: set[tuple[str, str, int]] = set()
    counts = audit.get("matrix", {}).get("counts", {})
    if isinstance(counts, dict):
        for key, info in counts.items():
            if "::" not in str(key):
                continue
            t, v = str(key).split("::", 1)
            for s in info.get("seeds", []):
                present.add(_run_tuple(t, v, int(s)))

    missing = required - present

    invalid: set[tuple[str, str, int]] = set()
    artifact = audit.get("artifact_integrity", {})
    for item in artifact.get("issue_runs", []):
        run = str(item.get("run", ""))
        parts = run.split("/")
        if len(parts) != 3:
            continue
        task, variant, seed_part = parts
        if not seed_part.startswith("seed_"):
            continue
        seed = int(seed_part.replace("seed_", ""))
        invalid.add(_run_tuple(task, variant, seed))

    rerun = missing | invalid

    by_task: dict[str, list[dict[str, Any]]] = {t: [] for t in tasks}
    by_variant: dict[str, list[dict[str, Any]]] = {v: [] for v in variants}

    def _row(t: str, v: str, s: int, reason: str) -> dict[str, Any]:
        return {"task": t, "variant": v, "seed": s, "reason": reason}

    rows: list[dict[str, Any]] = []
    for t, v, s in sorted(rerun):
        reason = "missing"
        if (t, v, s) in invalid and (t, v, s) in missing:
            reason = "missing_and_invalid"
        elif (t, v, s) in invalid:
            reason = "invalid"
        item = _row(t, v, s, reason)
        rows.append(item)
        if t in by_task:
            by_task[t].append(item)
        if v in by_variant:
            by_variant[v].append(item)

    out = {
        "audit": str(audit_path),
        "expected_total": len(required),
        "present_total": len(present),
        "missing_total": len(missing),
        "invalid_total": len(invalid),
        "rerun_total": len(rerun),
        "tasks": tasks,
        "variants": variants,
        "seeds": seeds,
        "rerun_rows": rows,
        "rerun_by_task": by_task,
        "rerun_by_variant": by_variant,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "out": str(out_path),
        "expected_total": len(required),
        "present_total": len(present),
        "rerun_total": len(rerun),
    }, ensure_ascii=False, indent=2))

    if args.print_commands:
        for task in tasks:
            rows_task = by_task.get(task, [])
            if not rows_task:
                continue
            variants_for_task = sorted({r["variant"] for r in rows_task})
            seeds_for_task = sorted({int(r["seed"]) for r in rows_task})
            print(
                "make run_stage_b "
                f"TASKS={task} "
                f"VARIANTS={','.join(variants_for_task)} "
                f"SEEDS={','.join(str(s) for s in seeds_for_task)}"
            )


if __name__ == "__main__":
    main()
