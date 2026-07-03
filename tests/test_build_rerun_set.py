from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_build_rerun_set_from_audit(tmp_path: Path) -> None:
    audit = {
        "matrix": {
            "expected": {
                "tasks": ["listops"],
                "variants": ["base", "dtr"],
                "seeds": [0, 1],
            },
            "counts": {
                "listops::base": {"seeds": [0]},
            },
        },
        "artifact_integrity": {
            "issue_runs": [
                {"run": "listops/base/seed_0", "issues": ["schema mismatch"]},
            ]
        },
    }

    audit_path = tmp_path / "readiness_audit.json"
    out_path = tmp_path / "rerun_set.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/build_rerun_set.py",
        "--audit",
        str(audit_path),
        "--out",
        str(out_path),
    ]
    proc = subprocess.run(cmd, cwd=_repo_root(), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout

    result = json.loads(out_path.read_text(encoding="utf-8"))
    # required total = 1 task * 2 variants * 2 seeds
    assert result["expected_total"] == 4
    # present only listops/base/seed_0
    assert result["present_total"] == 1
    # rerun includes missing + invalid
    assert result["rerun_total"] == 4

    rows = {(r["task"], r["variant"], int(r["seed"])): r["reason"] for r in result["rerun_rows"]}
    assert rows[("listops", "base", 0)] in {"invalid", "missing_and_invalid"}
    assert rows[("listops", "base", 1)] == "missing"
    assert rows[("listops", "dtr", 0)] == "missing"
    assert rows[("listops", "dtr", 1)] == "missing"
