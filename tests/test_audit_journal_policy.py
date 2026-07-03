from __future__ import annotations

from pathlib import Path

from scripts.audit_journal_readiness import _audit_methodology_policy, resolve_task_cfg_path


ROOT = Path(__file__).resolve().parents[1]


def test_resolve_task_cfg_path_with_pattern_and_config_dir(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    path = resolve_task_cfg_path(config_dir, "profiles/lra_{task}_fast.yaml", "text")
    assert path == ROOT / "profiles" / "lra_text_fast.yaml"


def test_audit_methodology_policy_flags_invalid_values(tmp_path: Path) -> None:
    cfg = {
        "strict_parity": False,
        "enforce_no_overlap": False,
        "enforce_param_budget": False,
        "max_cross_split_overlap_ratio": 0.02,
        "seq_len": 0,
        "batch_size": 0,
        "epochs": -1,
        "lr": 0.0,
        "pad_token_id": 1,
        "pin_memory": False,
        "num_workers": -3,
    }

    issues = _audit_methodology_policy(
        task="listops",
        cfg=cfg,
        cfg_path=tmp_path / "cfg.yaml",
        require_cuda=True,
    )

    messages = "\n".join(str(x.get("issue", "")) for x in issues)
    assert "strict_parity must be true" in messages
    assert "enforce_no_overlap must be true" in messages
    assert "enforce_param_budget must be true" in messages
    assert "max_cross_split_overlap_ratio must be <= 0.01" in messages
    assert "seq_len must be positive integer" in messages
    assert "batch_size must be positive integer" in messages
    assert "epochs must be positive integer" in messages
    assert "lr must be positive" in messages
    assert "pad_token_id must be 0" in messages
    assert "pin_memory should be true" in messages
    assert "num_workers must be >= 0" in messages
