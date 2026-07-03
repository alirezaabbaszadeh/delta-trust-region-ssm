from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional in minimal runtime envs
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional in minimal runtime envs
    plt = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.lra_light import LraLightDataConfig, build_lra_light_dataloaders, resolve_task_defaults
from src.data.lra_official import LraOfficialDataConfig, build_lra_official_dataloaders
from src.models.s4d_lite import S4DLiteClassifier, S4DLiteConfig
from src.models.seq_classifier import SequenceClassifier, SequenceClassifierConfig
from src.models.transformer_lite import TransformerLiteClassifier, TransformerLiteConfig
from src.optim.delta_trust_region import DeltaTrustRegionConfig, apply_delta_trust_region, compute_eps, snapshot_delta_params
from src.optim.lado import (
    LadoConfig,
    build_two_group_adamw,
    clip_delta_grads,
    compute_lr_delta,
    get_group_lr,
    grad_l2_norm,
    set_group_lr,
)
from src.utils.logging import CsvLogger, gather_system_info, make_run_dir, write_json
from src.utils.metrics import CollapseTracker, StabilityMetricConfig, classification_accuracy, spike_from_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one LRA-light task/variant run.")
    parser.add_argument("--task", required=True, choices=["listops", "text", "pathfinder"])
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", required=True, help="Task-specific YAML config")
    parser.add_argument("--model-config", default="configs/model_b2s6_1660ti.yaml")
    parser.add_argument("--method-config", default="configs/method_variants.yaml")
    parser.add_argument("--model-family", default="", help="Override model family in variant config")
    parser.add_argument("--out", default="output")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
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
    parser.add_argument("--max-train-steps", type=int, default=0, help="Optional hard cap for smoke tests")
    parser.add_argument("--param-budget-tolerance", type=float, default=0.10)
    parser.add_argument("--allow-param-budget-mismatch", action="store_true", default=False)

    parser.add_argument("--data-source", choices=["official_lra", "synthetic"], default="official_lra")
    parser.add_argument("--dataset-manifest", default="configs/datasets/lra_official_manifest.yaml")
    parser.add_argument("--prepare-official", action="store_true", default=False)

    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--deterministic-strict", action="store_true", default=False)
    parser.add_argument("--save-best", dest="save_best", action="store_true", default=True)
    parser.add_argument("--no-save-best", dest="save_best", action="store_false")
    parser.add_argument("--plot", dest="plot", action="store_true", default=True)
    parser.add_argument("--no-plots", dest="plot", action="store_false")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping.")
    return data


def _hash_json(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int, *, deterministic: bool, deterministic_strict: bool) -> None:
    if deterministic and torch.cuda.is_available():
        # Required by cuBLAS for deterministic GEMM on CUDA 10.2+.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Some CUDA ops still lack strict deterministic kernels in current PyTorch builds.
        torch.use_deterministic_algorithms(True, warn_only=not deterministic_strict)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_amp_dtype(name: str) -> torch.dtype:
    if str(name).lower() == "bf16":
        return torch.bfloat16
    return torch.float16


def _configure_cuda_runtime_for_speed(
    *,
    enabled: bool,
    deterministic: bool,
    allow_tf32: bool,
    cudnn_benchmark: bool,
) -> None:
    if not enabled or not torch.cuda.is_available():
        return

    # Determinism and fast-kernel knobs conflict by design; deterministic mode wins.
    if deterministic:
        return

    if allow_tf32:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if cudnn_benchmark and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def count_trainable_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def build_model(
    *,
    family: str,
    model_cfg: dict,
    vocab_size: int,
    num_classes: int,
    pad_token_id: int,
):
    family = family.lower().strip()
    if family == "b2s6":
        cfg = SequenceClassifierConfig(
            vocab_size=vocab_size,
            num_classes=num_classes,
            d_model=int(model_cfg.get("d_model", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            num_channel_blocks=int(model_cfg.get("num_channel_blocks", 4)),
            state_dim=int(model_cfg.get("state_dim", 16)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            pad_token_id=pad_token_id,
        )
        return SequenceClassifier(cfg)

    if family == "transformer_lite":
        cfg = TransformerLiteConfig(
            vocab_size=vocab_size,
            num_classes=num_classes,
            d_model=int(model_cfg.get("d_model", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            nhead=int(model_cfg.get("nhead", 4)),
            ffn_dim=int(model_cfg.get("ffn_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            max_seq_len=int(model_cfg.get("max_seq_len", 4096)),
            pad_token_id=pad_token_id,
        )
        return TransformerLiteClassifier(cfg)

    if family == "s4d_lite":
        cfg = S4DLiteConfig(
            vocab_size=vocab_size,
            num_classes=num_classes,
            d_model=int(model_cfg.get("d_model", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            pad_token_id=pad_token_id,
        )
        return S4DLiteClassifier(cfg)

    raise ValueError(f"Unsupported model_family: {family}")


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    accs = []
    non_blocking = device.type == "cuda"

    for batch in loader:
        input_ids = batch["input_ids"].to(device=device, non_blocking=non_blocking)
        attention_mask = batch["attention_mask"].to(device=device, non_blocking=non_blocking)
        labels = batch["labels"].to(device=device, non_blocking=non_blocking)

        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)

        losses.append(float(loss.detach().cpu()))
        accs.append(classification_accuracy(logits.detach(), labels))

    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "acc": float(sum(accs) / max(1, len(accs))),
    }


def _save_rng_state(path: Path) -> None:
    rng_state = {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(rng_state, path)


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_checkpoint(path: Path, model, optimizer, scaler) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler_state = ckpt.get("scaler_state")
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    return ckpt


def plot_curves(run_dir: Path, train_df, eval_df) -> None:
    if plt is None or pd is None:
        return

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not train_df.empty:
        plt.figure(figsize=(10, 4.5))
        plt.plot(train_df["step"], train_df["loss"], linewidth=1.2)
        plt.yscale("log")
        plt.xlabel("Step")
        plt.ylabel("Train Loss")
        plt.title("Training Loss")
        plt.tight_layout()
        plt.savefig(plot_dir / "loss.png", dpi=180)
        plt.close()

        plt.figure(figsize=(10, 4.5))
        plt.plot(train_df["step"], train_df["acc"], linewidth=1.2)
        plt.xlabel("Step")
        plt.ylabel("Train Accuracy")
        plt.title("Training Accuracy")
        plt.tight_layout()
        plt.savefig(plot_dir / "accuracy.png", dpi=180)
        plt.close()

        plt.figure(figsize=(10, 4.5))
        plt.plot(train_df["step"], train_df["grad_delta_l2"], linewidth=1.2)
        plt.yscale("log")
        plt.xlabel("Step")
        plt.ylabel("Delta Grad L2")
        plt.title("Delta Gradient Norm")
        plt.tight_layout()
        plt.savefig(plot_dir / "grad_delta.png", dpi=180)
        plt.close()

        if "drift_post_max" in train_df.columns and train_df["drift_post_max"].notna().any():
            plt.figure(figsize=(10, 4.5))
            sub = train_df[train_df["drift_post_max"].notna()]
            plt.plot(sub["step"], sub["drift_post_max"], linewidth=1.2)
            plt.yscale("log")
            plt.xlabel("Step")
            plt.ylabel("Post-TR Drift Max")
            plt.title("Delta Drift")
            plt.tight_layout()
            plt.savefig(plot_dir / "drift_post_max.png", dpi=180)
            plt.close()

    if not eval_df.empty:
        val_df = eval_df[eval_df["split"] == "val"]
        if not val_df.empty:
            plt.figure(figsize=(8, 4.5))
            plt.plot(val_df["epoch"], val_df["acc"], marker="o", linewidth=1.2)
            plt.xlabel("Epoch")
            plt.ylabel("Validation Accuracy")
            plt.title("Validation Accuracy by Epoch")
            plt.tight_layout()
            plt.savefig(plot_dir / "val_accuracy.png", dpi=180)
            plt.close()


def main() -> None:
    args = parse_args()

    task_cfg = load_yaml(Path(args.config))
    method_cfg = load_yaml(Path(args.method_config))

    variants = method_cfg.get("variants", {})
    if args.variant not in variants:
        raise SystemExit(f"Variant '{args.variant}' not found in {args.method_config}")
    variant_cfg = variants[args.variant]

    model_family = str(args.model_family).strip() or str(variant_cfg.get("model_family", "b2s6"))
    model_cfg_by_task = variant_cfg.get("model_config_by_task", {})
    if isinstance(model_cfg_by_task, dict) and args.task in model_cfg_by_task:
        model_cfg_path = Path(str(model_cfg_by_task[args.task]))
    else:
        model_cfg_path = Path(str(variant_cfg.get("model_config", args.model_config)))
    model_cfg_yaml = load_yaml(model_cfg_path)

    set_seed(
        args.seed,
        deterministic=bool(args.deterministic),
        deterministic_strict=bool(args.deterministic_strict),
    )
    device = resolve_device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("--require-cuda was set but CUDA device is not active.")
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = _resolve_amp_dtype(args.amp_dtype)
    use_grad_scaler = bool(use_amp and amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    _configure_cuda_runtime_for_speed(
        enabled=(device.type == "cuda"),
        deterministic=bool(args.deterministic),
        allow_tf32=bool(args.allow_tf32),
        cudnn_benchmark=bool(args.cudnn_benchmark),
    )

    defaults = resolve_task_defaults(args.task)
    num_classes = int(task_cfg.get("num_classes", defaults["num_classes"]))
    vocab_size = int(task_cfg.get("vocab_size", defaults["vocab_size"]))
    seq_len = int(task_cfg.get("seq_len", 512))
    batch_size = int(task_cfg.get("batch_size", 8))
    epochs = int(task_cfg.get("epochs", 2))
    pad_token_id = int(task_cfg.get("pad_token_id", 0))

    if args.data_source == "official_lra":
        official_cfg = LraOfficialDataConfig(
            task=args.task,
            seq_len=seq_len,
            batch_size=batch_size,
            manifest_path=args.dataset_manifest,
            raw_root=str(task_cfg.get("raw_root", "data/raw/lra_official")),
            processed_root=str(task_cfg.get("processed_root", "data/processed/lra_official")),
            pad_token_id=pad_token_id,
            num_workers=int(task_cfg.get("num_workers", 0)),
            pin_memory=bool(task_cfg.get("pin_memory", device.type == "cuda")),
            persistent_workers=bool(task_cfg.get("persistent_workers", int(task_cfg.get("num_workers", 0)) > 0)),
            prefetch_factor=int(task_cfg.get("prefetch_factor", 4)),
            strict_parity=bool(task_cfg.get("strict_parity", True)),
            enforce_no_overlap=bool(task_cfg.get("enforce_no_overlap", True)),
            max_cross_split_overlap_ratio=float(task_cfg.get("max_cross_split_overlap_ratio", 0.01)),
        )
        loaders, dataset_fingerprint = build_lra_official_dataloaders(
            official_cfg,
            seed=args.seed,
            ensure_prepared=bool(args.prepare_official),
            overwrite=False,
        )
    else:
        synthetic_cfg = LraLightDataConfig(
            task=args.task,
            seq_len=seq_len,
            batch_size=batch_size,
            vocab_size=vocab_size,
            num_classes=num_classes,
            train_size=int(task_cfg.get("train_size", 1024)),
            val_size=int(task_cfg.get("val_size", 256)),
            test_size=int(task_cfg.get("test_size", 256)),
            pad_token_id=pad_token_id,
            num_workers=int(task_cfg.get("num_workers", 0)),
            pin_memory=bool(task_cfg.get("pin_memory", device.type == "cuda")),
            persistent_workers=bool(task_cfg.get("persistent_workers", int(task_cfg.get("num_workers", 0)) > 0)),
            prefetch_factor=int(task_cfg.get("prefetch_factor", 2)),
        )
        loaders = build_lra_light_dataloaders(synthetic_cfg, seed=args.seed)
        dataset_fingerprint = {
            "source": "synthetic",
            "task": args.task,
            "seq_len": seq_len,
            "train_size": int(task_cfg.get("train_size", 1024)),
            "val_size": int(task_cfg.get("val_size", 256)),
            "test_size": int(task_cfg.get("test_size", 256)),
        }
        dataset_fingerprint["fingerprint"] = _hash_json(dataset_fingerprint)

    dataset_fingerprint_sha256 = _hash_json(dataset_fingerprint)

    model = build_model(
        family=model_family,
        model_cfg=model_cfg_yaml,
        vocab_size=vocab_size,
        num_classes=num_classes,
        pad_token_id=pad_token_id,
    )
    num_trainable_params = count_trainable_params(model)

    base_variant_cfg = variants.get("base", {})
    base_model_family = str(base_variant_cfg.get("model_family", "b2s6"))
    base_model_cfg_path = Path(str(base_variant_cfg.get("model_config", "configs/model_b2s6_1660ti.yaml")))
    base_model_cfg_yaml = load_yaml(base_model_cfg_path)
    base_model = build_model(
        family=base_model_family,
        model_cfg=base_model_cfg_yaml,
        vocab_size=vocab_size,
        num_classes=num_classes,
        pad_token_id=pad_token_id,
    )
    num_trainable_params_base = count_trainable_params(base_model)
    del base_model

    param_budget_tolerance = float(task_cfg.get("param_budget_tolerance", args.param_budget_tolerance))
    enforce_param_budget = bool(task_cfg.get("enforce_param_budget", True)) and (not bool(args.allow_param_budget_mismatch))
    param_diff_vs_base = abs(float(num_trainable_params - num_trainable_params_base)) / float(max(1, num_trainable_params_base))
    if enforce_param_budget and param_diff_vs_base > param_budget_tolerance:
        raise ValueError(
            f"Parameter budget mismatch for task={args.task}, variant={args.variant}: "
            f"num_trainable={num_trainable_params}, base={num_trainable_params_base}, "
            f"diff={param_diff_vs_base:.4f} > tolerance={param_budget_tolerance:.4f}."
        )

    model = model.to(device)
    if bool(args.torch_compile) and device.type == "cuda":
        model = torch.compile(model, mode=str(args.torch_compile_mode))
    delta_params = list(model.delta_parameters()) if hasattr(model, "delta_parameters") else []
    lr_base = float(task_cfg.get("lr", 3e-4))
    weight_decay = float(task_cfg.get("weight_decay", 0.01))
    lr_delta_scale = float(variant_cfg.get("lr_delta_scale", 1.0))

    lado_cfg = LadoConfig(**variant_cfg.get("lado", {}))
    dtr_cfg = DeltaTrustRegionConfig(**variant_cfg.get("dtr", {}))

    if not delta_params:
        lado_cfg = LadoConfig(enabled=False)
        dtr_cfg = DeltaTrustRegionConfig(enabled=False)

    if delta_params:
        if lado_cfg.enabled:
            lr_delta_target = compute_lr_delta(lr_base, seq_len=seq_len, l_ref=lado_cfg.l_ref, alpha=lado_cfg.alpha)
        else:
            lr_delta_target = lr_base * lr_delta_scale

        optimizer, delta_group_idx = build_two_group_adamw(
            model,
            lr_other=lr_base,
            lr_delta=lr_delta_target,
            weight_decay=weight_decay,
            delta_weight_decay=lado_cfg.delta_weight_decay,
        )

        if variant_cfg.get("freeze_delta", False):
            set_group_lr(optimizer, delta_group_idx, 0.0)
    else:
        lr_delta_target = lr_base
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_base, weight_decay=weight_decay)
        delta_group_idx = None

    run_dir = make_run_dir(Path(args.out), args.task, args.variant, args.seed)
    write_json(
        run_dir / "config.json",
        {
            "argv": list(sys.argv),
            "cwd": str(Path.cwd()),
            "python_executable": sys.executable,
            "hostname": socket.gethostname(),
            "task": args.task,
            "variant": args.variant,
            "seed": args.seed,
            "task_config": task_cfg,
            "model_family": model_family,
            "model_config_path": str(model_cfg_path),
            "model_config": model_cfg_yaml,
            "variant_config": variant_cfg,
            "device": str(device),
            "amp": use_amp,
            "amp_dtype": str(args.amp_dtype),
            "allow_tf32": bool(args.allow_tf32),
            "cudnn_benchmark": bool(args.cudnn_benchmark),
            "torch_compile": bool(args.torch_compile),
            "torch_compile_mode": str(args.torch_compile_mode),
            "data_source": args.data_source,
            "dataset_manifest": args.dataset_manifest,
            "dataset_fingerprint_sha256": dataset_fingerprint_sha256,
            "deterministic": bool(args.deterministic),
            "deterministic_strict": bool(args.deterministic_strict),
            "require_cuda": bool(args.require_cuda),
            "training_regime": "single_task",
            "num_trainable_params": num_trainable_params,
            "num_trainable_params_base": num_trainable_params_base,
            "param_diff_vs_base": param_diff_vs_base,
            "param_budget_tolerance": param_budget_tolerance,
            "param_budget_enforced": enforce_param_budget,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        },
    )
    write_json(run_dir / "system.json", gather_system_info(ROOT))
    write_json(run_dir / "dataset_fingerprint.json", dataset_fingerprint)

    ckpt_dir = run_dir / "checkpoints"
    latest_ckpt_path = ckpt_dir / "latest.pt"
    best_ckpt_path = ckpt_dir / "best.pt"
    rng_state_path = run_dir / "rng_state.pt"

    train_fields = [
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
    eval_fields = ["epoch", "split", "loss", "acc"]

    stability_cfg = StabilityMetricConfig(
        spike_ratio=float(task_cfg.get("spike_ratio", 2.0)),
        spike_window=int(task_cfg.get("spike_window", 10)),
        collapse_margin=float(task_cfg.get("collapse_margin", 0.05)),
        collapse_patience=int(task_cfg.get("collapse_patience", 3)),
    )
    collapse_tracker = CollapseTracker(stability_cfg)

    total_steps = len(loaders["train"]) * epochs
    warmup_steps = int(math.floor(total_steps * lado_cfg.warmup_frac)) if lado_cfg.enabled else 0

    start_epoch = 1
    global_step = 0
    best_val_acc = float("-inf")
    loss_history: list[float] = []

    if args.resume:
        if not latest_ckpt_path.exists():
            raise FileNotFoundError(f"--resume was set but checkpoint not found: {latest_ckpt_path}")
        ckpt = _load_checkpoint(latest_ckpt_path, model, optimizer, scaler)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_acc = float(ckpt.get("best_val_acc", float("-inf")))
        loss_history = [float(x) for x in ckpt.get("loss_history", [])]
        collapse_tracker._streak = int(ckpt.get("collapse_streak", 0))

    run_started_unix = time.time()
    run_started_utc = _utc_now_iso()
    start_time = run_started_unix
    last_epoch = max(0, start_epoch - 1)
    last_val_metrics: dict[str, float] | None = None
    final_test_metrics: dict[str, float] | None = None
    total_train_tokens = 0
    peak_gpu_mem_alloc_mb = 0.0
    peak_gpu_mem_reserved_mb = 0.0

    use_non_blocking = device.type == "cuda"

    with CsvLogger(run_dir / "metrics_train.csv", train_fields, append=bool(args.resume)) as train_logger, CsvLogger(
        run_dir / "metrics_eval.csv", eval_fields, append=bool(args.resume)
    ) as eval_logger:
        for epoch in range(start_epoch, epochs + 1):
            last_epoch = epoch
            model.train()

            for batch in loaders["train"]:
                if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                    break

                input_ids = batch["input_ids"].to(device=device, non_blocking=use_non_blocking)
                attention_mask = batch["attention_mask"].to(device=device, non_blocking=use_non_blocking)
                labels = batch["labels"].to(device=device, non_blocking=use_non_blocking)

                if delta_group_idx is not None:
                    if variant_cfg.get("freeze_delta", False):
                        set_group_lr(optimizer, delta_group_idx, 0.0)
                    elif lado_cfg.enabled:
                        if global_step < warmup_steps:
                            set_group_lr(optimizer, delta_group_idx, 0.0)
                        else:
                            set_group_lr(optimizer, delta_group_idx, lr_delta_target)
                    else:
                        set_group_lr(optimizer, delta_group_idx, lr_delta_target)

                old_snapshots = None
                delta_old = None
                eps_delta = None
                can_apply_dtr = bool(dtr_cfg.enabled and hasattr(model, "collect_delta_values") and delta_group_idx is not None)
                if can_apply_dtr:
                    old_snapshots = snapshot_delta_params(model)
                    delta_old = model.collect_delta_values(input_ids, attention_mask).detach()
                    eps_delta = compute_eps(
                        int(input_ids.shape[1]),
                        eps_ref=dtr_cfg.eps_ref,
                        l_ref=dtr_cfg.l_ref,
                        alpha=dtr_cfg.alpha,
                    )

                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                        logits = model(input_ids, attention_mask)
                        loss = F.cross_entropy(logits, labels)

                    if use_grad_scaler:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)

                        if (
                            lado_cfg.enabled
                            and lado_cfg.clip_delta_norm is not None
                            and lado_cfg.clip_delta_norm > 0
                            and delta_params
                        ):
                            clip_delta_grads(delta_params, max_norm=lado_cfg.clip_delta_norm)

                        grad_delta = grad_l2_norm(delta_params) if delta_params else 0.0
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()

                        if (
                            lado_cfg.enabled
                            and lado_cfg.clip_delta_norm is not None
                            and lado_cfg.clip_delta_norm > 0
                            and delta_params
                        ):
                            clip_delta_grads(delta_params, max_norm=lado_cfg.clip_delta_norm)

                        grad_delta = grad_l2_norm(delta_params) if delta_params else 0.0
                        optimizer.step()
                else:
                    logits = model(input_ids, attention_mask)
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()

                    if lado_cfg.enabled and lado_cfg.clip_delta_norm is not None and lado_cfg.clip_delta_norm > 0 and delta_params:
                        clip_delta_grads(delta_params, max_norm=lado_cfg.clip_delta_norm)

                    grad_delta = grad_l2_norm(delta_params) if delta_params else 0.0
                    optimizer.step()

                tr_metrics = None
                if can_apply_dtr and old_snapshots is not None and delta_old is not None and eps_delta is not None:
                    tr_metrics = apply_delta_trust_region(
                        model=model,
                        old_snapshots=old_snapshots,
                        delta_old=delta_old,
                        delta_fn=lambda: model.collect_delta_values(input_ids, attention_mask),
                        eps=eps_delta,
                        search_steps=dtr_cfg.search_steps,
                        sample_stride=dtr_cfg.sample_stride,
                        max_positions=dtr_cfg.max_positions,
                    )

                loss_value = float(loss.detach().cpu())
                acc_value = classification_accuracy(logits.detach(), labels)
                elapsed = time.time() - start_time
                batch_tokens = int(attention_mask.sum().detach().cpu())
                total_train_tokens += batch_tokens
                tokens_per_sec = float(total_train_tokens / max(elapsed, 1e-8))

                gpu_mem_alloc_mb = None
                gpu_mem_reserved_mb = None
                if device.type == "cuda":
                    gpu_mem_alloc_mb = float(torch.cuda.memory_allocated(device) / (1024 ** 2))
                    gpu_mem_reserved_mb = float(torch.cuda.memory_reserved(device) / (1024 ** 2))
                    peak_gpu_mem_alloc_mb = max(peak_gpu_mem_alloc_mb, gpu_mem_alloc_mb)
                    peak_gpu_mem_reserved_mb = max(peak_gpu_mem_reserved_mb, gpu_mem_reserved_mb)

                lr_current = float(optimizer.param_groups[0]["lr"])
                if delta_group_idx is not None:
                    lr_delta_current = get_group_lr(optimizer, delta_group_idx)
                else:
                    lr_delta_current = lr_current

                spike = int(spike_from_history(loss_history, loss_value, stability_cfg))
                collapse = int(collapse_tracker.update(acc_value, num_classes=num_classes))
                loss_history.append(loss_value)

                row = {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": loss_value,
                    "acc": acc_value,
                    "grad_delta_l2": grad_delta,
                    "lr": lr_current,
                    "lr_delta": lr_delta_current,
                    "spike": spike,
                    "collapse": collapse,
                    "collapse_streak": collapse_tracker.streak,
                    "eps_delta": eps_delta,
                    "drift_pre_max": None,
                    "drift_pre_p99": None,
                    "drift_post_max": None,
                    "drift_post_p99": None,
                    "trust_region_scale": None,
                    "time_sec": elapsed,
                    "batch_tokens": batch_tokens,
                    "tokens_per_sec": tokens_per_sec,
                    "gpu_mem_alloc_mb": gpu_mem_alloc_mb,
                    "gpu_mem_reserved_mb": gpu_mem_reserved_mb,
                }
                if tr_metrics is not None:
                    row.update(
                        {
                            "drift_pre_max": tr_metrics["drift_pre_max"],
                            "drift_pre_p99": tr_metrics["drift_pre_p99"],
                            "drift_post_max": tr_metrics["drift_post_max"],
                            "drift_post_p99": tr_metrics["drift_post_p99"],
                            "trust_region_scale": tr_metrics["trust_region_scale"],
                        }
                    )

                train_logger.log(row)
                global_step += 1

            val_metrics = evaluate(model, loaders["val"], device)
            last_val_metrics = dict(val_metrics)
            eval_logger.log({"epoch": epoch, "split": "val", **val_metrics})

            ckpt_payload = {
                "epoch": epoch,
                "global_step": global_step,
                "best_val_acc": best_val_acc,
                "loss_history": loss_history[-5000:],
                "collapse_streak": collapse_tracker.streak,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if use_grad_scaler else None,
            }
            _save_checkpoint(latest_ckpt_path, ckpt_payload)
            _save_rng_state(rng_state_path)

            if args.save_best and float(val_metrics["acc"]) >= best_val_acc:
                best_val_acc = float(val_metrics["acc"])
                ckpt_payload["best_val_acc"] = best_val_acc
                _save_checkpoint(best_ckpt_path, ckpt_payload)

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break

        if last_epoch == 0:
            last_epoch = 1

        test_metrics = evaluate(model, loaders["test"], device)
        final_test_metrics = dict(test_metrics)
        eval_logger.log({"epoch": last_epoch, "split": "test", **test_metrics})

    if args.save_best and not best_ckpt_path.exists():
        # Keep mandatory output contract for checkpoint artifact.
        _save_checkpoint(best_ckpt_path, torch.load(latest_ckpt_path, map_location="cpu"))

    if pd is not None and bool(args.plot):
        train_df = pd.read_csv(run_dir / "metrics_train.csv")
        eval_df = pd.read_csv(run_dir / "metrics_eval.csv")
        plot_curves(run_dir, train_df, eval_df)

    run_finished_unix = time.time()
    run_summary = {
        "run_dir": str(run_dir),
        "task": args.task,
        "variant": args.variant,
        "seed": int(args.seed),
        "data_source": args.data_source,
        "device": str(device),
        "amp": bool(use_amp),
        "amp_dtype": str(args.amp_dtype),
        "allow_tf32": bool(args.allow_tf32),
        "cudnn_benchmark": bool(args.cudnn_benchmark),
        "torch_compile": bool(args.torch_compile),
        "torch_compile_mode": str(args.torch_compile_mode),
        "deterministic": bool(args.deterministic),
        "deterministic_strict": bool(args.deterministic_strict),
        "start_utc": run_started_utc,
        "end_utc": _utc_now_iso(),
        "start_unix": float(run_started_unix),
        "end_unix": float(run_finished_unix),
        "wall_time_sec": float(run_finished_unix - run_started_unix),
        "epochs_completed": int(last_epoch),
        "global_step": int(global_step),
        "best_val_acc": (None if best_val_acc == float("-inf") else float(best_val_acc)),
        "last_val_metrics": last_val_metrics,
        "test_metrics": final_test_metrics,
        "num_trainable_params": int(num_trainable_params),
        "num_trainable_params_base": int(num_trainable_params_base),
        "param_diff_vs_base": float(param_diff_vs_base),
        "param_budget_tolerance": float(param_budget_tolerance),
        "param_budget_enforced": bool(enforce_param_budget),
        "plots_enabled": bool(args.plot),
        "total_train_tokens": int(total_train_tokens),
        "train_tokens_per_sec": float(total_train_tokens / max(run_finished_unix - run_started_unix, 1e-8)),
        "peak_gpu_mem_alloc_mb": float(peak_gpu_mem_alloc_mb),
        "peak_gpu_mem_reserved_mb": float(peak_gpu_mem_reserved_mb),
    }
    write_json(run_dir / "run_summary.json", run_summary)

    print(f"Saved run to: {run_dir}")


if __name__ == "__main__":
    main()
