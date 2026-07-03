#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_json, sha256_file, write_json, write_text


TASKS = ["listops", "text", "pathfinder"]
CORE_VARIANTS = ["base", "fixed_delta", "dtr", "dtrl"]
METHOD_VARIANTS = ["dtr", "dtrl"]
DISPLAY = {"listops": "ListOps", "text": "Text", "pathfinder": "Pathfinder", "base": "Base", "fixed_delta": "Fixed-$\\Delta$", "dtr": "DTR", "dtrl": "DTRL"}
COLORS = {"base": "#000000", "fixed_delta": "#D55E00", "dtr": "#0072B2", "dtrl": "#009E73", "pre": "#D55E00", "post": "#0072B2"}
LINESTYLES = {"base": "-", "fixed_delta": "--", "dtr": "-.", "dtrl": ":"}
MARKERS = {"fixed_delta": "s", "dtr": "o", "dtrl": "^"}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, stem: Path, *, metadata: dict[str, str]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), metadata=metadata)
    fig.savefig(stem.with_suffix(".png"), dpi=320, metadata=metadata)
    plt.close(fig)


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    if "step" in df.columns:
        return df.drop_duplicates(subset=["step"], keep="last").copy()
    return df.copy()


def _train(snapshot: Path, task: str, variant: str, seed: int) -> pd.DataFrame:
    return _dedup(pd.read_csv(snapshot / "runs" / task / variant / f"seed_{seed}" / "metrics_train.csv"))


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clean = np.sort(values[np.isfinite(values)])
    return clean, np.arange(1, len(clean) + 1) / len(clean) if len(clean) else np.array([])


def _write_method_tikz(path: Path) -> None:
    content = r"""%% AUTO-GENERATED: tools/render_figures.py
\resizebox{\linewidth}{!}{%
\begin{tikzpicture}[
  font=\small,
  node distance=7mm and 8mm,
  box/.style={draw, rounded corners=1.5pt, align=center, inner sep=4pt, minimum height=10mm, text width=25mm},
  decision/.style={draw, diamond, aspect=2.2, align=center, inner sep=2pt, text width=23mm},
  arrow/.style={-Latex, line width=0.7pt}
]
\node[box] (current) {current parameters\\$\theta_t$};
\node[box, right=of current] (proposal) {optimizer proposal\\$\theta_{t+1}^{\star}$};
\node[box, right=of proposal] (delta) {recompute\\$\Delta_t,\Delta_{t+1}^{\star}$};
\node[box, right=of delta] (measure) {measure realized drift\\$d_t=\|\Delta_{t+1}^{\star}-\Delta_t\|_{\infty}$};
\node[box, below=12mm of delta] (radius) {radius\\$\varepsilon$ (DTR) or $\varepsilon(L)$ (DTRL)};
\node[decision, right=of radius] (check) {$d_t\leq\varepsilon(L)$?};
\node[box, right=of check] (search) {bounded search for\\largest accepted scale $\rho$};
\node[box, below=12mm of search] (accept) {accept\\$\theta_t+\rho(\theta_{t+1}^{\star}-\theta_t)$\\for $\Delta$ parameters};
\node[box, left=of accept] (log) {log pre/post drift,\\radius, scale, violations};

\draw[arrow] (current) -- (proposal);
\draw[arrow] (proposal) -- (delta);
\draw[arrow] (delta) -- (measure);
\draw[arrow] (measure) |- (check);
\draw[arrow] (radius) -- (check);
\draw[arrow] (check) -- node[above]{no} (search);
\draw[arrow] (search) -- (accept);
\draw[arrow] (check) |- node[pos=0.18,right]{yes: $\rho=1$} (accept);
\draw[arrow] (accept) -- (log);
\end{tikzpicture}%
}
"""
    write_text(path, content)


def _forest(results: dict[str, Any], stem: Path, metadata: dict[str, str]) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), sharex=True, sharey=True, constrained_layout=True)
    ypos = np.arange(3)
    for ax, task in zip(axes, TASKS):
        rows = [row for row in results["comparisons"] if row["task"] == task]
        for y, row in zip(ypos, rows):
            mean = row["difference_summary"]["mean"]
            low, high = row["difference_summary"]["ci95_boot"]
            variant = row["variant"]
            ax.errorbar(
                mean,
                y,
                xerr=[[mean - low], [high - mean]],
                fmt=MARKERS[variant],
                color=COLORS[variant],
                markerfacecolor="white",
                markeredgewidth=1.2,
                capsize=2.5,
                linewidth=1.2,
            )
        ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
        ax.set_title(DISPLAY[task])
        ax.grid(axis="x", color="#d8d8d8", linewidth=0.5)
        ax.set_xlim(-0.067, 0.018)
        ax.set_xlabel("accuracy difference vs. base")
        ax.set_yticks(ypos, [DISPLAY[variant] for variant in ["fixed_delta", "dtr", "dtrl"]])
        ax.invert_yaxis()
    _save(fig, stem, metadata=metadata)


def _drift_ecdf(snapshot: Path, stem: Path, metadata: dict[str, str]) -> None:
    _style()
    fig, axes = plt.subplots(3, 2, figsize=(7.25, 7.0), sharex=False, sharey=True, constrained_layout=True)
    for row_index, task in enumerate(TASKS):
        for col_index, variant in enumerate(METHOD_VARIANTS):
            ax = axes[row_index, col_index]
            pre_all: list[np.ndarray] = []
            post_all: list[np.ndarray] = []
            run_count = 0
            active_steps = 0
            for seed in range(5):
                df = _train(snapshot, task, variant, seed).dropna(
                    subset=["eps_delta", "drift_pre_max", "drift_post_max", "trust_region_scale"]
                )
                active = df["trust_region_scale"].astype(float) < 1.0 - 1e-6
                if active.any():
                    run_count += 1
                    active_steps += int(active.sum())
                    eps = df.loc[active, "eps_delta"].to_numpy(float)
                    pre_all.append(df.loc[active, "drift_pre_max"].to_numpy(float) / eps)
                    post_all.append(df.loc[active, "drift_post_max"].to_numpy(float) / eps)
            if pre_all:
                for values, label, color, style in [
                    (np.concatenate(pre_all), "optimizer proposal", COLORS["pre"], "--"),
                    (np.concatenate(post_all), "accepted update", COLORS["post"], "-"),
                ]:
                    x, y = _ecdf(values)
                    if len(x) == 1:
                        ax.plot(x, y, marker="o", markersize=4, label=label, color=color, linestyle="none")
                    else:
                        ax.step(x, y, where="post", label=label, color=color, linestyle=style, linewidth=1.35)
            else:
                ax.text(0.5, 0.5, "No activated steps", transform=ax.transAxes, ha="center", va="center")
            ax.axvline(1.0, color="#555555", linewidth=0.8, linestyle=":")
            ax.set_title(f"{DISPLAY[task]}: {DISPLAY[variant]}")
            ax.set_xlabel("drift / radius on activated steps")
            ax.set_ylabel("empirical CDF")
            ax.grid(color="#dddddd", linewidth=0.45)
            ax.text(0.98, 0.04, f"{run_count} runs; {active_steps} steps", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
            if row_index == 0 and col_index == 0:
                ax.legend(loc="upper right", frameon=False)
    _save(fig, stem, metadata=metadata)


def _binned_loss(df: pd.DataFrame, bins: int = 120) -> tuple[np.ndarray, np.ndarray]:
    loss = df["loss"].to_numpy(float)
    progress = np.linspace(0.0, 1.0, len(loss), endpoint=True)
    edges = np.linspace(0.0, 1.0, bins + 1)
    x = (edges[:-1] + edges[1:]) / 2
    values = np.full(bins, np.nan)
    indices = np.clip(np.digitize(progress, edges) - 1, 0, bins - 1)
    for index in range(bins):
        selected = loss[indices == index]
        if len(selected):
            values[index] = np.median(selected)
    return x, values


def _training_dynamics(snapshot: Path, stem: Path, metadata: dict[str, str]) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.75), constrained_layout=True)
    for ax, task in zip(axes, TASKS):
        for variant in CORE_VARIANTS:
            trajectories = []
            x = None
            for seed in range(5):
                x, trajectory = _binned_loss(_train(snapshot, task, variant, seed))
                trajectories.append(trajectory)
            values = np.vstack(trajectories)
            median = np.nanmedian(values, axis=0)
            low = np.nanquantile(values, 0.25, axis=0)
            high = np.nanquantile(values, 0.75, axis=0)
            ax.plot(x, median, color=COLORS[variant], linestyle=LINESTYLES[variant], linewidth=1.2, label=DISPLAY[variant])
            ax.fill_between(x, low, high, color=COLORS[variant], alpha=0.10, linewidth=0)
        ax.set_title(DISPLAY[task])
        ax.set_xlabel("normalized training progress")
        ax.set_ylabel("cross-entropy loss")
        ax.grid(color="#dddddd", linewidth=0.45)
    axes[-1].legend(frameon=False, loc="upper right")
    _save(fig, stem, metadata=metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render journal figures from frozen evidence.")
    parser.add_argument("--results", type=Path, default=Path("results/results.json"))
    parser.add_argument("--snapshot", type=Path, default=Path("evidence_snapshot"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    results = load_json(args.results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in [
        "fig_collapse_rate_listops.pdf",
        "fig_drift_pre_post_listops.pdf",
        "fig_dtr_flow_tikz.tex",
    ]:
        stale = args.out_dir / stale_name
        if stale.exists():
            stale.unlink()
    metadata = {
        "Title": "Evidence-derived figure",
        "Author": "Alireza Abbaszadeh and Mohammad Hossein Moattar",
        "Subject": f"results_sha256={sha256_file(args.results)}",
    }
    _write_method_tikz(args.out_dir / "figure_01_method.tex")
    _forest(results, args.out_dir / "figure_02_accuracy_forest", metadata)
    _drift_ecdf(args.snapshot, args.out_dir / "figure_03_drift_ecdf", metadata)
    _training_dynamics(args.snapshot, args.out_dir / "figure_04_training_dynamics", metadata)
    generated = {
        "version": 1,
        "results_sha256": sha256_file(args.results),
        "snapshot_content_sha256": results["evidence_snapshot"]["content_sha256"],
        "figures": {
            "figure_01_method": {"format": "tikz", "source": "method definition"},
            "figure_02_accuracy_forest": {"format": ["pdf", "png"], "source": "results.json comparisons", "inference": "paired seed-level estimates"},
            "figure_03_drift_ecdf": {"format": ["pdf", "png"], "source": "metrics_train.csv", "scope": "activated steps pooled for visualization; inference remains run-level"},
            "figure_04_training_dynamics": {"format": ["pdf", "png"], "source": "metrics_train.csv", "aggregation": "120 progress bins per seed; median and interquartile range across five seeds"},
        },
    }
    write_json(args.out_dir / "figure_manifest.json", generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
