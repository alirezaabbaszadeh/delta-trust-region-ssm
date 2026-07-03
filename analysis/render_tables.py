#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import load_json, sha256_file, write_text


TASKS = ["listops", "text", "pathfinder"]
CORE_VARIANTS = ["base", "fixed_delta", "dtr", "dtrl"]
ALL_VARIANTS = ["base", "fixed_delta", "lr_only", "lado", "dtr", "dtrl", "transformer_lite", "s4d_lite"]
DISPLAY = {"listops": "ListOps", "text": "Text", "pathfinder": "Pathfinder", "base": "Base", "fixed_delta": "Fixed-$\\Delta$", "dtr": "DTR", "dtrl": "DTRL", "lr_only": "LR-only", "lado": "LADO", "transformer_lite": "Transformer-lite", "s4d_lite": "S4D-lite"}


def _fmt(value: float | None, digits: int = 4) -> str:
    return "--" if value is None else f"\\num{{{value:.{digits}f}}}"


def _pct(value: float | None, digits: int = 1) -> str:
    return "--" if value is None else f"\\num{{{100 * value:.{digits}f}}}"


def _ci(value: list[float] | None, digits: int = 4) -> str:
    return "--" if value is None else f"[{_fmt(value[0], digits)},\\,{_fmt(value[1], digits)}]"


def _header(results_sha: str) -> str:
    return f"%% AUTO-GENERATED: tools/render_tables.py; results_sha256={results_sha}\n"


def _cell(results: dict[str, Any], task: str, variant: str, seed: int = 0) -> dict[str, Any]:
    return results["cells"][f"{task}/{variant}/seed_{seed}"]


def render_protocol(results: dict[str, Any], result_sha: str) -> str:
    rows = []
    for task in TASKS:
        protocol = results["protocol"]["representative_by_task"][task]
        cfg = protocol["task_config"]
        params = _cell(results, task, "base", 0)["num_trainable_params"]
        rows.append(
            f"    {DISPLAY[task]} & {cfg['train_size']:,} & {cfg['val_size']:,} & {cfg['test_size']:,} & {cfg['seq_len']} & {cfg['batch_size']} & {cfg['epochs']} & {params:,} \\\\"
        )
    return _header(result_sha) + "\n".join(
        [
            "\\begin{table}[t]",
            "  \\centering",
            "  \\caption{Compute-controlled dataset and training configuration. All runs used official LRA splits, BF16, deterministic execution, and one NVIDIA RTX 4090.}",
            "  \\label{tab:protocol}",
            "  \\footnotesize",
            "  \\setlength{\\tabcolsep}{3.5pt}",
            "  \\begin{tabular}{lrrrrrrr}",
            "    \\toprule",
            "    Task & Train & Validation & Test & Length & Batch & Epochs & Parameters \\\\ ",
            "    \\midrule",
            *rows,
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\begin{minipage}{0.97\\linewidth}\\footnotesize",
            "  The shared B2S6 classifier used two layers, model width 128, state dimension 16, and four channel blocks. This lightweight protocol is not a reproduction of the canonical LRA leaderboard configuration. Evidence: snapshot \\EvidenceSnapshotShortID.",
            "  \\end{minipage}",
            "\\end{table}",
            "",
        ]
    )


def render_accuracy(results: dict[str, Any], result_sha: str) -> str:
    rows = []
    for task in TASKS:
        group = results["task_groups"][f"{task}_core_5seed"]["variants"]
        for index, variant in enumerate(CORE_VARIANTS):
            stats = group[variant]
            task_label = DISPLAY[task] if index == 0 else ""
            rows.append(
                f"    {task_label} & {DISPLAY[variant]} & {_fmt(stats['mean'])} & {_fmt(stats['sd'])} & {_ci(stats['ci95_boot'])} \\\\"
            )
        if task != TASKS[-1]:
            rows.append("    \\addlinespace[2pt]")
    return _header(result_sha) + "\n".join(
        [
            "\\begin{table}[t]",
            "  \\centering",
            "  \\caption{Test accuracy across five matched seeds. Intervals are 95\\% percentile-bootstrap confidence intervals for the mean (20,000 resamples).}",
            "  \\label{tab:core-accuracy}",
            "  \\begin{tabular}{llccc}",
            "    \\toprule",
            "    Task & Method & Mean & SD & 95\\% CI \\\\ ",
            "    \\midrule",
            *rows,
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\begin{minipage}{0.97\\linewidth}\\footnotesize",
            "  Each row contains seeds 0--4. The table supports estimation within the reported lightweight protocol, not benchmark leadership. Evidence: snapshot \\EvidenceSnapshotShortID.",
            "  \\end{minipage}",
            "\\end{table}",
            "",
        ]
    )


def render_effects_mechanism_compute(results: dict[str, Any], result_sha: str) -> str:
    effect_rows = []
    for row in results["comparisons"]:
        summary = row["difference_summary"]
        wil = row["wilcoxon_exact_two_sided"]
        directions = row["direction_counts"]
        effect_rows.append(
            "    "
            + " & ".join(
                [
                    DISPLAY[row["task"]],
                    DISPLAY[row["variant"]],
                    _fmt(summary["mean"]),
                    _ci(summary["ci95_boot"]),
                    _fmt(wil["p_raw"], 4),
                    _fmt(wil["p_holm_global_9"], 4),
                    _fmt(row["rank_biserial"], 3),
                    f"{directions['positive']}/{directions['zero']}/{directions['negative']}",
                ]
            )
            + " \\\\"
        )

    mechanism_rows = []
    for task in TASKS:
        for variant in ["dtr", "dtrl"]:
            mech = results["mechanism"]["tasks"][task][variant]
            overhead = results["compute"]["tasks"][task][variant]["paired_wall_time_overhead_fraction"]
            mechanism_rows.append(
                "    "
                + " & ".join(
                    [
                        DISPLAY[task],
                        DISPLAY[variant],
                        _pct(mech["activation_rate"]["mean"]),
                        _pct(mech["conditional_projection_success_rate"]["mean"]),
                        _fmt(mech["active_drift_post_over_pre_median"]["mean"], 3),
                        _pct(mech["post_violation_rate"]["mean"]),
                        _pct(overhead["mean"]),
                    ]
                )
                + " \\\\"
            )

    return _header(result_sha) + "\n".join(
        [
            "\\begin{table*}[t]",
            "  \\centering",
            "  \\caption{Paired accuracy effects, drift diagnostics, and computational cost. Accuracy differences are method minus base. Holm adjustment covers all nine comparisons. Mechanism quantities are first computed per run and then averaged across seeds; wall-time overhead uses matched seeds 1--4.}",
            "  \\label{tab:effects-mechanism}",
            "  \\scriptsize",
            "  \\setlength{\\tabcolsep}{3pt}",
            "  \\textbf{(a) Paired accuracy comparisons}\\par\\smallskip",
            "  \\begin{tabular}{llcccccc}",
            "    \\toprule",
            "    Task & Method & Mean diff. & 95\\% CI & $p$ & $p_{\\mathrm{Holm}}$ & $r_{rb}$ & $+ / 0 / -$ \\\\ ",
            "    \\midrule",
            *effect_rows,
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\par\\vspace{0.7em}",
            "  \\textbf{(b) Mechanism and cost}\\par\\smallskip",
            "  \\begin{tabular}{llccccc}",
            "    \\toprule",
            "    Task & Method & Act. (\\%) & Success (\\%) & Med. post/pre & Viol. (\\%) & Time (\\%) \\\\ ",
            "    \\midrule",
            *mechanism_rows,
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\begin{minipage}{0.98\\linewidth}\\footnotesize",
            "  Act.: activation rate; Success: conditional projection success among proposals outside the radius; Viol.: post-projection violation rate; Time: paired wall-time overhead. Exact Wilcoxon tests report effective nonzero sample sizes in the machine-readable results. Evidence: snapshot \\EvidenceSnapshotShortID.",
            "  \\end{minipage}",
            "\\end{table*}",
            "",
        ]
    )


def render_appendix(results: dict[str, Any], result_sha: str) -> str:
    rows = []
    for variant in ALL_VARIANTS:
        values = []
        for task in TASKS:
            cell = _cell(results, task, variant, 0)
            values.append("\\textsc{oom}" if cell["status"] == "oom" else _fmt(cell["test_acc"]))
        rows.append(f"    {DISPLAY[variant]} & " + " & ".join(values) + " \\\\ ")
    return _header(result_sha) + "\n".join(
        [
            "\\begin{table}[H]",
            "  \\centering",
            "  \\caption{Exploratory seed-0 test accuracy for the broader eight-variant matrix. These single-seed observations are not used for inferential claims or method ranking.}",
            "  \\label{tab:appendix-seed0}",
            "  \\begin{tabular}{lccc}",
            "    \\toprule",
            "    Method & ListOps & Text & Pathfinder \\\\ ",
            "    \\midrule",
            *rows,
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\begin{minipage}{0.97\\linewidth}\\footnotesize",
            "  Transformer-lite exhausted the available GPU memory on Text and Pathfinder under the locked configuration. This is a feasibility observation, not evidence of comparative scientific merit. Evidence: snapshot \\EvidenceSnapshotShortID.",
            "  \\end{minipage}",
            "\\end{table}",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render all manuscript tables from results.json.")
    parser.add_argument("--results", type=Path, default=Path("results/results.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("tables"))
    args = parser.parse_args()
    results = load_json(args.results)
    digest = sha256_file(args.results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "tab_protocol.tex": render_protocol(results, digest),
        "tab_core_accuracy.tex": render_accuracy(results, digest),
        "tab_effects_mechanism_compute.tex": render_effects_mechanism_compute(results, digest),
        "tab_appendix_seed0_matrix.tex": render_appendix(results, digest),
    }
    for name, content in outputs.items():
        write_text(args.out_dir / name, content)
    for stale in args.out_dir.glob("*.tex"):
        if stale.name not in outputs:
            stale.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
