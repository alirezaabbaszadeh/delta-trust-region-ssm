from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    lower: float
    upper: float
    n: int


@dataclass(frozen=True)
class WilcoxonResult:
    n: int
    w_plus: float
    w_minus: float
    statistic: float
    p_value: float
    rank_biserial: float


def _to_float_list(values) -> list[float]:
    return [float(v) for v in values]


def bootstrap_mean_ci(
    values,
    *,
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    xs = _to_float_list(values)
    n = len(xs)
    if n == 0:
        return BootstrapCI(mean=float("nan"), lower=float("nan"), upper=float("nan"), n=0)

    if n == 1:
        x = xs[0]
        return BootstrapCI(mean=x, lower=x, upper=x, n=1)

    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(int(n_resamples)):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        stats.append(sum(sample) / n)

    stats.sort()
    alpha = 1.0 - float(ci)
    lo_idx = max(0, min(len(stats) - 1, int(math.floor((alpha / 2.0) * len(stats)))))
    hi_idx = max(0, min(len(stats) - 1, int(math.ceil((1.0 - alpha / 2.0) * len(stats)) - 1)))

    mean = sum(xs) / n
    return BootstrapCI(mean=mean, lower=stats[lo_idx], upper=stats[hi_idx], n=n)


def _rankdata_average(values: list[float]) -> list[float]:
    pairs = sorted(enumerate(values), key=lambda kv: kv[1])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilcoxon_signed_rank(x, y) -> WilcoxonResult:
    xs = _to_float_list(x)
    ys = _to_float_list(y)
    if len(xs) != len(ys):
        raise ValueError("x and y must have equal length for paired Wilcoxon test.")

    diffs = [a - b for a, b in zip(xs, ys)]
    diffs = [d for d in diffs if d != 0.0]
    n = len(diffs)
    if n == 0:
        return WilcoxonResult(n=0, w_plus=0.0, w_minus=0.0, statistic=0.0, p_value=1.0, rank_biserial=0.0)

    abs_diffs = [abs(d) for d in diffs]
    ranks = _rankdata_average(abs_diffs)

    w_plus = sum(r for d, r in zip(diffs, ranks) if d > 0)
    w_minus = sum(r for d, r in zip(diffs, ranks) if d < 0)
    statistic = min(w_plus, w_minus)

    total_rank = sum(ranks)
    if n <= 20:
        deviations = []
        for signs in itertools.product([0, 1], repeat=n):
            w = 0.0
            for flag, rank in zip(signs, ranks):
                if flag == 1:
                    w += rank
            deviations.append(abs(w - total_rank / 2.0))
        observed = abs(w_plus - total_rank / 2.0)
        extreme = sum(1 for d in deviations if d >= observed - 1e-12)
        p_value = float(extreme) / float(len(deviations))
    else:
        mean_w = n * (n + 1) / 4.0

        tie_counts: dict[float, int] = {}
        for value in abs_diffs:
            tie_counts[value] = tie_counts.get(value, 0) + 1
        tie_corr = sum(t * (t + 1) * (2 * t + 1) for t in tie_counts.values() if t > 1)
        var_w = (n * (n + 1) * (2 * n + 1) - tie_corr) / 24.0

        if var_w <= 0:
            p_value = 1.0
        else:
            z = (abs(w_plus - mean_w) - 0.5) / math.sqrt(var_w)
            p_value = max(0.0, min(1.0, 2.0 * (1.0 - _normal_cdf(abs(z)))))

    denom = n * (n + 1) / 2.0
    rank_biserial = 0.0 if denom == 0 else (w_plus - w_minus) / denom

    return WilcoxonResult(
        n=n,
        w_plus=float(w_plus),
        w_minus=float(w_minus),
        statistic=float(statistic),
        p_value=float(p_value),
        rank_biserial=float(rank_biserial),
    )


def holm_bonferroni_correction(items: list[tuple[str, float]], alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    if not items:
        return {}

    sorted_items = sorted(items, key=lambda kv: kv[1])
    m = len(sorted_items)

    adjusted: dict[str, dict[str, float | bool]] = {}
    prev_adj = 0.0
    for i, (name, p) in enumerate(sorted_items, start=1):
        factor = m - i + 1
        p_adj = min(1.0, max(prev_adj, p * factor))
        prev_adj = p_adj
        adjusted[name] = {
            "p_raw": float(p),
            "p_holm": float(p_adj),
            "reject_h0": bool(p_adj < alpha),
        }

    return adjusted
