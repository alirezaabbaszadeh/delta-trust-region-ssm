from __future__ import annotations

import itertools
import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable


def bootstrap_ci_percentile(
    values: list[float],
    *,
    statistic: Callable[[list[float]], float] = statistics.mean,
    n_boot: int = 20_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap_ci_percentile requires non-empty values")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    rng = random.Random(seed)
    n = len(values)
    boots: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(float(statistic(sample)))
    boots.sort()
    lo_idx = int((alpha / 2.0) * n_boot)
    hi_idx = int((1.0 - alpha / 2.0) * n_boot) - 1
    lo_idx = max(0, min(lo_idx, n_boot - 1))
    hi_idx = max(0, min(hi_idx, n_boot - 1))
    return (boots[lo_idx], boots[hi_idx])


def _rankdata_average_ties(values: list[float]) -> list[float]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank_avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank_avg
        i = j
    return ranks


@dataclass(frozen=True)
class WilcoxonExactResult:
    n: int
    w_plus: float
    p_two_sided: float
    rank_biserial: float


def wilcoxon_signed_rank_exact(diffs: Iterable[float]) -> WilcoxonExactResult:
    diffs = [float(d) for d in diffs if float(d) != 0.0]
    n = len(diffs)
    if n == 0:
        return WilcoxonExactResult(n=0, w_plus=0.0, p_two_sided=1.0, rank_biserial=0.0)

    abs_vals = [abs(d) for d in diffs]
    ranks = _rankdata_average_ties(abs_vals)

    w_plus_obs = sum(r for d, r in zip(diffs, ranks) if d > 0)
    total_rank = float(sum(ranks))
    mean = total_rank / 2.0
    dist = {}
    for signs in itertools.product([0, 1], repeat=n):
        w = 0.0
        for s, r in zip(signs, ranks):
            if s:
                w += r
        dist[w] = dist.get(w, 0) + 1

    threshold = abs(w_plus_obs - mean) - 1e-12
    extreme = sum(count for w, count in dist.items() if abs(w - mean) >= threshold)
    p_two = extreme / (2**n)
    p_two = max(0.0, min(1.0, p_two))

    rank_biserial = (2.0 * w_plus_obs - total_rank) / total_rank
    if math.isnan(rank_biserial):
        rank_biserial = 0.0

    return WilcoxonExactResult(
        n=n,
        w_plus=w_plus_obs,
        p_two_sided=p_two,
        rank_biserial=rank_biserial,
    )

def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjustment.

    Returns adjusted p-values in the original order.
    """
    m = len(p_values)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: p_values[i])
    adj = [0.0] * m

    # Step-down adjustment.
    for k, i in enumerate(order):
        adj[i] = min(1.0, float((m - k) * p_values[i]))

    # Enforce monotonicity in sorted order.
    prev = 0.0
    for i in order:
        if adj[i] < prev:
            adj[i] = prev
        prev = adj[i]

    return adj
