from __future__ import annotations

from src.utils.statistics import bootstrap_mean_ci, holm_bonferroni_correction, wilcoxon_signed_rank


def test_bootstrap_ci_constant_values() -> None:
    ci = bootstrap_mean_ci([1.0, 1.0, 1.0, 1.0], n_resamples=2000, seed=0)
    assert ci.mean == 1.0
    assert ci.lower == 1.0
    assert ci.upper == 1.0


def test_wilcoxon_and_effect_size_on_controlled_data() -> None:
    x = [0.9, 0.88, 0.91, 0.87, 0.89]
    y = [0.8, 0.79, 0.81, 0.78, 0.8]
    res = wilcoxon_signed_rank(x, y)

    assert res.n == 5
    assert res.p_value <= 0.0625 + 1e-9
    assert res.rank_biserial > 0


def test_holm_bonferroni_returns_adjusted_pvalues() -> None:
    items = [("a", 0.01), ("b", 0.03), ("c", 0.2)]
    corrected = holm_bonferroni_correction(items, alpha=0.05)

    assert set(corrected.keys()) == {"a", "b", "c"}
    assert corrected["a"]["p_holm"] <= corrected["b"]["p_holm"]
