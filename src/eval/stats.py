"""
Statistical analysis tools for robotics experiment evaluation.
Computes:
- Paired bootstrap confidence intervals over trajectories (paired by trajectory & seed).
- Two One-Sided Tests (TOST) and paired bootstrap equivalence testing.
- Head x Size interaction test (two-way permutation test on ATE and failure rates).
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


def compute_paired_bootstrap_ci(
    data_a: np.ndarray,
    data_b: np.ndarray,
    num_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Computes paired bootstrap confidence interval for the difference (mean(A) - mean(B)).
    data_a and data_b must have identical lengths (paired by trajectory/seed).
    """
    rng = np.random.default_rng(seed)
    n = len(data_a)
    if n == 0 or len(data_b) != n:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0}

    diffs = data_a - data_b
    mean_diff = float(np.mean(diffs))

    # Bootstrap resampling
    boot_means = np.zeros(num_bootstrap)
    for i in range(num_bootstrap):
        sample_indices = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(diffs[sample_indices])

    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    # Two-sided empirical p-value for mean_diff != 0
    p_val = float(2.0 * min(np.mean(boot_means <= 0), np.mean(boot_means >= 0)))
    p_val = min(1.0, max(0.0, p_val))

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
    }


def tost_equivalence_test(
    sample_a: np.ndarray,
    sample_ref: np.ndarray,
    margin_rel: float = 0.15,  # Equivalence margin: within 15% of reference
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Tests whether model A is equivalent to reference (e.g. ResNet-50) within margin delta.
    H01: diff <= -delta or H02: diff >= +delta.
    If both one-sided nulls are rejected, equivalence is established.
    """
    ref_mean = np.mean(sample_ref)
    delta = margin_rel * max(1e-4, ref_mean)

    diff = sample_a - sample_ref
    mean_diff = np.mean(diff)
    se = np.std(diff, ddof=1) / np.sqrt(len(diff))

    if se < 1e-9:
        return {"is_equivalent": True, "margin_delta": float(delta), "p_tost": 0.0}

    # t-statistics for lower and upper tests
    t_lower = (mean_diff - (-delta)) / se
    t_upper = (delta - mean_diff) / se

    from scipy import stats
    df = len(diff) - 1
    p_lower = 1.0 - stats.t.cdf(t_lower, df=df)
    p_upper = 1.0 - stats.t.cdf(t_upper, df=df)

    p_tost = max(p_lower, p_upper)
    is_equivalent = bool(p_tost < alpha)

    return {
        "is_equivalent": is_equivalent,
        "mean_diff": float(mean_diff),
        "margin_delta": float(delta),
        "p_tost": float(p_tost),
    }


def permutation_test_interaction(
    df: pd.DataFrame,
    metric: str = "ate_rmse",
    factor_head: str = "head",
    factor_size: str = "backbone",
    num_permutations: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Two-way permutation test testing the significance of Head x Size interaction on metric.
    Computes interaction F-statistic / mean squared effect and compares against permuted null.
    """
    rng = np.random.default_rng(seed)
    
    # Calculate empirical interaction effect
    def compute_interaction_stat(data: pd.DataFrame) -> float:
        # Cell means
        cell_means = data.groupby([factor_head, factor_size])[metric].mean().unstack()
        grand_mean = data[metric].mean()
        row_means = data.groupby(factor_head)[metric].mean()
        col_means = data.groupby(factor_size)[metric].mean()

        # Interaction matrix: mu_ij - mu_i. - mu_.j + mu_..
        inter = cell_means.values - row_means.values[:, None] - col_means.values[None, :] + grand_mean
        return float(np.sum(inter**2))

    obs_stat = compute_interaction_stat(df)

    # Permutation under null
    perm_stats = np.zeros(num_permutations)
    vals = df[metric].values.copy()
    for i in range(num_permutations):
        shuffled = rng.permutation(vals)
        df_perm = df.copy()
        df_perm[metric] = shuffled
        perm_stats[i] = compute_interaction_stat(df_perm)

    p_interaction = float(np.mean(perm_stats >= obs_stat))

    return {
        "observed_interaction_stat": obs_stat,
        "p_value_interaction": p_interaction,
        "is_significant": bool(p_interaction < 0.05),
    }
