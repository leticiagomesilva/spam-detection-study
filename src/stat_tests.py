"""
Statistical procedures for the paired comparison of the two scenarios.

Decision rule (pre-specified)
-----------------------------
For each metric, the paired differences d_i = A_i - B_i are computed
(A = additive attention, B = scaled dot-product attention).

1. Shapiro-Wilk test on d (H0: d is normally distributed).
2. If normality is NOT rejected (p >= alpha)  -> paired Student t-test.
   If normality IS rejected (p < alpha)        -> Wilcoxon signed-rank test.
   The selected test gives the official decision about H0.
3. The paired sign-flip permutation test and the bootstrap CI of the mean
   difference are reported as robustness checks (both distribution-free).
4. Effect sizes: Cohen's d_z (mean / sd of differences) and the matched-pairs
   rank-biserial correlation (effect size associated with Wilcoxon).
5. If H0 is rejected, the direction of the mean difference indicates which
   scenario performed better (higher is better for quality metrics, lower is
   better for training time).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


# ──────────────────────────────────────────────────────────────────────
# Distribution-free procedures
# ──────────────────────────────────────────────────────────────────────

def paired_permutation_test(
    differences: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> dict:
    """Bilateral paired permutation test by sign flipping.

    Each difference d_i is multiplied by a random sign (+1 or -1), which is
    equivalent to swapping scenario labels within pair i.  The p-value is the
    proportion of permutations with |mean| >= |observed mean|.
    """
    d = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    observed = float(np.mean(d))

    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(d)))
    perm_means = (signs * d).mean(axis=1)
    # Small tolerance so that floating-point noise does not hide ties
    count_extreme = int(np.sum(np.abs(perm_means) >= abs(observed) - 1e-12))

    return {
        "test": "paired sign-flip permutation test",
        "observed_mean_difference": observed,
        "p_value": count_extreme / n_permutations,
        "n_permutations": n_permutations,
    }


def bootstrap_ci(
    differences: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap confidence interval for the mean paired difference."""
    d = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_bootstrap, len(d)))
    boot_means = d[idx].mean(axis=1)
    alpha = 1.0 - ci
    return {
        "lower": float(np.percentile(boot_means, 100 * alpha / 2)),
        "upper": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
        "ci_level": ci,
    }


# ──────────────────────────────────────────────────────────────────────
# Classical tests
# ──────────────────────────────────────────────────────────────────────

def shapiro_wilk(differences: np.ndarray) -> dict:
    """Shapiro-Wilk normality test on the paired differences."""
    d = np.asarray(differences, dtype=float)
    if len(d) < 3:
        return {"test": "Shapiro-Wilk", "statistic": None, "p_value": None,
                "note": "n < 3: test not applicable"}
    if np.allclose(d, d[0]):
        return {"test": "Shapiro-Wilk", "statistic": None, "p_value": None,
                "note": "all differences identical: test not applicable"}
    res = stats.shapiro(d)
    return {"test": "Shapiro-Wilk", "statistic": float(res.statistic),
            "p_value": float(res.pvalue)}


def paired_t_test(a: np.ndarray, b: np.ndarray, ci: float = 0.95) -> dict:
    """Two-sided paired Student t-test with CI for the mean difference."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    if np.allclose(d, d[0]):
        return {"test": "paired t-test", "statistic": None, "p_value": 1.0 if d[0] == 0 else 0.0,
                "df": len(d) - 1, "ci_lower": float(d[0]), "ci_upper": float(d[0]),
                "note": "zero variance in differences"}
    res = stats.ttest_rel(a, b)
    interval = res.confidence_interval(confidence_level=ci)
    return {
        "test": "paired t-test",
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "df": int(res.df),
        "ci_lower": float(interval.low),
        "ci_upper": float(interval.high),
        "ci_level": ci,
    }


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank test (zero differences discarded)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n_nonzero = int(np.sum(d != 0))
    if n_nonzero == 0:
        return {"test": "Wilcoxon signed-rank", "statistic": None, "p_value": 1.0,
                "n_nonzero": 0, "note": "all differences are zero"}
    res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {
        "test": "Wilcoxon signed-rank",
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "n_nonzero": n_nonzero,
        "n_zero_discarded": int(len(d) - n_nonzero),
    }


# ──────────────────────────────────────────────────────────────────────
# Effect sizes
# ──────────────────────────────────────────────────────────────────────

def cohens_dz(differences: np.ndarray) -> float:
    """Cohen's d_z for paired designs: mean(d) / sd(d)."""
    d = np.asarray(differences, dtype=float)
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    if sd == 0.0:
        return 0.0 if float(np.mean(d)) == 0.0 else math.copysign(math.inf, float(np.mean(d)))
    return float(np.mean(d) / sd)


def interpret_cohens_d(d: float) -> str:
    """Conventional labels (Cohen, 1988)."""
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def rank_biserial(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation, in [-1, 1].

    r = (W+ - W-) / (W+ + W-), zero differences discarded.  Positive values
    mean the first scenario tends to have larger values.
    """
    d = np.asarray(differences, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    return (w_plus - w_minus) / (w_plus + w_minus)


# ──────────────────────────────────────────────────────────────────────
# Full paired comparison
# ──────────────────────────────────────────────────────────────────────

def compare_paired(
    a: np.ndarray,
    b: np.ndarray,
    name_a: str,
    name_b: str,
    alpha: float = 0.05,
    higher_is_better: bool = True,
    n_permutations: int = 10_000,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Run the full pre-specified paired comparison between scenarios a and b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Paired samples must have the same shape: {a.shape} vs {b.shape}")
    d = a - b

    normality = shapiro_wilk(d)
    t_res = paired_t_test(a, b, ci=ci)
    w_res = wilcoxon_signed_rank(a, b)
    perm = paired_permutation_test(d, n_permutations=n_permutations, seed=seed)
    boot = bootstrap_ci(d, n_bootstrap=n_bootstrap, ci=ci, seed=seed)

    normal = normality["p_value"] is None or normality["p_value"] >= alpha
    primary = t_res if normal else w_res
    p_value = primary["p_value"]
    reject_h0 = p_value < alpha

    mean_diff = float(np.mean(d))
    better = None
    if reject_h0 and mean_diff != 0:
        a_better = (mean_diff > 0) if higher_is_better else (mean_diff < 0)
        better = name_a if a_better else name_b

    dz = cohens_dz(d)

    return {
        "definition": f"d_i = {name_a}_i - {name_b}_i",
        "n_pairs": int(len(d)),
        "higher_is_better": higher_is_better,
        "descriptive": {
            name_a: _describe(a),
            name_b: _describe(b),
            "difference": _describe(d),
        },
        "normality": {**normality, "normal_at_alpha": normal},
        "paired_t_test": t_res,
        "wilcoxon": w_res,
        "permutation_test": perm,
        "bootstrap_ci": boot,
        "effect_size": {
            "cohens_dz": dz,
            "cohens_dz_label": interpret_cohens_d(dz),
            "rank_biserial": rank_biserial(d),
        },
        "decision": {
            "alpha": alpha,
            "selected_test": primary["test"],
            "selection_rule": "Shapiro-Wilk p >= alpha -> paired t-test; otherwise Wilcoxon",
            "p_value": p_value,
            "reject_h0": bool(reject_h0),
            "conclusion": "Reject H0 (accept H1)" if reject_h0 else "Fail to reject H0",
            "better_scenario": better,
            "permutation_agrees": bool((perm["p_value"] < alpha) == reject_h0),
        },
    }


def _describe(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }
