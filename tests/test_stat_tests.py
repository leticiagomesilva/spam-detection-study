"""
Tests for the paired statistical procedures (src/stat_tests.py).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.stat_tests import (  # noqa: E402
    bootstrap_ci,
    cohens_dz,
    compare_paired,
    interpret_cohens_d,
    paired_permutation_test,
    paired_t_test,
    rank_biserial,
    shapiro_wilk,
    wilcoxon_signed_rank,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ──────────────────────────────────────────────────────────────────────
# Individual procedures
# ──────────────────────────────────────────────────────────────────────

def test_paired_t_test_matches_scipy(rng):
    a = rng.normal(0.93, 0.01, 30)
    b = a - rng.normal(0.005, 0.005, 30)
    res = paired_t_test(a, b)
    ref = stats.ttest_rel(a, b)
    assert res["p_value"] == pytest.approx(ref.pvalue)
    assert res["statistic"] == pytest.approx(ref.statistic)
    assert res["df"] == 29
    assert res["ci_lower"] < np.mean(a - b) < res["ci_upper"]


def test_wilcoxon_discards_zero_differences():
    a = np.array([0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97])
    b = np.array([0.90, 0.91, 0.90, 0.92, 0.93, 0.93, 0.95, 0.95])
    res = wilcoxon_signed_rank(a, b)
    assert res["n_nonzero"] == 6
    assert res["n_zero_discarded"] == 2
    assert 0 <= res["p_value"] <= 1


def test_wilcoxon_all_zero_returns_p_one():
    a = np.full(10, 0.93)
    res = wilcoxon_signed_rank(a, a.copy())
    assert res["p_value"] == 1.0
    assert res["n_nonzero"] == 0


def test_shapiro_not_applicable_for_constant():
    res = shapiro_wilk(np.zeros(30))
    assert res["p_value"] is None


def test_shapiro_detects_non_normal(rng):
    skewed = rng.exponential(1.0, 200)
    assert shapiro_wilk(skewed)["p_value"] < 0.05


def test_permutation_identical_is_not_significant():
    res = paired_permutation_test(np.zeros(30), n_permutations=1000)
    assert res["p_value"] == 1.0


def test_permutation_strong_effect_is_significant():
    d = np.full(30, 0.01) + np.linspace(-0.001, 0.001, 30)
    res = paired_permutation_test(d, n_permutations=2000)
    assert res["p_value"] < 0.01


def test_bootstrap_ci_contains_mean(rng):
    d = rng.normal(0.01, 0.005, 30)
    ci = bootstrap_ci(d, n_bootstrap=2000)
    assert ci["lower"] < np.mean(d) < ci["upper"]


# ──────────────────────────────────────────────────────────────────────
# Effect sizes
# ──────────────────────────────────────────────────────────────────────

def test_cohens_dz_value():
    d = np.array([1.0, 2.0, 3.0])
    assert cohens_dz(d) == pytest.approx(2.0 / 1.0)


def test_cohens_dz_zero():
    assert cohens_dz(np.zeros(5)) == 0.0


@pytest.mark.parametrize("d,label", [(0.1, "negligible"), (0.3, "small"),
                                     (0.6, "medium"), (-1.2, "large")])
def test_interpret_cohens_d(d, label):
    assert interpret_cohens_d(d) == label


def test_rank_biserial_extremes():
    assert rank_biserial(np.array([0.1, 0.2, 0.3])) == pytest.approx(1.0)
    assert rank_biserial(np.array([-0.1, -0.2, 0.0])) == pytest.approx(-1.0)
    assert rank_biserial(np.zeros(4)) == 0.0


# ──────────────────────────────────────────────────────────────────────
# Full comparison / decision rule
# ──────────────────────────────────────────────────────────────────────

def test_compare_paired_detects_better_scenario(rng):
    a = rng.normal(0.95, 0.005, 30)
    b = a - 0.02 + rng.normal(0, 0.003, 30)
    res = compare_paired(a, b, "additive", "scaled", n_permutations=2000, n_bootstrap=2000)
    assert res["decision"]["reject_h0"] is True
    assert res["decision"]["better_scenario"] == "additive"
    assert res["decision"]["permutation_agrees"] is True


def test_compare_paired_lower_is_better_for_time(rng):
    a = rng.normal(60, 3, 30)   # additive slower
    b = rng.normal(30, 3, 30)
    res = compare_paired(a, b, "additive", "scaled", higher_is_better=False,
                         n_permutations=2000, n_bootstrap=2000)
    assert res["decision"]["better_scenario"] == "scaled"


def test_compare_paired_no_difference(rng):
    a = rng.normal(0.93, 0.01, 30)
    b = a + rng.normal(0, 0.002, 30)
    res = compare_paired(a, b, "additive", "scaled", n_permutations=2000, n_bootstrap=2000)
    if not res["decision"]["reject_h0"]:
        assert res["decision"]["better_scenario"] is None
        assert res["decision"]["conclusion"] == "Fail to reject H0"


def test_compare_paired_uses_wilcoxon_when_not_normal(rng):
    a = np.full(40, 0.9)
    b = a - rng.exponential(0.01, 40)   # skewed differences
    res = compare_paired(a, b, "additive", "scaled", n_permutations=1000, n_bootstrap=1000)
    assert res["normality"]["normal_at_alpha"] is False
    assert res["decision"]["selected_test"] == "Wilcoxon signed-rank"


def test_compare_paired_uses_t_when_normal(rng):
    a = rng.normal(0.93, 0.01, 30)
    b = a - rng.normal(0.003, 0.004, 30)
    res = compare_paired(a, b, "additive", "scaled", n_permutations=1000, n_bootstrap=1000)
    if res["normality"]["normal_at_alpha"]:
        assert res["decision"]["selected_test"] == "paired t-test"


def test_compare_paired_shape_mismatch():
    with pytest.raises(ValueError):
        compare_paired(np.zeros(3), np.zeros(4), "a", "b")
