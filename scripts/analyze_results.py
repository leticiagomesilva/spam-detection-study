"""
Statistical analysis of experiment results.

Reads results/metrics.csv and produces results/statistical_analysis.json
and summary plots in results/figures/.

Analysis
--------
1. Descriptive statistics per scenario (mean, std, min, max of F1).
2. Paired permutation test (bilateral) on F1 scores:
     H0: E[F1_additive - F1_scaled_dot_product] = 0
     H1: E[F1_additive - F1_scaled_dot_product] ≠ 0
   Method: for each permutation, randomly swap the scenario labels within
   each pair (equivalent to flipping the sign of each difference with
   probability 0.5), then compute the mean difference.  The p-value is the
   proportion of permutations where |mean| >= |observed mean|.
3. 95 % bootstrap confidence interval on the mean paired difference.
4. Interpretation of effect magnitude.

Limitations (documented)
------------------------
- The 30 repetitions share a single train/val/test split.  They measure
  variability due to random initialisation and batch ordering, not independent
  draws from the population of messages.  This limits generalisation claims.
- Not rejecting H0 does not prove equivalence of the two approaches.
- A statistically significant difference does not necessarily imply practical
  relevance; the magnitude must be considered alongside the p-value.

Usage
-----
    python scripts/analyze_results.py --run-id campaign_20241115_143022
    python scripts/analyze_results.py --run-id campaign_20241115_143022 --no-plots
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("analyze_results")


# ──────────────────────────────────────────────────────────────────────
# Statistical procedures
# ──────────────────────────────────────────────────────────────────────

def paired_permutation_test(
    differences: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> dict:
    """Bilateral paired permutation test by sign flipping.

    For each permutation, each difference d_i is multiplied by a random sign
    (+1 or -1), equivalent to swapping scenario labels within pair i.

    Args:
        differences   : array of shape (n_pairs,), d_i = F1_add_i - F1_sdp_i.
        n_permutations: number of permutations.
        seed          : RNG seed for reproducibility.

    Returns:
        dict with 'observed_mean', 'p_value', and 'n_permutations'.
    """
    rng = np.random.default_rng(seed)
    observed = float(np.mean(differences))
    n = len(differences)

    count_extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_mean = float(np.mean(differences * signs))
        if abs(perm_mean) >= abs(observed):
            count_extreme += 1

    p_value = count_extreme / n_permutations
    return {
        "observed_mean_difference": observed,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


def bootstrap_ci(
    differences: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for the mean paired difference.

    Args:
        differences: array of shape (n_pairs,).
        n_bootstrap: number of bootstrap resamples.
        ci         : confidence level (e.g. 0.95 for 95 %).
        seed       : RNG seed.

    Returns:
        dict with 'lower', 'upper', 'ci_level'.
    """
    rng = np.random.default_rng(seed)
    n = len(differences)
    boot_means = np.array([
        np.mean(rng.choice(differences, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {"lower": lower, "upper": upper, "ci_level": ci}


def interpret_effect(mean_diff: float, std_diff: float) -> str:
    """Rough textual description of the effect magnitude.

    Based on the absolute mean difference in F1 score.  These thresholds are
    heuristic and should not substitute for domain-specific judgment.
    """
    abs_diff = abs(mean_diff)
    if abs_diff < 0.005:
        return "negligible (< 0.005 F1)"
    elif abs_diff < 0.02:
        return "small (0.005 – 0.02 F1)"
    elif abs_diff < 0.05:
        return "moderate (0.02 – 0.05 F1)"
    else:
        return f"large (>= 0.05 F1, |Δ| = {abs_diff:.4f})"


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_f1_distributions(df_add, df_sdp, fig_dir: Path, run_id: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box + strip plot
    df_plot = pd.concat([
        df_add.assign(scenario="Additive"),
        df_sdp.assign(scenario="Scaled Dot-Product"),
    ])
    sns.boxplot(data=df_plot, x="scenario", y="f1_spam", ax=axes[0], width=0.4)
    sns.stripplot(data=df_plot, x="scenario", y="f1_spam", ax=axes[0],
                  color="black", alpha=0.4, jitter=True, size=4)
    axes[0].set_title("F1-score (spam class) distribution")
    axes[0].set_xlabel("Attention mechanism")
    axes[0].set_ylabel("F1-score")

    # Paired difference histogram
    diffs = df_add["f1_spam"].values - df_sdp["f1_spam"].values
    axes[1].hist(diffs, bins=15, edgecolor="white", color="steelblue", alpha=0.8)
    axes[1].axvline(x=0, color="red", linestyle="--", label="Zero")
    axes[1].axvline(x=float(np.mean(diffs)), color="orange", linestyle="-",
                    label=f"Mean Δ = {np.mean(diffs):.4f}")
    axes[1].set_title("Paired difference in F1 (Additive − Scaled Dot-Product)")
    axes[1].set_xlabel("ΔF1")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.suptitle(f"Run: {run_id}", fontsize=11)
    fig.tight_layout()

    out_path = fig_dir / "f1_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", out_path)


def plot_metrics_summary(df_add, df_sdp, fig_dir: Path, run_id: str):
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_cols = ["f1_spam", "precision_spam", "recall_spam", "accuracy"]
    add_means = [df_add[m].mean() for m in metrics_cols]
    sdp_means = [df_sdp[m].mean() for m in metrics_cols]
    add_stds  = [df_add[m].std() for m in metrics_cols]
    sdp_stds  = [df_sdp[m].std() for m in metrics_cols]

    x = np.arange(len(metrics_cols))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, add_means, width, yerr=add_stds,
                   label="Additive", capsize=4, alpha=0.8)
    bars2 = ax.bar(x + width/2, sdp_means, width, yerr=sdp_stds,
                   label="Scaled Dot-Product", capsize=4, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(["F1 (spam)", "Precision (spam)", "Recall (spam)", "Accuracy"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (mean ± std over 30 seeds)")
    ax.set_title(f"Metrics comparison — Run: {run_id}")
    ax.legend()
    fig.tight_layout()

    out_path = fig_dir / "metrics_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", out_path)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse experiment results.")
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID to analyse (must exist in results/metrics.csv).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating figures.",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "experiment.yaml"),
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    sta_cfg = config["statistical_analysis"]
    alpha   = sta_cfg["significance_level"]

    # Load metrics
    metrics_path = REPO_ROOT / "results" / "metrics.csv"
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        logger.error("No metrics found at %s.  Run experiments first.", metrics_path)
        sys.exit(1)

    df = pd.read_csv(metrics_path)
    df_run = df[df["run_id"] == args.run_id]

    if df_run.empty:
        logger.error("Run ID '%s' not found in metrics CSV.", args.run_id)
        logger.info("Available run IDs: %s", df["run_id"].unique().tolist())
        sys.exit(1)

    df_add = df_run[df_run["scenario"] == "additive"].sort_values("seed").reset_index(drop=True)
    df_sdp = df_run[df_run["scenario"] == "scaled_dot_product"].sort_values("seed").reset_index(drop=True)

    if len(df_add) != len(df_sdp):
        logger.error(
            "Unequal number of results per scenario: additive=%d, scaled=%d. "
            "Ensure all pairs are complete before analysing.",
            len(df_add), len(df_sdp),
        )
        sys.exit(1)

    logger.info("Analysing %d pairs for run '%s'.", len(df_add), args.run_id)

    # Verify seeds match pair-wise
    if not (df_add["seed"].values == df_sdp["seed"].values).all():
        logger.error("Seed mismatch between scenarios — results may not be paired correctly.")
        sys.exit(1)

    # ── Descriptive statistics ────────────────────────────────────────
    def describe(series: pd.Series) -> dict:
        return {
            "mean": float(series.mean()),
            "std":  float(series.std(ddof=1)),
            "min":  float(series.min()),
            "max":  float(series.max()),
        }

    stats_add = describe(df_add["f1_spam"])
    stats_sdp = describe(df_sdp["f1_spam"])

    # ── Paired differences ────────────────────────────────────────────
    differences = df_add["f1_spam"].values - df_sdp["f1_spam"].values

    # ── Permutation test ──────────────────────────────────────────────
    perm_result = paired_permutation_test(
        differences,
        n_permutations=sta_cfg["n_permutations"],
        seed=sta_cfg["seed"],
    )

    # ── Bootstrap CI ─────────────────────────────────────────────────
    ci_result = bootstrap_ci(
        differences,
        n_bootstrap=sta_cfg["bootstrap_n_samples"],
        ci=sta_cfg["bootstrap_ci"],
        seed=sta_cfg["seed"],
    )

    # ── Decision ─────────────────────────────────────────────────────
    p_value   = perm_result["p_value"]
    reject_h0 = p_value < alpha
    decision  = "Reject H0" if reject_h0 else "Fail to reject H0"
    effect    = interpret_effect(perm_result["observed_mean_difference"], float(np.std(differences, ddof=1)))

    # ── Assemble output ───────────────────────────────────────────────
    analysis = {
        "run_id":      args.run_id,
        "n_pairs":     int(len(df_add)),
        "scenarios": {
            "additive": {
                "n": len(df_add),
                "f1_spam": stats_add,
                "train_time_s": describe(df_add["train_time_s"]),
            },
            "scaled_dot_product": {
                "n": len(df_sdp),
                "f1_spam": stats_sdp,
                "train_time_s": describe(df_sdp["train_time_s"]),
            },
        },
        "paired_difference": {
            "definition": "F1_additive - F1_scaled_dot_product, per pair",
            "differences": differences.tolist(),
            "mean_diff": perm_result["observed_mean_difference"],
            "std_diff": float(np.std(differences, ddof=1)),
        },
        "permutation_test": {
            **perm_result,
            "alternative": "two-sided",
            "method": "paired sign-swap permutation test",
        },
        "bootstrap_ci_95": ci_result,
        "statistical_decision": {
            "alpha":     alpha,
            "p_value":   p_value,
            "reject_h0": reject_h0,
            "decision":  decision,
            "effect_magnitude": effect,
        },
        "limitations": [
            "The 30 pairs share a single train/val/test split; this measures "
            "variability of training, not independent population samples.",
            "Failing to reject H0 does not prove the approaches are equivalent.",
            "A statistically significant result must be evaluated alongside "
            "the magnitude of the difference.",
        ],
    }

    # Save JSON
    out_path = REPO_ROOT / "results" / "statistical_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(analysis, fh, indent=2)
    logger.info("Statistical analysis saved to %s", out_path)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Statistical Analysis — Run: {args.run_id}")
    print("=" * 60)
    print(f"  Additive     F1: {stats_add['mean']:.4f} ± {stats_add['std']:.4f}")
    print(f"  Scaled DP    F1: {stats_sdp['mean']:.4f} ± {stats_sdp['std']:.4f}")
    print(f"  Mean Δ F1   : {perm_result['observed_mean_difference']:+.4f}")
    print(f"  95% CI      : [{ci_result['lower']:+.4f}, {ci_result['upper']:+.4f}]")
    print(f"  p-value     : {p_value:.4f}")
    print(f"  Decision    : {decision}  (α = {alpha})")
    print(f"  Effect      : {effect}")
    print("=" * 60)

    # Plots
    if not args.no_plots:
        fig_dir = REPO_ROOT / "results" / "figures" / args.run_id
        try:
            plot_f1_distributions(df_add, df_sdp, fig_dir, args.run_id)
            plot_metrics_summary(df_add, df_sdp, fig_dir, args.run_id)
        except ImportError as e:
            logger.warning("Plotting skipped (%s). Install matplotlib and seaborn.", e)


if __name__ == "__main__":
    main()
