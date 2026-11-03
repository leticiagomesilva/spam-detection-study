"""
Statistical analysis of experiment results.

Reads results/metrics.csv and produces results/statistical_analysis.json
and summary plots in results/figures/<run_id>/.

Analysis (see src/stat_tests.py for the pre-specified decision rule)
--------------------------------------------------------------------
Primary metric: F1-score of the spam class.
    H0: E[F1_additive - F1_scaled_dot_product] = 0
    H1: E[F1_additive - F1_scaled_dot_product] != 0

    1. Shapiro-Wilk on the paired differences.
    2. Paired t-test (if normal) or Wilcoxon signed-rank (otherwise).
    3. Permutation test + bootstrap CI as robustness checks.
    4. Effect sizes (Cohen's d_z, rank-biserial).

Secondary metrics (exploratory, same procedure, no multiplicity correction):
    precision, recall, accuracy, average precision, training time.

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
    python scripts/analyze_results.py --run-id campaign_full_30
    python scripts/analyze_results.py --run-id campaign_full_30 --no-plots
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

from src.stat_tests import compare_paired  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("analyze_results")

SCENARIO_A = "additive"
SCENARIO_B = "scaled_dot_product"
LABELS = {SCENARIO_A: "Additive", SCENARIO_B: "Scaled Dot-Product"}

PRIMARY_METRIC = "f1_spam"
# metric -> higher_is_better
SECONDARY_METRICS = {
    "precision_spam":    True,
    "recall_spam":       True,
    "accuracy":          True,
    "average_precision": True,
    "train_time_s":      False,
}


def interpret_f1_magnitude(mean_diff: float) -> str:
    """Practical magnitude of the mean F1 difference (heuristic thresholds)."""
    a = abs(mean_diff)
    if a < 0.005:
        return "negligible (< 0.005 F1)"
    if a < 0.02:
        return "small (0.005 - 0.02 F1)"
    if a < 0.05:
        return "moderate (0.02 - 0.05 F1)"
    return f"large (>= 0.05 F1, |delta| = {a:.4f})"


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_f1_distributions(df_add, df_sdp, fig_dir: Path, run_id: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df_plot = pd.concat([
        df_add.assign(scenario=LABELS[SCENARIO_A]),
        df_sdp.assign(scenario=LABELS[SCENARIO_B]),
    ])
    sns.boxplot(data=df_plot, x="scenario", y="f1_spam", ax=axes[0], width=0.4)
    sns.stripplot(data=df_plot, x="scenario", y="f1_spam", ax=axes[0],
                  color="black", alpha=0.4, jitter=True, size=4)
    axes[0].set_title("F1-score (spam class) distribution")
    axes[0].set_xlabel("Attention mechanism")
    axes[0].set_ylabel("F1-score")

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
    add_stds = [df_add[m].std() for m in metrics_cols]
    sdp_stds = [df_sdp[m].std() for m in metrics_cols]

    x = np.arange(len(metrics_cols))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, add_means, width, yerr=add_stds,
           label=LABELS[SCENARIO_A], capsize=4, alpha=0.8)
    ax.bar(x + width / 2, sdp_means, width, yerr=sdp_stds,
           label=LABELS[SCENARIO_B], capsize=4, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["F1 (spam)", "Precision (spam)", "Recall (spam)", "Accuracy"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(f"Score (mean ± std over {len(df_add)} seeds)")
    ax.set_title(f"Metrics comparison — Run: {run_id}")
    ax.legend()
    fig.tight_layout()
    out_path = fig_dir / "metrics_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", out_path)


def plot_qq_differences(df_add, df_sdp, fig_dir: Path, shapiro_p):
    """Normal Q-Q plot of the paired F1 differences (supports Shapiro-Wilk)."""
    import matplotlib.pyplot as plt
    from scipy import stats

    fig_dir.mkdir(parents=True, exist_ok=True)
    diffs = df_add["f1_spam"].values - df_sdp["f1_spam"].values
    fig, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(diffs, dist="norm", plot=ax)
    p_txt = "n/a" if shapiro_p is None else f"{shapiro_p:.4f}"
    ax.set_title(f"Normal Q-Q plot of ΔF1 (Shapiro-Wilk p = {p_txt})")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Observed ΔF1")
    fig.tight_layout()
    out_path = fig_dir / "qq_f1_differences.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", out_path)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def load_paired_runs(run_id: str):
    metrics_path = REPO_ROOT / "results" / "metrics.csv"
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        logger.error("No metrics found at %s.  Run experiments first.", metrics_path)
        sys.exit(1)

    df = pd.read_csv(metrics_path)
    df_run = df[df["run_id"] == run_id]
    if df_run.empty:
        logger.error("Run ID '%s' not found in metrics CSV.", run_id)
        logger.info("Available run IDs: %s", df["run_id"].unique().tolist())
        sys.exit(1)

    df_add = df_run[df_run["scenario"] == SCENARIO_A].sort_values("seed").reset_index(drop=True)
    df_sdp = df_run[df_run["scenario"] == SCENARIO_B].sort_values("seed").reset_index(drop=True)

    if len(df_add) != len(df_sdp):
        logger.error(
            "Unequal number of results per scenario: additive=%d, scaled=%d. "
            "Ensure all pairs are complete before analysing.",
            len(df_add), len(df_sdp),
        )
        sys.exit(1)
    if not (df_add["seed"].values == df_sdp["seed"].values).all():
        logger.error("Seed mismatch between scenarios — results may not be paired correctly.")
        sys.exit(1)
    if len(df_add) < 3:
        logger.error("At least 3 complete pairs are required (found %d).", len(df_add))
        sys.exit(1)
    return df_add, df_sdp


def main():
    # Windows consoles may default to a legacy code page (Δ, α, ± would fail)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="Analyse experiment results.")
    parser.add_argument("--run-id", required=True,
                        help="Run ID to analyse (must exist in results/metrics.csv).")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating figures.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "experiment.yaml"))
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    sta_cfg = config["statistical_analysis"]
    alpha = sta_cfg["significance_level"]

    df_add, df_sdp = load_paired_runs(args.run_id)
    logger.info("Analysing %d pairs for run '%s'.", len(df_add), args.run_id)

    def run(metric: str, higher_is_better: bool) -> dict:
        return compare_paired(
            df_add[metric].values, df_sdp[metric].values,
            name_a=SCENARIO_A, name_b=SCENARIO_B,
            alpha=alpha,
            higher_is_better=higher_is_better,
            n_permutations=sta_cfg["n_permutations"],
            n_bootstrap=sta_cfg["bootstrap_n_samples"],
            ci=sta_cfg["bootstrap_ci"],
            seed=sta_cfg["seed"],
        )

    primary = run(PRIMARY_METRIC, True)
    primary["practical_magnitude"] = interpret_f1_magnitude(
        primary["descriptive"]["difference"]["mean"]
    )
    primary["differences"] = (df_add[PRIMARY_METRIC].values - df_sdp[PRIMARY_METRIC].values).tolist()
    secondary = {m: run(m, hib) for m, hib in SECONDARY_METRICS.items()}

    analysis = {
        "run_id": args.run_id,
        "n_pairs": int(len(df_add)),
        "seeds": df_add["seed"].astype(int).tolist(),
        "hypotheses": {
            "H0": "E[F1_additive - F1_scaled_dot_product] = 0",
            "H1": "E[F1_additive - F1_scaled_dot_product] != 0",
        },
        "primary_metric": PRIMARY_METRIC,
        "primary": primary,
        "secondary_metrics_note": (
            "Exploratory; same procedure; no correction for multiple comparisons."
        ),
        "secondary": secondary,
        "limitations": [
            "The 30 pairs share a single train/val/test split; this measures "
            "variability of training, not independent population samples.",
            "Failing to reject H0 does not prove the approaches are equivalent.",
            "A statistically significant result must be evaluated alongside "
            "the magnitude of the difference.",
        ],
    }

    out_path = REPO_ROOT / "results" / "statistical_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2, ensure_ascii=False)
    logger.info("Statistical analysis saved to %s", out_path)

    print_summary(analysis)

    if not args.no_plots:
        fig_dir = REPO_ROOT / "results" / "figures" / args.run_id
        try:
            plot_f1_distributions(df_add, df_sdp, fig_dir, args.run_id)
            plot_metrics_summary(df_add, df_sdp, fig_dir, args.run_id)
            plot_qq_differences(df_add, df_sdp, fig_dir, primary["normality"]["p_value"])
        except ImportError as e:
            logger.warning("Plotting skipped (%s). Install matplotlib and seaborn.", e)


def _fmt_p(p):
    return "n/a" if p is None else f"{p:.4f}"


def print_summary(analysis: dict):
    p = analysis["primary"]
    desc = p["descriptive"]
    dec = p["decision"]
    print("\n" + "=" * 68)
    print(f"Statistical Analysis — Run: {analysis['run_id']}  ({analysis['n_pairs']} pairs)")
    print("=" * 68)
    print("Primary metric: F1 (spam)")
    print(f"  Additive        : {desc[SCENARIO_A]['mean']:.4f} ± {desc[SCENARIO_A]['std']:.4f}")
    print(f"  Scaled DP       : {desc[SCENARIO_B]['mean']:.4f} ± {desc[SCENARIO_B]['std']:.4f}")
    print(f"  Mean ΔF1        : {desc['difference']['mean']:+.4f}")
    print(f"  Shapiro-Wilk    : p = {_fmt_p(p['normality']['p_value'])} "
          f"({'normal' if p['normality']['normal_at_alpha'] else 'not normal'})")
    t = p["paired_t_test"]
    print(f"  Paired t-test   : p = {_fmt_p(t['p_value'])}  "
          f"95% CI [{t['ci_lower']:+.4f}, {t['ci_upper']:+.4f}]")
    print(f"  Wilcoxon        : p = {_fmt_p(p['wilcoxon']['p_value'])}")
    print(f"  Permutation     : p = {_fmt_p(p['permutation_test']['p_value'])}")
    b = p["bootstrap_ci"]
    print(f"  Bootstrap 95% CI: [{b['lower']:+.4f}, {b['upper']:+.4f}]")
    e = p["effect_size"]
    print(f"  Cohen's d_z     : {e['cohens_dz']:+.3f} ({e['cohens_dz_label']})   "
          f"rank-biserial: {e['rank_biserial']:+.3f}")
    print(f"  Practical size  : {p['practical_magnitude']}")
    print(f"  Selected test   : {dec['selected_test']}  (α = {dec['alpha']})")
    print(f"  DECISION        : {dec['conclusion']}"
          + (f" — better: {dec['better_scenario']}" if dec["better_scenario"] else ""))
    print(f"  Permutation test agrees: {dec['permutation_agrees']}")

    print("\nSecondary metrics (exploratory):")
    print(f"  {'metric':<18}{'additive':>10}{'scaled_dp':>11}{'Δ mean':>10}"
          f"{'test':>10}{'p':>9}  result")
    for m, r in analysis["secondary"].items():
        d = r["descriptive"]
        dd = r["decision"]
        short = "t" if dd["selected_test"] == "paired t-test" else "Wilcoxon"
        res = dd["better_scenario"] + " better" if dd["better_scenario"] else "n.s."
        print(f"  {m:<18}{d[SCENARIO_A]['mean']:>10.4f}{d[SCENARIO_B]['mean']:>11.4f}"
              f"{d['difference']['mean']:>+10.4f}{short:>10}{dd['p_value']:>9.4f}  {res}")
    print("=" * 68)


if __name__ == "__main__":
    main()
