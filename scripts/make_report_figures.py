"""
Publication figures for the IEEE report (and the slides).

Reads the results of one campaign and writes PDF (vector, for LaTeX) and PNG
(300 dpi, for slides) files to results/figures/<run_id>/report/.

Figures
-------
fig_f1_distribution   F1 (spam) per seed, box + points, per scenario
fig_f1_differences    Histogram of paired ΔF1 with mean and 95 % CI
fig_f1_paired         Paired (slope) plot: F1 of each seed in both scenarios
fig_metrics           Mean ± sd of F1, precision, recall, accuracy (dot plot)
fig_train_time        Training time per run and seconds per epoch
fig_learning_curves   Mean training / validation loss per epoch
fig_confusion         Mean confusion matrix (test set) per scenario
fig_qq                Normal Q-Q plot of ΔF1
fig_attention         Attention weights of both mechanisms on test messages
                      (requires TensorFlow and the saved checkpoints)

Usage
-----
    python scripts/make_report_figures.py --run-id campaign_full_30
    python scripts/make_report_figures.py --run-id campaign_full_30 --no-attention
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("make_report_figures")

# ── Visual identity (same as docs/apresentacao/guia_slide.html) ────────
SCEN = ["additive", "scaled_dot_product"]
LABEL = {"additive": "Additive", "scaled_dot_product": "Scaled dot-product"}
COLOR = {"additive": "#1f77b4", "scaled_dot_product": "#ff7f0e"}
INK = "#1d1d1b"
MUTED = "#6b6a64"
GRID = "#e5e3dc"

COL_W = 3.5    # IEEE single column width (in)
PAGE_W = 7.16  # IEEE double column width (in)


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(out_dir / f"{name}.{ext}", **kw)
    plt.close(fig)
    logger.info("Saved %s.{pdf,png}", out_dir / name)


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_runs(run_id: str):
    df = pd.read_csv(REPO_ROOT / "results" / "metrics.csv")
    df = df[df["run_id"] == run_id]
    runs = {s: df[df["scenario"] == s].sort_values("seed").reset_index(drop=True) for s in SCEN}
    if len(runs[SCEN[0]]) == 0 or len(runs[SCEN[0]]) != len(runs[SCEN[1]]):
        raise SystemExit(f"Incomplete or missing run '{run_id}' in metrics.csv")
    return runs


def load_histories(run_id: str):
    hist_dir = REPO_ROOT / "results" / "campaigns" / run_id / "histories"
    out = {s: [] for s in SCEN}
    for s in SCEN:
        for p in sorted(hist_dir.glob(f"{s}_seed*.json")):
            with open(p) as fh:
                out[s].append(json.load(fh))
    return out


# ──────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────

def _box_strip(ax, runs, col, rng):
    data = [runs[s][col].values for s in SCEN]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.45, patch_artist=True,
                    showfliers=False, medianprops={"color": INK, "linewidth": 1})
    for patch, s in zip(bp["boxes"], SCEN):
        patch.set_facecolor(COLOR[s] + "33")
        patch.set_edgecolor(COLOR[s])
        patch.set_linewidth(0.9)
    for el in ("whiskers", "caps"):
        for line in bp[el]:
            line.set_color(MUTED)
            line.set_linewidth(0.7)
    for i, s in enumerate(SCEN):
        x = i + rng.uniform(-0.12, 0.12, len(data[i]))
        ax.scatter(x, data[i], s=9, color=COLOR[s], edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks([0, 1], [LABEL[s] for s in SCEN])
    ax.grid(axis="x", visible=False)


def fig_f1_distribution(runs, out):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    _box_strip(ax, runs, "f1_spam", rng)
    for i, s in enumerate(SCEN):
        v = runs[s]["f1_spam"]
        ax.text(i + 0.3, v.mean(), f"{v.mean():.4f}\n±{v.std(ddof=1):.4f}",
                va="center", fontsize=6.5, color=INK)
    ax.set_ylabel("F1-score (spam)")
    ax.set_xlim(-0.5, 1.75)
    save(fig, out, "fig_f1_distribution")


def fig_f1_differences(runs, analysis, out):
    d = runs[SCEN[0]]["f1_spam"].values - runs[SCEN[1]]["f1_spam"].values
    t = analysis["primary"]["paired_t_test"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    ax.axvspan(t["ci_lower"], t["ci_upper"], color=MUTED, alpha=0.15, lw=0,
               label="95% CI of mean (t)")
    counts, _, _ = ax.hist(d, bins=12, color="#5b7aa6", edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, counts.max() * 1.6)  # headroom for the legend
    ax.axvline(0, color=INK, lw=0.8, ls="--", label="No difference")
    ax.axvline(d.mean(), color="#2b4c7e", lw=1.4, label=f"Mean ΔF1 = {d.mean():+.4f}")
    ax.set_xlabel("ΔF1 = F1(additive) − F1(scaled dot-product)")
    ax.set_ylabel("Number of seeds")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig_f1_differences")


def fig_f1_paired(runs, out):
    a = runs[SCEN[0]]["f1_spam"].values
    b = runs[SCEN[1]]["f1_spam"].values
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for ya, yb in zip(a, b):
        ax.plot([0, 1], [ya, yb], color=MUTED, lw=0.5, alpha=0.6, zorder=1)
    for i, (s, v) in enumerate(zip(SCEN, (a, b))):
        ax.scatter(np.full(len(v), i), v, s=12, color=COLOR[s], edgecolor="white",
                   linewidth=0.4, zorder=3)
    n_up = int(np.sum(a > b))
    n_down = int(np.sum(a < b))
    n_tie = int(np.sum(a == b))
    ax.set_xticks([0, 1], [LABEL[s] for s in SCEN])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("F1-score (spam)")
    ax.set_title(f"Same seed linked · additive higher in {n_up}, lower in {n_down}, tied in {n_tie}",
                 fontsize=7, color=MUTED)
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig_f1_paired")


def fig_metrics(runs, out):
    metrics = [("f1_spam", "F1 (spam)"), ("precision_spam", "Precision (spam)"),
               ("recall_spam", "Recall (spam)"), ("accuracy", "Accuracy")]
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    y = np.arange(len(metrics))[::-1]
    for k, s in enumerate(SCEN):
        off = 0.14 if k == 0 else -0.14
        means = [runs[s][m].mean() for m, _ in metrics]
        sds = [runs[s][m].std(ddof=1) for m, _ in metrics]
        ax.errorbar(means, y + off, xerr=sds, fmt="o", ms=4, color=COLOR[s],
                    ecolor=COLOR[s], elinewidth=0.9, capsize=2, label=LABEL[s])
    ax.set_yticks(y, [lbl for _, lbl in metrics])
    ax.set_xlabel("Score (mean ± sd over 30 seeds)")
    ax.legend(loc="lower left", ncol=1)
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig_metrics")


def fig_train_time(runs, out):
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.2))
    _box_strip(axes[0], runs, "train_time_s", rng)
    axes[0].set_ylabel("Training time per run (s)")
    per_epoch = {s: runs[s]["train_time_s"] / runs[s]["total_epochs"] for s in SCEN}
    tmp = {s: pd.DataFrame({"spe": per_epoch[s]}) for s in SCEN}
    _box_strip(axes[1], tmp, "spe", rng)
    axes[1].set_ylabel("Seconds per epoch")
    for ax in axes:
        ax.set_xticks([0, 1], ["Additive", "Scaled\ndot-product"])
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "fig_train_time")


def fig_learning_curves(histories, out, min_runs=5):
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.0), sharey=True)
    for ax, key, title in ((axes[0], "loss", "Training loss"),
                           (axes[1], "val_loss", "Validation loss")):
        for s in SCEN:
            curves = [h[key] for h in histories[s]]
            max_len = max(len(c) for c in curves)
            means, sds, epochs = [], [], []
            for e in range(max_len):
                vals = [c[e] for c in curves if len(c) > e]
                if len(vals) < min_runs:
                    break
                epochs.append(e + 1)
                means.append(np.mean(vals))
                sds.append(np.std(vals, ddof=1))
            means, sds = np.array(means), np.array(sds)
            ax.plot(epochs, means, color=COLOR[s], lw=1.4, label=LABEL[s])
            ax.fill_between(epochs, means - sds, means + sds, color=COLOR[s], alpha=0.15, lw=0)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].legend(loc="upper right")
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig_learning_curves")


def fig_confusion(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.85))
    for ax, s in zip(axes, SCEN):
        r = runs[s]
        mean = np.array([[r.tn.mean(), r.fp.mean()], [r.fn.mean(), r.tp.mean()]])
        sd = np.array([[r.tn.std(ddof=1), r.fp.std(ddof=1)], [r.fn.std(ddof=1), r.tp.std(ddof=1)]])
        # log scale on colour so the small off-diagonal cells remain visible
        ax.imshow(np.log10(mean + 1), cmap="Blues", vmin=0, vmax=3.2)
        for i in range(2):
            for j in range(2):
                dark = np.log10(mean[i, j] + 1) > 2.0
                ax.text(j, i, f"{mean[i, j]:.1f}\n±{sd[i, j]:.1f}", ha="center", va="center",
                        fontsize=7, color="white" if dark else INK)
        ax.set_xticks([0, 1], ["ham", "spam"])
        ax.set_yticks([0, 1], ["ham", "spam"])
        ax.set_xlabel("Predicted")
        ax.set_title(LABEL[s])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("True")
    fig.tight_layout(w_pad=1.0)
    save(fig, out, "fig_confusion")


def fig_qq(runs, analysis, out):
    d = runs[SCEN[0]]["f1_spam"].values - runs[SCEN[1]]["f1_spam"].values
    (osm, osr), (slope, intercept, _) = stats.probplot(d, dist="norm")
    p = analysis["primary"]["normality"]["p_value"]
    fig, ax = plt.subplots(figsize=(COL_W * 0.75, 2.2))
    ax.scatter(osm, osr, s=10, color="#5b7aa6", edgecolor="white", linewidth=0.4, zorder=3)
    xs = np.array([osm.min(), osm.max()])
    ax.plot(xs, slope * xs + intercept, color=INK, lw=0.8)
    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_ylabel("Observed ΔF1")
    ax.text(0.03, 0.95, f"Shapiro–Wilk p = {p:.3f}", transform=ax.transAxes,
            va="top", fontsize=7)
    save(fig, out, "fig_qq")


# ──────────────────────────────────────────────────────────────────────
# Attention heatmap
# ──────────────────────────────────────────────────────────────────────

def fig_attention(run_id: str, seed: int, out: Path, n_spam=2, n_ham=1, max_tokens=18):
    import tensorflow as tf

    from src.data import apply_split_ids, handle_duplicates, load_raw_data, load_split_ids
    from src.predict import load_artifacts
    from src.preprocessing import clean_text, encode_texts

    df = load_raw_data(REPO_ROOT / "data" / "raw" / "SMSSpamCollection")
    df, _ = handle_duplicates(df)
    split_ids = load_split_ids(REPO_ROOT / "data" / "processed" / "split_ids.json")
    _, _, df_test = apply_split_ids(df, split_ids)

    models = {}
    for s in SCEN:
        ckpt = REPO_ROOT / "checkpoints" / run_id / f"{s}_seed{seed}" / "weights.weights.h5"
        models[s], tokenizer, config = load_artifacts(
            ckpt, REPO_ROOT / "data" / "processed" / "tokenizer.pkl",
            REPO_ROOT / "configs" / "experiment.yaml")
    max_len = config["preprocessing"]["max_length"]

    def attend(model, X):
        x = model.embedding(X)
        seq, h, _ = model.lstm(x)
        _, w = model.attention([h, seq], mask=tf.not_equal(X, 0))
        prob = model(X, training=False)
        return w.numpy(), prob.numpy().ravel()

    texts = df_test["text"].tolist()
    labels = df_test["label"].tolist()
    X = encode_texts(tokenizer, [clean_text(t) for t in texts], max_len)
    X_tf = tf.constant(X, dtype=tf.int32)
    weights, probs = {}, {}
    for s in SCEN:
        weights[s], probs[s] = attend(models[s], X_tf)

    lengths = (X != 0).sum(axis=1)
    chosen = []
    for want, n in (("spam", n_spam), ("ham", n_ham)):
        k = 0
        for i, (lbl, L) in enumerate(zip(labels, lengths)):
            if lbl != want or not (8 <= L <= max_tokens):
                continue
            ok = all((probs[s][i] >= 0.5) == (want == "spam") for s in SCEN)
            if ok:
                chosen.append(i)
                k += 1
                if k == n:
                    break

    index_word = tokenizer.index_word
    fig, axes = plt.subplots(len(chosen), 1, figsize=(PAGE_W, 0.95 * len(chosen) + 0.3))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, chosen):
        L = int(lengths[i])
        toks = [index_word.get(int(t), "<OOV>") for t in X[i, :L]]
        mat = np.vstack([weights[s][i, :L] for s in SCEN])
        ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=max(0.3, mat.max()))
        ax.set_xticks(range(L), toks, rotation=40, ha="right", fontsize=7)
        ax.set_yticks([0, 1], [f"{LABEL[s]}  ·  P(spam) = {probs[s][i]:.2f}" for s in SCEN], fontsize=7)
        for r in range(2):
            j = int(np.argmax(mat[r]))
            ax.text(j, r, f"{mat[r, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if mat[r, j] > 0.2 else INK)
        ax.set_title(f"True label: {labels[i]}", loc="left", fontsize=7, color=MUTED)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout(h_pad=0.6)
    save(fig, out, "fig_attention")
    logger.info("Attention figure uses seed %d and test messages %s", seed, chosen)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate report figures.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--no-attention", action="store_true",
                        help="Skip the attention heatmap (needs TensorFlow and checkpoints).")
    parser.add_argument("--attention-seed", type=int, default=1001)
    args = parser.parse_args()

    setup_style()
    out = REPO_ROOT / "results" / "figures" / args.run_id / "report"

    runs = load_runs(args.run_id)
    with open(REPO_ROOT / "results" / "statistical_analysis.json", encoding="utf-8") as fh:
        analysis = json.load(fh)
    if analysis.get("run_id") != args.run_id:
        raise SystemExit("statistical_analysis.json belongs to another run; run analyze_results.py first.")

    fig_f1_distribution(runs, out)
    fig_f1_differences(runs, analysis, out)
    fig_f1_paired(runs, out)
    fig_metrics(runs, out)
    fig_train_time(runs, out)
    fig_confusion(runs, out)
    fig_qq(runs, analysis, out)

    histories = load_histories(args.run_id)
    if all(histories[s] for s in SCEN):
        fig_learning_curves(histories, out)
    else:
        logger.warning("Training histories not found — skipping learning curves.")

    if not args.no_attention:
        fig_attention(args.run_id, args.attention_seed, out)


if __name__ == "__main__":
    main()
