"""
Evaluation utilities for the spam-detection study.

Metrics computed per model
--------------------------
Primary:
  - F1-score (spam class, label = 1)

Complementary:
  - Precision   (spam class)
  - Recall      (spam class)
  - Accuracy    (overall)
  - Average precision  (computed from predicted probabilities via
                        sklearn.metrics.average_precision_score)
  - Confusion matrix   (stored as flattened list [TN, FP, FN, TP])
  - Training time (seconds)
  - Best epoch and total epochs

The test set must only be used for final evaluation.  It must not inform
any hyperparameter choices or model selection.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    batch_size: int = 32,
) -> dict:
    """Compute all metrics on a test partition.

    Args:
        model      : trained tf.keras.Model.
        X_test     : integer sequences (n_samples, max_length).
        y_test     : binary labels (n_samples,).
        threshold  : classification threshold (default 0.5).
        batch_size : batch size for prediction.

    Returns:
        dict with keys: f1, precision, recall, accuracy, average_precision,
                        confusion_matrix (list), probabilities (np.ndarray).
    """
    probabilities = model.predict(X_test, batch_size=batch_size, verbose=0).flatten()
    predictions = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(y_test.astype(int), predictions, labels=[0, 1])

    metrics = {
        "f1_spam":            float(f1_score(y_test,          predictions, pos_label=1, zero_division=0)),
        "precision_spam":     float(precision_score(y_test,   predictions, pos_label=1, zero_division=0)),
        "recall_spam":        float(recall_score(y_test,      predictions, pos_label=1, zero_division=0)),
        "accuracy":           float(accuracy_score(y_test.astype(int), predictions)),
        "average_precision":  float(average_precision_score(y_test, probabilities, pos_label=1)),
        "confusion_matrix":   cm.flatten().tolist(),  # [TN, FP, FN, TP]
        "probabilities":      probabilities,
        "predictions":        predictions,
        "threshold":          threshold,
    }

    logger.info(
        "Evaluation — F1: %.4f  Precision: %.4f  Recall: %.4f  Acc: %.4f  AP: %.4f",
        metrics["f1_spam"],
        metrics["precision_spam"],
        metrics["recall_spam"],
        metrics["accuracy"],
        metrics["average_precision"],
    )

    return metrics


# ──────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────

def append_metrics_row(
    metrics_csv_path: str | Path,
    run_id: str,
    scenario: str,
    seed: int,
    metrics: dict,
    train_time: float,
    best_epoch: int,
    total_epochs: int,
):
    """Append a single result row to the shared metrics CSV.

    The CSV is created with a header if it does not exist yet.  Existing rows
    from completed runs are never overwritten.

    Columns:
        run_id, scenario, seed, f1_spam, precision_spam, recall_spam,
        accuracy, average_precision, tn, fp, fn, tp,
        train_time_s, best_epoch, total_epochs
    """
    path = Path(metrics_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cm = metrics["confusion_matrix"]
    row = {
        "run_id":            run_id,
        "scenario":          scenario,
        "seed":              seed,
        "f1_spam":           metrics["f1_spam"],
        "precision_spam":    metrics["precision_spam"],
        "recall_spam":       metrics["recall_spam"],
        "accuracy":          metrics["accuracy"],
        "average_precision": metrics["average_precision"],
        "tn":                cm[0],
        "fp":                cm[1],
        "fn":                cm[2],
        "tp":                cm[3],
        "train_time_s":      round(train_time, 3),
        "best_epoch":        best_epoch,
        "total_epochs":      total_epochs,
    }

    df_row = pd.DataFrame([row])
    write_header = not path.exists() or path.stat().st_size == 0
    df_row.to_csv(path, mode="a", header=write_header, index=False)
    logger.info("Metrics appended to %s (run=%s, scenario=%s, seed=%d)", path, run_id, scenario, seed)


def save_predictions(
    output_path: str | Path,
    message_ids: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
):
    """Save per-message predictions to CSV.

    Columns: message_id, true_label, probability, predicted_label
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "message_id":      message_ids,
        "true_label":      y_true.astype(int),
        "probability":     probabilities,
        "predicted_label": predictions.astype(int),
    })
    df.to_csv(path, index=False)
    logger.info("Predictions saved to %s (%d rows)", path, len(df))


def save_training_history(output_path: str | Path, history):
    """Save the Keras training history dict to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(path, "w") as fh:
        json.dump(hist_dict, fh, indent=2)
    logger.info("Training history saved to %s", path)


def load_metrics(metrics_csv_path: str | Path) -> pd.DataFrame:
    """Load the accumulated metrics CSV; return an empty DataFrame if absent."""
    path = Path(metrics_csv_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def is_run_completed(
    metrics_csv_path: str | Path,
    run_id: str,
    scenario: str,
    seed: int,
) -> bool:
    """Return True if this (run_id, scenario, seed) row already exists in the CSV."""
    df = load_metrics(metrics_csv_path)
    if df.empty:
        return False
    mask = (
        (df["run_id"] == run_id)
        & (df["scenario"] == scenario)
        & (df["seed"] == seed)
    )
    return bool(mask.any())
