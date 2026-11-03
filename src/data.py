"""
Data loading, integrity checking, and train/val/test splitting.

Splitting strategy
------------------
Stratified split: 70 % train, 15 % validation, 15 % test.
The split seed is fixed (config['split']['seed'] = 42) so the same partition
is used across all 30 experimental pairs.  Split indices are saved to disk so
the partition can be reproduced exactly.

Duplicate handling
------------------
Rule 1  — Same text, same label:  exact duplicates.  Keep the first occurrence.
Rule 2  — Same text, different labels: conflicting duplicates.  Remove all
          occurrences of that text and log them.

Vocabulary leakage prevention
------------------------------
The tokeniser (Tokenizer object) is fitted exclusively on training texts.
Validation and test sets are only transformed, never used to fit the vocabulary.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────

def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load the raw SMS Spam Collection tab-separated file.

    Returns a DataFrame with columns ['label', 'text'] and an integer 'id'
    column (0-based) that serves as a stable row identifier throughout the
    experiment pipeline.
    """
    filepath = Path(filepath)
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8",
    )
    df.insert(0, "id", range(len(df)))
    logger.info("Loaded %d rows from %s", len(df), filepath)
    return df


def compute_file_hash(filepath: str | Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ──────────────────────────────────────────────────────────────────────
# Integrity checks
# ──────────────────────────────────────────────────────────────────────

def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return per-column missing-value counts."""
    missing = df.isnull().sum()
    if missing.any():
        logger.warning("Missing values found:\n%s", missing[missing > 0])
    return missing


def check_invalid_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose label is not 'ham' or 'spam'."""
    valid = {"ham", "spam"}
    invalid = df[~df["label"].isin(valid)]
    if not invalid.empty:
        logger.warning("Found %d rows with invalid labels.", len(invalid))
    return invalid


# ──────────────────────────────────────────────────────────────────────
# Duplicate handling
# ──────────────────────────────────────────────────────────────────────

def handle_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Remove duplicate messages following documented rules.

    Rules (applied in order):
      1. Same text + same label   → keep first occurrence, drop the rest.
      2. Same text + diff. labels → remove ALL occurrences and record them.

    Args:
        df: DataFrame with at least ['text', 'label'] columns.

    Returns:
        clean_df : deduplicated DataFrame, index reset.
        report   : dict with statistics and list of conflicting texts.
    """
    report: dict = {}
    n_original = len(df)

    # ── Rule 2: conflicting texts ─────────────────────────────────────
    label_counts = df.groupby("text")["label"].nunique()
    conflict_texts = label_counts[label_counts > 1].index.tolist()
    report["conflicting_texts"] = conflict_texts
    report["n_conflicting_texts"] = len(conflict_texts)
    report["n_rows_removed_conflicting"] = int(df["text"].isin(conflict_texts).sum())

    if conflict_texts:
        logger.warning(
            "Removing %d rows with %d conflicting texts (same text, different labels).",
            report["n_rows_removed_conflicting"],
            report["n_conflicting_texts"],
        )

    df_no_conflict = df[~df["text"].isin(conflict_texts)].copy()

    # ── Rule 1: exact duplicates (same text + same label) ─────────────
    n_before_dedup = len(df_no_conflict)
    df_clean = df_no_conflict.drop_duplicates(subset=["text", "label"], keep="first")
    report["n_exact_duplicates_removed"] = n_before_dedup - len(df_clean)

    if report["n_exact_duplicates_removed"] > 0:
        logger.info(
            "Removed %d exact duplicates (same text, same label).",
            report["n_exact_duplicates_removed"],
        )

    df_clean = df_clean.reset_index(drop=True)
    report["n_original"] = n_original
    report["n_clean"] = len(df_clean)

    logger.info(
        "After deduplication: %d rows (from %d original).",
        report["n_clean"],
        report["n_original"],
    )
    return df_clean, report


# ──────────────────────────────────────────────────────────────────────
# Stratified split
# ──────────────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split into train, validation, and test sets.

    The split is performed once and the resulting integer IDs are saved to
    disk (see save_split_ids / load_split_ids) to guarantee exact reproduction.
    Duplicates must be removed before calling this function.

    Balancing or augmentation must NOT be applied before the split.

    Args:
        df          : clean DataFrame with an 'id' and 'label' column.
        train_ratio : fraction for training (default 0.70).
        val_ratio   : fraction for validation (default 0.15).
        test_ratio  : fraction for test (default 0.15).
        seed        : random seed for reproducibility.

    Returns:
        df_train, df_val, df_test
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.")

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    df_train, df_val_test = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df["label"],
        random_state=seed,
    )

    # Second split: val vs test within the remaining fraction
    relative_val = val_ratio / val_test_ratio
    df_val, df_test = train_test_split(
        df_val_test,
        test_size=1.0 - relative_val,
        stratify=df_val_test["label"],
        random_state=seed,
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(df_train),
        len(df_val),
        len(df_test),
    )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def save_split_ids(df_train, df_val, df_test, output_dir: str | Path):
    """Persist the row IDs of each split partition to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": df_train["id"].tolist(),
        "val": df_val["id"].tolist(),
        "test": df_test["id"].tolist(),
    }
    out_path = output_dir / "split_ids.json"
    with open(out_path, "w") as fh:
        json.dump(splits, fh, indent=2)
    logger.info("Split IDs saved to %s", out_path)


def load_split_ids(path: str | Path) -> dict:
    """Load previously saved split IDs."""
    with open(path) as fh:
        return json.load(fh)


def apply_split_ids(df: pd.DataFrame, split_ids: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Re-apply a previously saved partition by matching 'id' values."""
    df_train = df[df["id"].isin(split_ids["train"])].reset_index(drop=True)
    df_val = df[df["id"].isin(split_ids["val"])].reset_index(drop=True)
    df_test = df[df["id"].isin(split_ids["test"])].reset_index(drop=True)
    return df_train, df_val, df_test
