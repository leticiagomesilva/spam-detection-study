"""
Tests for data loading, integrity checks, and splitting (src/data.py).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import (
    apply_split_ids,
    check_invalid_labels,
    check_missing_values,
    compute_file_hash,
    handle_duplicates,
    split_data,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal DataFrame mimicking the raw dataset format."""
    return pd.DataFrame({
        "id":    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "label": ["ham", "spam", "ham", "ham", "spam",
                  "ham", "spam", "ham", "ham", "spam"],
        "text":  [
            "Hey how are you",
            "Win a free prize now!!!",
            "See you at 6pm",
            "Are you coming?",
            "CALL NOW to claim your reward",
            "Dinner at 7?",
            "Free entry to our competition",
            "Ok thanks",
            "On my way",
            "Urgent! Reply to claim cash",
        ],
    })


@pytest.fixture
def df_with_exact_dupes():
    """DataFrame containing same-text-same-label duplicates."""
    return pd.DataFrame({
        "id":    [0, 1, 2, 3],
        "label": ["ham", "ham", "spam", "spam"],
        "text":  ["Hello", "Hello", "Win now", "Win now"],
    })


@pytest.fixture
def df_with_conflict_dupes():
    """DataFrame containing same-text-different-label duplicates."""
    return pd.DataFrame({
        "id":    [0, 1, 2, 3, 4],
        "label": ["ham", "spam", "ham", "ham", "spam"],
        "text":  ["Conflict msg", "Conflict msg", "Normal ham", "Another ham", "Real spam"],
    })


# ──────────────────────────────────────────────────────────────────────
# Tests: integrity checks
# ──────────────────────────────────────────────────────────────────────

def test_check_missing_values_no_missing(sample_df):
    missing = check_missing_values(sample_df)
    assert missing.sum() == 0


def test_check_missing_values_detects_nan():
    df = pd.DataFrame({"id": [0, 1], "label": ["ham", None], "text": ["Hi", "Bye"]})
    missing = check_missing_values(df)
    assert missing["label"] == 1


def test_check_invalid_labels_clean(sample_df):
    invalid = check_invalid_labels(sample_df)
    assert invalid.empty


def test_check_invalid_labels_detects_bad():
    df = pd.DataFrame({
        "id":    [0, 1],
        "label": ["ham", "junk"],
        "text":  ["Hi", "Bad label"],
    })
    invalid = check_invalid_labels(df)
    assert len(invalid) == 1
    assert invalid.iloc[0]["label"] == "junk"


# ──────────────────────────────────────────────────────────────────────
# Tests: duplicate handling
# ──────────────────────────────────────────────────────────────────────

def test_handle_exact_duplicates(df_with_exact_dupes):
    clean, report = handle_duplicates(df_with_exact_dupes)
    # "Hello" appears twice with "ham" → one kept
    # "Win now" appears twice with "spam" → one kept
    assert report["n_exact_duplicates_removed"] == 2
    assert report["n_conflicting_texts"] == 0
    assert len(clean) == 2


def test_handle_conflicting_duplicates(df_with_conflict_dupes):
    clean, report = handle_duplicates(df_with_conflict_dupes)
    # "Conflict msg" has both ham and spam → removed entirely
    assert report["n_conflicting_texts"] == 1
    assert "Conflict msg" in report["conflicting_texts"]
    assert all(clean["text"] != "Conflict msg")


def test_no_duplicates_unchanged(sample_df):
    clean, report = handle_duplicates(sample_df)
    assert report["n_exact_duplicates_removed"] == 0
    assert report["n_conflicting_texts"] == 0
    assert len(clean) == len(sample_df)


# ──────────────────────────────────────────────────────────────────────
# Tests: splitting
# ──────────────────────────────────────────────────────────────────────

def test_split_sizes():
    """Split produces approximately correct proportions."""
    n = 200
    labels = ["ham"] * 160 + ["spam"] * 40
    df = pd.DataFrame({
        "id":    range(n),
        "label": labels,
        "text":  [f"msg {i}" for i in range(n)],
    })
    df_train, df_val, df_test = split_data(df, seed=42)
    total = len(df_train) + len(df_val) + len(df_test)
    assert total == n
    # Train should be ~70 %
    assert abs(len(df_train) / n - 0.70) < 0.05


def test_split_stratification():
    """Both classes should be present in all three splits."""
    n = 200
    labels = ["ham"] * 160 + ["spam"] * 40
    df = pd.DataFrame({
        "id":    range(n),
        "label": labels,
        "text":  [f"msg {i}" for i in range(n)],
    })
    df_train, df_val, df_test = split_data(df, seed=42)
    for subset in (df_train, df_val, df_test):
        assert "ham" in subset["label"].values
        assert "spam" in subset["label"].values


def test_split_no_overlap():
    """No message should appear in more than one split."""
    n = 300
    labels = (["ham"] * 250 + ["spam"] * 50)
    df = pd.DataFrame({
        "id":    range(n),
        "label": labels,
        "text":  [f"unique msg {i}" for i in range(n)],
    })
    df_train, df_val, df_test = split_data(df, seed=42)
    train_ids = set(df_train["id"])
    val_ids   = set(df_val["id"])
    test_ids  = set(df_test["id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_reproducibility():
    """Same seed always produces the same split."""
    n = 100
    labels = ["ham"] * 80 + ["spam"] * 20
    df = pd.DataFrame({
        "id":    range(n),
        "label": labels,
        "text":  [f"msg {i}" for i in range(n)],
    })
    tr1, va1, te1 = split_data(df, seed=7)
    tr2, va2, te2 = split_data(df, seed=7)
    assert list(tr1["id"]) == list(tr2["id"])
    assert list(te1["id"]) == list(te2["id"])


# ──────────────────────────────────────────────────────────────────────
# Tests: apply_split_ids
# ──────────────────────────────────────────────────────────────────────

def test_apply_split_ids_roundtrip():
    """apply_split_ids should reconstruct the same partitions."""
    n = 100
    labels = ["ham"] * 80 + ["spam"] * 20
    df = pd.DataFrame({
        "id":    range(n),
        "label": labels,
        "text":  [f"msg {i}" for i in range(n)],
    })
    df_train, df_val, df_test = split_data(df, seed=42)
    split_ids = {
        "train": df_train["id"].tolist(),
        "val":   df_val["id"].tolist(),
        "test":  df_test["id"].tolist(),
    }
    tr2, va2, te2 = apply_split_ids(df, split_ids)
    assert set(tr2["id"]) == set(df_train["id"])
    assert set(te2["id"]) == set(df_test["id"])


# ──────────────────────────────────────────────────────────────────────
# Tests: file hash
# ──────────────────────────────────────────────────────────────────────

def test_compute_file_hash_deterministic(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    h1 = compute_file_hash(f)
    h2 = compute_file_hash(f)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_compute_file_hash_changes_with_content(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("content A")
    h1 = compute_file_hash(f)
    f.write_text("content B")
    h2 = compute_file_hash(f)
    assert h1 != h2
