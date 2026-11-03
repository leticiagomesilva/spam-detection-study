"""
Text preprocessing for the spam-detection study.

Design choices (intentionally minimal):
- Only lowercasing is applied.
- Numbers, punctuation, URLs, and special characters are preserved because
  they carry potential signal for spam detection (prize amounts, phone numbers,
  exclamation marks, shortcode URLs, etc.).
- Tokenisation is handled by Keras Tokenizer with out-of-vocabulary (OOV)
  token '<OOV>' mapped to index 1 (index 0 is reserved for padding).

Vocabulary leakage prevention:
- build_tokenizer() must be called only on training texts.
- encode_texts() can then be applied to any split.
"""

import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Text cleaning
# ──────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Minimal normalisation: lowercase only.

    Preserved intentionally:
        - Digits and numbers   (e.g., "Win £1000", "call 08007")
        - Punctuation          (e.g., "!!!", "FREE!")
        - URLs                 (e.g., "http://claim-prize.com")
        - Special characters   (e.g., "£", "@", "$")

    These features are strong spam indicators and must not be removed.
    """
    return text.lower()


# ──────────────────────────────────────────────────────────────────────
# Tokenisation
# ──────────────────────────────────────────────────────────────────────

def build_tokenizer(train_texts: List[str], max_vocab_size: int) -> Tokenizer:
    """Fit a Keras Tokenizer on training texts only.

    Index 0 is padding (reserved by Keras when mask_zero=True in Embedding).
    Index 1 is the OOV token '<OOV>'.
    Indices 2 … max_vocab_size are real vocabulary tokens.

    Note: the Keras default `filters` are used, so most punctuation
    (!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~) is stripped at tokenisation time even
    though clean_text() preserves it.  '£', apostrophes and digits are kept.
    Kept unchanged so results stay reproducible; see docs/metodologia.md §3.1.

    Args:
        train_texts   : list of cleaned training texts.
        max_vocab_size: maximum vocabulary size (num_words parameter).

    Returns:
        A fitted Tokenizer instance.
    """
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    vocab_used = min(len(tokenizer.word_index) + 1, max_vocab_size)
    logger.info("Vocabulary size: %d tokens (capped at %d)", vocab_used, max_vocab_size)
    return tokenizer


def encode_texts(
    tokenizer: Tokenizer,
    texts: List[str],
    max_length: int,
) -> np.ndarray:
    """Convert texts to zero-padded integer sequences.

    Sequences longer than max_length are truncated from the right.
    Sequences shorter than max_length are zero-padded on the right.

    Args:
        tokenizer  : a previously fitted Tokenizer (must NOT be re-fitted).
        texts      : list of cleaned texts.
        max_length : maximum sequence length.

    Returns:
        Integer array of shape (n_samples, max_length).
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
    return padded.astype(np.int32)


# ──────────────────────────────────────────────────────────────────────
# Label encoding
# ──────────────────────────────────────────────────────────────────────

def encode_labels(labels) -> np.ndarray:
    """Encode 'spam' → 1, 'ham' → 0."""
    mapping = {"ham": 0, "spam": 1}
    encoded = np.array([mapping[lbl] for lbl in labels], dtype=np.float32)
    return encoded


# ──────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────

def save_tokenizer(tokenizer: Tokenizer, path: str | Path):
    """Serialise tokenizer to disk using pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(tokenizer, fh)
    logger.info("Tokenizer saved to %s", path)


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load a previously saved tokenizer."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ──────────────────────────────────────────────────────────────────────
# Convenience: prepare all splits
# ──────────────────────────────────────────────────────────────────────

def prepare_splits(df_train, df_val, df_test, max_vocab_size: int, max_length: int):
    """Clean, tokenise, and encode all three splits.

    The tokeniser is fitted exclusively on training texts.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, tokenizer
    """
    train_texts = [clean_text(t) for t in df_train["text"]]
    val_texts   = [clean_text(t) for t in df_val["text"]]
    test_texts  = [clean_text(t) for t in df_test["text"]]

    tokenizer = build_tokenizer(train_texts, max_vocab_size)

    X_train = encode_texts(tokenizer, train_texts, max_length)
    X_val   = encode_texts(tokenizer, val_texts,   max_length)
    X_test  = encode_texts(tokenizer, test_texts,  max_length)

    y_train = encode_labels(df_train["label"])
    y_val   = encode_labels(df_val["label"])
    y_test  = encode_labels(df_test["label"])

    return X_train, y_train, X_val, y_val, X_test, y_test, tokenizer
