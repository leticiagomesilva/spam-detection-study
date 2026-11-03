"""
Inference utilities: load a saved model and predict on new messages.

Usage example (command line not required — import and call directly):

    from src.predict import load_artifacts, predict_messages

    model, tokenizer, config = load_artifacts(
        checkpoint_path="checkpoints/my_run/additive_seed1001/weights.h5",
        tokenizer_path="data/processed/tokenizer.pkl",
        config_path="configs/experiment.yaml",
    )

    texts = [
        "Congratulations! You have won a £1000 prize. Call 09061701461 now.",
        "Hey, are you coming to the party tonight?",
    ]
    probs, labels = predict_messages(model, tokenizer, texts, config)
    for t, p, l in zip(texts, probs, labels):
        print(f"[{'spam' if l else 'ham':4s}  {p:.3f}]  {t}")
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

from src.preprocessing import clean_text, encode_texts, load_tokenizer

logger = logging.getLogger(__name__)


def load_artifacts(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    config_path: str | Path,
):
    """Load model weights, tokenizer, and configuration.

    The model architecture is reconstructed from the configuration file, then
    the saved weights are loaded.  Both the attention type and all hyper-
    parameters must match the checkpoint.

    Args:
        checkpoint_path: path to a .weights.h5 or directory saved by ModelCheckpoint.
        tokenizer_path : path to the pickled Tokenizer.
        config_path    : path to experiment.yaml.

    Returns:
        (model, tokenizer, config)
    """
    import tensorflow as tf
    from src.models.classifier import SpamClassifier

    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    tokenizer = load_tokenizer(tokenizer_path)

    # Determine attention type from checkpoint path (convention: path contains
    # "additive" or "scaled_dot_product")
    cp_str = str(checkpoint_path)
    if "scaled_dot_product" in cp_str:
        attention_type = "scaled_dot_product"
    elif "additive" in cp_str:
        attention_type = "additive"
    else:
        raise ValueError(
            "Cannot infer attention_type from checkpoint path. "
            "Path must contain 'additive' or 'scaled_dot_product'."
        )

    model = SpamClassifier(
        vocab_size=config["preprocessing"]["max_vocab_size"],
        embedding_dim=config["model"]["embedding_dim"],
        lstm_units=config["model"]["lstm_units"],
        attention_type=attention_type,
        attention_dim=config["model"]["attention_dim"],
        dropout_rate=config["model"]["dropout_rate"],
    )

    # Build the model before loading weights
    max_length = config["preprocessing"]["max_length"]
    dummy = tf.zeros((1, max_length), dtype=tf.int32)
    model(dummy, training=False)

    model.load_weights(str(checkpoint_path))
    logger.info("Model weights loaded from %s", checkpoint_path)

    return model, tokenizer, config


def predict_messages(
    model,
    tokenizer,
    texts: List[str],
    config: dict,
    threshold: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict spam probability and class for a list of raw text messages.

    Args:
        model     : loaded SpamClassifier.
        tokenizer : fitted Tokenizer (from training).
        texts     : list of raw (uncleaned) SMS messages.
        config    : experiment config dict.
        threshold : override the default threshold from config if provided.

    Returns:
        probabilities : float array of shape (n,) in [0, 1].
        labels        : int array of shape (n,) with 0 = ham, 1 = spam.
    """
    if threshold is None:
        threshold = config["evaluation"]["threshold"]

    max_length = config["preprocessing"]["max_length"]
    cleaned = [clean_text(t) for t in texts]
    X = encode_texts(tokenizer, cleaned, max_length)

    probs = model.predict(X, verbose=0).flatten()
    labels = (probs >= threshold).astype(int)

    return probs, labels
