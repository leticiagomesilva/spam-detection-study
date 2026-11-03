"""
Training utilities for the paired-experiment campaign.

Paired training protocol
------------------------
For each seed k in {1001 … 1030}:
  1. Build the additive-attention model and initialise it with seed k.
  2. Build the scaled-dot-product model with a different initialisation seed.
  3. Copy the weights of the shared layers (Embedding, LSTM, Dense output)
     from the additive model to the scaled model.
  4. Verify the copy is exact (raises if not).
  5. Train both models on the SAME pre-shuffled training data
     (shuffled once with seed k, no epoch-level reshuffling).
  6. Evaluate each model on the held-out test set.

Reproducibility
---------------
- Global seeds (Python random, NumPy, TensorFlow) are set before each model
  construction via set_global_seeds().
- tf.config.experimental.enable_op_determinism() is attempted; if unsupported
  (e.g. older TF or GPU), a warning is logged and training continues.
- The training data is shuffled once per pair with the pair seed; both models
  in the pair see exactly the same sequence of batches across all epochs.
"""

import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Determinism helpers
# ──────────────────────────────────────────────────────────────────────

def set_global_seeds(seed: int):
    """Set all global random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_determinism() -> bool:
    """Attempt to enable TensorFlow operation determinism.

    Returns True on success, False if unsupported (e.g. some GPU ops).
    Limitations are documented: certain CUDA ops may remain non-deterministic
    on GPU even when this flag is set.
    """
    try:
        tf.config.experimental.enable_op_determinism()
        logger.info("TF op determinism enabled.")
        return True
    except Exception as exc:
        logger.warning(
            "Could not enable TF op determinism (%s). "
            "Results may vary slightly across runs on GPU.",
            exc,
        )
        return False


# ──────────────────────────────────────────────────────────────────────
# Model construction
# ──────────────────────────────────────────────────────────────────────

def build_model(config: dict, attention_type: str, init_seed: int):
    """Build and compile a SpamClassifier.

    The model weights are initialised using init_seed as the global TF seed
    at construction time.

    Args:
        config        : loaded experiment.yaml as a dict.
        attention_type: "additive" or "scaled_dot_product".
        init_seed     : seed used to initialise layer weights.

    Returns:
        Compiled tf.keras.Model.
    """
    from src.models.classifier import SpamClassifier

    set_global_seeds(init_seed)

    model = SpamClassifier(
        vocab_size=config["preprocessing"]["max_vocab_size"],
        embedding_dim=config["model"]["embedding_dim"],
        lstm_units=config["model"]["lstm_units"],
        attention_type=attention_type,
        attention_dim=config["model"]["attention_dim"],
        dropout_rate=config["model"]["dropout_rate"],
        name=f"spam_classifier_{attention_type}",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["training"]["learning_rate"]
    )
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Initialise weights by doing a dummy forward pass
    max_length = config["preprocessing"]["max_length"]
    dummy = tf.zeros((1, max_length), dtype=tf.int32)
    model(dummy, training=False)

    return model


def verify_weight_copy(source_model, target_model):
    """Raise AssertionError if shared-layer weights differ between models."""
    src = source_model.get_shared_weights()
    tgt = target_model.get_shared_weights()
    for layer_name, src_weights in src.items():
        tgt_weights = tgt[layer_name]
        for sw, tw in zip(src_weights, tgt_weights):
            if not np.allclose(sw, tw, atol=1e-7):
                raise AssertionError(
                    f"Weight copy verification failed for layer '{layer_name}'."
                )
    logger.debug("Shared weight copy verified for all layers.")


def create_paired_models(config: dict, pair_seed: int):
    """Create a matched pair of models with identical shared-layer weights.

    Steps:
      1. Build additive model with pair_seed.
      2. Build scaled dot-product model with a derived seed.
      3. Copy shared layers from additive → scaled.
      4. Verify the copy.

    Returns:
        (model_additive, model_scaled)
    """
    model_add = build_model(config, "additive", init_seed=pair_seed)
    # Use a different seed for the second model; shared layers will be overwritten
    model_sdp = build_model(config, "scaled_dot_product", init_seed=pair_seed + 100_000)

    # Copy shared layer weights from additive to scaled dot-product
    model_sdp.set_shared_weights(model_add.get_shared_weights())
    verify_weight_copy(model_add, model_sdp)

    return model_add, model_sdp


# ──────────────────────────────────────────────────────────────────────
# Dataset preparation
# ──────────────────────────────────────────────────────────────────────

def preshuffle_training_data(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    """Shuffle training data once with a fixed seed.

    Both models in a pair receive the same shuffled training order (no
    epoch-level reshuffling), guaranteeing identical batch sequences.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_train))
    return X_train[idx], y_train[idx]


def make_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from NumPy arrays."""
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(X, dtype=tf.int32), tf.constant(y, dtype=tf.float32))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ──────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────

def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    pair_seed: int,
    checkpoint_path: Optional[str | Path] = None,
    verbose: int = 0,
):
    """Train a single model and return training artefacts.

    Args:
        model          : compiled tf.keras.Model.
        X_train        : pre-shuffled training sequences (int32 array).
        y_train        : pre-shuffled training labels (float32 array).
        X_val          : validation sequences.
        y_val          : validation labels.
        config         : experiment config dict.
        pair_seed      : used to set global seeds before training.
        checkpoint_path: if provided, save best model weights here.
        verbose        : Keras verbosity (0 = silent).

    Returns:
        history    : Keras History object.
        train_time : wall-clock seconds for the fit() call.
        best_epoch : epoch index (1-based) with the best val_loss.
        total_epochs: total number of epochs actually run.
    """
    # Reset global seeds before training to ensure reproducible weight updates
    set_global_seeds(pair_seed)

    tcfg = config["training"]
    batch_size = tcfg["batch_size"]
    max_epochs = tcfg["max_epochs"]
    es_cfg = tcfg["early_stopping"]

    train_ds = make_tf_dataset(X_train, y_train, batch_size, shuffle=False)
    val_ds   = make_tf_dataset(X_val,   y_val,   batch_size, shuffle=False)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=es_cfg["monitor"],
            patience=es_cfg["patience"],
            restore_best_weights=es_cfg["restore_best_weights"],
            verbose=0,
        )
    ]

    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=es_cfg["monitor"],
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
            )
        )

    # Optional class weights
    class_weight = tcfg.get("class_weight", None)
    if class_weight == "balanced":
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        total = n_neg + n_pos
        class_weight = {0: total / (2 * n_neg), 1: total / (2 * n_pos)}
        logger.info("Using balanced class weights: %s", class_weight)

    start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=verbose,
    )
    train_time = time.perf_counter() - start

    val_losses = history.history["val_loss"]
    best_epoch = int(np.argmin(val_losses)) + 1
    total_epochs = len(val_losses)

    logger.info(
        "Training complete — best epoch: %d/%d, val_loss: %.4f, time: %.1f s",
        best_epoch,
        total_epochs,
        val_losses[best_epoch - 1],
        train_time,
    )

    return history, train_time, best_epoch, total_epochs
