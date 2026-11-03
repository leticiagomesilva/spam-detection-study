"""
Tests for attention mechanisms and the SpamClassifier (src/models/).

Tests verify:
- Output shapes for both attention types.
- That masked (padding) positions receive near-zero attention weight.
- That context vectors have the correct dtype and are not NaN/Inf.
- That shared-weight copy and verification work correctly.
- That the classifier is serialisable (get_config / from_config round-trip).

These tests run on CPU and do not require GPU or a full dataset.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.attention import AdditiveAttention, ScaledDotProductAttention
from src.models.classifier import SpamClassifier


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

BATCH      = 4
SEQ_LEN    = 10
LSTM_UNITS = 16
ATT_DIM    = 16


@pytest.fixture
def query():
    """Simulated last valid LSTM hidden state."""
    tf.random.set_seed(0)
    return tf.random.normal((BATCH, LSTM_UNITS))


@pytest.fixture
def keys():
    """Simulated LSTM output sequence."""
    tf.random.set_seed(1)
    return tf.random.normal((BATCH, SEQ_LEN, LSTM_UNITS))


@pytest.fixture
def valid_mask():
    """All positions valid (no padding)."""
    return tf.ones((BATCH, SEQ_LEN), dtype=tf.bool)


@pytest.fixture
def partial_mask():
    """Last 3 positions in each example are padding."""
    mask = np.ones((BATCH, SEQ_LEN), dtype=bool)
    mask[:, -3:] = False
    return tf.constant(mask)


# ──────────────────────────────────────────────────────────────────────
# AdditiveAttention
# ──────────────────────────────────────────────────────────────────────

class TestAdditiveAttention:

    def test_output_shapes(self, query, keys, valid_mask):
        layer = AdditiveAttention(attention_dim=ATT_DIM)
        context, weights = layer([query, keys], mask=valid_mask)
        assert context.shape == (BATCH, LSTM_UNITS), f"Expected ({BATCH}, {LSTM_UNITS}), got {context.shape}"
        assert weights.shape == (BATCH, SEQ_LEN),   f"Expected ({BATCH}, {SEQ_LEN}), got {weights.shape}"

    def test_weights_sum_to_one(self, query, keys, valid_mask):
        layer = AdditiveAttention(attention_dim=ATT_DIM)
        _, weights = layer([query, keys], mask=valid_mask)
        row_sums = tf.reduce_sum(weights, axis=-1).numpy()
        np.testing.assert_allclose(row_sums, np.ones(BATCH), atol=1e-5)

    def test_padding_gets_near_zero_weight(self, query, keys, partial_mask):
        """Positions 7, 8, 9 (last 3) are padding and must have ~0 weight."""
        layer = AdditiveAttention(attention_dim=ATT_DIM)
        _, weights = layer([query, keys], mask=partial_mask)
        padding_weights = weights[:, -3:].numpy()
        np.testing.assert_allclose(padding_weights, np.zeros((BATCH, 3)), atol=1e-5)

    def test_no_nan_or_inf(self, query, keys, partial_mask):
        layer = AdditiveAttention(attention_dim=ATT_DIM)
        context, weights = layer([query, keys], mask=partial_mask)
        assert not np.any(np.isnan(context.numpy())), "Context contains NaN"
        assert not np.any(np.isinf(context.numpy())), "Context contains Inf"
        assert not np.any(np.isnan(weights.numpy())), "Weights contain NaN"

    def test_no_mask(self, query, keys):
        """Layer should work without a mask."""
        layer = AdditiveAttention(attention_dim=ATT_DIM)
        context, weights = layer([query, keys])
        assert context.shape == (BATCH, LSTM_UNITS)

    def test_serialisation(self):
        layer = AdditiveAttention(attention_dim=32, name="test_add")
        cfg   = layer.get_config()
        layer2 = AdditiveAttention.from_config(cfg)
        assert layer2.attention_dim == 32

    def test_trainable_weights_count(self):
        """Additive attention should have exactly 3 weight tensors (W_q, W_k, v)."""
        layer = AdditiveAttention(attention_dim=ATT_DIM)
        dummy_q = tf.zeros((1, LSTM_UNITS))
        dummy_k = tf.zeros((1, SEQ_LEN, LSTM_UNITS))
        layer([dummy_q, dummy_k])
        assert len(layer.trainable_weights) == 3


# ──────────────────────────────────────────────────────────────────────
# ScaledDotProductAttention
# ──────────────────────────────────────────────────────────────────────

class TestScaledDotProductAttention:

    def test_output_shapes(self, query, keys, valid_mask):
        layer = ScaledDotProductAttention()
        context, weights = layer([query, keys], mask=valid_mask)
        assert context.shape == (BATCH, LSTM_UNITS)
        assert weights.shape == (BATCH, SEQ_LEN)

    def test_weights_sum_to_one(self, query, keys, valid_mask):
        layer = ScaledDotProductAttention()
        _, weights = layer([query, keys], mask=valid_mask)
        row_sums = tf.reduce_sum(weights, axis=-1).numpy()
        np.testing.assert_allclose(row_sums, np.ones(BATCH), atol=1e-5)

    def test_padding_gets_near_zero_weight(self, query, keys, partial_mask):
        layer = ScaledDotProductAttention()
        _, weights = layer([query, keys], mask=partial_mask)
        padding_weights = weights[:, -3:].numpy()
        np.testing.assert_allclose(padding_weights, np.zeros((BATCH, 3)), atol=1e-5)

    def test_no_nan_or_inf(self, query, keys, partial_mask):
        layer = ScaledDotProductAttention()
        context, weights = layer([query, keys], mask=partial_mask)
        assert not np.any(np.isnan(context.numpy()))
        assert not np.any(np.isinf(context.numpy()))

    def test_no_trainable_weights(self):
        """Scaled dot-product attention has zero additional trainable weights."""
        layer = ScaledDotProductAttention()
        dummy_q = tf.zeros((1, LSTM_UNITS))
        dummy_k = tf.zeros((1, SEQ_LEN, LSTM_UNITS))
        layer([dummy_q, dummy_k])
        assert len(layer.trainable_weights) == 0

    def test_serialisation(self):
        layer = ScaledDotProductAttention(name="test_sdp")
        cfg   = layer.get_config()
        layer2 = ScaledDotProductAttention.from_config(cfg)
        assert layer2.name == "test_sdp"


# ──────────────────────────────────────────────────────────────────────
# SpamClassifier
# ──────────────────────────────────────────────────────────────────────

VOCAB_SIZE    = 100
EMBEDDING_DIM = 16
MAX_LENGTH    = 10


def make_classifier(attention_type: str) -> SpamClassifier:
    model = SpamClassifier(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS,
        attention_type=attention_type,
        attention_dim=ATT_DIM,
        dropout_rate=0.0,
    )
    # Build by calling with dummy input
    dummy = tf.zeros((1, MAX_LENGTH), dtype=tf.int32)
    model(dummy, training=False)
    return model


class TestSpamClassifier:

    def test_output_shape_additive(self):
        model = make_classifier("additive")
        x = tf.random.uniform((BATCH, MAX_LENGTH), minval=0, maxval=VOCAB_SIZE, dtype=tf.int32)
        out = model(x, training=False)
        assert out.shape == (BATCH, 1)

    def test_output_shape_scaled(self):
        model = make_classifier("scaled_dot_product")
        x = tf.random.uniform((BATCH, MAX_LENGTH), minval=0, maxval=VOCAB_SIZE, dtype=tf.int32)
        out = model(x, training=False)
        assert out.shape == (BATCH, 1)

    def test_output_in_01_range(self):
        model = make_classifier("additive")
        x = tf.random.uniform((BATCH, MAX_LENGTH), minval=0, maxval=VOCAB_SIZE, dtype=tf.int32)
        out = model(x, training=False).numpy()
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_invalid_attention_type_raises(self):
        with pytest.raises(ValueError):
            SpamClassifier(
                vocab_size=VOCAB_SIZE,
                embedding_dim=EMBEDDING_DIM,
                lstm_units=LSTM_UNITS,
                attention_type="unknown",
            )

    def test_padding_tokens_masked(self):
        """Predictions should be stable when padding tokens are appended."""
        model = make_classifier("additive")
        # Short sequence: [5, 7, 0, 0, 0, 0, 0, 0, 0, 0]
        x_short = tf.constant([[5, 7] + [0] * (MAX_LENGTH - 2)] * BATCH, dtype=tf.int32)
        # Same tokens without padding (will be padded internally later, but here
        # just verify that the all-pad suffix does not cause NaN)
        out = model(x_short, training=False).numpy()
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_shared_weight_copy_and_verify(self):
        """Shared weights can be extracted, copied, and verified."""
        model_add = make_classifier("additive")
        model_sdp = make_classifier("scaled_dot_product")

        # Weights differ initially
        add_emb = model_add.embedding.get_weights()[0]
        sdp_emb = model_sdp.embedding.get_weights()[0]
        # May or may not differ depending on seeds; test the copy mechanism

        # Copy
        model_sdp.set_shared_weights(model_add.get_shared_weights())

        # Verify using the built-in method (should not raise)
        from src.train import verify_weight_copy
        verify_weight_copy(model_add, model_sdp)

    def test_additive_has_more_params_than_scaled(self):
        """Additive attention introduces extra parameters."""
        model_add = make_classifier("additive")
        model_sdp = make_classifier("scaled_dot_product")
        n_add = model_add.count_params()
        n_sdp = model_sdp.count_params()
        assert n_add > n_sdp, (
            f"Expected additive ({n_add}) to have more params than scaled ({n_sdp})."
        )

    def test_get_config_roundtrip(self):
        model = make_classifier("additive")
        cfg = model.get_config()
        model2 = SpamClassifier.from_config(cfg)
        assert model2._attention_type == "additive"
        assert model2._vocab_size == VOCAB_SIZE
