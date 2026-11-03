"""
Attention mechanisms used in the spam-detection comparative study.

Two mechanisms are implemented as serialisable Keras layers:

1. AdditiveAttention  (Bahdanau-style)
   score_i = v^T · tanh(W_q · query + W_k · key_i)

2. ScaledDotProductAttention
   score_i = (query · key_i) / sqrt(d_k)

Both layers:
- Accept inputs as [query, keys] where
    query : (batch, lstm_units)        — last valid LSTM hidden state
    keys  : (batch, seq_len, lstm_units) — full sequence of hidden states
- Accept a boolean `mask` argument (batch, seq_len) where True = valid token.
  Padding positions receive a large negative value before softmax so their
  attention weight approaches zero.
- Return (context, weights):
    context : (batch, lstm_units)  — weighted sum of value vectors
    weights : (batch, seq_len)     — attention distribution
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="SpamDetection")
class AdditiveAttention(tf.keras.layers.Layer):
    """Bahdanau-style additive (concat) attention.

    Trainable parameters:
        W_q : (lstm_units, attention_dim)
        W_k : (lstm_units, attention_dim)
        v   : (attention_dim, 1)

    The additive mechanism has more parameters than ScaledDotProductAttention
    because of the learned projection matrices W_q, W_k and the scoring vector
    v.  This difference is intentional and is part of the comparison.
    """

    def __init__(self, attention_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim

    # ------------------------------------------------------------------
    # Layer lifecycle
    # ------------------------------------------------------------------

    def build(self, input_shape):
        # input_shape is a list: [query_shape, keys_shape]
        # query_shape = (batch, lstm_units)
        # keys_shape  = (batch, seq_len, lstm_units)
        lstm_units = input_shape[0][-1]

        self.W_q = self.add_weight(
            name="W_q",
            shape=(lstm_units, self.attention_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.W_k = self.add_weight(
            name="W_k",
            shape=(lstm_units, self.attention_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.v = self.add_weight(
            name="v",
            shape=(self.attention_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        """
        Args:
            inputs: list [query, keys]
                query : (batch, lstm_units)
                keys  : (batch, seq_len, lstm_units)
            mask  : boolean tensor (batch, seq_len), True = valid position.

        Returns:
            context : (batch, lstm_units)
            weights : (batch, seq_len)
        """
        query, keys = inputs

        # Project query: (batch, 1, attention_dim)
        query_proj = tf.matmul(tf.expand_dims(query, axis=1), self.W_q)

        # Project keys: (batch, seq_len, attention_dim)
        keys_proj = tf.matmul(keys, self.W_k)

        # Combined score (broadcast over seq_len): (batch, seq_len, attention_dim)
        combined = tf.tanh(query_proj + keys_proj)

        # Scalar scores: (batch, seq_len)
        scores = tf.squeeze(tf.matmul(combined, self.v), axis=-1)

        # Mask padding positions with a large negative value
        if mask is not None:
            mask_f = tf.cast(mask, dtype=scores.dtype)
            scores = scores + (1.0 - mask_f) * (-1e9)

        # Normalised attention weights
        weights = tf.nn.softmax(scores, axis=-1)  # (batch, seq_len)

        # Context vector: weighted sum of values (keys used as values)
        # (batch, 1, seq_len) @ (batch, seq_len, lstm_units) → (batch, lstm_units)
        context = tf.squeeze(
            tf.matmul(tf.expand_dims(weights, axis=1), keys), axis=1
        )

        return context, weights

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_config(self):
        config = super().get_config()
        config.update({"attention_dim": self.attention_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="SpamDetection")
class ScaledDotProductAttention(tf.keras.layers.Layer):
    """Scaled dot-product attention (Vaswani et al., 2017).

    score_i = (query · key_i) / sqrt(d_k)

    This mechanism has no additional trainable parameters beyond the shared
    embedding and LSTM weights.  The comparison in parameter count is therefore
    inherent to the choice of mechanism.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        """
        Args:
            inputs: list [query, keys]
                query : (batch, lstm_units)
                keys  : (batch, seq_len, lstm_units)
            mask  : boolean tensor (batch, seq_len), True = valid position.

        Returns:
            context : (batch, lstm_units)
            weights : (batch, seq_len)
        """
        query, keys = inputs

        d_k = tf.cast(tf.shape(keys)[-1], dtype=keys.dtype)
        scale = tf.sqrt(d_k)

        # Dot-product scores: (batch, seq_len)
        # query expanded to (batch, 1, lstm_units), then matmul with keys transposed
        scores = tf.squeeze(
            tf.matmul(tf.expand_dims(query, axis=1), keys, transpose_b=True),
            axis=1,
        ) / scale

        # Mask padding positions
        if mask is not None:
            mask_f = tf.cast(mask, dtype=scores.dtype)
            scores = scores + (1.0 - mask_f) * (-1e9)

        weights = tf.nn.softmax(scores, axis=-1)  # (batch, seq_len)

        context = tf.squeeze(
            tf.matmul(tf.expand_dims(weights, axis=1), keys), axis=1
        )

        return context, weights

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)
