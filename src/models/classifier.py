"""
Spam classifier with configurable attention mechanism.

Architecture (identical for both scenarios except for the attention layer):

    Input (token IDs, padded)
        ↓
    Embedding (trainable, mask_zero=True)
        ↓
    LSTM (unidirectional, return_sequences=True, return_state=True)
        ↓
    Attention (additive  OR  scaled dot-product)
        query  = last valid LSTM hidden state  h_final
        keys   = full sequence of hidden states
        values = same as keys
        ↓
    Dropout
        ↓
    Dense(1, activation='sigmoid')

Padding is handled explicitly: the boolean mask derived from the input
(token_id != 0) is forwarded to the attention layer so that padding positions
receive negligible weight after softmax.

Parameter counts differ between the two attention mechanisms:
  - AdditiveAttention adds W_q, W_k ∈ R^{lstm_units × attention_dim} and
    v ∈ R^{attention_dim}.  With default settings (all dims = 64) this adds
    64*64 + 64*64 + 64 = 8,256 parameters.
  - ScaledDotProductAttention adds zero parameters.

This difference is intentional and documented as part of the comparison.
"""

import tensorflow as tf

from src.models.attention import AdditiveAttention, ScaledDotProductAttention


@tf.keras.utils.register_keras_serializable(package="SpamDetection")
class SpamClassifier(tf.keras.Model):
    """Single configurable model for both experimental scenarios."""

    ATTENTION_TYPES = ("additive", "scaled_dot_product")

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        lstm_units: int,
        attention_type: str,
        attention_dim: int = 64,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        """
        Args:
            vocab_size     : number of tokens in the vocabulary (padding = 0,
                             so Embedding input_dim = vocab_size + 1).
            embedding_dim  : output dimension of the Embedding layer.
            lstm_units     : number of units in the LSTM layer.
            attention_type : "additive" or "scaled_dot_product".
            attention_dim  : internal dimension for additive attention only.
            dropout_rate   : dropout applied after the attention context vector.
        """
        if attention_type not in self.ATTENTION_TYPES:
            raise ValueError(
                f"attention_type must be one of {self.ATTENTION_TYPES}, "
                f"got '{attention_type}'."
            )
        super().__init__(**kwargs)

        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._lstm_units = lstm_units
        self._attention_type = attention_type
        self._attention_dim = attention_dim
        self._dropout_rate = dropout_rate

        # ── shared layers ────────────────────────────────────────────
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size + 1,  # +1 because index 0 = padding
            output_dim=embedding_dim,
            mask_zero=True,
            name="embedding",
        )
        self.lstm = tf.keras.layers.LSTM(
            units=lstm_units,
            return_sequences=True,
            return_state=True,
            name="lstm",
        )
        self.drop = tf.keras.layers.Dropout(rate=dropout_rate, name="dropout")
        self.out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")

        # ── attention layer (varies between scenarios) ───────────────
        if attention_type == "additive":
            self.attention = AdditiveAttention(
                attention_dim=attention_dim, name="additive_attention"
            )
        else:
            self.attention = ScaledDotProductAttention(
                name="scaled_dot_product_attention"
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(self, inputs, training: bool = False):
        """
        Args:
            inputs   : integer tensor (batch, seq_len) of token IDs.
            training : passed to Dropout.

        Returns:
            Sigmoid probability tensor (batch, 1).
        """
        # Boolean mask: True = valid token, False = padding (index 0)
        mask = tf.cast(tf.not_equal(inputs, 0), dtype=tf.bool)  # (batch, seq_len)

        # Embeddings — Keras propagates _keras_mask to LSTM automatically
        x = self.embedding(inputs)  # (batch, seq_len, embedding_dim)

        # LSTM — Keras uses the propagated mask; h_final is the last valid state
        sequences, h_final, _ = self.lstm(x)  # sequences: (batch, seq_len, lstm_units)

        # Attention with explicit mask to zero out padding positions
        context, _ = self.attention([h_final, sequences], mask=mask)  # (batch, lstm_units)

        x = self.drop(context, training=training)
        return self.out(x)  # (batch, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def shared_layer_names(self):
        """Names of layers that are kept identical between paired models."""
        return ["embedding", "lstm", "output"]

    def get_shared_weights(self):
        """Return weights of shared layers as a dict."""
        return {
            "embedding": self.embedding.get_weights(),
            "lstm": self.lstm.get_weights(),
            "output": self.out.get_weights(),
        }

    def set_shared_weights(self, weights_dict: dict):
        """Apply shared-layer weights from another model's get_shared_weights()."""
        self.embedding.set_weights(weights_dict["embedding"])
        self.lstm.set_weights(weights_dict["lstm"])
        self.out.set_weights(weights_dict["output"])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self._vocab_size,
                "embedding_dim": self._embedding_dim,
                "lstm_units": self._lstm_units,
                "attention_type": self._attention_type,
                "attention_dim": self._attention_dim,
                "dropout_rate": self._dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
