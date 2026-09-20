"""Shared CNN encoder for compact chess boards."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="ChessRating")
class BoardEncoder(layers.Layer):
    """Encode positions in one CNN batch, with shared weights.

    Compact input stores signed piece IDs (0, +/-1..6), not one-hot floats.
    """

    def __init__(self, skip_padding=False, **kwargs):
        super().__init__(**kwargs)
        self.skip_padding = skip_padding
        self.cnn = keras.Sequential([
            layers.Conv2D(32, 3, padding="same", activation="gelu"),
            layers.Conv2D(64, 3, padding="same", activation="gelu"),
            layers.Conv2D(128, 3, padding="same", activation="gelu"),
            layers.Flatten(),
            layers.Dense(128),
        ], name="board_cnn")

    def build(self, input_shape):
        self.cnn.build((None, 8, 8, 12))
        super().build(input_shape)

    def call(self, inputs):
        boards, valid = inputs
        batch, steps = tf.shape(boards)[0], tf.shape(boards)[1]
        # Stable B*T shapes let cuDNN reuse an algorithm selection across steps.
        # Gathering valid positions changes the convolution batch size on almost
        # every step, potentially triggering repeated GPU autotuning.
        indices = (tf.cast(tf.where(tf.reshape(valid, [-1]))[:, 0], tf.int32)
                   if self.skip_padding else None)
        selected = tf.reshape(boards, [-1, 8, 8])
        if self.skip_padding:
            selected = tf.gather(selected, indices)
        pieces = tf.cast(selected, tf.int32)
        channels = tf.abs(pieces) - 1 + tf.cast(pieces < 0, tf.int32) * 6
        # Empty squares have index -1: one_hot returns twelve zeros.
        selected = tf.one_hot(channels, 12, dtype=self.compute_dtype)
        if not self.skip_padding:
            encoded = tf.reshape(self.cnn(selected), [batch, steps, 128])
            return tf.where(valid[..., None], encoded, tf.zeros_like(encoded))
        # A dummy position also makes an all-padding batch safe for Conv2D.
        selected = tf.concat([tf.zeros((1, 8, 8, 12), selected.dtype), selected], axis=0)
        encoded = self.cnn(selected)[1:]
        scattered = tf.scatter_nd(indices[:, None], encoded, [batch * steps, 128])
        return tf.reshape(scattered, [batch, steps, 128])

    def compute_output_shape(self, input_shape):
        return (*input_shape[0][:2], 128)

    def get_config(self):
        return {**super().get_config(), "skip_padding": self.skip_padding}


