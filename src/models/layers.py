"""Serializable position, attention-mask, and pooling layers."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="ChessRating")
class PositionIndices(layers.Layer):
    def call(self, inputs):
        return tf.tile(tf.range(tf.shape(inputs)[1])[None, :], [tf.shape(inputs)[0], 1])

    def compute_output_shape(self, input_shape):
        return input_shape[:2]


@keras.utils.register_keras_serializable(package="ChessRating")
class AttentionMask(layers.Layer):
    def call(self, inputs):
        return inputs[:, None, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, input_shape[1])


@keras.utils.register_keras_serializable(package="ChessRating")
class MaskedMean(layers.Layer):
    def call(self, inputs):
        values, valid = inputs
        # Accumulate in float32 even under mixed precision.
        values = tf.cast(values, tf.float32)
        weights = tf.cast(valid, tf.float32)
        return tf.reduce_sum(values * weights[..., None], axis=1) / tf.maximum(
            tf.reduce_sum(weights, axis=1, keepdims=True), 1.0
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])


