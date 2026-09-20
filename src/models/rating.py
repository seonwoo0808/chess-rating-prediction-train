"""Build and compile the original CNN–Transformer rating model."""
from tensorflow import keras
from tensorflow.keras import layers

from .cnn import BoardEncoder
from .layers import MaskedMean
from .transformer import encode_sequence


def build_model(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")], 
        jit_compile=False, 
        skip_padding=False):
    # Fixed board orientation; channels: six white pieces, then six black pieces.
    # Each position follows one ply. valid_steps excludes padded positions.
    boards = keras.Input((128, 8, 8), dtype="int8", name="boards")
    valid = keras.Input((128,), dtype="bool", name="valid_steps")
    x = BoardEncoder(skip_padding=skip_padding,
                     name="encode_positions")([boards, valid])
    x = encode_sequence(x, valid)
    x = MaskedMean(name="masked_mean", dtype="float32")([x, valid])
    x = layers.Dense(128, activation="gelu")(x)
    output = layers.Dense(2, dtype="float32", name="white_black_rating")(x)
    model = keras.Model([boards, valid], output, name="chess_rating_cnn_transformer")
    model.compile(
        optimizer=optimizer,
        loss="mse", 
        metrics=metrics,
        jit_compile=jit_compile
    )
    return model

