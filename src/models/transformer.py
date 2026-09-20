"""Position embeddings and the four-block Transformer stack."""
from tensorflow.keras import layers

from .layers import AttentionMask, PositionIndices


def encode_sequence(x, valid):
    positions = PositionIndices(name="position_indices")(valid)
    x = layers.Add()([x, layers.Embedding(128, 128)(positions)])
    mask = AttentionMask(name="attention_mask")(valid)
    for i in range(4):
        y = layers.LayerNormalization()(x)
        y = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(
            y, y, attention_mask=mask
        )
        x = layers.Add()([x, y])
        y = layers.LayerNormalization()(x)
        y = layers.Dense(256, activation="gelu")(y)
        y = layers.Dropout(0.1)(y)
        y = layers.Dense(128)(y)
        x = layers.Add(name=f"transformer_{i + 1}")([x, y])
    x = layers.LayerNormalization()(x)
    return x
