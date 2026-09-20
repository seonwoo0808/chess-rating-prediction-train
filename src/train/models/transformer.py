"""Pre-normalized attention and feed-forward blocks."""
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(128, eps=1e-3)
        self.attention = nn.MultiheadAttention(128, 4, dropout=0.1, batch_first=True)
        self.feed_forward_norm = nn.LayerNorm(128, eps=1e-3)
        self.feed_forward = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 128),
        )

    def forward(self, values, padding_mask):
        normalized = self.attention_norm(values)
        attended, _ = self.attention(
            normalized, normalized, normalized,
            key_padding_mask=padding_mask, need_weights=False,
        )
        values = values + attended
        return values + self.feed_forward(self.feed_forward_norm(values))
