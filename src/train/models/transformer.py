"""Pre-normalized attention and feed-forward blocks with rotary positions."""
import torch
from torch import nn
from torch.nn import functional as F


class RotaryPositionEmbedding(nn.Module):
    """Apply RoPE to attention heads using their zero-based ply indices."""

    def __init__(self, head_dim, base=10_000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        frequencies = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inverse_frequencies", frequencies, persistent=False)

    def forward(self, values):
        # values: [batch, heads, plies, head_dim]. Frequencies and angles are
        # calculated in float32 for stability under bfloat16/float16 training.
        steps = values.shape[-2]
        positions = torch.arange(steps, device=values.device, dtype=torch.float32)
        angles = torch.outer(positions, self.inverse_frequencies.float())
        angles = torch.cat((angles, angles), dim=-1)
        cos = angles.cos().to(dtype=values.dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=values.dtype)[None, None, :, :]
        half = values.shape[-1] // 2
        rotated = torch.cat((-values[..., half:], values[..., :half]), dim=-1)
        return values * cos + rotated * sin


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(128, eps=1e-3)
        self.num_heads = 4
        self.head_dim = 128 // self.num_heads
        self.query = nn.Linear(128, 128)
        self.key = nn.Linear(128, 128)
        self.value = nn.Linear(128, 128)
        self.rotary = RotaryPositionEmbedding(self.head_dim)
        self.attention_output = nn.Linear(128, 128)
        self.attention_dropout = 0.1
        self.feed_forward_norm = nn.LayerNorm(128, eps=1e-3)
        self.feed_forward = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 128),
        )

    def forward(self, values, padding_mask):
        normalized = self.attention_norm(values)
        batch, steps, _ = normalized.shape

        def split_heads(projection):
            return projection(normalized).reshape(
                batch, steps, self.num_heads, self.head_dim
            ).transpose(1, 2)

        query = self.rotary(split_heads(self.query))
        key = self.rotary(split_heads(self.key))
        value = split_heads(self.value)
        # SDPA's boolean mask uses True for keys that may be attended to.
        attention_mask = (~padding_mask)[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(batch, steps, 128)
        values = values + self.attention_output(attended)
        return values + self.feed_forward(self.feed_forward_norm(values))
