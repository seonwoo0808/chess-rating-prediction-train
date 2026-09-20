"""CNN–Transformer regression of white and black ratings."""
import torch
from torch import nn

from ..data.constants import MAX_PLIES
from .cnn import BoardEncoder
from .transformer import TransformerBlock


class RatingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.board_encoder = BoardEncoder()
        self.positions = nn.Embedding(MAX_PLIES, 128)
        self.blocks = nn.ModuleList(TransformerBlock() for _ in range(4))
        self.norm = nn.LayerNorm(128, eps=1e-3)
        self.head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 2))

    def forward(self, boards, valid_steps):
        if boards.ndim != 4 or boards.shape[2:] != (8, 8):
            raise ValueError("boards must have shape [batch, steps, 8, 8]")
        if valid_steps.shape != boards.shape[:2] or valid_steps.dtype != torch.bool:
            raise ValueError("valid_steps must be bool [batch, steps]")
        steps = boards.shape[1]
        if not 1 <= steps <= MAX_PLIES:
            raise ValueError(f"steps must be in [1, {MAX_PLIES}]")
        values = self.board_encoder(boards, valid_steps)
        values = values + self.positions(torch.arange(steps, device=boards.device))
        padding_mask = ~valid_steps.clone()
        # An empty game still needs one attention key to avoid all-masked NaNs.
        # Pooling below excludes that key, so the game has a zero representation.
        padding_mask[:, 0] &= valid_steps.any(dim=1)
        for block in self.blocks:
            values = block(values, padding_mask)
        values = self.norm(values).float()
        weights = valid_steps.unsqueeze(-1)
        pooled = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        # Ratings and regression loss stay float32 during mixed precision training.
        with torch.autocast(device_type=boards.device.type, enabled=False):
            return self.head(pooled)


def build_model():
    return RatingModel()
