"""Shared CNN for signed piece IDs: empty=0, white=1..6, black=-1..-6."""
import torch
from torch import nn
from torch.nn import functional as F


class BoardEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(128 * 8 * 8, 128),
        )

    def forward(self, boards, valid):
        batch, steps = valid.shape
        indices = valid.reshape(-1).nonzero(as_tuple=True)[0]
        pieces = boards.reshape(-1, 8, 8)[indices].long()
        # Empty squares use channel 0, which is removed after one-hot encoding.
        channels = pieces.abs() + (pieces < 0) * 6
        encoded = F.one_hot(channels, 13)[..., 1:].permute(0, 3, 1, 2).float()
        # Keep every parameter in the graph even for an entirely empty batch.
        dummy = encoded.new_zeros((1, 12, 8, 8))
        features = self.cnn(torch.cat((dummy, encoded)))[1:]
        output = features.new_zeros((batch * steps, 128))
        return output.index_copy(0, indices, features).reshape(batch, steps, 128)
