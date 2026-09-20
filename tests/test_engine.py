import unittest

import torch

from train.engine import run_epoch


class ZeroModel(torch.nn.Module):
    def forward(self, boards, valid):
        return torch.zeros((len(boards), 2))


class EngineTests(unittest.TestCase):
    def test_metrics_weight_partial_batch_by_game_count(self):
        batches = [((torch.zeros(n, 1, 8, 8, dtype=torch.int8), torch.ones(n, 1, dtype=torch.bool)),
                    torch.full((n, 2), rating)) for n, rating in ((3, 2.), (1, 10.))]
        metrics = run_epoch(ZeroModel(), batches, device=torch.device("cpu"), precision="float32")
        self.assertAlmostEqual(metrics["loss"], 17.139675)
        self.assertAlmostEqual(metrics["origin_mae"], 1654.0)

    def test_nonfinite_loss_fails(self):
        batches = [((torch.zeros(1, 1, 8, 8), torch.ones(1, 1, dtype=torch.bool)),
                    torch.full((1, 2), float("nan")))]
        with self.assertRaises(FloatingPointError):
            run_epoch(ZeroModel(), batches, device=torch.device("cpu"), precision="float32")
