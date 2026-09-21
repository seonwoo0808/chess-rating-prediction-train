import unittest

import torch

from train.engine import run_epoch


class ZeroModel(torch.nn.Module):
    def forward(self, boards, valid):
        return torch.zeros((len(boards), 2))


class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output = torch.nn.Parameter(torch.full((2,), 10.0))

    def forward(self, boards, valid):
        return self.output.expand(len(boards), -1)


class EngineTests(unittest.TestCase):
    def test_metrics_weight_partial_batch_by_game_count(self):
        batches = [((torch.zeros(n, 1, 8, 8, dtype=torch.int8), torch.ones(n, 1, dtype=torch.bool)),
                    torch.full((n, 2), rating)) for n, rating in ((3, 2060.), (1, 2860.))]
        metrics = run_epoch(ZeroModel(), batches, device=torch.device("cpu"), precision="float32")
        self.assertEqual(metrics, {"loss": 3.0, "origin_mae": 600.0})

    def test_clipping_bounds_update_after_unscaling(self):
        batches = [((torch.zeros(1, 1, 8, 8), torch.ones(1, 1, dtype=torch.bool)),
                    torch.full((1, 2), 1660.0))]
        for precision, scaling in (("float32", False), ("bfloat16", False), ("float32", True)):
            with self.subTest(precision=precision, scaling=scaling):
                model = ConstantModel()
                before = model.output.detach().clone()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                scaler = torch.amp.GradScaler("cpu", enabled=scaling, init_scale=8.0)
                metrics = run_epoch(model, batches, device=torch.device("cpu"),
                                    precision=precision, optimizer=optimizer, scaler=scaler)
                self.assertAlmostEqual((before - model.output).norm().item(), 0.1, places=5)
                self.assertEqual(metrics, {"loss": 100.0, "origin_mae": 4000.0})

    def test_nonfinite_gradient_stops_before_optimizer_update(self):
        model = ConstantModel()
        before = model.output.detach().clone()
        model.output.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        optimizer = torch.optim.Adam(model.parameters())
        batches = [((torch.zeros(1, 1, 8, 8), torch.ones(1, 1, dtype=torch.bool)),
                    torch.full((1, 2), 1660.0))]
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            run_epoch(model, batches, device=torch.device("cpu"), precision="bfloat16",
                      optimizer=optimizer, scaler=torch.amp.GradScaler("cpu", enabled=False))
        torch.testing.assert_close(model.output, before, rtol=0, atol=0)
        self.assertFalse(optimizer.state)

    def test_nonfinite_loss_fails(self):
        batches = [((torch.zeros(1, 1, 8, 8), torch.ones(1, 1, dtype=torch.bool)),
                    torch.full((1, 2), float("nan")))]
        with self.assertRaises(FloatingPointError):
            run_epoch(ZeroModel(), batches, device=torch.device("cpu"), precision="float32")
