import unittest

import torch

from train.models import build_model


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = build_model()
        self.boards = torch.randint(-6, 7, (3, 8, 8, 8), dtype=torch.int8)
        self.valid = torch.tensor([[True] * 8, [True] * 3 + [False] * 5, [False] * 8])

    def test_shape_padding_invariance_and_empty_game(self):
        self.model.eval()
        with torch.no_grad():
            expected = self.model(self.boards, self.valid)
            changed = self.boards.clone()
            changed[~self.valid] = 99  # Padding is never expanded into piece channels.
            actual = self.model(changed, self.valid)
        self.assertEqual(actual.shape, (3, 2))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_backward_and_optimizer_change_parameters(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        previous = self.model.board_encoder.cnn[0].weight.detach().clone()
        loss = (self.model(self.boards, self.valid) - 1500).square().mean()
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        self.assertFalse(torch.equal(previous, self.model.board_encoder.cnn[0].weight))

    def test_all_padding_backward_is_finite(self):
        result = self.model(self.boards, torch.zeros_like(self.valid))
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(result).all())
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_cpu_autocast_keeps_output_float32(self):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = self.model(self.boards, self.valid)
        result.square().mean().backward()
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.isfinite(result).all())
