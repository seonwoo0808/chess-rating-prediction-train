import copy
import unittest

import torch
from torch.nn import functional as F

from train.models import build_model


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = build_model()
        self.boards = torch.randint(-6, 7, (3, 8, 8, 8), dtype=torch.int8)
        self.clocks = torch.stack((torch.arange(24).reshape(3, 8).float(),
                                   torch.ones(3, 8)), dim=-1)
        self.valid = torch.tensor([[True] * 8, [True] * 3 + [False] * 5, [False] * 8])

    def test_shape_padding_invariance_and_empty_game(self):
        self.model.eval()
        with torch.no_grad():
            expected = self.model(self.boards, self.valid, self.clocks)
            changed = self.boards.clone()
            changed[~self.valid] = 99  # Padding is sanitized before piece-channel expansion.
            clocks = self.clocks.clone()
            clocks[~self.valid] = float('nan')
            actual = self.model(changed, self.valid, clocks)
        self.assertEqual(actual.shape, (3, 2))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_clock_conditions_predictions_and_receives_gradients(self):
        self.model.eval()
        boards = self.boards[:1].expand(3, -1, -1, -1)
        valid = self.valid[:1].expand(3, -1)
        clocks = torch.zeros(3, 8, 2)
        clocks[1, :, 1] = 1  # Real zero seconds differs from missing.
        clocks[2, :, 0] = 180
        clocks[2, :, 1] = 1
        result = self.model(boards, valid, clocks)
        self.assertGreater((result[0] - result[1]).abs().max().item(), 1e-6)
        self.assertGreater((result[1] - result[2]).abs().max().item(), 1e-6)
        result.square().sum().backward()
        grad = self.model.clock_projection.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue((grad.abs().sum(dim=0) > 0).all())
        with self.assertRaisesRegex(ValueError, "clocks"):
            self.model(boards, valid, torch.ones(3, 4))

    def test_clock_scaling_missing_padding_and_saved_constants(self):
        clocks = torch.tensor([[[0., 1.], [60., 1.], [float('nan'), 0.],
                                [float('nan'), 1.]]])
        valid = torch.tensor([[True, True, True, False]])
        features = self.model.normalize_clocks(clocks, valid)
        expected = torch.tensor([[[-4 / 1.2, 1.],
                                  [(torch.log1p(torch.tensor(60.)).item() - 4) / 1.2, 1.],
                                  [0., 0.], [0., 0.]]])
        torch.testing.assert_close(features, expected)
        restored = build_model()
        restored.load_state_dict(self.model.state_dict())
        self.assertEqual(restored.clock_log_mean.item(), 4.0)
        self.assertAlmostEqual(restored.clock_log_std.item(), 1.2)
        self.model.eval()
        clocks = self.clocks.clone()
        clocks[..., 1] = 0
        with torch.no_grad():
            expected = self.model(self.boards, self.valid, clocks)
            clocks[..., 0] = float('nan')
            actual = self.model(self.boards, self.valid, clocks)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cnn_size_does_not_depend_on_valid_count(self):
        encoder = self.model.board_encoder
        shapes = []
        hook = encoder.cnn.register_forward_pre_hook(
            lambda module, inputs: shapes.append(tuple(inputs[0].shape)))
        try:
            with torch.no_grad():
                for valid in (self.valid, torch.ones_like(self.valid),
                              torch.zeros_like(self.valid)):
                    features = encoder(self.boards, valid)
                    self.assertTrue(torch.equal(features[~valid], torch.zeros_like(features[~valid])))
                encoder(self.boards[:1], self.valid[:1])
        finally:
            hook.remove()
        self.assertEqual(shapes, [(25, 12, 8, 8)] * 3 + [(9, 12, 8, 8)])

    def test_encoder_matches_valid_only_outputs_and_gradients(self):
        encoder = self.model.board_encoder
        reference_cnn = copy.deepcopy(encoder.cnn)
        indices = self.valid.reshape(-1).nonzero(as_tuple=True)[0]
        pieces = self.boards.reshape(-1, 8, 8)[indices].long()
        channels = pieces.abs() + (pieces < 0) * 6
        encoded = F.one_hot(channels, 13)[..., 1:].permute(0, 3, 1, 2).float()
        selected = reference_cnn(torch.cat((encoded.new_zeros((1, 12, 8, 8)), encoded)))[1:]
        expected = selected.new_zeros((24, 128)).index_copy(0, indices, selected).reshape(3, 8, 128)
        actual = encoder(self.boards, self.valid)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
        upstream = torch.randn_like(actual)
        (actual * upstream).sum().backward()
        (expected * upstream).sum().backward()
        for parameter, reference in zip(encoder.cnn.parameters(), reference_cnn.parameters()):
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=3e-4, atol=3e-5)

    def test_backward_and_optimizer_change_parameters(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        previous = self.model.board_encoder.cnn[0].weight.detach().clone()
        loss = (self.model(self.boards, self.valid, self.clocks) - 1500).square().mean()
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        self.assertFalse(torch.equal(previous, self.model.board_encoder.cnn[0].weight))

    def test_all_padding_backward_is_finite(self):
        result = self.model(self.boards, torch.zeros_like(self.valid), self.clocks)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(result).all())
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_cpu_autocast_keeps_output_float32(self):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = self.model(self.boards, self.valid, self.clocks)
        result.square().mean().backward()
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.isfinite(result).all())
