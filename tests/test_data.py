"""Board replay, Parquet slicing, tensor batches and split coverage."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from train.data import board_sequence, build_datasets
from train.data.batches import row_batches
from train.data.split import split_counts
import torch


def moves(*uci):
    def square(s):
        return (ord(s[0]) - ord('a')) + (int(s[1]) - 1) * 8
    return [square(m[:2]) | (square(m[2:4]) << 6) |
            ((" nbrq".index(m[4]) if len(m) == 5 else 0) << 12) for m in uci]


CASTLE = moves('e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'g8f6', 'e1g1')
EN_PASSANT = moves('e2e4', 'a7a6', 'e4e5', 'd7d5', 'e5d6')
PROMOTION = moves('a2a4', 'h7h5', 'a4a5', 'h5h4', 'a5a6', 'h4h3',
                  'a6b7', 'h3g2', 'b7a8q')


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'games.parquet'
        self.games = [CASTLE, EN_PASSANT, PROMOTION, [], None,
                      moves('e2e4'), moves('d2d4', 'd7d5')]
        ply_type = pa.list_(pa.struct([('movement', pa.binary(2)), ('time', pa.uint32())]))
        self.times = [None if g is None else [None if j % 3 == 1 else i * 100 + j
                                            for j in range(len(g))]
                      for i, g in enumerate(self.games)]
        plies = [None if g is None else [dict(movement=m.to_bytes(2, 'little'), time=t)
                                       for m, t in zip(g, ts)]
                 for g, ts in zip(self.games, self.times)]
        pq.write_table(pa.table({
            'ply_list': pa.array(plies, type=ply_type),
            'white_elo': pa.array(range(1000, 1007), type=pa.uint32()),
            'black_elo': pa.array(range(1500, 1507), type=pa.uint32()),
        }), self.path, row_group_size=3)

    def test_special_moves(self):
        board, valid = board_sequence(CASTLE)
        self.assertEqual(valid.sum(), 7)
        self.assertEqual(board[6, 0, 6], 6)
        self.assertEqual(board[6, 0, 5], 4)
        self.assertEqual(board[6, 0, 7], 0)
        board, _ = board_sequence(EN_PASSANT)
        self.assertEqual(board[4, 5, 3], 1)
        self.assertEqual(board[4, 4, 3], 0)
        board, _ = board_sequence(PROMOTION)
        self.assertEqual(board[8, 7, 0], 5)

    def test_decoders_and_partial_ranges(self):
        for decoder in ('python', 'numba'):
            blocks = list(row_batches(self.path, None, start=1, stop=7,
                          read_batch_size=2, generator_batch_size=4, decoder=decoder))
            self.assertEqual([len(y) for _, y in blocks], [4, 2])
            boards = np.concatenate([x[0] for x, _ in blocks])
            valid = np.concatenate([x[1] for x, _ in blocks])
            targets = np.concatenate([y for _, y in blocks])
            clocks = np.concatenate([x[2] for x, _ in blocks])
            for i in range(6):
                np.testing.assert_array_equal(clocks[i], self.expected_clocks(i + 1))
            for i, game in enumerate(self.games[1:]):
                expected_board, expected_valid = board_sequence(game)
                np.testing.assert_array_equal(boards[i], expected_board)
                np.testing.assert_array_equal(valid[i], expected_valid)
            np.testing.assert_array_equal(targets[:, 0], range(1001, 1007))
            np.testing.assert_array_equal(targets[:, 1], range(1501, 1507))

    def test_split_and_model_input(self):
        self.assertEqual(split_counts(self.path, None, 0.3), (7, 4))
        self.assertEqual(split_counts(self.path, 5, 0.3), (5, 3))
        train, validation = build_datasets(self.path, batch_size=3, validation_size=0.3,
                               read_batch_size=2, decoder='python')
        training = list(train)
        checking = list(validation)
        self.assertEqual([len(y) for _, y in training], [3, 1])
        self.assertEqual([len(y) for _, y in checking], [3])
        self.assertEqual(set(np.concatenate([y[:, 0] for _, y in training])), set(range(1000,1004)))
        np.testing.assert_array_equal(checking[0][1][:, 0], range(1004, 1007))
        for (boards, valid, clocks), targets in training + checking:
            for clock, target in zip(clocks, targets):
                np.testing.assert_array_equal(clock.numpy(), self.expected_clocks(int(target[0]) - 1000))
        from train.models import build_model
        model = build_model()
        model.eval()
        with torch.no_grad():
            prediction = model(*training[0][0]).numpy()
        self.assertEqual(prediction.shape, (3, 2))
        self.assertTrue(np.isfinite(prediction).all())

    def expected_clocks(self, index):
        result = np.zeros((128, 2), np.float32)
        for j, seconds in enumerate(self.times[index] or []):
            if seconds is not None:
                result[j] = [seconds, 1]
        return result

    def test_invalid_clocks_and_missing_column(self):
        from train.data.parquet import arrow_numpy_columns
        def batch(times, dtype=pa.float64()):
            return pa.record_batch({
                'white_elo': [1200], 'black_elo': [1400],
                'ply_list': pa.array([[{'movement': 12 | (28 << 6), 'time': t} for t in times]],
                                    type=pa.list_(pa.struct([('movement', pa.uint16()), ('time', dtype)]))),
            })
        for value in (-1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'time'):
                arrow_numpy_columns(batch([value]))
        with self.assertRaisesRegex(TypeError, 'time'):
            arrow_numpy_columns(batch(['bad'], pa.string()))
        np.testing.assert_array_equal(arrow_numpy_columns(batch([None, 0, 60]))[-1],
                                      [[0, 0], [0, 1], [60, 1]])
        np.testing.assert_array_equal(arrow_numpy_columns(batch([None], pa.null()))[-1], [[0, 0]])
        # Legacy inputs with no clock field stay usable, but contribute no time embedding.
        for plies in (pa.array([[12 | (28 << 6)]], type=pa.list_(pa.uint16())),
                      pa.array([[{'movement': 12 | (28 << 6)}]],
                               type=pa.list_(pa.struct([('movement', pa.uint16())])))):
            pq.write_table(pa.table({'white_elo': [1200], 'black_elo': [1400],
                                     'ply_list': plies}), self.path)
            (_, _, clocks), _ = next(row_batches(self.path, None, decoder='python'))
            self.assertFalse(clocks.any())

    def test_clock_truncation_and_early_replay_stop(self):
        # Repeated knight cycles provide a legal sequence longer than MAX_PLIES.
        game = moves('g1f3', 'g8f6', 'f3g1', 'f6g8') * 33
        broken = moves('e2e4', 'e2e3', 'e7e5')  # Empty source ends decoding at ply 2.
        plies = [[{'movement': move, 'time': i} for i, move in enumerate(g)]
                 for g in (game, broken)]
        pq.write_table(pa.table({'white_elo': [1200, 1300], 'black_elo': [1400, 1500],
                                 'ply_list': pa.array(plies)}), self.path)
        for decoder in ('python', 'numba'):
            (_, valid, clocks), _ = next(row_batches(self.path, None, decoder=decoder))
            self.assertEqual(valid.sum(axis=1).tolist(), [128, 1])
            np.testing.assert_array_equal(clocks[0, :, 0], np.arange(128))
            self.assertTrue((clocks[0, :, 1] == 1).all())
            np.testing.assert_array_equal(clocks[1, 0], [0, 1])
            self.assertFalse(clocks[1, 1:].any())

    def test_invalid_data(self):
        with self.assertRaises(ValueError):
            board_sequence([12 | (28 << 6) | (5 << 12)])
        from train.data.parquet import movement_numpy
        with self.assertRaises(ValueError):
            movement_numpy(pa.array([b'x'], type=pa.binary()))
        with self.assertRaises(ValueError):
            movement_numpy(pa.array([None], type=pa.binary(2)))


if __name__ == '__main__':
    unittest.main()
