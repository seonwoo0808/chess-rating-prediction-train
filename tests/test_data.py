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
        plies = [None if g is None else [dict(movement=m.to_bytes(2, 'little'), time=None)
                                       for m in g] for g in self.games]
        pq.write_table(pa.table({
            'ply_list': pa.array(plies, type=ply_type),
            "game_type": pa.array([[j == i % 4 for j in range(4)] for i in range(7)], type=pa.list_(pa.bool_())),
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
            types = np.concatenate([x[2] for x, _ in blocks])
            np.testing.assert_array_equal(types, np.eye(4)[np.arange(1, 7) % 4])
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
        for (boards, valid, game_type), targets in training + checking:
            expected = torch.eye(4)[(targets[:, 0].long() - 1000) % 4]
            torch.testing.assert_close(game_type, expected)
        from train.models import build_model
        model = build_model()
        model.eval()
        with torch.no_grad():
            prediction = model(*training[0][0]).numpy()
        self.assertEqual(prediction.shape, (3, 2))
        self.assertTrue(np.isfinite(prediction).all())

    def test_invalid_game_types_and_sliced_fixed_lists(self):
        from train.data.parquet import arrow_numpy_columns, projected_columns
        table = pq.read_table(self.path).combine_chunks()
        index = table.schema.get_field_index("game_type")
        for bad in (None, [True, False], [False] * 4, [True] * 4,
                    [True, None, False, False], [float("nan"), 0., 0., 0.],
                    [0.5, 0.5, 0., 0.], ["1", "0", "0", "0"]):
            with self.subTest(bad=bad):
                values = pa.array([bad] * len(table))
                batch = table.set_column(index, "game_type", values).to_batches()[0]
                with self.assertRaisesRegex(ValueError, "game_type"):
                    arrow_numpy_columns(batch)
        without = table.drop(["game_type"])
        with self.assertRaisesRegex(ValueError, "game_type"):
            arrow_numpy_columns(without.to_batches()[0])
        pq.write_table(without, self.path)
        with pq.ParquetFile(self.path) as source:
            with self.assertRaisesRegex(ValueError, "game_type"):
                projected_columns(source)
        for dtype in (pa.list_(pa.bool_(), 4), pa.large_list(pa.bool_())):
            values = pa.array([[j == i % 4 for j in range(4)] for i in range(7)], type=dtype)
            batch = table.set_column(index, "game_type", values).slice(2, 3).to_batches()[0]
            np.testing.assert_array_equal(arrow_numpy_columns(batch)[-1],
                                          np.eye(4)[np.arange(2, 5) % 4])

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
