"""Asynchronous residency, cleanup, error propagation and multi-file coverage."""
from contextlib import closing
from pathlib import Path
from threading import Event
from unittest.mock import patch
import gc
import tempfile
import unittest
import weakref

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from train.data import build_datasets
from train.data.batches import async_row_batches
from train.data.prefetch import prefetched_columns
from train.data.split import FileSelection, split_plan


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = []
        for i in range(3):
            path = Path(self.tmp.name) / f'{i}.parquet'
            move = (12 | (28 << 6)).to_bytes(2, 'little')
            table = pa.table({
                'white_elo': pa.array(range(i * 5 + 1000, i * 5 + 1005), type=pa.uint32()),
                'black_elo': pa.array(range(i * 5 + 1500, i * 5 + 1505), type=pa.uint32()),
                'ply_list': pa.array([[{'movement': move, 'time': 1000 + i * 5 + j}] for j in range(5)],
                                     type=pa.list_(pa.struct([('movement', pa.binary(2)),
                                                            ('time', pa.uint32())]))),
            })
            pq.write_table(table, path, row_group_size=2)
            self.paths.append(path)
        self.selections = [FileSelection(path, 0, 5) for path in self.paths]

    def test_split_coverage_batches_and_repeated_epochs(self):
        total, split, training, validation = split_plan(self.paths, 13, 0.3)
        self.assertEqual((total, split), (13, 9))
        for decoder in ('python', 'numba'):
            blocks = list(async_row_batches(training, read_batch_size=2,
                          generator_batch_size=4, decoder=decoder))
            self.assertEqual([len(y) for _, y in blocks], [4, 4, 1])
            np.testing.assert_array_equal(np.concatenate([y[:, 0] for _, y in blocks]),
                                          range(1000, 1009))
        train, val = build_datasets(self.paths, max_games=13, validation_size=0.3,
                                   batch_size=4, decoder='python')
        for _ in range(2):
            batches = list(train)
            for (_, _, clocks), targets in batches:
                torch.testing.assert_close(clocks[:, 0, 0], targets[:, 0])
                self.assertTrue((clocks[:, 0, 1] == 1).all())
                self.assertFalse(clocks[:, 1:].any())
            ys = np.concatenate([y[:, 0] for _, y in batches])
            np.testing.assert_array_equal(np.sort(ys), range(1000, 1009))
        ys = np.concatenate([y[:, 0] for _, y in val])
        np.testing.assert_array_equal(ys, range(1009, 1013))

    def test_overlap_two_file_limit_and_projection(self):
        from train.data.prefetch import load_file
        references = []
        second_ready = Event()
        def tracked(selection, cancelled):
            # Loading counts as a slot, so at most one older table may remain.
            self.assertLessEqual(sum(r() is not None for r in references), 1)
            table = load_file(selection, cancelled)
            self.assertEqual([f.name for f in table.column('ply_list').type.value_type],
                             ['movement', 'time'])
            references.append(weakref.ref(table))
            if selection.path == self.paths[1]:
                second_ready.set()
            return table
        with patch('train.data.prefetch.load_file', side_effect=tracked):
            with closing(prefetched_columns(self.selections, read_batch_size=2)) as source:
                first = next(source)
                self.assertTrue(second_ready.wait(5), 'Next file was not loaded during consumption')
                self.assertEqual(sum(r() is not None for r in references), 2)
                del first
                count = 2
                for item in source:
                    count += len(item[3])
                    del item
                self.assertEqual(count, 15)
        gc.collect()
        self.assertTrue(all(r() is None for r in references))

    def test_early_close_cancels_worker(self):
        from train.data.prefetch import load_file
        started, finished = Event(), Event()
        def waiting(selection, cancelled):
            if selection.path == self.paths[1]:
                started.set()
                try:
                    self.assertTrue(cancelled.wait(5))
                    return None
                finally:
                    finished.set()
            return load_file(selection, cancelled)
        with patch('train.data.prefetch.load_file', side_effect=waiting):
            source = async_row_batches(self.selections, generator_batch_size=2, decoder='python')
            next(source)
            self.assertTrue(started.wait(5))
            source.close()
            self.assertTrue(finished.is_set())

    def test_worker_exception_reaches_consumer(self):
        from train.data.prefetch import load_file
        def failing(selection, cancelled):
            if selection.path == self.paths[1]:
                raise OSError('simulated read failure')
            return load_file(selection, cancelled)
        with patch('train.data.prefetch.load_file', side_effect=failing):
            with self.assertRaisesRegex(OSError, 'simulated read failure'):
                list(async_row_batches(self.selections, decoder='python'))

    def test_duplicate_and_empty_files(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            split_plan([self.paths[0], self.paths[0]], None, 0.2)
        empty = Path(self.tmp.name) / 'empty.parquet'
        pq.write_table(pq.read_table(self.paths[0]).slice(0, 0), empty)
        total, split, train, val = split_plan([empty, *self.paths], None, 0.2)
        self.assertEqual((total, split), (15, 12))
        self.assertNotIn(empty, [s.path for s in (*train, *val)])

    def test_batch_prefetch_preserves_all_tensors_epochs_and_partial_batches(self):
        for decoder in ('python', 'numba'):
            options = dict(max_games=13, validation_size=0.3, batch_size=4,
                           shuffle_buffer=5, decoder=decoder, read_batch_size=2, seed=11)
            inline = build_datasets(self.paths, prefetch_batches=0, **options)
            ahead = build_datasets(self.paths, prefetch_batches=2, **options)
            for epoch in range(3):
                for original, prefetched in zip(inline, ahead):
                    original.set_epoch(epoch)
                    prefetched.set_epoch(epoch)
                    expected, actual = list(original), list(prefetched)
                    self.assertEqual(len(expected), len(actual))
                    for ((eb, ev, et), ey), ((ab, av, at), ay) in zip(expected, actual):
                        for left, right in ((eb, ab), (ev, av), (et, at), (ey, ay)):
                            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_dataset_close_cancels_nested_file_load(self):
        from train.data.prefetch import load_file
        started, finished = Event(), Event()
        def waiting(selection, cancelled, *, stop_event=None):
            if selection.path == self.paths[1]:
                started.set()
                try:
                    self.assertIsNotNone(stop_event)
                    self.assertTrue(stop_event.wait(5), 'Batch cancellation did not reach file load')
                    return None
                finally:
                    finished.set()
            return load_file(selection, cancelled, stop_event=stop_event)
        train, _ = build_datasets(self.paths, batch_size=2, shuffle_buffer=2,
                                  decoder='python', prefetch_batches=2)
        with patch('train.data.prefetch.load_file', side_effect=waiting):
            with closing(iter(train)) as batches:
                next(batches)
                self.assertTrue(started.wait(5))
            self.assertTrue(finished.is_set())

    def test_invalid_batch_prefetch_size(self):
        for size in (-1, 1.5):
            with self.assertRaisesRegex(ValueError, 'prefetch_batches'):
                build_datasets(self.paths, prefetch_batches=size)


if __name__ == '__main__':
    unittest.main()
