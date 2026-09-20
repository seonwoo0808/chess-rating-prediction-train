from contextlib import closing
from threading import Event, current_thread
import unittest
from unittest.mock import patch

from train.data.batch_prefetch import prefetched_batches


class BatchPrefetchTests(unittest.TestCase):
    def test_produces_while_consumer_is_idle_and_bounds_read_ahead(self):
        pending, finished = Event(), Event()
        produced, workers = [], []

        def source(cancelled):
            workers.append(current_thread())
            try:
                for index in range(100):
                    produced.append(index)
                    if index == 3:
                        pending.set()
                    yield index
            finally:
                finished.set()

        with closing(prefetched_batches(source, 2)) as batches:
            self.assertEqual(next(batches), 0)
            # Consumer does no more reads: producer fills two slots, then holds
            # one pending item. The next item must wait for queue capacity.
            self.assertTrue(pending.wait(5))
            self.assertEqual(produced, [0, 1, 2, 3])
        self.assertTrue(finished.is_set())
        self.assertFalse(workers[0].is_alive())

    def test_order_eof_and_empty_input(self):
        closed = Event()
        def source(cancelled):
            try:
                yield from range(17)
            finally:
                closed.set()
        self.assertEqual(list(prefetched_batches(source, 2)), list(range(17)))
        self.assertTrue(closed.is_set())
        self.assertEqual(list(prefetched_batches(lambda cancelled: (item for item in ()), 1)), [])

    def test_errors_are_delivered_after_prior_batches_and_source_closes(self):
        closed, workers = Event(), []
        def source(cancelled):
            workers.append(current_thread())
            try:
                yield 1
                yield 2
                raise OSError("decoder failure")
            finally:
                closed.set()
        with closing(prefetched_batches(source, 1)) as batches:
            self.assertEqual(next(batches), 1)
            self.assertEqual(next(batches), 2)
            with self.assertRaisesRegex(OSError, "decoder failure"):
                next(batches)
        self.assertTrue(closed.is_set())
        self.assertFalse(workers[0].is_alive())

    def test_factory_error_and_unstarted_close(self):
        def broken(cancelled):
            raise ValueError("cannot open source")
        with self.assertRaisesRegex(ValueError, "cannot open source"):
            next(prefetched_batches(broken, 1))
        with patch("train.data.batch_prefetch.Thread") as thread:
            prefetched_batches(broken, 1).close()
        thread.assert_not_called()

    def test_close_signals_blocked_source_before_joining(self):
        waiting, closed = Event(), Event()
        workers = []
        def source(cancelled):
            workers.append(current_thread())
            try:
                yield 1
                waiting.set()
                if not cancelled.wait(5):
                    raise AssertionError("Cancellation did not reach producer")
            finally:
                closed.set()
        source_iter = prefetched_batches(source, 1)
        self.assertEqual(next(source_iter), 1)
        self.assertTrue(waiting.wait(5))
        source_iter.close()
        self.assertTrue(closed.is_set())
        self.assertFalse(workers[0].is_alive())


if __name__ == "__main__":
    unittest.main()
