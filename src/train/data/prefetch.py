"""One active Arrow file and one background load; no byte-based memory limit."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Event
import logging
import time

import pyarrow as pa
import pyarrow.parquet as pq

from .parquet import arrow_numpy_columns, projected_columns

logger = logging.getLogger(__name__)


def load_file(selection, cancelled):
    """Load selected row groups, keeping Arrow chunks instead of copying them."""
    started = time.perf_counter()
    parts = []
    try:
        with pq.ParquetFile(selection.path) as source:
            columns = projected_columns(source)
            offset = 0
            last_log = started
            for index in range(source.num_row_groups):
                if cancelled.is_set():
                    return None
                end = offset + source.metadata.row_group(index).num_rows
                if offset >= selection.stop:
                    break
                if end > selection.start:
                    table = source.read_row_group(index, columns=columns)
                    left = max(0, selection.start - offset)
                    right = min(end, selection.stop) - offset
                    parts.append(table.slice(left, right - left))
                    del table
                offset = end
                if time.perf_counter() - last_log >= 5:
                    logger.info('Loading %s: row group %d/%d', selection.path,
                                index + 1, source.num_row_groups)
                    last_log = time.perf_counter()
        if cancelled.is_set():
            return None
        result = pa.concat_tables(parts)
        if result.num_rows != selection.stop - selection.start:
            raise ValueError(f'Parquet row count changed: {selection.path}')
        logger.info('Loaded %s: %s games, %.3f GiB Arrow, %.2fs', selection.path,
                    result.num_rows, result.nbytes / 1024**3, time.perf_counter() - started)
        return result
    finally:
        parts.clear()


def _table_columns(table, read_batch_size):
    with table.to_reader(max_chunksize=read_batch_size) as reader:
        for batch in reader:
            yield arrow_numpy_columns(batch)
            del batch


def prefetched_columns(selections, *, read_batch_size=512):
    """Yield NumPy columns while at most two file selections are resident/loading.

    close() cancels subsequent row groups and joins the worker. An in-flight
    Arrow row-group read must finish before shutdown; it cannot be interrupted.
    Consumers must release zero-copy NumPy views before requesting the next item.
    """
    if read_batch_size < 1:
        raise ValueError('read_batch_size must be positive')
    selections = iter(selections)
    first = next(selections, None)
    if first is None:
        return
    cancelled = Event()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='parquet-prefetch')
    future = executor.submit(load_file, first, cancelled)
    table = None
    try:
        while future is not None:
            table = future.result()
            future = None  # Future also owns its result: drop it before advancing.
            if table is None:
                return
            following = next(selections, None)
            if following is not None:
                future = executor.submit(load_file, following, cancelled)
            try:
                with closing(_table_columns(table, read_batch_size)) as columns:
                    yield from columns
            finally:
                table = None
            # Only after releasing A do we receive B and start loading C.
    finally:
        cancelled.set()
        if future is not None:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        future = None
        table = None
