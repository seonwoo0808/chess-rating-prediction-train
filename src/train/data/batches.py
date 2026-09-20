"""Decode games in bounded batches, preserving partial batches."""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Optional

import numpy as np

from .constants import MAX_PLIES
from .decoder import board_sequence, compiled_decoder, readonly_array
from .parquet import column_batches

def row_batches(path: Path, max_games: Optional[int], *,
                start: int = 0, stop: Optional[int] = None, read_batch_size: int = 512,
                generator_batch_size: int = 512,
                decoder: str = "numba"):
    """Decode directly into fresh blocks; preserve boundaries and partial batches."""
    yield from decode_batches(
        column_batches(path, max_games, start=start, stop=stop,
                       read_batch_size=read_batch_size),
        generator_batch_size=generator_batch_size, decoder=decoder)


def async_row_batches(selections, *, read_batch_size=512,
                      generator_batch_size=512, decoder="numba", stop_event=None):
    """Decode across file boundaries; only the epoch's last block is partial."""
    from .prefetch import prefetched_columns
    yield from decode_batches(
        prefetched_columns(selections, read_batch_size=read_batch_size, stop_event=stop_event),
        generator_batch_size=generator_batch_size, decoder=decoder)


def decode_batches(source, *, generator_batch_size=512, decoder="numba"):
    """Shared decoder for disk-streaming and asynchronous Arrow sources."""
    if generator_batch_size < 1:
        raise ValueError("generator_batch_size must be positive")
    if decoder not in ("numba", "python"):
        raise ValueError("decoder must be numba or python")
    kernel = compiled_decoder() if decoder == "numba" else None
    boards = valid = targets = None
    filled = 0
    with closing(iter(source)) as source:
        for moves, offsets, present, white, black in source:
            moves = readonly_array(moves, np.uint16)
            offsets = readonly_array(offsets, np.int64)
            present = readonly_array(np.ones(len(white), bool) if present is None else present,
                                     np.bool_)
            cursor = 0
            while cursor < len(white):
                if boards is None:
                    boards = np.empty((generator_batch_size, MAX_PLIES, 8, 8), np.int8)
                    valid = np.empty((generator_batch_size, MAX_PLIES), bool)
                    targets = np.empty((generator_batch_size, 2), np.float32)
                count = min(generator_batch_size-filled, len(white)-cursor)
                end = cursor+count
                target_end = filled+count
                targets[filled:target_end, 0] = white[cursor:end]
                targets[filled:target_end, 1] = black[cursor:end]
                if kernel is not None:
                    kernel(moves, offsets[cursor:end+1], present[cursor:end],
                           boards[filled:target_end], valid[filled:target_end])
                else:
                    for i in range(count):
                        row = cursor+i
                        boards[filled+i], valid[filled+i] = board_sequence(
                            moves[offsets[row]:offsets[row+1]] if present[row] else moves[:0])
                filled, cursor = target_end, end
                if filled == generator_batch_size:
                    yield (boards, valid), targets
                    boards = valid = targets = None
                    filled = 0
            # Release Arrow-backed views before the source advances to another file.
            del moves, offsets, present, white, black
        if filled:
            yield (boards[:filled], valid[:filled]), targets[:filled]
