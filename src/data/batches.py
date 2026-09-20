"""Decode games in bounded batches, preserving partial batches."""
from __future__ import annotations

import time
from contextlib import closing
from pathlib import Path
from typing import Optional

import numpy as np

from .constants import MAX_PLIES
from .decoder import _decode_row, board_sequence, compiled_decoder, readonly_array
from .parquet import column_batches

def rows(path: Path, max_games: Optional[int], **kwargs):
    """Reference per-game path, retained for independent regression checks."""
    for moves, offsets, present, white, black in column_batches(path, max_games, **kwargs):
        for i in range(len(white)):
            yield _decode_row((moves[offsets[i]:offsets[i+1]]
                               if present is None or present[i] else moves[:0],
                               white[i], black[i]))


def row_batches(path: Path, max_games: Optional[int], *,
                start: int = 0, stop: Optional[int] = None, read_batch_size: int = 512,
                generator_batch_size: int = 512, profile: Optional[dict] = None,
                decoder: str = "numba"):
    """Decode directly into fresh blocks; preserve boundaries and partial batches."""
    yield from decode_batches(
        column_batches(path, max_games, start=start, stop=stop,
                       read_batch_size=read_batch_size, profile=profile),
        generator_batch_size=generator_batch_size, decoder=decoder, profile=profile)


def async_row_batches(selections, *, read_batch_size=512,
                      generator_batch_size=512, decoder="numba"):
    """Decode across file boundaries; only the epoch's last block is partial."""
    from .prefetch import prefetched_columns
    yield from decode_batches(
        prefetched_columns(selections, read_batch_size=read_batch_size),
        generator_batch_size=generator_batch_size, decoder=decoder)


def decode_batches(source, *, generator_batch_size=512, decoder="numba", profile=None):
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
            started = time.perf_counter() if profile is not None else 0
            moves = readonly_array(moves, np.uint16)
            offsets = readonly_array(offsets, np.int64)
            present = readonly_array(np.ones(len(white), bool) if present is None else present,
                                     np.bool_)
            if profile is not None:
                profile["arrow_numpy_seconds"] += time.perf_counter() - started
            cursor = 0
            while cursor < len(white):
                started = time.perf_counter() if profile is not None else 0
                if boards is None:
                    boards = np.empty((generator_batch_size, MAX_PLIES, 8, 8), np.int8)
                    valid = np.empty((generator_batch_size, MAX_PLIES), bool)
                    targets = np.empty((generator_batch_size, 2), np.float32)
                count = min(generator_batch_size-filled, len(white)-cursor)
                end = cursor+count
                target_end = filled+count
                targets[filled:target_end, 0] = white[cursor:end]
                targets[filled:target_end, 1] = black[cursor:end]
                if profile is not None:
                    profile["batch_pack_seconds"] += time.perf_counter() - started
                started = time.perf_counter() if profile is not None else 0
                if kernel is not None:
                    kernel(moves, offsets[cursor:end+1], present[cursor:end],
                           boards[filled:target_end], valid[filled:target_end])
                else:
                    for i in range(count):
                        row = cursor+i
                        boards[filled+i], valid[filled+i] = board_sequence(
                            moves[offsets[row]:offsets[row+1]] if present[row] else moves[:0])
                if profile is not None:
                    profile["board_decode_seconds"] += time.perf_counter() - started
                    profile["rows"] += count
                filled, cursor = target_end, end
                if filled == generator_batch_size:
                    if profile is not None:
                        profile["generator_batches"] += 1
                    yield (boards, valid), targets
                    boards = valid = targets = None
                    filled = 0
            # Release Arrow-backed views before the source advances to another file.
            del moves, offsets, present, white, black
        if filled:
            if profile is not None:
                profile["generator_batches"] += 1
            yield (boards[:filled], valid[:filled]), targets[:filled]
