"""Read bounded Parquet row ranges through Arrow buffers."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def projected_columns(source):
    """Use physical nested paths (including LIST wrappers) to omit clock data."""
    names = source.schema_arrow.names
    for name in ("white_elo", "black_elo", "ply_list"):
        if name not in names:
            raise ValueError(f"Parquet 필수 열이 없습니다: {name}")
    paths = [source.schema.column(i).path for i in range(len(source.schema))
             if source.schema.column(i).path.startswith("ply_list.")]
    movements = [path for path in paths if path.endswith(".movement")]
    if movements:
        return ["white_elo", "black_elo", *movements]
    # Retain support for older list<uint16>/list<binary> inputs.
    return ["white_elo", "black_elo", "ply_list"]


def movement_numpy(values):
    """View Arrow movement storage as integers, without Python bytes/dicts.

    The Go writer stores fixed_size_binary[2], little-endian uint16. Numeric
    movement columns and two-byte binary/large_binary columns are also accepted.
    Array.offset must be included: validation can start inside a row group.
    """
    if pa.types.is_dictionary(values.type):
        values = values.dictionary_decode()
    if values.null_count:
        raise ValueError("movement에 null 값이 있습니다.")
    if len(values) == 0:
        return np.empty(0, dtype=np.uint16)
    if pa.types.is_integer(values.type):
        result = values.to_numpy(zero_copy_only=False)
        if np.any(result < 0) or np.any(result > 65535):
            raise ValueError("movement 값은 uint16 범위여야 합니다.")
        return result
    if pa.types.is_fixed_size_binary(values.type):
        if values.type.byte_width != 2:
            raise ValueError(f"movement는 2바이트여야 합니다: {values.type}")
        byte_start = values.offset * 2
        buffer = values.buffers()[1]
    elif pa.types.is_binary(values.type) or pa.types.is_large_binary(values.type):
        dtype = np.int64 if pa.types.is_large_binary(values.type) else np.int32
        offsets = np.frombuffer(values.buffers()[1], dtype=dtype,
                                count=len(values)+1, offset=values.offset*np.dtype(dtype).itemsize)
        if np.any(np.diff(offsets) != 2):
            raise ValueError("movement binary 값은 각각 2바이트여야 합니다.")
        byte_start = int(offsets[0])
        buffer = values.buffers()[2]
    else:
        raise TypeError(f"지원하지 않는 movement 타입: {values.type}")
    # np.frombuffer retains a reference to the Arrow buffer for its lifetime.
    return np.frombuffer(buffer, dtype="<u2", count=len(values), offset=byte_start)


def arrow_numpy_columns(batch):
    """Return flat moves, per-game offsets/validity, and numeric ratings."""
    for name in ("ply_list", "white_elo", "black_elo"):
        if batch.schema.get_field_index(name) < 0:
            raise ValueError(f"Parquet 필수 열이 없습니다: {name}")
    plies = batch.column(batch.schema.get_field_index("ply_list"))
    if not (pa.types.is_list(plies.type) or pa.types.is_large_list(plies.type)):
        raise TypeError(f"ply_list는 Arrow list/large_list여야 합니다: {plies.type}")
    offsets = plies.offsets.to_numpy(zero_copy_only=False)
    first, last = int(offsets[0]), int(offsets[-1])
    children = plies.values.slice(first, last-first)
    if pa.types.is_struct(children.type):
        if children.null_count:
            raise ValueError("ply_list 내부에 null 수가 있습니다.")
        if children.type.get_field_index("movement") < 0:
            raise ValueError("ply_list struct에 movement 필드가 없습니다.")
        children = children.field("movement")
    moves = movement_numpy(children)
    offsets = offsets-first
    present = plies.is_valid().to_numpy(zero_copy_only=False) if plies.null_count else None

    def rating(name):
        values = batch.column(batch.schema.get_field_index(name))
        if values.null_count:
            raise ValueError(f"{name}에 null 값이 있습니다.")
        if not (pa.types.is_integer(values.type) or pa.types.is_floating(values.type)):
            raise TypeError(f"{name}은 숫자 타입이어야 합니다: {values.type}")
        result = values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise ValueError(f"{name}에 유한하지 않은 값이 있습니다.")
        return result

    return moves, offsets, present, rating("white_elo"), rating("black_elo")


def column_batches(path: Path, max_games: Optional[int], *,
         start: int = 0, stop: Optional[int] = None, read_batch_size: int = 512,
         profile: Optional[dict] = None):
    """Read/decode only [start, stop); skip other row groups by metadata.

    A row group crossing the split may be read by both streams, but each row
    is decoded by exactly one of them. No decoded samples are filtered out.
    """
    if read_batch_size < 1 or (max_games is not None and max_games < 0):
        raise ValueError("read_batch_size must be positive; max_games must be >= 0")
    with pq.ParquetFile(path) as parquet_file:
        columns = projected_columns(parquet_file)
        total_rows = parquet_file.metadata.num_rows
        if max_games:
            total_rows = min(total_rows, max_games)
        stop = total_rows if stop is None else min(stop, total_rows)
        if not 0 <= start <= stop:
            raise ValueError(f"Invalid row interval: [{start}, {stop})")
        if start == stop:
            return
        print(f"[Parquet] rows [{start:,}, {stop:,}): {stop-start:,} games; "
              f"read batch={read_batch_size}, input=arrow_numpy", flush=True)
        group_start = 0
        for group in range(parquet_file.num_row_groups):
            group_end = group_start + parquet_file.metadata.row_group(group).num_rows
            if group_start >= stop:
                break
            if group_end <= start:
                group_start = group_end
                continue
            offset = group_start
            batches = iter(parquet_file.iter_batches(
                row_groups=[group], batch_size=read_batch_size, columns=columns
            ))
            while True:
                started = time.perf_counter() if profile is not None else 0
                try:
                    batch = next(batches)
                except StopIteration:
                    if profile is not None:
                        profile["arrow_read_seconds"] += time.perf_counter() - started
                    break
                if profile is not None:
                    profile["arrow_read_seconds"] += time.perf_counter() - started
                    profile["arrow_batches"] += 1
                left, right = max(0, start-offset), min(batch.num_rows, stop-offset)
                offset += batch.num_rows
                if left < right:
                    started = time.perf_counter() if profile is not None else 0
                    moves, offsets, present, white, black = arrow_numpy_columns(
                        batch.slice(left, right-left))
                    if profile is not None:
                        profile["arrow_numpy_seconds"] += time.perf_counter() - started
                    yield moves, offsets, present, white, black
                if offset >= stop:
                    return
            group_start = group_end
