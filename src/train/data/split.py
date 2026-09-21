"""Fixed train/validation ranges across an ordered list of Parquet files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pyarrow.parquet as pq


@dataclass(frozen=True)
class FileSelection:
    path: Path
    start: int
    stop: int


def file_counts(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = tuple(Path(path) for path in paths)
    if not paths:
        raise ValueError('At least one Parquet file is required')
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError('Duplicate Parquet files are not allowed')
    result = []
    for path in paths:
        with pq.ParquetFile(path) as source:
            result.append((path, source.metadata.num_rows))
    return result


def split_plan(paths, max_games, test_size):
    if not 0 < test_size < 1 or (max_games is not None and max_games < 0):
        raise ValueError('test_size must be in (0, 1); max_games must be >= 0')
    counts = file_counts(paths)
    total = sum(count for _, count in counts)
    if max_games:
        total = min(total, max_games)
    if total < 2:
        raise ValueError('학습/검증 분할에는 최소 2게임이 필요합니다.')
    split = min(total - 1, max(1, int(total * (1.0 - test_size))))

    def select(start, stop):
        selected = []
        offset = 0
        for path, count in counts:
            left, right = max(start, offset), min(stop, offset + count)
            if left < right:
                selected.append(FileSelection(path, left - offset, right - offset))
            offset += count
        return tuple(selected)

    return total, split, select(0, split), select(split, total)


def split_counts(path, max_games, test_size):
    total, split, _, _ = split_plan(path, max_games, test_size)
    return total, split


def slice_selections(selections, start, stop):
    """Select a global interval without reading/decoding other ranks' games."""
    result = []
    offset = 0
    for part in selections:
        count = part.stop - part.start
        left, right = max(start, offset), min(stop, offset + count)
        if left < right:
            result.append(FileSelection(part.path, part.start + left - offset,
                                        part.start + right - offset))
        offset += count
    return tuple(result)
