"""Bounded, epoch-seeded shuffle of decoded games into PyTorch batches."""
from contextlib import closing

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .batches import async_row_batches
from .batch_prefetch import prefetched_batches
from .split import slice_selections, split_plan


class GameDataset(IterableDataset):
    """Already batched tensors; use directly or DataLoader(batch_size=None).

    File loading and bounded batch production own their background workers.
    Extra DataLoader workers would duplicate games and file buffers.
    """
    def __init__(self, selections, *, batch_size, shuffle_buffer, seed, decoder,
                 read_batch_size, prefetch_batches=2):
        if not isinstance(prefetch_batches, int) or prefetch_batches < 0:
            raise ValueError("prefetch_batches must be a non-negative integer")
        self.selections = selections
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.decoder = decoder
        self.read_batch_size = read_batch_size
        self.prefetch_batches = prefetch_batches
        self.epoch = 0
        self.game_count = sum(part.stop - part.start for part in selections)

    def __len__(self):
        return (self.game_count + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError("Use num_workers=0; the dataset owns its prefetch workers")
        epoch = self.epoch
        if self.prefetch_batches:
            yield from prefetched_batches(
                lambda cancelled: self._iter_batches(epoch, cancelled), self.prefetch_batches)
        else:
            yield from self._iter_batches(epoch)

    def _iter_batches(self, epoch, cancelled=None):
        rng = np.random.default_rng(self.seed + epoch)
        source = async_row_batches(
            self.selections, read_batch_size=self.read_batch_size,
            generator_batch_size=max(self.batch_size, self.shuffle_buffer),
            decoder=self.decoder,
            stop_event=cancelled,
        )
        pending = None
        with closing(source):
            for (boards, valid, clocks), targets in source:
                if cancelled is not None and cancelled.is_set():
                    return
                arrays = (boards, valid, targets, clocks)
                if self.shuffle_buffer > 1:
                    order = rng.permutation(len(targets))
                    arrays = tuple(array[order] for array in arrays)
                if pending is not None:
                    arrays = tuple(np.concatenate((left, right))
                                   for left, right in zip(pending, arrays))
                    pending = None
                count = len(arrays[2])
                complete = count - count % self.batch_size
                for start in range(0, complete, self.batch_size):
                    if cancelled is not None and cancelled.is_set():
                        return
                    yield self._tensors(arrays, start, start + self.batch_size)
                if complete < count:
                    # Copy the short tail so it does not keep a large block alive.
                    pending = tuple(array[complete:].copy() for array in arrays)
            if pending is not None:
                yield self._tensors(pending, 0, len(pending[2]))

    @staticmethod
    def _tensors(arrays, start, stop):
        boards, valid, targets, clocks = (torch.from_numpy(array[start:stop]) for array in arrays)
        return (boards, valid, clocks), targets


def build_datasets(path, *, batch_size=128, max_games=None, validation_size=0.05,
                   read_batch_size=512, decoder="numba", shuffle_buffer=4096, seed=42,
                   prefetch_batches=2, rank=0, world_size=1):
    if batch_size < 1 or shuffle_buffer < 1 or read_batch_size < 1:
        raise ValueError("batch_size, shuffle_buffer and read_batch_size must be positive")
    if decoder not in ("numba", "python"):
        raise ValueError("decoder must be numba or python")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Require world_size >= 1 and 0 <= rank < world_size")
    if batch_size % world_size:
        raise ValueError("Global batch_size must be divisible by world_size")
    total, split, training, validation = split_plan(path, max_games, validation_size)
    train_count = split - split % world_size
    if train_count == 0:
        raise ValueError("Training requires at least one game per rank")
    local_count = train_count // world_size
    training = slice_selections(training, rank * local_count, (rank + 1) * local_count)
    validation = slice_selections(validation, (total - split) * rank // world_size,
                                  (total - split) * (rank + 1) // world_size)
    options = dict(batch_size=batch_size // world_size, seed=seed + rank, decoder=decoder,
                   read_batch_size=read_batch_size, prefetch_batches=prefetch_batches)
    train = GameDataset(training, shuffle_buffer=shuffle_buffer, **options)
    val = GameDataset(validation, shuffle_buffer=1, **options)
    train.global_game_count = train_count
    train.dropped_game_count = split - train_count
    val.global_game_count = total - split
    return train, val
