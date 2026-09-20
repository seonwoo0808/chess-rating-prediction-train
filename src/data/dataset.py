"""Create TensorFlow datasets with bounded shuffle and prefetch."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import tensorflow as tf

from .constants import MAX_PLIES
from .batches import async_row_batches
from .split import split_plan

def dataset_for(path: str | Path | Iterable[str | Path], max_games: Optional[int],
                test_size: float, validation: bool, read_batch_size: int = 512,
                generator_batch_size: int = 512, decoder: str = "numba"):
    if read_batch_size < 1 or generator_batch_size < 1:
        raise ValueError("read_batch_size/generator_batch_size must be positive")
    if decoder not in ("numba", "python"):
        raise ValueError("decoder must be numba or python")
    total, split, training, checking = split_plan(path, max_games, test_size)
    selections = checking if validation else training
    count = total - split if validation else split

    def generator():
        yield from async_row_batches(selections, read_batch_size=read_batch_size,
                                     generator_batch_size=generator_batch_size,
                                     decoder=decoder)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            (tf.TensorSpec((None, MAX_PLIES, 8, 8), tf.int8),
             tf.TensorSpec((None, MAX_PLIES), tf.bool)),
            tf.TensorSpec((None, 2), tf.float32),
        ),
    )
    blocks = (count+generator_batch_size-1)//generator_batch_size
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(blocks))
    # Unbatch inside TensorFlow, so existing per-game shuffle/cache semantics
    # and training batch sizes are preserved without per-game Python callbacks.
    dataset = dataset.unbatch().apply(tf.data.experimental.assert_cardinality(count))
    return dataset, total, split


def build_datasets(path: str | Path | Iterable[str | Path], *, batch_size: int = 128,
                   max_games: Optional[int] = None, validation_size: float = 0.05,
                   read_batch_size: int = 512, generator_batch_size: int = 512,
                   decoder: str = "numba", shuffle_buffer: int = 4096,
                   prefetch: int = 2, seed: int = 42):
    """Return (train, validation) datasets ready for model.fit.

    Accept a path or ordered sequence of paths. Split before shuffling.
    Each iterator holds at most two Arrow file selections, with no byte limit.
    Re-read files each epoch and preserve the final partial batch.
    """
    if batch_size < 1 or shuffle_buffer < 1 or prefetch < 0:
        raise ValueError("batch_size/shuffle_buffer must be positive; prefetch must be >= 0")
    if read_batch_size < 1 or generator_batch_size < 1:
        raise ValueError("read_batch_size/generator_batch_size must be positive")
    if decoder not in ("numba", "python"):
        raise ValueError("decoder must be numba or python")
    if not isinstance(path, (str, Path)):
        path = tuple(path)
    options = dict(path=path, max_games=max_games, test_size=validation_size,
                   read_batch_size=read_batch_size,
                   generator_batch_size=generator_batch_size, decoder=decoder)
    train, _, split = dataset_for(validation=False, **options)
    validation, _, _ = dataset_for(validation=True, **options)
    train = train.shuffle(min(split, shuffle_buffer), seed=seed,
                          reshuffle_each_iteration=True)
    return (train.batch(batch_size).prefetch(prefetch),
            validation.batch(batch_size).prefetch(prefetch))
