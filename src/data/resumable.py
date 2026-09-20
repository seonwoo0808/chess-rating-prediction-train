"""Deterministic epoch datasets and a manifest for checkpoint compatibility."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .dataset import dataset_for
from .split import split_plan


class ResumableData:
    """Replay shuffle from the epoch start, then skip completed training batches.

    This avoids attempting to checkpoint an unsupported Python-generator iterator.
    Files must remain immutable. The manifest verifies paths, sizes and mtimes,
    not a full content hash of hundreds of GB of input.
    """

    def __init__(self, paths, *, batch_size=128, validation_size=0.05,
                 max_games=None, read_batch_size=512, generator_batch_size=512,
                 decoder="numba", shuffle_buffer=4096, prefetch=2, seed=42):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = tuple(Path(p).resolve() for p in paths)
        if min(batch_size, read_batch_size, generator_batch_size, shuffle_buffer) < 1:
            raise ValueError('Batch sizes and shuffle_buffer must be positive')
        if prefetch < 0 or not 0 <= seed < 2**31 - 1:
            raise ValueError('prefetch must be nonnegative; seed must be in [0, 2**31-1)')
        if decoder not in ('numba', 'python'):
            raise ValueError('decoder must be numba or python')
        self.config = dict(batch_size=batch_size, validation_size=validation_size,
                           max_games=max_games, read_batch_size=read_batch_size,
                           generator_batch_size=generator_batch_size, decoder=decoder,
                           shuffle_buffer=shuffle_buffer, prefetch=prefetch, seed=seed)
        self.total, self.train_count, _, _ = split_plan(self.paths, max_games, validation_size)
        self.steps_per_epoch = (self.train_count + batch_size - 1) // batch_size
        self.seed = seed
        self._manifest = self._current_manifest()

    def _current_manifest(self):
        files = []
        for path in self.paths:
            stat = path.stat()
            files.append(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
        # Fingerprint the preprocessing code too: changed decoding cannot silently resume.
        root = Path(__file__).parent
        code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.glob('*.py'))}
        return dict(format=1, files=files, config=self.config.copy(),
                    total=self.total, train_count=self.train_count,
                    steps_per_epoch=self.steps_per_epoch, preprocessing=code)

    def manifest(self):
        current = self._current_manifest()
        if current != self._manifest:
            raise ValueError('Input files or preprocessing changed after dataset construction')
        return current

    def _dataset(self, validation):
        c = self.config
        return dataset_for(self.paths, c['max_games'], c['validation_size'], validation,
                           c['read_batch_size'], c['generator_batch_size'], c['decoder'])[0]

    def training(self, epoch, start_step=0):
        if epoch < 0 or not 0 <= start_step <= self.steps_per_epoch:
            raise ValueError('Invalid epoch/start_step')
        import tensorflow as tf
        c = self.config
        dataset = self._dataset(False)
        dataset = dataset.shuffle(min(self.train_count, c['shuffle_buffer']),
                                  seed=(c['seed'] + epoch) % (2**31 - 1),
                                  reshuffle_each_iteration=False)
        dataset = dataset.batch(c['batch_size']).skip(start_step)
        options = tf.data.Options()
        options.experimental_deterministic = True
        return dataset.with_options(options).prefetch(c['prefetch'])

    def validation(self):
        c = self.config
        return self._dataset(True).batch(c['batch_size']).prefetch(c['prefetch'])
