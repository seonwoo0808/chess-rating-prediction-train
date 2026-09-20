"""Atomic full training snapshots at completed batch boundaries."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import uuid

import numpy as np
import tensorflow as tf
from tensorflow import keras


def runtime():
    return dict(python=sys.version.split()[0], tensorflow=tf.__version__,
                keras=keras.__version__, numpy=np.__version__,
                precision=keras.mixed_precision.global_policy().name)


def metric_variables(model):
    return [v for metric in model.metrics for v in metric.variables]


def capture_rng():
    numpy_state = np.random.get_state()
    generator = tf.random.get_global_generator()
    return dict(python=random.getstate(),
                numpy=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                tensorflow={'algorithm': int(generator.algorithm),
                            'state': generator.state.numpy().tolist()})


def restore_rng(state):
    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, (list, tuple)) else value
    random.setstate(tuples(state['python']))
    value = state['numpy']
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    value = state['tensorflow']
    tf.random.set_global_generator(tf.random.Generator.from_state(
        value['state'], alg=value['algorithm']))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with open(path, 'w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


class StepCheckpoint(keras.callbacks.Callback):
    """Use with fit_resumable; save every N completed training batches.

    Epoch-end snapshots are also saved after validation, so restarting does not
    silently omit validation at the last training batch of an epoch.
    State counters are zero-based epoch + number of completed batches in it.
    """

    def __init__(self, directory, every_n_steps=1000):
        super().__init__()
        if not isinstance(every_n_steps, int) or isinstance(every_n_steps, bool) or every_n_steps < 1:
            raise ValueError('every_n_steps must be a positive integer')
        self.directory = Path(directory)
        self.every_n_steps = every_n_steps
        self.epoch = 0
        self.step_in_epoch = 0
        self.global_step = 0
        self.last_checkpoint = None
        self.data_manifest = None
        self.history = []
        self.topology = None

    def on_train_begin(self, logs=None):
        if self.data_manifest is None:
            raise RuntimeError('Use StepCheckpoint with callbacks.fit_resumable and ResumableData')

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        self.step_in_epoch = batch + 1
        self.global_step += 1
        if self.global_step % self.every_n_steps == 0:
            self.save('train')

    def on_epoch_end(self, epoch, logs=None):
        self.history.append(dict(epoch=epoch + 1, **(logs or {})))
        self.epoch = epoch + 1
        self.step_in_epoch = 0
        self.save('epoch_complete')

    def save(self, phase):
        if self.data_manifest is None:
            raise RuntimeError('No resumable dataset manifest is attached')
        self.directory.mkdir(parents=True, exist_ok=True)
        name = (f'step-{self.global_step:012d}-epoch-{self.epoch:05d}'
                f'-batch-{self.step_in_epoch:09d}-{uuid.uuid4().hex[:8]}')
        staging = self.directory / ('.pending-' + name)
        destination = self.directory / name
        staging.mkdir()
        rng = capture_rng()
        variables = {'model': list(self.model.variables),
                     'optimizer': list(self.model.optimizer.variables),
                     'metrics': metric_variables(self.model)}
        arrays = {f'{group}_{i}': v.numpy() for group, values in variables.items()
                  for i, v in enumerate(values)}
        metadata = dict(format=1, epoch=self.epoch, step_in_epoch=self.step_in_epoch,
                        epoch_number=self.epoch + 1, step_number=self.step_in_epoch,
                        global_step=self.global_step, phase=phase,
                        optimizer_iterations=int(self.model.optimizer.iterations.numpy()),
                        every_n_steps=self.every_n_steps,
                        runtime=runtime(), data=self.data_manifest,
                        topology=self.topology, history=self.history,
                        counts={k: len(v) for k, v in variables.items()})
        try:
            self.model.save(staging / 'model.keras')
            # Includes Keras SeedGenerator variables, which .keras alone may omit.
            np.savez(staging / 'variables.npz', **arrays)
            write_json(staging / 'rng.json', rng)
            metadata['sha256'] = {name: digest(staging / name)
                                  for name in ('model.keras', 'variables.npz', 'rng.json')}
            write_json(staging / 'state.json', metadata)
            for path in staging.iterdir():
                with open(path, 'rb') as stream:
                    os.fsync(stream.fileno())
            os.replace(staging, destination)
            pointer = self.directory / ('.latest-' + uuid.uuid4().hex + '.json')
            write_json(pointer, {'checkpoint': name})
            os.replace(pointer, self.directory / 'latest.json')
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self.last_checkpoint = destination
        finally:
            if staging.exists():
                shutil.rmtree(staging)
            restore_rng(rng)
        return destination


def load_checkpoint(path, data):
    """Load a complete snapshot and reject changed data/runtime or corrupt files."""
    path = Path(path)
    if (path / 'latest.json').is_file():
        name = json.loads((path / 'latest.json').read_text())['checkpoint']
        if Path(name).name != name:
            raise ValueError('Invalid checkpoint pointer')
        path = path / name
    state = json.loads((path / 'state.json').read_text())
    if state['format'] != 1 or state['runtime'] != runtime():
        raise ValueError('Checkpoint format/runtime/precision differs from this environment')
    if state['data'] != data.manifest():
        raise ValueError('Checkpoint dataset, file order, preprocessing or settings differ')
    if state['phase'] not in ('train', 'epoch_complete'):
        raise ValueError('Invalid checkpoint phase')
    if not 0 <= state['step_in_epoch'] <= data.steps_per_epoch or state['epoch'] < 0:
        raise ValueError('Invalid checkpoint position')
    if state['global_step'] != state['epoch'] * data.steps_per_epoch + state['step_in_epoch']:
        raise ValueError('Inconsistent checkpoint step counters')
    if state['phase'] == 'epoch_complete' and state['step_in_epoch'] != 0:
        raise ValueError('Invalid completed epoch position')
    for name in ('model.keras', 'variables.npz', 'rng.json'):
        if digest(path / name) != state['sha256'][name]:
            raise ValueError(f'Corrupt checkpoint file: {name}')
    import models  # Register custom layers before deserializing.
    model = keras.models.load_model(path / 'model.keras')
    # Build lazy compile metrics without a training/optimizer update.
    dummy = tf.zeros((1, *model.output_shape[1:]), dtype=tf.float32)
    model.compute_metrics(None, dummy, dummy)
    groups = {'model': list(model.variables), 'optimizer': list(model.optimizer.variables),
              'metrics': metric_variables(model)}
    with np.load(path / 'variables.npz', allow_pickle=False) as arrays:
        for group, variables in groups.items():
            if len(variables) != state['counts'][group]:
                raise ValueError(f'Checkpoint variable count mismatch: {group}')
            for i, variable in enumerate(variables):
                value = arrays[f'{group}_{i}']
                if tuple(variable.shape) != value.shape or np.dtype(variable.dtype) != value.dtype:
                    raise ValueError(f'Checkpoint variable mismatch: {group}[{i}]')
                variable.assign(value)
    if int(model.optimizer.iterations.numpy()) != state['optimizer_iterations']:
        raise ValueError('Optimizer iteration restore failed')
    restore_rng(json.loads((path / 'rng.json').read_text()))
    return model, state
