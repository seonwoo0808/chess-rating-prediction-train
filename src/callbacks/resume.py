"""Epoch/step-aware training entry point for StepCheckpoint."""
from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .step_checkpoint import StepCheckpoint, load_checkpoint


def fit_resumable(model, data, *, epochs, checkpoint: StepCheckpoint,
                  resume_from=None, verbose=1, strategy=None):
    """Train until the total epoch count, optionally resuming a full snapshot.

    Uses public Keras train_step/test_step with persistent epoch metrics, rather
    than train_on_batch (which resets metrics on every call in Keras 3).
    Returns (model, history). Pass model=None when resuming.
    This entry point owns training; do not also call model.fit on its datasets.
    """
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError('epochs must be a positive integer')
    if not isinstance(checkpoint, StepCheckpoint):
        raise TypeError('checkpoint must be a StepCheckpoint')
    if resume_from is not None and model is not None:
        raise ValueError('Pass model=None when resume_from is specified')
    # Avoid tf_keras 2.17's Python 3.12-incompatible set_random_seed helper.
    random.seed(data.seed)
    np.random.seed(data.seed)
    tf.random.set_seed(data.seed)
    tf.random.set_global_generator(tf.random.Generator.from_seed(data.seed))
    strategy = strategy or (model.distribute_strategy if model is not None
                            else tf.distribute.get_strategy())
    state = None
    with strategy.scope():
        if resume_from is not None:
            model, state = load_checkpoint(resume_from, data)
        if model is None or getattr(model, 'optimizer', None) is None:
            raise ValueError('A compiled model is required for a fresh run')
        if getattr(model, 'steps_per_execution', 1) != 1:
            raise ValueError('Exact N-step checkpoints require steps_per_execution=1')
        # This loop preserves optimizer EMA state; it does not replace weights
        # with EMA weights at the end of a fit call.
        model.optimizer.build(model.trainable_variables)
        if state is None:
            dummy = tf.zeros((1, *model.output_shape[1:]), tf.float32)
            model.compute_metrics(None, dummy, dummy, None)
            model.reset_metrics()
        # Legacy tf_keras 2.17 cannot construct integer-input models after
        # determinism mode has been enabled. Enable it only after deserialization
        # and model construction, before the first training step.
        tf.config.experimental.enable_op_determinism()

    topology = dict(strategy=type(strategy).__name__, replicas=strategy.num_replicas_in_sync)
    if state is not None and state.get('topology') != topology:
        raise ValueError('Use the same distribution strategy and replica count when resuming')
    checkpoint.topology = topology
    checkpoint.set_model(model)
    checkpoint.data_manifest = data.manifest()
    checkpoint.set_params(dict(epochs=epochs, steps=data.steps_per_epoch, verbose=verbose))
    checkpoint.epoch = state['epoch'] if state else 0
    checkpoint.step_in_epoch = state['step_in_epoch'] if state else 0
    checkpoint.global_step = state['global_step'] if state else 0
    checkpoint.history = state.get('history', []) if state else []
    if checkpoint.epoch > epochs:
        raise ValueError('epochs is smaller than the saved epoch count')
    checkpoint.on_train_begin()
    model.stop_training = False

    @tf.function(reduce_retracing=True, jit_compile=bool(model.jit_compile))
    def train_step(batch):
        return strategy.run(model.train_step, args=(batch,))

    @tf.function(reduce_retracing=True)
    def test_step(batch):
        return strategy.run(model.test_step, args=(batch,))

    def logs():
        return {name: float(value.numpy()) for name, value in model.get_metrics_result().items()}

    first_epoch = checkpoint.epoch
    first_step = checkpoint.step_in_epoch
    for epoch in range(first_epoch, epochs):
        data.manifest()  # Reject inputs changed while the process was running.
        start = first_step if epoch == first_epoch else 0
        if start == 0:
            model.reset_metrics()
        checkpoint.on_epoch_begin(epoch)
        if verbose:
            print(f'Epoch {epoch + 1}/{epochs}: resume at batch {start}/{data.steps_per_epoch}', flush=True)
        if start < data.steps_per_epoch:
            dataset = strategy.experimental_distribute_dataset(data.training(epoch, start))
            iterator = iter(dataset)
            interrupted = False
            try:
                completed = start
                for step, batch in enumerate(iterator, start=start):
                    train_step(batch)
                    checkpoint.on_train_batch_end(step, logs())
                    completed = step + 1
                    if model.stop_training:
                        interrupted = True
                        break
                if completed != data.steps_per_epoch:
                    if not interrupted:
                        raise RuntimeError('Dataset ended at an unexpected training step')
            finally:
                del iterator, dataset
            if interrupted:
                checkpoint.on_train_end()
                return model, checkpoint.history
        epoch_logs = logs()
        # A checkpoint at the final train batch resumes here, including validation.
        model.reset_metrics()
        dataset = strategy.experimental_distribute_dataset(data.validation())
        iterator = iter(dataset)
        try:
            for batch in iterator:
                test_step(batch)
        finally:
            del iterator, dataset
        epoch_logs.update({f'val_{name}': value for name, value in logs().items()})
        checkpoint.on_epoch_end(epoch, epoch_logs)
        if verbose:
            print(json.dumps({'epoch': epoch + 1, **epoch_logs}), flush=True)
    checkpoint.on_train_end()
    return model, checkpoint.history
