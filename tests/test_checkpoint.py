import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
from tensorflow import keras

from callbacks import StepCheckpoint, fit_resumable
from data import ResumableData


def tiny_model():
    boards = keras.Input((128, 8, 8), dtype="int8", name="boards")
    valid = keras.Input((128,), dtype="bool", name="valid_steps")
    board_values = keras.layers.Flatten(dtype="float32")(boards)
    valid_values = keras.layers.Flatten(dtype="float32")(valid)
    x = keras.layers.Concatenate()([board_values, valid_values])
    # Keep the test model small while exercising a compiled optimizer.
    output = keras.layers.Dense(2)(x)
    model = keras.Model([boards, valid], output)
    model.compile(keras.optimizers.Adam(1e-3), loss="mse", metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return model


class StopAt(StepCheckpoint):
    def __init__(self, directory, stop_step, **kwargs):
        super().__init__(directory, **kwargs)
        self.stop_step = stop_step

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs)
        if self.global_step == self.stop_step:
            self.model.stop_training = True


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "games.parquet"
        move = (12 | (28 << 6)).to_bytes(2, "little")
        table = pa.table({
            "white_elo": pa.array([1000 + i for i in range(12)], type=pa.uint32()),
            "black_elo": pa.array([1500 + i for i in range(12)], type=pa.uint32()),
            "ply_list": pa.array(
                [[{"movement": move}] for _ in range(12)],
                type=pa.list_(pa.struct([("movement", pa.binary(2))])),
            ),
        })
        pq.write_table(table, path, row_group_size=4)
        self.path = path

    def data(self):
        return ResumableData(self.path, batch_size=2, validation_size=0.25,
                             generator_batch_size=3, decoder="python", seed=11,
                             prefetch=0)

    def test_interrupted_step_resumes_exactly(self):
        data = self.data()
        with tempfile.TemporaryDirectory() as directory:
            keras.utils.set_random_seed(123)
            interrupted = StopAt(directory, stop_step=2, every_n_steps=2)
            partial, _ = fit_resumable(tiny_model(), data, epochs=2,
                                       checkpoint=interrupted, verbose=0)
            self.assertEqual(interrupted.global_step, 2)
            checkpoint = Path(directory) / "latest.json"
            self.assertTrue(checkpoint.is_file())
            state_dir = Path(directory) / __import__("json").loads(checkpoint.read_text())["checkpoint"]
            state = __import__("json").loads((state_dir / "state.json").read_text())
            self.assertEqual((state["epoch"], state["step_in_epoch"], state["global_step"]), (0, 2, 2))

            resumed_callback = StepCheckpoint(directory, every_n_steps=2)
            resumed, _ = fit_resumable(None, self.data(), epochs=2,
                                       checkpoint=resumed_callback,
                                       resume_from=directory, verbose=0)

            keras.utils.set_random_seed(123)
            uninterrupted_callback = StepCheckpoint(Path(directory) / "full", every_n_steps=2)
            uninterrupted, _ = fit_resumable(tiny_model(), self.data(), epochs=2,
                                              checkpoint=uninterrupted_callback, verbose=0)
            for left, right in zip(resumed.variables, uninterrupted.variables):
                np.testing.assert_allclose(left.numpy(), right.numpy(), rtol=0, atol=1e-6)
            self.assertEqual(int(resumed.optimizer.iterations.numpy()),
                             int(uninterrupted.optimizer.iterations.numpy()))

    def test_manifest_and_interval_validation(self):
        data = self.data()
        with self.assertRaises(ValueError):
            StepCheckpoint("unused", every_n_steps=0)
        self.assertEqual(data.steps_per_epoch, 5)
        with self.assertRaises(ValueError):
            data.training(0, 6)


if __name__ == "__main__":
    unittest.main()
