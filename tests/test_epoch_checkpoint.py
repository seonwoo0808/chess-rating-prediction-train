import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from callbacks import EpochCheckpoint, load_epoch_checkpoint


def tiny_model():
    boards = keras.Input((128, 8, 8), dtype="int8", name="boards")
    valid = keras.Input((128,), dtype="bool", name="valid_steps")
    x = keras.layers.Flatten()(keras.layers.Rescaling(1.0)(boards))
    mask = keras.layers.Flatten()(keras.layers.Rescaling(1.0)(valid))
    x = keras.layers.Concatenate()([x, mask])
    output = keras.layers.Dense(2, dtype="float32")(x)
    model = keras.Model([boards, valid], output)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def tiny_dataset():
    boards = np.zeros((4, 128, 8, 8), dtype=np.int8)
    valid = np.zeros((4, 128), dtype=np.bool_)
    valid[:, :1] = True
    targets = np.zeros((4, 2), dtype=np.float32)
    return tf.data.Dataset.from_tensor_slices(((boards, valid), targets)).batch(2)


class EpochCheckpointTests(unittest.TestCase):
    def test_fit_saves_and_loads_complete_epoch(self):
        manifest = {"format": 1, "files": [], "config": {"batch_size": 2}}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            model = tiny_model()
            model.fit(
                tiny_dataset(), epochs=1,
                callbacks=[EpochCheckpoint(checkpoint_dir, manifest)],
                verbose=0,
            )

            latest = json.loads((checkpoint_dir / "latest.json").read_text())
            state = json.loads(
                (checkpoint_dir / latest["checkpoint"] / "state.json").read_text()
            )
            self.assertEqual(state["completed_epoch"], 1)
            restored, restored_state = load_epoch_checkpoint(checkpoint_dir, manifest)
            self.assertEqual(restored_state["completed_epoch"], 1)
            self.assertGreater(int(restored.optimizer.iterations.numpy()), 0)

            restored.fit(
                tiny_dataset(), epochs=2, initial_epoch=1,
                callbacks=[EpochCheckpoint(checkpoint_dir, manifest)],
                verbose=0,
            )
            latest = json.loads((checkpoint_dir / "latest.json").read_text())
            state = json.loads(
                (checkpoint_dir / latest["checkpoint"] / "state.json").read_text()
            )
            self.assertEqual(state["completed_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
