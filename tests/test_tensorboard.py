"""TensorBoard events contain live rank-0 windows and epoch validation metrics."""
from pathlib import Path
import tempfile
import unittest

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from test_epoch_checkpoint import write_games
from train.main import run_training


class TensorBoardTests(unittest.TestCase):
    def test_training_writes_step_and_epoch_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "games.parquet"
            write_games(parquet)
            _, history = run_training(
                parquet, epochs=1, batch_size=2, validation_size=0.2,
                device="cpu", decoder="python", prefetch_batches=0, verbose=0,
                checkpoint_dir=root / "checkpoints", output_dir=root / "run",
                tensorboard_dir=root / "tensorboard", log_every_steps=1,
            )
            events = list((root / "tensorboard").glob("events.out.tfevents.*"))
            self.assertEqual(len(events), 1)
            reader = EventAccumulator(str(root / "tensorboard"))
            reader.Reload()
            tags = set(reader.Tags()["scalars"])
            self.assertTrue({"train_rank0/window_loss", "train_rank0/window_mae",
                             "train_rank0/learning_rate", "epoch/train_mae", "epoch/val_mae"} <= tags)
            self.assertEqual([event.step for event in reader.Scalars("train_rank0/window_mae")], [1, 2])
            self.assertAlmostEqual(reader.Scalars("epoch/val_mae")[0].value,
                                   history["val_origin_mae"][0], places=4)


if __name__ == "__main__":
    unittest.main()
