"""Check that checkpoint comparison uses fixed games and detects changed weights."""
from pathlib import Path
import tempfile
import unittest

import torch

from check_checkpoints import compare
from test_epoch_checkpoint import write_games
from train.checkpoint import PRE_MONITORING_CODE_HASHES
from train.main import training_manifest
from train.models import build_model


class CheckCheckpointsTests(unittest.TestCase):
    def test_comparison_uses_same_windows_and_shows_checkpoint_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "games.parquet"
            write_games(parquet)
            options = dict(batch_size=3, validation_size=0.2, max_games=None,
                           read_batch_size=2, decoder="python", shuffle_buffer=3,
                           seed=7, prefetch_batches=0, world_size=1)
            manifest = training_manifest(
                [parquet], options, device=torch.device("cpu"),
                precision="float32", learning_rate=1e-4)
            model = build_model()
            first = root / "epoch-000001.pt"
            second = root / "epoch-000002.pt"
            torch.save({"format": 3, "completed_epoch": 1, "manifest": manifest,
                        "model": model.state_dict()}, first)
            with torch.no_grad():
                model.head[-1].bias.add_(1.0)
            torch.save({"format": 3, "completed_epoch": 2, "manifest": manifest,
                        "model": model.state_dict()}, second)

            result = compare([first, second], sample_games=2, batch_size=2,
                             device="cpu", precision="float32")
            self.assertEqual(set(result), {"train_head", "train_middle", "train_tail", "val_head"})
            self.assertTrue(any(abs(values[1] - values[0]) > 1 for values in result.values()))

            original_first = torch.load(first, weights_only=True)
            legacy = dict(original_first)
            legacy["manifest"] = dict(legacy["manifest"])
            legacy["manifest"].pop("lr_schedule")
            legacy["manifest"]["model_code"] = dict(legacy["manifest"]["model_code"])
            legacy["manifest"]["model_code"].update(
                {"main.py": "old-main", "checkpoint.py": "old-checkpoint"})
            torch.save(legacy, first)
            self.assertEqual(compare([first, second], sample_games=2, batch_size=2,
                                     device="cpu", precision="float32"), result)
            torch.save(original_first, first)

            current = torch.load(second, weights_only=True)
            current["manifest"] = dict(current["manifest"])
            current["manifest"]["model_code"] = dict(current["manifest"]["model_code"],
                                                      **PRE_MONITORING_CODE_HASHES)
            torch.save(current, second)
            self.assertEqual(compare([first, second], sample_games=2, batch_size=2,
                                     device="cpu", precision="float32"), result)

            parquet.touch()
            with self.assertRaisesRegex(ValueError, "Parquet file changed"):
                compare([first, second], sample_games=2, batch_size=2, device="cpu")


if __name__ == "__main__":
    unittest.main()
