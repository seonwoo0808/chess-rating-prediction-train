import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from train.checkpoint import load_checkpoint, save_checkpoint
from train.main import run_training
from train.models import build_model


def write_games(path):
    move = (12 | (28 << 6)).to_bytes(2, "little")
    ply_type = pa.list_(pa.struct([("movement", pa.binary(2))]))
    pq.write_table(pa.table({
        "game_type": pa.array([[j == i % 4 for j in range(4)] for i in range(5)], type=pa.list_(pa.bool_())),
        "white_elo": [1000., 1200., 1400., 1600., 1800.],
        "black_elo": [1100., 1300., 1500., 1700., 1900.],
        "ply_list": pa.array([[{"movement": move}]] * 5, type=ply_type),
    }), path, row_group_size=2)


class EpochCheckpointTests(unittest.TestCase):
    def test_actual_training_resume_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "games.parquet"
            write_games(path)
            options = dict(batch_size=3, validation_size=0.2, device="cpu", decoder="python",
                           shuffle_buffer=3, seed=7, verbose=0)
            full_model, full_history = run_training(
                path, epochs=2, checkpoint_dir=root / "full", output_dir=root / "full-out", **options)
            run_training(path, epochs=1, checkpoint_dir=root / "resume",
                         output_dir=root / "first-out", **options)
            resumed_model, resumed_history = run_training(
                path, epochs=2, resume_from=root / "resume", checkpoint_dir=root / "resume",
                output_dir=root / "resumed-out", **options)
            self.assertEqual(full_history, resumed_history)
            for name, value in full_model.state_dict().items():
                torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=0, atol=0)
            self.assertEqual(len(resumed_history["loss"]), 2)
            state = torch.load(root / "resume" / "epoch-000002.pt", weights_only=True)
            self.assertEqual(state["completed_epoch"], 2)
            self.assertTrue(all(item["step"].item() == 4 for item in state["optimizer"]["state"].values()))
            restored = build_model()
            restored.load_state_dict(torch.load(root / "resumed-out" / "model.pt", weights_only=True))
            self.assertEqual(json.loads((root / "resumed-out" / "run.json").read_text())["initial_epoch"], 1)
            with self.assertRaisesRegex(ValueError, "settings differ"):
                run_training(path, epochs=3, resume_from=root / "resume", learning_rate=0.01,
                             checkpoint_dir=root / "resume", output_dir=root / "bad", **options)

    def test_failed_save_keeps_previous_checkpoint_and_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 2)
            options = dict(model=model, optimizer=torch.optim.Adam(model.parameters()),
                           scaler=torch.amp.GradScaler("cuda", enabled=False), manifest={}, history={})
            save_checkpoint(directory, completed_epoch=1, **options)
            pointer = (Path(directory) / "latest.json").read_bytes()
            with patch("train.checkpoint.torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    save_checkpoint(directory, completed_epoch=2, **options)
            self.assertEqual((Path(directory) / "latest.json").read_bytes(), pointer)
            self.assertFalse(list(Path(directory).glob("*.tmp")))
            epoch, _ = load_checkpoint(directory, **{k: options[k] for k in
                                      ("model", "optimizer", "scaler", "manifest")})
            self.assertEqual(epoch, 1)
