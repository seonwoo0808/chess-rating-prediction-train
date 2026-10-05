import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from train.checkpoint import PRE_MONITORING_CODE_HASHES, load_checkpoint, save_checkpoint
from train.main import run_training
from train.models import build_model


def write_games(path):
    move = (12 | (28 << 6)).to_bytes(2, "little")
    ply_type = pa.list_(pa.struct([("movement", pa.binary(2)), ("time", pa.uint32())]))
    pq.write_table(pa.table({
        "white_elo": [1000., 1200., 1400., 1600., 1800.],
        "black_elo": [1100., 1300., 1500., 1700., 1900.],
        "ply_list": pa.array([[{"movement": move, "time": t}] for t in (None, 0, 60, 180, 600)], type=ply_type),
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
            self.assertAlmostEqual(state["optimizer"]["param_groups"][0]["lr"], 9e-6)
            self.assertEqual(state["scheduler"]["last_epoch"], 2)
            restored = build_model()
            restored.load_state_dict(torch.load(root / "resumed-out" / "model.pt", weights_only=True))
            self.assertEqual(json.loads((root / "resumed-out" / "run.json").read_text())["initial_epoch"], 1)
            with self.assertRaisesRegex(ValueError, "settings differ"):
                run_training(path, epochs=3, resume_from=root / "resume", learning_rate=0.01,
                             checkpoint_dir=root / "resume", output_dir=root / "bad", **options)

    def test_old_first_epoch_checkpoint_can_resume_with_step_lr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "games.parquet"
            write_games(path)
            options = dict(batch_size=3, validation_size=0.2, device="cpu", decoder="python",
                           shuffle_buffer=3, seed=7, verbose=0)
            full_model, full_history = run_training(
                path, epochs=2, checkpoint_dir=root / "full",
                output_dir=root / "full-out", **options)
            old_path = root / "full" / "epoch-000001.pt"
            old = torch.load(old_path, weights_only=True)
            old["manifest"] = dict(old["manifest"])
            old["manifest"].pop("lr_schedule")
            old["manifest"]["model_code"] = dict(old["manifest"]["model_code"])
            old["manifest"]["model_code"].update(
                {"main.py": "old-main", "checkpoint.py": "old-checkpoint"})
            old.pop("scheduler")
            old["optimizer"]["param_groups"][0]["lr"] = 1e-4
            torch.save(old, root / "old-epoch-1.pt")
            resumed_model, resumed_history = run_training(
                path, epochs=2, resume_from=root / "old-epoch-1.pt",
                checkpoint_dir=root / "migrated", output_dir=root / "migrated-out", **options)
            self.assertEqual(full_history, resumed_history)
            for name, value in full_model.state_dict().items():
                torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=0, atol=0)
            migrated = torch.load(root / "migrated" / "epoch-000002.pt", weights_only=True)
            self.assertAlmostEqual(migrated["optimizer"]["param_groups"][0]["lr"], 9e-6)

            with self.assertRaisesRegex(ValueError, "data.config.batch_size: saved=3, current=6"):
                run_training(path, epochs=2, batch_size=6, validation_size=0.2,
                             device="cpu", decoder="python", shuffle_buffer=3,
                             seed=7, verbose=0, resume_from=root / "old-epoch-1.pt",
                             checkpoint_dir=root / "invalid", output_dir=root / "invalid-out")

    def test_pre_monitoring_checkpoint_can_resume_with_tensorboard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "games.parquet"
            write_games(path)
            options = dict(batch_size=3, validation_size=0.2, device="cpu", decoder="python",
                           shuffle_buffer=3, seed=7, verbose=0)
            full_model, full_history = run_training(
                path, epochs=2, checkpoint_dir=root / "full",
                output_dir=root / "full-out", **options)
            old = torch.load(root / "full" / "epoch-000001.pt", weights_only=True)
            old["manifest"] = dict(old["manifest"])
            old["manifest"]["model_code"] = dict(old["manifest"]["model_code"],
                                                  **PRE_MONITORING_CODE_HASHES)
            torch.save(old, root / "pre-monitoring.pt")
            resumed_model, resumed_history = run_training(
                path, epochs=2, resume_from=root / "pre-monitoring.pt",
                checkpoint_dir=root / "resumed", output_dir=root / "resumed-out",
                tensorboard_dir=root / "tensorboard", log_every_steps=1, **options)
            self.assertEqual(full_history, resumed_history)
            for name, value in full_model.state_dict().items():
                torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=0, atol=0)
            events = EventAccumulator(str(root / "tensorboard"))
            events.Reload()
            self.assertEqual([point.step for point in events.Scalars("epoch/val_mae")], [1, 2])
            self.assertEqual([point.step for point in events.Scalars("train_rank0/window_mae")], [3, 4])

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
