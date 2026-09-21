"""Real two-process Gloo checks for sharding, gradients, metrics and restart."""
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from train.data import build_datasets
from train.engine import run_epoch
from train.main import run_training


class SmallModel(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, boards, valid):
        return self.linear(self.dropout(boards.float()))


class SmallRatingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = SmallModel(0.4)

    def forward(self, boards, valid):
        return self.inner(boards[:, 0, 0, :2], valid)


def batches(n, rating=1860.0):
    return [((torch.ones(n, 2), torch.ones(n, 1, dtype=torch.bool)),
             torch.full((n, 2), rating))] if n else []


def write_games(path, start, count):
    pq.write_table(pa.table({
        "white_elo": [1000.0 + i for i in range(start, start + count)],
        "black_elo": [1500.0 + i for i in range(start, start + count)],
        "ply_list": pa.array([[12 | (28 << 6)]] * count, type=pa.list_(pa.uint16())),
    }), path, row_group_size=3)


def distributed_worker(rank, directory):
    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        # One synchronous update must match the equivalent global batch.
        torch.manual_seed(9)
        model = SmallModel()
        reference = SmallModel()
        reference.load_state_dict(model.state_dict())
        ddp = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        run_epoch(ddp, batches(1, 1660.0 + rank * 400), device=torch.device("cpu"),
                  precision="float32", optimizer=optimizer, scaler=scaler)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        targets = torch.tensor([[0., 0.], [1., 1.]])
        (reference(torch.ones(2, 2), None) - targets).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        ref_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)

        # Uneven validation and an entirely empty rank still reduce correctly.
        for parameter in model.parameters():
            parameter.data.zero_()
        metrics = run_epoch(ddp, batches(0 if rank == 0 else 3),
                            device=torch.device("cpu"), precision="float32")
        assert metrics == {"loss": 0.25, "origin_mae": 200.0}, metrics
        metrics = run_epoch(ddp, batches(1 if rank == 0 else 3, 2060. if rank == 0 else 2860.),
                            device=torch.device("cpu"), precision="float32")
        assert metrics == {"loss": 7.0, "origin_mae": 1000.0}, metrics

        # Rank-specific dropout RNG and Adam must survive an epoch restart.
        options = dict(batch_size=4, validation_size=0.1, device="cpu", decoder="python",
                       shuffle_buffer=3, seed=7, verbose=0, prefetch_batches=1)
        with patch("train.main.build_model", SmallRatingModel):
            full, full_history = run_training(
                root / "games.parquet", epochs=2, checkpoint_dir=root / "full",
                output_dir=root / "full-out", **options)
            run_training(root / "games.parquet", epochs=1, checkpoint_dir=root / "resume",
                         output_dir=root / "first-out", **options)
            resumed, resumed_history = run_training(
                root / "games.parquet", epochs=2, resume_from=root / "resume",
                checkpoint_dir=root / "resume", output_dir=root / "resumed-out", **options)
        assert full_history == resumed_history, (full_history, resumed_history)
        for name, value in full.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
        dist.barrier()
        saved = torch.load(root / "resume" / "epoch-000002.pt", weights_only=True)
        assert len(saved["rng_by_rank"]) == 2
        assert not torch.equal(saved["rng_by_rank"][0]["cpu"], saved["rng_by_rank"][1]["cpu"])
        assert not any(name.startswith("module.") for name in saved["model"])
    finally:
        dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def test_shards_cross_files_without_duplicates_and_keep_partial_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{i}.parquet" for i in range(2)]
            write_games(paths[0], 0, 5)
            write_games(paths[1], 5, 8)
            train_ids, val_ids = [], []
            for rank in range(2):
                training, validation = build_datasets(
                    paths, batch_size=6, validation_size=0.15, decoder="python",
                    world_size=2, rank=rank, shuffle_buffer=3, prefetch_batches=1)
                train = list(training)
                self.assertEqual([len(y) for _, y in train], [3, 2])
                train_ids.extend(int(y) for _, targets in train for y in targets[:, 0])
                val_ids.extend(int(y) for _, targets in validation for y in targets[:, 0])
                first = [targets.tolist() for _, targets in training]
                training.set_epoch(1)
                second = [targets.tolist() for _, targets in training]
                training.set_epoch(1)
                self.assertEqual(second, [targets.tolist() for _, targets in training])
                self.assertNotEqual(first, second)
            self.assertEqual(sorted(train_ids), list(range(1000, 1010)))
            self.assertEqual(sorted(val_ids), [1011, 1012])
            with self.assertRaisesRegex(ValueError, "divisible"):
                build_datasets(paths, batch_size=3, world_size=2)
            with self.assertRaisesRegex(ValueError, "one game per rank"):
                build_datasets(paths, batch_size=4, world_size=4, max_games=2)

    @unittest.skipUnless(dist.is_gloo_available(), "Gloo is required")
    def test_two_process_gradients_validation_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            write_games(Path(directory) / "games.parquet", 0, 9)
            workers = mp.spawn(distributed_worker, args=(directory,), nprocs=2, join=False)
            try:
                deadline = time.monotonic() + 90
                while not workers.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("DDP test exceeded 90 seconds (possible collective deadlock)")
            finally:
                for process in workers.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)

    @unittest.skipUnless(dist.is_gloo_available(), "Gloo is required")
    def test_torchrun_actual_model_and_single_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_games(root / "games.parquet", 0, 5)
            # Explicit loopback avoids hostname lookup dependence on macOS.
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            command = [sys.executable, "-m", "torch.distributed.run",
                       "--master-addr=127.0.0.1", f"--master-port={port}",
                       "--nproc-per-node=2", "-m", "train", str(root / "games.parquet"),
                       "--device", "cpu", "--decoder", "python", "--batch-size", "4",
                       "--epochs", "1", "--verbose", "0",
                       "--checkpoint-dir", str(root / "checkpoints"),
                       "--output-dir", str(root / "output")]
            result = subprocess.run(command, capture_output=True, text=True, timeout=90,
                                    env={**os.environ, "OMP_NUM_THREADS": "1"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / "output" / "run.json").read_text())
            self.assertEqual(report["replicas"], 2)
            self.assertEqual(report["train_games"], 4)
            self.assertEqual(report["validation_games"], 1)
            self.assertEqual(report["per_rank_batch_size"], 2)
            self.assertEqual(len(list((root / "checkpoints").glob("epoch-*.pt"))), 1)
            self.assertFalse(list(root.rglob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
