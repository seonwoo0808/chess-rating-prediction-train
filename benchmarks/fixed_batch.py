"""Repeat one real batch using the existing training loop; save no training state."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import logging
from pathlib import Path
import statistics
import sys
import time

# Import this checkout, even when another version of train is installed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from train.data import build_datasets, warmup_decoder
from train.engine import PRECISIONS, run_epoch
from train.main import collect_parquet_files, configure_runtime
from train.models import build_model


class RepeatedBatch:
    def __init__(self, batch, steps):
        self.batch = batch
        self.steps = steps

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            yield self.batch


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", nargs="+", type=Path)
    parser.add_argument("--batch-size", type=positive_int, default=1024)
    parser.add_argument("--precision", choices=tuple(PRECISIONS), default="bfloat16")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--steps", type=positive_int, default=100)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--sample-games", type=positive_int, default=8192,
                        help="Maximum games to load before selecting one batch")
    parser.add_argument("--shuffle-buffer", type=positive_int, default=4096)
    parser.add_argument("--read-batch-size", type=positive_int, default=512)
    parser.add_argument("--decoder", choices=("numba", "python"), default="numba")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", action="store_true",
                        help="Include the original progress bar and 20-step metric reads")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = configure_runtime(args.seed, args.device, args.precision)
    replicas = torch.cuda.device_count() if device.type == "cuda" else 1
    if args.batch_size < replicas:
        raise ValueError("batch-size must be at least the number of visible GPUs")
    paths = collect_parquet_files(args.parquet)
    warmup_decoder(args.decoder)
    dataset, _ = build_datasets(
        paths, batch_size=args.batch_size, max_games=args.sample_games,
        validation_size=0.05, read_batch_size=args.read_batch_size,
        decoder=args.decoder, shuffle_buffer=args.shuffle_buffer, seed=args.seed,
    )
    if dataset.game_count < args.batch_size:
        raise ValueError("Not enough training games for a full batch; increase "
                         "--sample-games / available data, or reduce --batch-size")
    print("Preparing one real batch; file loading and decoding are not timed.", flush=True)
    with closing(iter(dataset)) as batches:
        (boards, valid, clocks), targets = next(batches)
    # The loader is closed and its background thread has joined before timing.
    valid_by_replica = [int(part.sum()) for part in valid.chunk(replicas, dim=0)]
    valid_fraction = float(valid.float().mean())
    fixed = ((boards.to(device), valid.to(device), clocks.to(device)), targets.to(device))
    del boards, valid, clocks, targets, dataset

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-7)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.precision == "float16",
    )
    parallel_model = torch.nn.DataParallel(model) if replicas > 1 else model

    def synchronize():
        if device.type == "cuda":
            for index in range(replicas):
                torch.cuda.synchronize(index)

    def run(steps, description):
        return run_epoch(
            parallel_model, RepeatedBatch(fixed, steps), device=device,
            precision=args.precision, optimizer=optimizer, scaler=scaler,
            progress=args.progress, description=description,
        )

    print(f"device={device} GPUs={replicas if device.type == 'cuda' else 0} "
          f"batch={args.batch_size} precision={args.precision} "
          f"valid_boards_per_replica={valid_by_replica}", flush=True)
    print(f"Warmup: {args.warmup} steps (excluded)", flush=True)
    run(args.warmup, "Warmup")
    synchronize()
    timings = []
    for repeat in range(args.repeats):
        print(f"Measuring round {repeat + 1}/{args.repeats}: {args.steps} steps", flush=True)
        synchronize()
        started = time.perf_counter()
        run(args.steps, f"Round {repeat + 1}")
        synchronize()
        ms_per_step = (time.perf_counter() - started) * 1000 / args.steps
        timings.append(ms_per_step)
        print(f"fixed_batch_ms_per_step={ms_per_step:.3f}", flush=True)

    median = statistics.median(timings)
    print(json.dumps({
        "benchmark": "fixed_batch",
        "source_root": str(PROJECT_ROOT / "src"),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "devices": ([torch.cuda.get_device_name(i) for i in range(replicas)]
                    if device.type == "cuda" else ["cpu"]),
        "replicas": replicas,
        "parallelism": "DataParallel" if replicas > 1 else "single_device",
        "batch_size": args.batch_size,
        "precision": args.precision,
        "warmup_steps": args.warmup,
        "steps_per_round": args.steps,
        "seed": args.seed,
        "sample_games_limit": args.sample_games,
        "shuffle_buffer": args.shuffle_buffer,
        "progress": args.progress,
        "valid_fraction": valid_fraction,
        "valid_boards_per_replica": valid_by_replica,
        "round_ms_per_step": timings,
        "fixed_batch_median_ms_per_step": median,
        "games_per_second": args.batch_size * 1000 / median,
        "scope": "Existing run_epoch, forward/backward/Adam/metrics and DataParallel "
                 "scatter/replication/gather included. File I/O, decode, shuffle and "
                 "initial CPU-to-device transfer excluded. No per-step timer sync.",
        "note": "One real batch, fresh model, no checkpoint loaded or saved. "
                "Weights update continuously across warmup and rounds. "
                "Fixed game lengths may not represent the full dataset.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
