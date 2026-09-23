"""Read-only training-code diagnostics: transfer, varying batches and input waits."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from train.data import build_datasets, warmup_decoder
from train.engine import PRECISIONS, run_epoch
from train.main import collect_parquet_files, configure_runtime
from train.models import build_model

MODES = ("fixed_gpu", "fixed_cpu", "varied_gpu", "varied_cpu", "streaming", "input_only")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("parquet", nargs="+", type=Path)
    result.add_argument("--batch-size", type=positive_int, default=1024)
    result.add_argument("--precision", choices=tuple(PRECISIONS), default="bfloat16")
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--warmup", type=positive_int, default=20)
    result.add_argument("--steps", type=positive_int, default=100)
    result.add_argument("--repeats", type=positive_int, default=3)
    result.add_argument("--bank-batches", type=positive_int, default=16)
    result.add_argument("--sample-games", type=int, default=None,
                        help="Default: enough for all steps; 0: full original dataset")
    result.add_argument("--shuffle-buffer", type=positive_int, default=4096)
    result.add_argument("--prefetch-batches", type=int, default=2,
                        help="CPU batch prefetch queue size; 0 reproduces inline preparation")
    result.add_argument("--read-batch-size", type=positive_int, default=512)
    result.add_argument("--decoder", choices=("numba", "python"), default="numba")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--progress", action="store_true")
    result.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    result.add_argument("--output", type=Path,
                        help="New JSON path; default: benchmarks/results/pipeline-TIMESTAMP.json")
    return result


def distribution(values):
    return {"count": len(values), "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "p95_ms": float(np.percentile(values, 95)), "max_ms": max(values)}


def batch_info(batch, replicas):
    valid = batch[0][1]
    # CPU-only metadata: no CUDA tensor read or extra GPU synchronization.
    return {"valid_fraction": float(valid.numpy().mean()),
            "valid_boards_per_replica": [int(part.numpy().sum())
                                         for part in valid.chunk(replicas)]}


def move_batch(batch, device):
    (boards, valid, clocks), targets = batch
    return ((boards.to(device), valid.to(device), clocks.to(device)), targets.to(device))


class Cycle:
    def __init__(self, bank):
        self.bank = bank
        self.index = 0

    def __next__(self):
        batch = self.bank[self.index % len(self.bank)]
        self.index += 1
        return batch


class Window:
    """A bounded view; closing this view must not close the ongoing source."""
    def __init__(self, source, count):
        self.source, self.count = source, count

    def __len__(self):
        return self.count

    def __iter__(self):
        for _ in range(self.count):
            try:
                yield next(self.source)
            except StopIteration as exc:
                raise ValueError("Dataset ended before the requested benchmark steps") from exc


class TimedInput:
    def __init__(self, source, args, replicas):
        self.source = source
        self.args = args
        self.replicas = replicas
        self.rows = []
        self.phase = "warmup"
        block = max(args.batch_size, args.shuffle_buffer)
        self.period = block // args.batch_size if block % args.batch_size == 0 else None

    def __next__(self):
        started = time.perf_counter()
        batch = next(self.source)
        wait_ms = (time.perf_counter() - started) * 1000
        index = len(self.rows)
        self.rows.append({"batch_index": index, "phase": self.phase,
                          "block_phase": index % self.period if self.period else None,
                          "next_wait_ms": wait_ms, **batch_info(batch, self.replicas)})
        return batch

    def summary(self):
        measured = [row for row in self.rows if row["phase"] != "warmup"]
        grouped = {}
        if self.period:
            for phase in sorted({row["block_phase"] for row in measured}):
                grouped[str(phase)] = distribution([
                    row["next_wait_ms"] for row in measured if row["block_phase"] == phase])
        return {"first_batch_prepare_ms": self.rows[0]["next_wait_ms"],
                "next_wait": distribution([row["next_wait_ms"] for row in measured]),
                "block_period_batches": self.period,
                "next_wait_by_block_phase": grouped,
                "measured_valid_fraction_mean": statistics.mean(
                    row["valid_fraction"] for row in measured),
                "slowest_batches": sorted(measured, key=lambda row: row["next_wait_ms"],
                                          reverse=True)[:10]}


def main(argv=None):
    args = parser().parse_args(argv)
    if args.sample_games is not None and args.sample_games < 0:
        raise ValueError("--sample-games must be >= 0")
    if len(set(args.modes)) != len(args.modes):
        raise ValueError("Duplicate benchmark modes")
    if args.bank_batches < 2 and any(mode.startswith("varied") for mode in args.modes):
        raise ValueError("Varying-batch measurements require --bank-batches >= 2")
    output = (args.output or Path(__file__).parent / "results" /
              f"pipeline-{datetime.now():%Y%m%d-%H%M%S-%f}.json").resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing file: {output}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = configure_runtime(args.seed, args.device, args.precision)
    replicas = torch.cuda.device_count() if device.type == "cuda" else 1
    if args.batch_size < replicas:
        raise ValueError("batch-size must be at least the visible GPU count")
    paths = collect_parquet_files(args.parquet)
    bank_count = (args.bank_batches if any(m.startswith("varied") for m in args.modes)
                  else int(any(m.startswith("fixed") for m in args.modes)))
    sequence_steps = args.warmup + args.steps * args.repeats
    needs_stream = any(m in ("streaming", "input_only") for m in args.modes)
    needed_batches = max(bank_count, sequence_steps if needs_stream else 0)
    sample_games = args.sample_games
    if sample_games is None:
        sample_games = math.ceil((needed_batches * args.batch_size +
                                  max(args.batch_size, args.shuffle_buffer)) / 0.95)
    warmup_decoder(args.decoder)
    dataset, _ = build_datasets(
        paths, batch_size=args.batch_size, max_games=sample_games or None,
        validation_size=0.05, read_batch_size=args.read_batch_size,
        decoder=args.decoder, shuffle_buffer=args.shuffle_buffer, seed=args.seed,
        prefetch_batches=args.prefetch_batches,
    )
    if dataset.game_count < needed_batches * args.batch_size:
        raise ValueError(f"Need at least {needed_batches * args.batch_size} training games; "
                         f"have {dataset.game_count}. Increase sample/data size or reduce steps.")

    def sync():
        if device.type == "cuda":
            for index in range(replicas):
                torch.cuda.synchronize(index)

    cpu_bank = []
    if bank_count:
        print(f"Preparing {bank_count} real batches (untimed)", flush=True)
        with closing(iter(dataset)) as source:
            for _ in range(bank_count):
                batch = next(source)
                # Own only each batch, not the entire shuffled NumPy block.
                cpu_bank.append(((batch[0][0].clone(), batch[0][1].clone()), batch[1].clone()))
        del batch
    bank_metadata = [batch_info(batch, replicas) for batch in cpu_bank]

    def scenario(mode):
        input_only = mode == "input_only"
        model = optimizer = scaler = parallel_model = None
        if not input_only:
            configure_runtime(args.seed, args.device, args.precision)
            model = build_model().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-7)
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda"
                                         and args.precision == "float16")
            parallel_model = torch.nn.DataParallel(model) if replicas > 1 else model
        raw = None
        if mode in ("streaming", "input_only"):
            dataset.set_epoch(0)
            raw = iter(dataset)
            source = TimedInput(raw, args, replicas)
        else:
            bank = cpu_bank[:1] if mode.startswith("fixed") else cpu_bank
            if mode.endswith("gpu"):
                bank = [move_batch(batch, device) for batch in bank]
            source = Cycle(bank)
        sync()

        def run(count, label):
            window = Window(source, count)
            if input_only:
                for _ in window:
                    pass
            else:
                run_epoch(parallel_model, window, device=device, precision=args.precision,
                          optimizer=optimizer, scaler=scaler, progress=args.progress,
                          description=f"{mode} {label}")

        try:
            print(f"[{mode}] Warmup: {args.warmup} steps", flush=True)
            started = time.perf_counter()
            run(args.warmup, "warmup")
            sync()
            warmup_seconds = time.perf_counter() - started
            timings = []
            for repeat in range(args.repeats):
                if isinstance(source, TimedInput):
                    source.phase = f"round_{repeat + 1}"
                print(f"[{mode}] Round {repeat + 1}/{args.repeats}", flush=True)
                sync()
                started = time.perf_counter()
                run(args.steps, f"round {repeat + 1}")
                sync()
                timings.append((time.perf_counter() - started) * 1000 / args.steps)
                print(f"[{mode}] {timings[-1]:.3f} ms/step", flush=True)
            result = {"warmup_seconds_excluded": warmup_seconds,
                      "round_ms_per_step": timings,
                      "median_ms_per_step": statistics.median(timings)}
            if isinstance(source, TimedInput):
                result.update(source.summary())
                result["input_wait_trace"] = source.rows
            return result
        finally:
            # Joining a background file reader must happen outside the timer.
            if raw is not None:
                raw.close()

    report = {
        "benchmark": "pipeline", "source_root": str(PROJECT_ROOT / "src"),
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(
            path.read_bytes()).hexdigest() for path in sorted((PROJECT_ROOT / "src/train").rglob("*.py"))},
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "devices": ([torch.cuda.get_device_name(i) for i in range(replicas)]
                    if device.type == "cuda" else ["cpu"]),
        "replicas": replicas, "parallelism": "DataParallel" if replicas > 1 else "single_device",
        "torch_cpu_threads": torch.get_num_threads(),
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        "batch_size": args.batch_size, "precision": args.precision,
        "warmup_steps": args.warmup, "steps_per_round": args.steps, "repeats": args.repeats,
        "seed": args.seed, "sample_games_limit": sample_games, "train_games": dataset.game_count,
        "shuffle_buffer": args.shuffle_buffer, "read_batch_size": args.read_batch_size,
        "prefetch_batches": args.prefetch_batches,
        "decoder": args.decoder, "progress": args.progress, "mode_order": args.modes,
        "bank_batches": bank_count, "bank_metadata": bank_metadata,
        "cpu_bank_pinned": bool(cpu_bank and cpu_bank[0][0][0].is_pinned()),
        "bank_payload_mib": sum(t.numel() * t.element_size() for b in cpu_bank
                                for t in (*b[0], b[1])) / 1024**2,
        "scenarios": {},
        "notes": [
            "Original run_epoch and DataParallel retained; no training state saved.",
            "Each training scenario starts from the same seed with a fresh model/Adam.",
            "CPU banks use ordinary unpinned memory and original blocking transfers.",
            "GPU banks reside on the first GPU; DataParallel redistribution remains timed.",
            "Varying banks cycle through a finite set of shapes; this does not reproduce "
            "all unseen shapes or the full data distribution of streaming.",
            "Warmup batches are consumed, not replayed, before measured streaming rounds.",
            "Only round boundaries add GPU synchronization. next_wait_ms is host time "
            "inside next(dataset) and may overlap previous GPU work; do not add to step time.",
            "Per-step CPU metadata overhead is included in streaming/input_only totals, "
            "but excluded from next_wait_ms. No per-step GPU time is claimed.",
            "Automatic sample cap changes whole-file prefetch sizes. Use --sample-games 0 "
            "to reproduce full-file residency/loading; this can use much more RAM.",
            "Input-only and training run separately; OS caching and file prefetch overlap differ.",
            "With batch prefetch enabled, next_wait_ms measures queue wait; decoder and shuffle "
            "run in the producer. Block phase no longer necessarily predicts consumer stalls.",
            "CPU mode is a smoke check; *_gpu names mean device-resident, not measured CUDA.",
        ],
    }
    for mode in args.modes:
        report["scenarios"][mode] = scenario(mode)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    medians = {name: result["median_ms_per_step"] for name, result in report["scenarios"].items()}
    comparisons = {}
    for label, left, right in (
        ("fixed_cpu_minus_gpu_ms", "fixed_cpu", "fixed_gpu"),
        ("varied_cpu_minus_gpu_ms", "varied_cpu", "varied_gpu"),
        ("varied_gpu_minus_fixed_gpu_ms", "varied_gpu", "fixed_gpu"),
        ("streaming_minus_varied_cpu_ms", "streaming", "varied_cpu"),
    ):
        if left in medians and right in medians:
            comparisons[label] = medians[left] - medians[right]
    report["comparisons_not_additive"] = comparisons
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    summary = {"median_ms_per_step": medians, "comparisons_not_additive": comparisons,
               "input_summaries": {name: {key: value for key, value in result.items()
                                           if key not in ("input_wait_trace", "slowest_batches")}
                                   for name, result in report["scenarios"].items()
                                   if "input_wait_trace" in result},
               "result_file": str(output)}
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
