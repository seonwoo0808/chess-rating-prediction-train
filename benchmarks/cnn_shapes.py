"""Compare benchmark-only valid-board selection with the production fixed CNN."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime
import hashlib
import json
import logging
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch import nn
from torch.nn import functional as F

from train.data import build_datasets, warmup_decoder
from train.engine import PRECISIONS, run_epoch
from train.main import collect_parquet_files, configure_runtime
from train.models import build_model
from train.models.cnn import BoardEncoder


class FixedShapeEncoder(BoardEncoder):
    """Reuse the production fixed-shape forward and original CNN weights."""
    def __init__(self, original):
        nn.Module.__init__(self)
        self.cnn = original.cnn


class DynamicShapeEncoder(FixedShapeEncoder):
    """Preserve the former valid-only path as a benchmark baseline."""
    def forward(self, boards, valid):
        batch, steps = valid.shape
        indices = valid.reshape(-1).nonzero(as_tuple=True)[0]
        pieces = boards.reshape(-1, 8, 8)[indices].long()
        channels = pieces.abs() + (pieces < 0) * 6
        encoded = F.one_hot(channels, 13)[..., 1:].permute(0, 3, 1, 2).float()
        dummy = encoded.new_zeros((1, 12, 8, 8))
        features = self.cnn(torch.cat((dummy, encoded)))[1:]
        output = features.new_zeros((batch * steps, 128))
        return output.index_copy(0, indices, features).reshape(batch, steps, 128)


class BatchList:
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


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
    result.add_argument("--steps", type=positive_int, default=100,
                        help="Distinct batch positions per pass, not a short cyclic bank")
    result.add_argument("--passes", type=positive_int, default=3)
    result.add_argument("--residency", choices=("gpu", "cpu"), default="gpu",
                        help="gpu: preload on device; cpu: include original host transfers")
    result.add_argument("--sample-games", type=int, default=None,
                        help="Default: enough for warmup + one pass; 0: full file selections")
    result.add_argument("--shuffle-buffer", type=positive_int, default=4096)
    result.add_argument("--read-batch-size", type=positive_int, default=512)
    result.add_argument("--decoder", choices=("numba", "python"), default="numba")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--progress", action="store_true")
    result.add_argument("--variant", choices=("both", "dynamic", "fixed"), default="both")
    result.add_argument("--output", type=Path)
    return result


def make_model(variant):
    model = build_model()
    if variant == "fixed":
        model.board_encoder = FixedShapeEncoder(model.board_encoder)
    else:
        model.board_encoder = DynamicShapeEncoder(model.board_encoder)
    return model


def worker(args):
    device = configure_runtime(args.seed, args.device, args.precision)
    replicas = torch.cuda.device_count() if device.type == "cuda" else 1
    if args.batch_size % replicas:
        raise ValueError("For equal fixed shapes, batch-size must be divisible by GPU count")
    paths = collect_parquet_files(args.parquet)
    needed = args.warmup + args.steps
    sample_games = args.sample_games
    if sample_games is None:
        sample_games = math.ceil((needed * args.batch_size +
                                  max(args.batch_size, args.shuffle_buffer)) / 0.95)
    warmup_decoder(args.decoder)
    dataset, _ = build_datasets(
        paths, batch_size=args.batch_size, max_games=sample_games or None,
        validation_size=0.05, read_batch_size=args.read_batch_size,
        decoder=args.decoder, shuffle_buffer=args.shuffle_buffer, seed=args.seed,
    )
    if dataset.game_count < needed * args.batch_size:
        raise ValueError(f"Need {needed * args.batch_size} training games; have "
                         f"{dataset.game_count}. Reduce steps or increase available sample.")

    def sync():
        if device.type == "cuda":
            for index in range(replicas):
                torch.cuda.synchronize(index)

    bank, metadata = [], []
    fingerprint = hashlib.sha256()
    payload_bytes = 0
    print(f"[{args.variant}] Preparing {needed} real batches; not timed", flush=True)
    with closing(iter(dataset)) as source:
        for index in range(needed):
            (boards, valid), targets = next(source)
            arrays = (boards, valid, targets)
            counts = [int(part.numpy().sum()) for part in valid.chunk(replicas)]
            metadata.append({"batch_index": index,
                             "phase": "warmup" if index < args.warmup else "measured",
                             "valid_fraction": float(valid.numpy().mean()),
                             "dynamic_cnn_boards_per_replica": [count + 1 for count in counts]})
            for tensor in arrays:
                fingerprint.update(memoryview(tensor.numpy()).cast("B"))
                payload_bytes += tensor.numel() * tensor.element_size()
            if args.residency == "gpu":
                saved = [tensor.to(device) for tensor in arrays]
                # CPU smoke mode still owns each batch instead of a whole NumPy block.
                if device.type == "cpu":
                    saved = [tensor.clone() for tensor in saved]
            else:
                saved = [tensor.clone() for tensor in arrays]
            bank.append(((saved[0], saved[1]), saved[2]))
            if (index + 1) % 20 == 0:
                print(f"[{args.variant}] Prepared {index + 1}/{needed}", flush=True)
    del boards, valid, targets, arrays, saved, dataset
    sync()
    # Both variants initialize from the same seed, after identical data preparation.
    configure_runtime(args.seed, args.device, args.precision)
    model = make_model(args.variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda"
                                 and args.precision == "float16")
    parallel = torch.nn.DataParallel(model) if replicas > 1 else model

    def train(batches, label):
        return run_epoch(parallel, BatchList(batches), device=device,
                         precision=args.precision, optimizer=optimizer, scaler=scaler,
                         progress=args.progress, description=f"{args.variant} {label}")

    print(f"[{args.variant}] Warmup on {args.warmup} separate batches", flush=True)
    train(bank[:args.warmup], "warmup")
    sync()
    measured = bank[args.warmup:]
    results = []
    for index in range(args.passes):
        print(f"[{args.variant}] Pass {index + 1}/{args.passes}: "
              f"{args.steps} batches in the same order", flush=True)
        sync()
        if device.type == "cuda":
            for gpu in range(replicas):
                torch.cuda.reset_peak_memory_stats(gpu)
        started = time.perf_counter()
        metrics = train(measured, f"pass {index + 1}")
        sync()
        ms = (time.perf_counter() - started) * 1000 / args.steps
        results.append({"pass": index + 1, "ms_per_step": ms,
                        "games_per_second": args.batch_size * 1000 / ms,
                        "metrics": metrics,
                        "peak_allocated_mib_per_gpu": ([torch.cuda.max_memory_allocated(gpu) / 1024**2
                                                       for gpu in range(replicas)]
                                                      if device.type == "cuda" else [])})
        print(f"[{args.variant}] {ms:.3f} ms/step", flush=True)

    warm_shapes = [set(row["dynamic_cnn_boards_per_replica"][gpu]
                       for row in metadata[:args.warmup]) for gpu in range(replicas)]
    measured_shapes = [set(row["dynamic_cnn_boards_per_replica"][gpu]
                           for row in metadata[args.warmup:]) for gpu in range(replicas)]
    source_hashes = {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted((PROJECT_ROOT / "src/train").rglob("*.py"))}
    return {
        "variant": args.variant, "source_sha256": source_hashes,
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "devices": ([torch.cuda.get_device_name(i) for i in range(replicas)]
                    if device.type == "cuda" else ["cpu"]),
        "replicas": replicas, "precision": args.precision, "batch_size": args.batch_size,
        "parallelism": "DataParallel" if replicas > 1 else "single_device",
        "residency": args.residency, "seed": args.seed, "progress": args.progress,
        "sample_games_limit": sample_games, "shuffle_buffer": args.shuffle_buffer,
        "warmup_steps": args.warmup, "steps_per_pass": args.steps,
        "input_sha256": fingerprint.hexdigest(), "bank_payload_mib": payload_bytes / 1024**2,
        "fixed_cnn_shape_per_replica": [args.batch_size // replicas * 128 + 1, 12, 8, 8],
        "measured_valid_fraction": statistics.mean(row["valid_fraction"]
                                                   for row in metadata[args.warmup:]),
        "unique_measured_dynamic_sizes_per_gpu": [len(values) for values in measured_shapes],
        "measured_dynamic_sizes_not_in_warmup_per_gpu": [len(measured_shapes[i] - warm_shapes[i])
                                                         for i in range(replicas)],
        "passes": results, "first_pass_ms": results[0]["ms_per_step"],
        "replay_median_ms": (statistics.median(row["ms_per_step"] for row in results[1:])
                             if len(results) > 1 else None),
        "batch_shapes": metadata,
    }


def main(argv=None):
    args = parser().parse_args(argv)
    if args.sample_games is not None and args.sample_games < 0:
        raise ValueError("sample-games must be >= 0")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = (args.output or Path(__file__).parent / "results" /
              f"cnn-shapes-{datetime.now():%Y%m%d-%H%M%S-%f}.json").resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if args.variant != "both":
        report = worker(args)
    else:
        reports = {}
        # Fresh processes isolate PyTorch/cuDNN contexts and process-local caches.
        # No serialized training data or model state is written to disk.
        with tempfile.TemporaryDirectory(prefix="chess-cnn-shapes-") as directory:
            for variant in ("dynamic", "fixed"):
                child_output = Path(directory) / f"{variant}.json"
                command = [sys.executable, "-B", str(Path(__file__).resolve()),
                           *[str(path.resolve()) for path in args.parquet],
                           "--variant", variant, "--output", str(child_output)]
                for name in ("batch_size", "precision", "device", "warmup", "steps", "passes",
                             "residency", "shuffle_buffer", "read_batch_size", "decoder", "seed"):
                    command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
                if args.sample_games is not None:
                    command += ["--sample-games", str(args.sample_games)]
                if args.progress:
                    command.append("--progress")
                subprocess.run(command, check=True)
                reports[variant] = json.loads(child_output.read_text())
        dynamic, fixed = reports["dynamic"], reports["fixed"]
        if dynamic["input_sha256"] != fixed["input_sha256"]:
            raise RuntimeError("Input batches differed between variants; comparison rejected")
        if dynamic["source_sha256"] != fixed["source_sha256"]:
            raise RuntimeError("Original source changed between variants; comparison rejected")
        comparisons = [{"pass": left["pass"], "dynamic_ms": left["ms_per_step"],
                        "fixed_ms": right["ms_per_step"],
                        "dynamic_minus_fixed_ms": left["ms_per_step"] - right["ms_per_step"],
                        "fixed_speedup_ratio": left["ms_per_step"] / right["ms_per_step"]}
                       for left, right in zip(dynamic["passes"], fixed["passes"])]
        report = {"benchmark": "cnn_shapes", "identical_input_verified": True,
                  "variants": reports, "comparisons": comparisons,
                  "notes": [
                      "Fixed uses production BoardEncoder.forward. Dynamic uses the former "
                      "valid-only selection path in a benchmark-local wrapper sharing the CNN.",
                      "Fixed processes all B*T boards plus the original dummy, then masks invalid "
                      "features to zero before positions/Transformer. Padding semantics preserved.",
                      "Float32 one-hot intermediates, memory layout, precision, Adam, DataParallel "
                      "and original run_epoch are retained. No torch.compile/cuDNN setting changes.",
                      "First measured pass uses batches separate from warmup. Their CNN sizes may "
                      "overlap warmup; actual unique/new sizes are reported. Later passes replay "
                      "exactly the same measured batches. Model/optimizer continue updating.",
                      "GPU residency excludes input preparation and initial host transfers. "
                      "CPU residency includes original host transfers but still excludes decoding.",
                      "Timing synchronizes all GPUs only at pass boundaries; no extra per-step sync.",
                      "Fresh subprocesses isolate framework caches, but not OS cache or GPU temperature.",
                      "Fixed shape also changes padding compute and gather/scatter work. A speedup "
                      "does not by itself prove that cuDNN plan creation was the cause.",
                      "Peak allocated memory includes resident batch bank, weights and optimizer; "
                      "it excludes cached/reserved memory and external CUDA allocations.",
                      "Only JSON reports are saved. CPU mode is functional validation, not GPU timing.",
                  ]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if args.variant == "both":
        print(json.dumps({"comparisons": report["comparisons"],
                          "dynamic_first_pass_ms": dynamic["first_pass_ms"],
                          "dynamic_replay_median_ms": dynamic["replay_median_ms"],
                          "fixed_first_pass_ms": fixed["first_pass_ms"],
                          "fixed_replay_median_ms": fixed["replay_median_ms"],
                          "result_file": str(output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
