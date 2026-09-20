"""Collect data, construct a model, then train/validate/checkpoint each epoch."""
import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

import torch

from .checkpoint import atomic_save, load_checkpoint, save_checkpoint
from .data import build_datasets, warmup_decoder
from .data.manifest import dataset_manifest
from .engine import PRECISIONS, run_epoch
from .models import build_model

LOGGER = logging.getLogger(__name__)


def collect_parquet_files(inputs):
    """Expand directories in filename order; preserve explicit file order."""
    paths = []
    for value in inputs:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            found = sorted(child for child in path.iterdir()
                           if child.is_file() and child.suffix.lower() == ".parquet")
            if not found:
                raise ValueError(f"No .parquet files found in directory: {path}")
            paths.extend(found)
        else:
            if not path.is_file():
                raise FileNotFoundError(f"Parquet file not found: {path}")
            paths.append(path)
    if not paths:
        raise ValueError("No Parquet files were found")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate Parquet files were specified")
    return tuple(paths)


def configure_runtime(seed, device, precision):
    if not 0 <= seed < 2**31 - 1:
        raise ValueError("seed must be in [0, 2**31-1)")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if precision not in PRECISIONS:
        raise ValueError(f"Unsupported precision: {precision}")
    if precision == "float16" and device.type != "cuda":
        raise ValueError("float16 training requires CUDA; use float32 or bfloat16 on CPU")
    if precision == "bfloat16" and device.type == "cuda":
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError(f"CUDA device {index} does not support bfloat16")
    torch.manual_seed(seed)
    return device


def training_manifest(paths, data_options, *, device, precision, learning_rate):
    root = Path(__file__).parent
    code_files = [*sorted((root / "models").glob("*.py")), root / "engine.py", root / "main.py"]
    return {
        "data": dataset_manifest(paths, **data_options),
        "model_code": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in code_files},
        "torch": str(torch.__version__),
        "device": device.type,
        "replicas": torch.cuda.device_count() if device.type == "cuda" else 1,
        "precision": precision,
        "learning_rate": learning_rate,
    }


def run_training(data_paths, *, epochs=10, batch_size=128, validation_size=0.05,
                 max_games=None, read_batch_size=512, decoder="numba",
                 shuffle_buffer=4096, prefetch_batches=2, seed=42, precision="float32", learning_rate=1e-4,
                 device="auto", checkpoint_dir="outputs/checkpoints", resume_from=None,
                 output_dir="outputs/run", verbose=1):
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if verbose not in (0, 1, 2):
        raise ValueError("verbose must be 0, 1 or 2")
    if isinstance(data_paths, (str, Path)):
        data_paths = (data_paths,)
    paths = collect_parquet_files(data_paths)
    device = configure_runtime(seed, device, precision)
    data_options = dict(
        batch_size=batch_size, validation_size=validation_size, max_games=max_games or None,
        read_batch_size=read_batch_size, decoder=decoder, shuffle_buffer=shuffle_buffer, seed=seed,
        prefetch_batches=prefetch_batches,
    )
    training, validation = build_datasets(paths, **data_options)
    manifest = training_manifest(paths, data_options, device=device, precision=precision,
                                 learning_rate=learning_rate)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "float16")
    initial_epoch = 0
    history = {"loss": [], "origin_mae": [], "val_loss": [], "val_origin_mae": []}
    if resume_from is not None:
        initial_epoch, history = load_checkpoint(
            resume_from, model=model, optimizer=optimizer, scaler=scaler, manifest=manifest,
        )
        if initial_epoch > epochs:
            raise ValueError("epochs is smaller than the saved checkpoint epoch")
    warmup_decoder(decoder)
    # One process keeps a single pair of Arrow file buffers for all visible GPUs.
    parallel_model = (torch.nn.DataParallel(model) if device.type == "cuda"
                      and torch.cuda.device_count() > 1 else model)
    LOGGER.info("device=%s replicas=%d games=%d train=%d validation=%d batch=%d precision=%s",
                device, manifest["replicas"], training.game_count + validation.game_count,
                training.game_count, validation.game_count, batch_size, precision)
    LOGGER.info("batch_prefetch=%d (0=inline preparation)", prefetch_batches)
    for epoch in range(initial_epoch, epochs):
        training.set_epoch(epoch)
        train_metrics = run_epoch(
            parallel_model, training, device=device, precision=precision,
            optimizer=optimizer, scaler=scaler, progress=verbose == 1,
            description=f"Epoch {epoch + 1}/{epochs}",
        )
        val_metrics = run_epoch(parallel_model, validation, device=device, precision=precision)
        for name, value in train_metrics.items():
            history[name].append(value)
        for name, value in val_metrics.items():
            history[f"val_{name}"].append(value)
        save_checkpoint(checkpoint_dir, model=model, optimizer=optimizer, scaler=scaler,
                        completed_epoch=epoch + 1, manifest=manifest, history=history)
        if verbose:
            LOGGER.info("epoch=%d loss=%.6f origin_mae=%.3f val_loss=%.6f val_origin_mae=%.3f", epoch + 1,
                        train_metrics["loss"], train_metrics["origin_mae"],
                        val_metrics["loss"], val_metrics["origin_mae"])
    destination = Path(output_dir)
    atomic_save(destination / "model.pt", model.state_dict())
    (destination / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    run_info = dict(manifest, completed_epoch=epochs, initial_epoch=initial_epoch,
                    train_games=training.game_count, validation_games=validation.game_count,
                    steps_per_epoch=len(training))
    (destination / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
    return model, history


def build_parser():
    parser = argparse.ArgumentParser(description="Train the PyTorch chess rating model")
    parser.add_argument("parquet", nargs="+", type=Path, help="Parquet files or directories")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128, help="Global batch across visible GPUs")
    parser.add_argument("--validation-size", type=float, default=0.05)
    parser.add_argument("--max-games", type=int, default=0, help="0 means all games")
    parser.add_argument("--read-batch-size", type=int, default=512)
    parser.add_argument("--decoder", choices=("numba", "python"), default="numba")
    parser.add_argument("--shuffle-buffer", type=int, default=4096)
    parser.add_argument("--prefetch-batches", type=int, default=2,
                        help="CPU batches to prepare ahead; 0 disables batch prefetch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=tuple(PRECISIONS), default="float32")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--resume-from", type=Path, help="Checkpoint directory or epoch .pt file")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run"))
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1,
                        help="0=no progress, 1=progress bar, 2=epoch summaries")
    return parser


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    options = vars(build_parser().parse_args(argv))
    paths = options.pop("parquet")
    run_training(paths, **options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
