"""Production training loop for the chess rating model."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
from typing import Iterable, Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras

from callbacks import EpochCheckpoint, load_epoch_checkpoint
from data import build_datasets, warmup_decoder
from data.manifest import dataset_manifest
from data.split import split_plan
from models import build_model

LOGGER = logging.getLogger(__name__)


def build_strategy():
    """Use all visible GPUs when more than one is available."""
    gpus = tf.config.list_logical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy(
            devices=[device.name for device in gpus]
        )
        LOGGER.info(
            "multi-GPU strategy=%s devices=%s replicas=%d",
            type(strategy).__name__, [device.name for device in gpus],
            strategy.num_replicas_in_sync,
        )
        return strategy
    strategy = tf.distribute.get_strategy()
    LOGGER.info(
        "single-device strategy=%s visible_gpus=%d replicas=%d",
        type(strategy).__name__, len(gpus), strategy.num_replicas_in_sync,
    )
    return strategy


def collect_parquet_files(inputs: Iterable[str | Path]) -> tuple[Path, ...]:
    """Expand directories into sorted direct-child Parquet files.

    Explicit files keep their argument order. Directory expansion is sorted by
    filename so the monthly training order is stable across process restarts.
    """
    paths: list[Path] = []
    for value in inputs:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            found = sorted(
                (child for child in path.iterdir()
                 if child.is_file() and child.suffix.lower() == ".parquet"),
                key=lambda child: child.name,
            )
            if not found:
                raise ValueError(f"No .parquet files found in directory: {path}")
            paths.extend(found)
        else:
            paths.append(path)
    if not paths:
        raise ValueError("No Parquet files were found")
    if len({path for path in paths}) != len(paths):
        raise ValueError("Duplicate Parquet files were specified")
    return tuple(paths)


def configure_runtime(*, seed: int, precision: str) -> None:
    """Set process-wide training settings before constructing a fresh model."""
    if not 0 <= seed < 2**31 - 1:
        raise ValueError("seed must be in [0, 2**31-1)")
    # tf_keras 2.17 uses Python's random.Random.randint with a float upper
    # bound under Python 3.12. Seed the three runtimes directly instead.
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.random.set_global_generator(tf.random.Generator.from_seed(seed))
    if precision not in {"float32", "mixed_float16", "mixed_bfloat16"}:
        raise ValueError(f"Unsupported precision: {precision}")
    keras.mixed_precision.set_global_policy(precision)
    # Legacy tf_keras 2.17 cannot build this integer-input model after enabling
    # operation-level deterministic execution, so keep the process-level seed
    # setup here and let the standard Keras fit loop manage the epoch lifecycle.


def run_training(
    data_paths: str | Path | Sequence[str | Path],
    *,
    epochs: int = 10,
    batch_size: int = 128,
    validation_size: float = 0.05,
    max_games: int | None = None,
    read_batch_size: int = 512,
    generator_batch_size: int = 512,
    decoder: str = "numba",
    shuffle_buffer: int = 4096,
    prefetch: int = 2,
    seed: int = 42,
    precision: str = "float32",
    learning_rate: float = 1e-4,
    jit_compile: bool = False,
    skip_padding: bool = False,
    checkpoint_dir: str | Path = "outputs/checkpoints",
    resume_from: str | Path | None = None,
    output_dir: str | Path = "outputs/run",
    verbose: int = 1,
):
    """Run or resume a complete training job.

    ``data_paths`` is ordered and is passed to the two-file asynchronous loader.
    Checkpoints are written after complete epochs and resumed with Keras'
    ``initial_epoch`` API.
    """
    if isinstance(data_paths, (str, Path)):
        data_paths = (data_paths,)
    if not data_paths:
        raise ValueError("At least one Parquet file is required")
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    paths = collect_parquet_files(data_paths)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Parquet file not found: {missing[0]}")

    configure_runtime(seed=seed, precision=precision)
    total_games, train_games, _, _ = split_plan(paths, max_games, validation_size)
    manifest = dataset_manifest(
        paths, batch_size=batch_size, validation_size=validation_size,
        max_games=max_games, read_batch_size=read_batch_size,
        generator_batch_size=generator_batch_size, decoder=decoder,
        shuffle_buffer=shuffle_buffer, prefetch=prefetch, seed=seed,
    )
    if decoder == "numba":
        warmup_decoder(decoder)

    checkpoint = EpochCheckpoint(checkpoint_dir, manifest)
    initial_epoch = 0
    strategy = build_strategy()
    with strategy.scope():
        if resume_from is None:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            model = build_model(
                optimizer=optimizer,
                jit_compile=jit_compile,
                skip_padding=skip_padding,
            )
        else:
            model, state = load_epoch_checkpoint(resume_from, manifest)
            initial_epoch = int(state["completed_epoch"])
            if initial_epoch > epochs:
                raise ValueError("epochs is smaller than the saved checkpoint epoch")

    train_ds, validation_ds = build_datasets(
        paths, batch_size=batch_size, max_games=max_games,
        validation_size=validation_size, read_batch_size=read_batch_size,
        generator_batch_size=generator_batch_size, decoder=decoder,
        shuffle_buffer=shuffle_buffer, prefetch=prefetch, seed=seed,
    )

    LOGGER.info(
        "training files=%d games=%d train=%d validation=%d steps/epoch=%d "
        "batch=%d precision=%s decoder=%s",
        len(paths), total_games, train_games, total_games - train_games,
        manifest["steps_per_epoch"], batch_size, precision, decoder,
    )
    LOGGER.info(
        "strategy=%s replicas=%d",
        type(strategy).__name__, strategy.num_replicas_in_sync,
    )
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint],
        shuffle=False,
        verbose=verbose,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model.save(destination / "model.keras")
    (destination / "history.json").write_text(
        json.dumps(history.history, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    run_info = {
        "files": [str(path) for path in paths],
        "epochs": epochs,
        "total_games": total_games,
        "train_games": train_games,
        "validation_games": total_games - train_games,
        "steps_per_epoch": manifest["steps_per_epoch"],
        "batch_size": batch_size,
        "precision": keras.mixed_precision.global_policy().name,
        "decoder": decoder,
        "strategy": type(strategy).__name__,
        "replicas": strategy.num_replicas_in_sync,
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "resume_from": None if resume_from is None else str(Path(resume_from).resolve()),
        "completed_epoch": epochs,
    }
    (destination / "run.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return model, history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the chess rating model")
    parser.add_argument("parquet", nargs="+", type=Path,
                        help="Parquet files or directories; directory files are sorted")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--validation-size", type=float, default=0.01)
    parser.add_argument("--max-games", type=int, default=0,
                        help="0 means all games across all files")
    parser.add_argument("--read-batch-size", type=int, default=512)
    parser.add_argument("--generator-batch-size", type=int, default=512)
    parser.add_argument("--decoder", choices=("numba", "python"), default="numba")
    parser.add_argument("--shuffle-buffer", type=int, default=4096)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("float32", "mixed_float16", "mixed_bfloat16"),
                        default="float32")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--jit-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-padding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Checkpoint directory or a specific checkpoint directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run"))
    parser.add_argument(
        "--verbose", type=int, choices=(0, 1, 2), default=1,
        help="Keras fit verbosity: 0=silent, 1=progress bar, 2=one line per epoch",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    run_training(
        args.parquet,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_size=args.validation_size,
        max_games=None if args.max_games == 0 else args.max_games,
        read_batch_size=args.read_batch_size,
        generator_batch_size=args.generator_batch_size,
        decoder=args.decoder,
        shuffle_buffer=args.shuffle_buffer,
        prefetch=args.prefetch,
        seed=args.seed,
        precision=args.precision,
        learning_rate=args.learning_rate,
        jit_compile=args.jit_compile,
        skip_padding=args.skip_padding,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
