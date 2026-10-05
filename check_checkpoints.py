"""Compare epoch checkpoints on identical, fixed training/validation games.

Run from the train project directory, on the machine holding the original data:
    python check_checkpoints.py outputs/checkpoints/epoch-000001.pt \
        outputs/checkpoints/epoch-000002.pt --sample-games 4096
"""
import argparse
import hashlib
from pathlib import Path

import torch

from train.checkpoint import (FORMAT_VERSION, LEGACY_ORCHESTRATION_FILES,
                              PRE_MONITORING_CODE_HASHES, is_monitoring_only_manifest,
                              is_pre_step_lr_manifest)
from train.data.dataset import GameDataset
from train.data.split import slice_selections, split_plan
from train.engine import PRECISIONS, run_epoch
from train.models import build_model


def load_state(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return state


def check_data_and_code(manifest):
    data = manifest["data"]
    paths = []
    for entry in data["files"]:
        path = Path(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Original Parquet file is missing: {path}")
        stat = path.stat()
        if stat.st_size != entry["size"] or stat.st_mtime_ns != entry["mtime_ns"]:
            raise ValueError(f"Parquet file changed since training: {path}")
        paths.append(path)
    source_root = Path(__file__).parent / "src" / "train"
    for relative, expected in {**data["preprocessing"], **manifest["model_code"]}.items():
        path = (source_root / "data" / relative) if relative in data["preprocessing"] else (source_root / relative)
        legacy_ok = ("lr_schedule" not in manifest and relative in LEGACY_ORCHESTRATION_FILES)
        legacy_ok |= (relative in PRE_MONITORING_CODE_HASHES
                      and expected == PRE_MONITORING_CODE_HASHES[relative])
        if not path.is_file() or (hashlib.sha256(path.read_bytes()).hexdigest() != expected
                                  and not legacy_ok):
            raise ValueError(f"Training code changed since checkpoint: {path}")
    return paths


def fixed_windows(selections, count, sample_games, prefix):
    size = min(sample_games, count)
    starts = (("head", 0), ("middle", (count - size) // 2), ("tail", count - size))
    seen = set()
    for label, start in starts:
        if (start, start + size) in seen:
            continue
        seen.add((start, start + size))
        yield f"{prefix}_{label}", slice_selections(selections, start, start + size)


def compare(checkpoint_paths, *, sample_games=4096, batch_size=128, device="auto", precision=None):
    if sample_games < 1 or batch_size < 1:
        raise ValueError("sample-games and batch-size must be positive")
    states = [load_state(path) for path in checkpoint_paths]
    source_root = Path(__file__).parent / "src" / "train"
    current_main_hash = hashlib.sha256((source_root / "main.py").read_bytes()).hexdigest()
    manifest = next((state["manifest"] for state in states
                     if state["manifest"]["model_code"]["main.py"] == current_main_hash),
                    next((state["manifest"] for state in states
                          if "lr_schedule" in state["manifest"]), states[0]["manifest"]))
    if any(state["manifest"] != manifest
           and not (state["completed_epoch"] == 1
                    and is_pre_step_lr_manifest(state["manifest"], manifest))
           and not is_monitoring_only_manifest(state["manifest"], manifest)
           for state in states):
        raise ValueError("Checkpoints come from different training runs or settings")
    paths = check_data_and_code(manifest)
    config = manifest["data"]["config"]
    total, split, training, validation = split_plan(
        paths, config["max_games"], config["validation_size"])
    train_count = split - split % manifest["replicas"]
    training = slice_selections(training, 0, train_count)
    windows = [*fixed_windows(training, train_count, sample_games, "train"),
               *fixed_windows(validation, total - split, sample_games, "val")]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    device = torch.device(device)
    precision = precision or manifest["precision"]
    if precision not in PRECISIONS:
        raise ValueError(f"Unsupported precision: {precision}")
    if precision == "float16" and device.type != "cuda":
        raise ValueError("float16 evaluation requires CUDA")
    if precision == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA device does not support bfloat16")

    models = []
    for state in states:
        model = build_model().to(device)
        model.load_state_dict(state["model"])
        model.eval()
        models.append(model)

    print(f"device={device} precision={precision} sample_games={sample_games}", flush=True)
    print("window\tgames\t" + "\t".join(
        f"epoch_{state['completed_epoch']}_mae" for state in states) + "\tdelta_last_minus_first", flush=True)
    results = {}
    for name, selections in windows:
        dataset = GameDataset(
            selections, batch_size=batch_size, shuffle_buffer=1, seed=0,
            decoder=config["decoder"], read_batch_size=config["read_batch_size"],
            prefetch_batches=0,
        )
        values = []
        for model in models:
            with torch.inference_mode():
                values.append(run_epoch(model, dataset, device=device, precision=precision)["origin_mae"])
        results[name] = values
        print(f"{name}\t{dataset.game_count}\t" + "\t".join(f"{value:.3f}" for value in values)
              + f"\t{values[-1] - values[0]:+.3f}", flush=True)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+", help="Epoch .pt files to compare")
    parser.add_argument("--sample-games", type=int, default=4096,
                        help="Games per fixed window (default: 4096)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Evaluation batch size (default: 128)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=tuple(PRECISIONS),
                        help="Evaluation precision; default is the checkpoint's training precision")
    args = parser.parse_args(argv)
    if len(args.checkpoints) < 2:
        parser.error("Pass at least two epoch checkpoint .pt files")
    compare(args.checkpoints, sample_games=args.sample_games, batch_size=args.batch_size,
            device=args.device, precision=args.precision)


if __name__ == "__main__":
    main()
