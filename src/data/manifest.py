"""Dataset identity used by epoch-level checkpoints."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .split import split_plan


def dataset_manifest(paths, *, batch_size, validation_size, max_games,
                     read_batch_size, generator_batch_size, decoder,
                     shuffle_buffer, prefetch, seed):
    if isinstance(paths, (str, Path)):
        paths = (paths,)
    paths = tuple(Path(path).resolve() for path in paths)
    files = []
    for path in paths:
        stat = path.stat()
        files.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    total, train_count, _, _ = split_plan(paths, max_games, validation_size)
    code_root = Path(__file__).parent
    preprocessing = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(code_root.glob("*.py"))
    }
    return {
        "format": 1,
        "files": files,
        "config": {
            "batch_size": batch_size,
            "validation_size": validation_size,
            "max_games": max_games,
            "read_batch_size": read_batch_size,
            "generator_batch_size": generator_batch_size,
            "decoder": decoder,
            "shuffle_buffer": shuffle_buffer,
            "prefetch": prefetch,
            "seed": seed,
        },
        "total_games": total,
        "train_games": train_count,
        "steps_per_epoch": (train_count + batch_size - 1) // batch_size,
        "preprocessing": preprocessing,
    }
