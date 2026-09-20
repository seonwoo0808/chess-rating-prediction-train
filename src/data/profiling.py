"""Measure CPU input stages independently of model execution."""
from __future__ import annotations

import time

from .batches import row_batches
from .split import split_counts

def profile_input_stages(path, max_games, test_size, read_batch_size,
                         generator_batch_size, sample_games, batch_size, decoder="numba"):
    """A separate bounded CPU pass; these are wall times, not GPU step times."""
    _, split = split_counts(path, max_games, test_size)
    count = min(split, sample_games)
    timings = dict.fromkeys(("arrow_read_seconds", "arrow_numpy_seconds",
                             "board_decode_seconds", "batch_pack_seconds"), 0.0)
    timings.update(rows=0, arrow_batches=0, generator_batches=0)
    started = time.perf_counter()
    for _ in row_batches(path, max_games, stop=count, read_batch_size=read_batch_size,
                         generator_batch_size=generator_batch_size, profile=timings, decoder=decoder):
        pass
    elapsed = time.perf_counter() - started
    assert timings["rows"] == count
    scale = batch_size * 1000 / count
    return {
        "sample_games": count, "decoder": decoder, "read_batch_size": read_batch_size,
        "generator_batch_size": generator_batch_size,
        "generator_batches": timings["generator_batches"],
        "arrow_batches": timings["arrow_batches"],
        "parquet_read_ms_per_batch": round(timings["arrow_read_seconds"] * scale, 3),
        "arrow_numpy_ms_per_batch": round(timings["arrow_numpy_seconds"] * scale, 3),
        "input_format": "Arrow buffers -> NumPy (no to_pydict)",
        "board_decode_ms_per_batch": round(timings["board_decode_seconds"] * scale, 3),
        "batch_pack_ms_per_batch": round(timings["batch_pack_seconds"] * scale, 3),
        "other_ms_per_batch": round(max(0, elapsed - sum(timings[k] for k in timings
                                                          if k.endswith("_seconds"))) * scale, 3),
        "total_ms_per_batch": round(elapsed * scale, 3),
        "note": "Separate post-benchmark CPU pass normalized to training batch size. Arrow read includes decompression and OS cache effects; decode is serial. Excludes tf.data shuffle/transfer/prefetch; do not add these times to end_to_end_ms_per_step.",
    }
