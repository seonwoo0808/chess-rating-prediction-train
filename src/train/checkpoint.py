"""Atomic epoch checkpoints containing weights, optimizer, scaler and RNG state."""
import json
import os
from pathlib import Path
import uuid

import torch
import torch.distributed as dist

from .distributed import is_primary, rank, world_size


FORMAT_VERSION = 3


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(directory, *, model, optimizer, scaler, completed_epoch,
                    manifest, history):
    # All ranks participate; only rank zero writes model/optimizer and files.
    local_rng = {
        "cpu": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state().cpu()
                 if next(model.parameters()).device.type == "cuda" else None),
    }
    rng_by_rank = [None] * world_size() if is_primary() else None
    if dist.is_initialized():
        dist.gather_object(local_rng, rng_by_rank, dst=0)
    else:
        rng_by_rank[0] = local_rng
    if not is_primary():
        return None
    directory = Path(directory)
    path = directory / f"epoch-{completed_epoch:06d}.pt"
    state = {
        "format": FORMAT_VERSION,
        "completed_epoch": completed_epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "manifest": manifest,
        "history": history,
        "rng_by_rank": rng_by_rank,
    }
    atomic_save(path, state)
    # The pointer is replaced only after the entire checkpoint is durable.
    temporary = directory / f".latest-{uuid.uuid4().hex}.json"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump({"checkpoint": path.name}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "latest.json")
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_checkpoint(path, *, model, optimizer, scaler, manifest):
    path = Path(path)
    if path.is_dir():
        name = json.loads((path / "latest.json").read_text())["checkpoint"]
        if Path(name).name != name or not name.endswith(".pt"):
            raise ValueError("Invalid checkpoint pointer")
        path = path / name
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format; start a new PyTorch run")
    if state["manifest"] != manifest:
        raise ValueError("Checkpoint data, model code or training settings differ")
    if len(state["rng_by_rank"]) != world_size():
        raise ValueError("Checkpoint world size differs")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    rng = state["rng_by_rank"][rank()]
    torch.set_rng_state(rng["cpu"])
    if rng["cuda"] is not None:
        torch.cuda.set_rng_state(rng["cuda"])
    return state["completed_epoch"], state["history"]
