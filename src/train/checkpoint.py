"""Atomic epoch checkpoints containing weights, optimizer, scaler and RNG state."""
import json
import os
from pathlib import Path
import uuid

import torch


FORMAT_VERSION = 2


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
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
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
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["completed_epoch"], state["history"]
