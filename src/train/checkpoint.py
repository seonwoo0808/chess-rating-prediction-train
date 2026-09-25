"""Atomic epoch checkpoints containing weights, optimizer, scaler and RNG state."""
import json
import os
from pathlib import Path
import uuid

import torch
import torch.distributed as dist

from .distributed import is_primary, rank, world_size


FORMAT_VERSION = 3
LEGACY_ORCHESTRATION_FILES = ("main.py", "checkpoint.py")


def pre_step_lr_expected_manifest(current, saved):
    """Retain all data/model/optimizer settings; permit old orchestration code."""
    expected = dict(current)
    expected.pop("lr_schedule", None)
    codes = dict(current["model_code"])
    for name in LEGACY_ORCHESTRATION_FILES:
        if name in saved.get("model_code", {}):
            codes[name] = saved["model_code"][name]
    expected["model_code"] = codes
    return expected


def is_pre_step_lr_manifest(saved, current):
    """The old first epoch may resume when model/data and settings still match."""
    return saved == pre_step_lr_expected_manifest(current, saved)


def manifest_mismatch_details(saved, expected):
    """Summarize relevant differences without printing every file or hash."""
    differences = []
    saved_data, expected_data = saved.get("data", {}), expected.get("data", {})
    for key in ("files", "preprocessing"):
        if saved_data.get(key) != expected_data.get(key):
            differences.append(f"data.{key}")
    saved_config, expected_config = saved_data.get("config", {}), expected_data.get("config", {})
    for key in sorted(saved_config.keys() | expected_config.keys()):
        if saved_config.get(key) != expected_config.get(key):
            differences.append(f"data.config.{key}: saved={saved_config.get(key)!r}, current={expected_config.get(key)!r}")
    saved_code, expected_code = saved.get("model_code", {}), expected.get("model_code", {})
    for key in sorted(saved_code.keys() | expected_code.keys()):
        if saved_code.get(key) != expected_code.get(key):
            differences.append(f"model_code.{key}")
    for key in sorted((saved.keys() | expected.keys()) - {"data", "model_code"}):
        if saved.get(key) != expected.get(key):
            differences.append(f"{key}: saved={saved.get(key)!r}, current={expected.get(key)!r}")
    return "; ".join(differences) or "unknown difference"


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
                    manifest, history, scheduler=None):
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
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
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


def load_checkpoint(path, *, model, optimizer, scaler, manifest, scheduler=None,
                    allow_pre_step_lr=False):
    path = Path(path)
    if path.is_dir():
        name = json.loads((path / "latest.json").read_text())["checkpoint"]
        if Path(name).name != name or not name.endswith(".pt"):
            raise ValueError("Invalid checkpoint pointer")
        path = path / name
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format; start a new PyTorch run")
    legacy = (allow_pre_step_lr and scheduler is not None
              and state["completed_epoch"] == 1
              and state.get("scheduler") is None
              and is_pre_step_lr_manifest(state["manifest"], manifest))
    if state["manifest"] != manifest and not legacy:
        expected = (pre_step_lr_expected_manifest(manifest, state["manifest"])
                    if allow_pre_step_lr and state["completed_epoch"] == 1
                    and state.get("scheduler") is None else manifest)
        raise ValueError("Checkpoint data, model code or training settings differ: "
                         + manifest_mismatch_details(state["manifest"], expected))
    if scheduler is not None and not legacy and state.get("scheduler") is None:
        raise ValueError("Checkpoint has no StepLR state")
    if len(state["rng_by_rank"]) != world_size():
        raise ValueError("Checkpoint world size differs")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    if scheduler is not None:
        if legacy:
            # The old first epoch used the same initial LR. Prepare the first
            # resumed epoch at the LR specified by this StepLR schedule.
            scheduler_state = scheduler.state_dict()
            scheduler_state["last_epoch"] = 1
            scheduler_state["_step_count"] = 2
            scheduler_state["_last_lr"] = [
                base * scheduler.gamma ** (1 // scheduler.step_size)
                for base in scheduler.base_lrs
            ]
            scheduler.load_state_dict(scheduler_state)
            for group, next_lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
                group["lr"] = next_lr
        else:
            scheduler.load_state_dict(state["scheduler"])
    rng = state["rng_by_rank"][rank()]
    torch.set_rng_state(rng["cpu"])
    if rng["cuda"] is not None:
        torch.cuda.set_rng_state(rng["cuda"])
    return state["completed_epoch"], state["history"]
