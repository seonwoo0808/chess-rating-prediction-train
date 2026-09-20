"""Atomic full-model checkpoints written after completed epochs."""
from __future__ import annotations

import hashlib
import json
import os
from importlib import metadata
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np
import tensorflow as tf
from tensorflow import keras


def runtime():
    keras_version = getattr(keras, "__version__", None)
    if keras_version is None:
        package = "tf-keras" if keras.__name__.startswith("tf_keras") else "keras"
        keras_version = metadata.version(package)
    return {
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "keras": keras_version,
        "keras_module": keras.__name__,
        "numpy": np.__version__,
        "precision": keras.mixed_precision.global_policy().name,
    }


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class EpochCheckpoint(keras.callbacks.Callback):
    """Save the complete model and optimizer after every completed epoch."""

    def __init__(self, directory, data_manifest):
        super().__init__()
        self.directory = Path(directory)
        self.data_manifest = data_manifest
        self.last_checkpoint = None

    def on_epoch_end(self, epoch, logs=None):
        self.save(epoch + 1, logs or {})

    def save(self, completed_epoch, logs):
        self.directory.mkdir(parents=True, exist_ok=True)
        name = f"epoch-{completed_epoch:06d}-{uuid.uuid4().hex[:8]}"
        staging = self.directory / (".pending-" + name)
        destination = self.directory / name
        staging.mkdir()
        try:
            self.model.save(staging / "model.keras")
            clean_logs = {key: float(value) for key, value in logs.items()}
            state = {
                "format": 1,
                "completed_epoch": completed_epoch,
                "logs": clean_logs,
                "runtime": runtime(),
                "data": self.data_manifest,
                "model_sha256": digest(staging / "model.keras"),
            }
            write_json(staging / "state.json", state)
            for path in staging.iterdir():
                with open(path, "rb") as stream:
                    os.fsync(stream.fileno())
            os.replace(staging, destination)
            pointer = self.directory / (".latest-" + uuid.uuid4().hex + ".json")
            write_json(pointer, {"checkpoint": name})
            os.replace(pointer, self.directory / "latest.json")
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self.last_checkpoint = destination
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return destination


def load_epoch_checkpoint(path, data_manifest):
    """Load the latest or a specific epoch checkpoint after validating identity."""
    path = Path(path)
    if (path / "latest.json").is_file():
        name = json.loads((path / "latest.json").read_text(encoding="utf-8"))["checkpoint"]
        if Path(name).name != name:
            raise ValueError("Invalid checkpoint pointer")
        path = path / name
    state = json.loads((path / "state.json").read_text(encoding="utf-8"))
    if state.get("format") != 1:
        raise ValueError("Unsupported epoch checkpoint format")
    if state["runtime"] != runtime():
        raise ValueError("Checkpoint runtime, Keras version, or precision differs")
    if state["data"] != data_manifest:
        raise ValueError("Checkpoint dataset, file order, or preprocessing differs")
    model_path = path / "model.keras"
    if digest(model_path) != state["model_sha256"]:
        raise ValueError("Corrupt epoch checkpoint model")
    import models  # Register custom layers before deserializing.
    model = keras.models.load_model(model_path)
    return model, state
