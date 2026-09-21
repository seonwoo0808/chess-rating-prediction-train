"""torchrun process lifecycle and rank helpers (NCCL on CUDA, Gloo on CPU)."""
from functools import wraps
import os

import torch
import torch.distributed as dist


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def is_primary():
    return rank() == 0


def distributed_session(function):
    """Initialize only for torchrun; always release groups owned by this call."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        owned = False
        try:
            if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
                requested = kwargs.get("device", "auto")
                if requested not in ("auto", "cpu", "cuda"):
                    raise ValueError("device must be auto, cpu or cuda")
                cuda = requested != "cpu" and torch.cuda.is_available()
                if requested == "cuda" and not cuda:
                    raise ValueError("CUDA was requested but is unavailable")
                if cuda:
                    local_rank = int(os.environ["LOCAL_RANK"])
                    if not 0 <= local_rank < torch.cuda.device_count():
                        raise ValueError("LOCAL_RANK exceeds the visible CUDA devices")
                    torch.cuda.set_device(local_rank)
                dist.init_process_group(backend="nccl" if cuda else "gloo")
                owned = True
            return function(*args, **kwargs)
        finally:
            if owned:
                dist.destroy_process_group()
    return wrapped
