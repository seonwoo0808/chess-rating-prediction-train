"""One shared pass for training and validation; metrics are weighted by games."""
from contextlib import closing, nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm


PRECISIONS = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

# Ratings are optimized in standardized units. The loss stays standardized;
# only the reported MAE is converted back to the original rating scale.
RATING_MEAN = 1660.0
RATING_STD = 400.0
MAX_GRAD_NORM = 1.0


def run_epoch(model, dataset, *, device, precision, optimizer=None, scaler=None,
              progress=False, description=""):
    training = optimizer is not None
    model.train(training)
    # Validation ranks may have different numbers of batches (including zero).
    # Avoid DDP forward collectives; this model has no mutable training buffers.
    if not training and isinstance(model, DistributedDataParallel):
        model = model.module
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    elements = 0
    iterator = iter(dataset)
    scope = closing(iterator) if hasattr(iterator, "close") else nullcontext(iterator)
    with scope as batches, tqdm(
        total=len(dataset), desc=description, disable=not progress,
    ) as bar:
        for (boards, valid, game_type), targets in batches:
            boards, valid, game_type, targets = (
                tensor.to(device) for tensor in (boards, valid, game_type, targets))
            raw_targets = targets.float()
            if not torch.isfinite(raw_targets).all():
                raise FloatingPointError("Non-finite rating target")
            normalized_targets = (raw_targets - RATING_MEAN) / RATING_STD
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training), torch.autocast(
                device_type=device.type, dtype=PRECISIONS[precision],
                enabled=precision != "float32",
            ):
                predictions = model(boards, valid, game_type)
                # Optimize the standardized target; this keeps the regression
                # loss and its gradients at a numerically well-scaled magnitude.
                error = predictions.float() - normalized_targets
                loss = error.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite regression loss")
            if training:
                scaler.scale(loss).backward()
                # Clip actual gradients after loss scaling has been undone.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=MAX_GRAD_NORM, error_if_nonfinite=True,
                )
                scaler.step(optimizer)
                scaler.update()
            # Keep loss in standardized units. Convert only the MAE to the
            # original rating scale, using float64 and multiplication by sigma
            # instead of explicitly materializing prediction * sigma + mean.
            detached = error.detach().double()
            metric_error = detached * RATING_STD
            if not torch.isfinite(metric_error).all():
                raise FloatingPointError("Non-finite original-scale MAE")
            totals += torch.stack((detached.square().sum(), metric_error.abs().sum()))
            elements += raw_targets.numel()
            bar.update(1)
            if progress and (bar.n % 20 == 0 or bar.n == len(dataset)):
                loss_value, origin_mae = (totals / elements).tolist()
                bar.set_postfix(loss=f"{loss_value:.4f}", origin_mae=f"{origin_mae:.1f}")
    if dist.is_initialized():
        combined = torch.cat((totals, totals.new_tensor([elements])))
        dist.all_reduce(combined, op=dist.ReduceOp.SUM)
        totals, elements = combined[:2], combined[2].item()
    if not elements:
        raise ValueError("Dataset yielded no games")
    loss_value, origin_mae = (totals / elements).tolist()
    return {"loss": loss_value, "origin_mae": origin_mae}
