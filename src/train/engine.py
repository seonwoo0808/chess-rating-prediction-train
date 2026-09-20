"""One shared pass for training and validation; metrics are weighted by games."""
from contextlib import closing, nullcontext

import torch
from tqdm.auto import tqdm


PRECISIONS = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def run_epoch(model, dataset, *, device, precision, optimizer=None, scaler=None,
              progress=False, description=""):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    elements = 0
    iterator = iter(dataset)
    scope = closing(iterator) if hasattr(iterator, "close") else nullcontext(iterator)
    with scope as batches, tqdm(
        total=len(dataset), desc=description, disable=not progress,
    ) as bar:
        for (boards, valid), targets in batches:
            boards, valid, targets = (tensor.to(device) for tensor in (boards, valid, targets))
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training), torch.autocast(
                device_type=device.type, dtype=PRECISIONS[precision],
                enabled=precision != "float32",
            ):
                predictions = model(boards, valid)
                error = predictions.float() - targets
                loss = error.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite regression loss")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            detached = error.detach()
            totals += torch.stack((detached.square().sum(), detached.abs().sum())).double()
            elements += targets.numel()
            bar.update(1)
            if progress and (bar.n % 20 == 0 or bar.n == len(dataset)):
                mse, mae = (totals / elements).tolist()
                bar.set_postfix(mse=f"{mse:.1f}", mae=f"{mae:.1f}")
    if not elements:
        raise ValueError("Dataset yielded no games")
    mse, mae = (totals / elements).tolist()
    return {"loss": mse, "mae": mae}
