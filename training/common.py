from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import torch

from evaluation import MetricAccumulator


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.inference_mode()
def evaluate_reconstruction(
    prediction_function,
    loader: Iterable[dict],
    device: torch.device,
    use_amp: bool,
) -> dict:
    accumulator = MetricAccumulator()
    for batch in loader:
        noisy = batch["noisy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        with autocast_context(device, use_amp):
            prediction = prediction_function(noisy, clean)
        accumulator.update(
            noisy,
            clean,
            prediction,
            batch["mu"],
            batch["sigma"],
        )
    return accumulator.finalize()


def build_stage2_optimizer(model, config: dict):
    trainable = {
        id(parameter): (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    normalization_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn_norm_types())
        for parameter in module.parameters(recurse=False)
    }
    decay = []
    no_decay = []
    for parameter_id, (name, parameter) in trainable.items():
        if parameter_id in normalization_ids or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    weight_decay = float(config["optimizer"]["weight_decay"])
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    assigned = {id(parameter) for group in groups for parameter in group["params"]}
    if assigned != set(trainable):
        raise AssertionError("Optimizer groups do not match Stage 2 trainables")
    return torch.optim.AdamW(
        groups,
        lr=float(config["optimizer"]["lr"]),
        weight_decay=weight_decay,
    )


def nn_norm_types():
    return (
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.modules.batchnorm._NormBase,
    )
