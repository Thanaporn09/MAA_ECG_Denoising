from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.nn import functional as F

from models.stage2 import MAAOutput, MemoryAccessAlignment

from .reconstruction import reconstruction_objective


def _probabilities(value: torch.Tensor, eps: float) -> torch.Tensor:
    value = value.clamp_min(eps)
    return value / value.sum(dim=-1, keepdim=True)


def teacher_to_student_kl(
    teacher: torch.Tensor, student: torch.Tensor, eps: float
) -> torch.Tensor:
    target = _probabilities(teacher.detach(), eps)
    prediction = _probabilities(student, eps)
    return (target * (target.log() - prediction.log())).sum(dim=-1).mean()


def memory_access_alignment_objective(
    model: MemoryAccessAlignment,
    x_N: torch.Tensor,
    x_C: torch.Tensor,
    config: dict,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], MAAOutput]:
    output = model(x_N, x_C)
    losses = config["maa_losses"]
    eps = float(config["loss"]["eps"])
    bottleneck = F.mse_loss(
        output.student.bottleneck, output.teacher.bottleneck.detach()
    )
    query = F.mse_loss(output.student.query, output.teacher.query.detach())
    route = teacher_to_student_kl(
        output.teacher.probabilities, output.student.probabilities, eps
    )
    memory = F.l1_loss(output.student.memory, output.teacher.memory.detach())
    reconstruction, reconstruction_parts = reconstruction_objective(
        output.student.reconstruction, x_C, config["loss"]
    )
    raw = {
        "bottleneck_alignment": bottleneck,
        "query_alignment": query,
        "route_alignment": route,
        "memory_readout_alignment": memory,
        "student_reconstruction": reconstruction,
    }
    weighted: Dict[str, torch.Tensor] = {}
    total = x_N.new_zeros(())
    for name, value in raw.items():
        settings = losses[name]
        contribution = (
            value * float(settings["weight"])
            if settings["enabled"]
            else x_N.new_zeros(())
        )
        weighted[f"{name}_weighted"] = contribution
        total = total + contribution
    return total, {**raw, **reconstruction_parts, **weighted, "total_loss": total}, output
