from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .lunet import same_padding


class RoutingOutput(NamedTuple):
    query: torch.Tensor
    probabilities: torch.Tensor
    top_indices: torch.Tensor
    top_weights: torch.Tensor


class MemoryRouter(nn.Module):
    def __init__(self, input_dim: int, key_dim: int, window_size: int) -> None:
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("router window_size must be odd")
        self.projection = nn.Conv1d(
            input_dim,
            key_dim,
            kernel_size=window_size,
            padding=same_padding(window_size, dilation=1),
            bias=False,
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(bottleneck).transpose(1, 2), dim=-1)


class ValueMixingHead(nn.Module):
    def __init__(self, key_dim: int, num_bases: int) -> None:
        super().__init__()
        self.projection = nn.Linear(key_dim, num_bases)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.projection(query), dim=-1)


def route_memory(
    query: torch.Tensor,
    keys: torch.Tensor,
    top_k: int,
    temperature: float,
) -> RoutingOutput:
    normalized_keys = F.normalize(keys, dim=-1)
    logits = torch.matmul(query.float(), normalized_keys.float().t()) / temperature
    probabilities = torch.softmax(logits, dim=-1)
    top_probabilities, top_indices = probabilities.topk(top_k, dim=-1)
    top_weights = top_probabilities / top_probabilities.sum(dim=-1, keepdim=True)
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("Memory routing contains NaN or Inf")
    return RoutingOutput(query, probabilities, top_indices, top_weights)


def read_memory(
    routing: RoutingOutput,
    values: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_values = values[routing.top_indices]
    value_bases = torch.einsum(
        "btk,btkmd->btmd", routing.top_weights, selected_values
    )
    tokens = torch.einsum("btm,btmd->btd", beta, value_bases)
    return tokens.transpose(1, 2), value_bases
