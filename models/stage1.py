from __future__ import annotations

from typing import Dict, NamedTuple, Tuple

import torch
from torch import nn

from .lunet import LUNet1D, build_lunet
from .memory import MemoryRouter, RoutingOutput, ValueMixingHead, read_memory, route_memory


class Stage1Output(NamedTuple):
    reconstruction: torch.Tensor
    target_bottleneck: torch.Tensor
    query: torch.Tensor
    probabilities: torch.Tensor
    top_indices: torch.Tensor
    top_weights: torch.Tensor
    value_bases: torch.Tensor
    beta: torch.Tensor
    memory: torch.Tensor
    decoder_aux: tuple[int, int, int]


class CleanMemoryCuration(nn.Module):
    memory_content = "clean"

    def __init__(
        self,
        teacher: LUNet1D,
        *,
        num_slots: int = 48,
        top_k: int = 8,
        key_dim: int = 32,
        value_dim: int = 64,
        num_value_bases: int = 6,
        router_window_size: int = 13,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if teacher.use_skip:
            raise ValueError("Stage 1 requires no-skip LUNet")
        if top_k > num_slots:
            raise ValueError("top_k cannot exceed num_slots")
        if value_dim != teacher.bottleneck_channels:
            raise ValueError(
                f"value_dim={value_dim} does not match bottleneck={teacher.bottleneck_channels}"
            )
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.backbone = teacher
        self.num_slots = int(num_slots)
        self.top_k = int(top_k)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.num_value_bases = int(num_value_bases)
        self.temperature = float(temperature)
        self.R_T = MemoryRouter(value_dim, key_dim, router_window_size)
        self.C_T = ValueMixingHead(key_dim, num_value_bases)
        self.K = nn.Parameter(torch.randn(num_slots, key_dim) * key_dim**-0.5)
        self.V = nn.Parameter(
            torch.randn(num_slots, num_value_bases, value_dim) * value_dim**-0.5
        )
        self.initialization_report = None

    @property
    def teacher_encoder(self) -> Tuple[nn.Module, ...]:
        return tuple(getattr(self.backbone, name) for name in self.backbone.encoder_attr_names)

    @property
    def teacher_decoder(self) -> Tuple[nn.Module, ...]:
        return tuple(getattr(self.backbone, name) for name in self.backbone.decoder_attr_names)

    def encode(self, x_C: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        return self.backbone.encode(x_C)

    def decode(
        self, memory_readout: torch.Tensor, decoder_aux: tuple[int, int, int]
    ) -> torch.Tensor:
        return self.backbone.decode(memory_readout, decoder_aux)

    def route(self, query: torch.Tensor) -> RoutingOutput:
        return route_memory(query, self.K, self.top_k, self.temperature)

    def memory_readout(
        self, z_T: torch.Tensor
    ) -> tuple[torch.Tensor, RoutingOutput, torch.Tensor, torch.Tensor]:
        routing = self.route(self.R_T(z_T))
        beta = self.C_T(routing.query)
        memory_readout, value_bases = read_memory(routing, self.V, beta)
        return memory_readout, routing, value_bases, beta

    def teacher_pathway(self, x_C: torch.Tensor) -> Stage1Output:
        z_T, decoder_aux = self.encode(x_C)
        memory_readout, routing, value_bases, beta = self.memory_readout(z_T)
        x_hat_C_T = self.decode(memory_readout, decoder_aux)
        return Stage1Output(
            x_hat_C_T,
            z_T,
            routing.query,
            routing.probabilities,
            routing.top_indices,
            routing.top_weights,
            value_bases,
            beta,
            memory_readout,
            decoder_aux,
        )

    def forward(self, x_C: torch.Tensor) -> Stage1Output:
        return self.teacher_pathway(x_C)

    def trainable_groups(self) -> Dict[str, Tuple[nn.Parameter, ...]]:
        return {
            "E_T": tuple(
                parameter
                for module in self.teacher_encoder
                for parameter in module.parameters()
            ),
            "R_T": tuple(self.R_T.parameters()),
            "C_T": tuple(self.C_T.parameters()),
            "K": (self.K,),
            "V": (self.V,),
            "D_T": tuple(
                parameter
                for module in self.teacher_decoder
                for parameter in module.parameters()
            ),
        }

    def assert_gradient_contract(self) -> None:
        expected = {
            id(parameter)
            for parameters in self.trainable_groups().values()
            for parameter in parameters
        }
        actual = {
            id(parameter) for parameter in self.parameters() if parameter.requires_grad
        }
        if expected != actual or expected != {id(parameter) for parameter in self.parameters()}:
            raise AssertionError("Stage 1 must train E_T, R_T, C_T, K, V, and D_T only")


def build_stage1(config: dict) -> CleanMemoryCuration:
    return CleanMemoryCuration(build_lunet(config["backbone"]), **config["memory"])


def split_lunet_state(
    lunet: LUNet1D,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    state = lunet.state_dict()
    encoder_prefixes = tuple(f"{name}." for name in lunet.encoder_attr_names)
    decoder_prefixes = tuple(f"{name}." for name in lunet.decoder_attr_names)
    encoder = {
        name: value.detach().cpu()
        for name, value in state.items()
        if name.startswith(encoder_prefixes)
    }
    decoder = {
        name: value.detach().cpu()
        for name, value in state.items()
        if name.startswith(decoder_prefixes)
    }
    if len(encoder) + len(decoder) != len(state):
        raise AssertionError("LUNet state contains unknown modules")
    return encoder, decoder


def load_stage1_components(
    model: CleanMemoryCuration, checkpoint: dict, device: torch.device
) -> None:
    backbone_state = dict(checkpoint["E_T"])
    backbone_state.update(checkpoint["D"])
    model.backbone.load_state_dict(backbone_state, strict=True)
    model.R_T.load_state_dict(checkpoint["R_T"], strict=True)
    model.C_T.load_state_dict(checkpoint["C_T"], strict=True)
    if checkpoint["K"].shape != model.K.shape or checkpoint["V"].shape != model.V.shape:
        raise RuntimeError("Stage 1 memory shape mismatch")
    with torch.no_grad():
        model.K.copy_(checkpoint["K"].to(device=device, dtype=model.K.dtype))
        model.V.copy_(checkpoint["V"].to(device=device, dtype=model.V.dtype))
    model.initialization_report = checkpoint.get("memory_initialization")
