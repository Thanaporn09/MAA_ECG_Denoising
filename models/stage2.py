from __future__ import annotations

from copy import deepcopy
from typing import Dict, NamedTuple, Tuple

import torch
from torch import nn

from .memory import MemoryRouter, RoutingOutput, read_memory, route_memory
from .stage1 import CleanMemoryCuration


class MAAPathwayOutput(NamedTuple):
    reconstruction: torch.Tensor
    encoder1: torch.Tensor
    encoder2: torch.Tensor
    encoder3: torch.Tensor
    bottleneck: torch.Tensor
    query: torch.Tensor
    probabilities: torch.Tensor
    top_indices: torch.Tensor
    top_weights: torch.Tensor
    value_bases: torch.Tensor
    beta: torch.Tensor
    memory: torch.Tensor


class MAAOutput(NamedTuple):
    teacher: MAAPathwayOutput
    student: MAAPathwayOutput


class MemoryAccessAlignment(nn.Module):
    def __init__(self, stage1: CleanMemoryCuration) -> None:
        super().__init__()
        self.backbone = stage1.backbone
        self.memory_content = "clean"
        self.num_slots = stage1.num_slots
        self.top_k = stage1.top_k
        self.key_dim = stage1.key_dim
        self.value_dim = stage1.value_dim
        self.num_value_bases = stage1.num_value_bases
        self.temperature = stage1.temperature
        self.R_T = stage1.R_T
        self.C_T = stage1.C_T
        self.K = stage1.K
        self.V = stage1.V
        self.E_S = deepcopy(stage1.backbone)
        self.R_S = deepcopy(stage1.R_T)
        self._set_gradient_contract()

    @property
    def teacher_encoder(self) -> Tuple[nn.Module, ...]:
        return tuple(getattr(self.backbone, name) for name in self.backbone.encoder_attr_names)

    @property
    def teacher_decoder(self) -> Tuple[nn.Module, ...]:
        return tuple(getattr(self.backbone, name) for name in self.backbone.decoder_attr_names)

    @property
    def student_encoder(self) -> Tuple[nn.Module, ...]:
        return tuple(getattr(self.E_S, name) for name in self.E_S.encoder_attr_names)

    @property
    def student_decoder(self) -> Tuple[nn.Module, ...]:
        return tuple(getattr(self.E_S, name) for name in self.E_S.decoder_attr_names)

    def _set_gradient_contract(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.E_S.bottleneck.parameters():
            parameter.requires_grad_(True)
        for parameter in self.R_S.parameters():
            parameter.requires_grad_(True)
        self._set_module_modes(False)

    def _set_module_modes(self, training: bool) -> None:
        self.backbone.eval()
        self.E_S.eval()
        self.R_T.eval()
        self.C_T.eval()
        self.E_S.bottleneck.train(training)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._NormBase):
                module.eval()

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self._set_module_modes(mode)
        return self

    def route(self, query: torch.Tensor) -> RoutingOutput:
        return route_memory(query, self.K, self.top_k, self.temperature)

    def _pathway(
        self,
        encoder: nn.Module,
        router: MemoryRouter,
        signal: torch.Tensor,
    ) -> MAAPathwayOutput:
        encoder1, encoder2, encoder3, bottleneck, decoder_aux = encoder.encode_stages(
            signal
        )
        routing = self.route(router(bottleneck))
        beta = self.C_T(routing.query)
        memory_readout, value_bases = read_memory(routing, self.V, beta)
        reconstruction = self.backbone.decode(memory_readout, decoder_aux)
        return MAAPathwayOutput(
            reconstruction,
            encoder1,
            encoder2,
            encoder3,
            bottleneck,
            routing.query,
            routing.probabilities,
            routing.top_indices,
            routing.top_weights,
            value_bases,
            beta,
            memory_readout,
        )

    def teacher_pathway(self, x_C: torch.Tensor) -> MAAPathwayOutput:
        with torch.no_grad():
            return self._pathway(self.backbone, self.R_T, x_C)

    def student_pathway(self, x_N: torch.Tensor) -> MAAPathwayOutput:
        return self._pathway(self.E_S, self.R_S, x_N)

    def forward(self, x_N: torch.Tensor, x_C: torch.Tensor) -> MAAOutput:
        student = self.student_pathway(x_N)
        teacher = self.teacher_pathway(x_C)
        return MAAOutput(teacher, student)

    def trainable_groups(self) -> Dict[str, Tuple[nn.Parameter, ...]]:
        return {
            "E_S.bottleneck": tuple(self.E_S.bottleneck.parameters()),
            "R_S": tuple(self.R_S.parameters()),
        }

    def trainable_parameter_names(self) -> list[str]:
        return [
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        ]

    def assert_gradient_contract(self) -> None:
        expected = {
            id(parameter)
            for parameters in self.trainable_groups().values()
            for parameter in parameters
        }
        actual = {
            id(parameter) for parameter in self.parameters() if parameter.requires_grad
        }
        if actual != expected:
            raise AssertionError("MAA trainables must be E_S.bottleneck and R_S only")
        frozen = {
            "teacher LUNet": tuple(self.backbone.parameters()),
            "memory": (self.K, self.V),
            "teacher router": tuple(self.R_T.parameters()),
            "value mixing": tuple(self.C_T.parameters()),
            "student decoder": tuple(
                parameter
                for module in self.student_decoder
                for parameter in module.parameters()
            ),
            "student early encoder": tuple(
                parameter
                for module in self.student_encoder[:-1]
                for parameter in module.parameters()
            ),
        }
        for name, parameters in frozen.items():
            if any(parameter.requires_grad for parameter in parameters):
                raise AssertionError(f"Frozen MAA component is trainable: {name}")

    def frozen_state(self) -> Dict[str, torch.Tensor]:
        trainable_names = set(self.trainable_parameter_names())
        return {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
            if name not in trainable_names
        }


def load_stage2_components(model: MemoryAccessAlignment, checkpoint: dict) -> None:
    for name in ("encoder1", "encoder2", "encoder3", "bottleneck"):
        getattr(model.E_S, name).load_state_dict(checkpoint["E_S"][name], strict=True)
    model.R_S.load_state_dict(checkpoint["R_S"], strict=True)
    model.assert_gradient_contract()
