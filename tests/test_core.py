from __future__ import annotations

import unittest

import torch

from losses import memory_access_alignment_objective, reconstruction_objective
from models import MemoryAccessAlignment, build_stage1


def configuration():
    return {
        "backbone": {
            "in_channels": 1,
            "out_channels": 1,
            "enc_channels": [16, 32, 64],
            "dec_channels": [32, 16, 16],
            "encoder_kernels": [7, 5, 3, 3],
            "decoder_kernels": [3, 5, 7],
            "dilation": 2,
            "pool_kernels": [5, 2, 2],
            "use_norm": True,
            "use_skip": False,
        },
        "memory": {
            "num_slots": 48,
            "top_k": 8,
            "key_dim": 32,
            "value_dim": 64,
            "num_value_bases": 6,
            "router_window_size": 13,
            "temperature": 0.1,
        },
        "loss": {"lambda_l1": 1.0, "lambda_spec": 0.1, "eps": 1e-8},
        "maa_losses": {
            "bottleneck_alignment": {"enabled": True, "weight": 1.0},
            "query_alignment": {"enabled": False, "weight": 0.0},
            "route_alignment": {"enabled": False, "weight": 0.0},
            "memory_readout_alignment": {"enabled": True, "weight": 1.0},
            "student_reconstruction": {"enabled": False, "weight": 0.0},
        },
    }


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = configuration()
        self.stage1 = build_stage1(self.config)

    def test_stage1_shapes_and_loss(self):
        x_C = torch.randn(2, 1, 64)
        output = self.stage1(x_C)
        self.assertEqual(tuple(output.reconstruction.shape), (2, 1, 64))
        self.assertEqual(tuple(output.target_bottleneck.shape), (2, 64, 4))
        self.assertEqual(tuple(output.query.shape), (2, 4, 32))
        self.assertEqual(tuple(output.memory.shape), (2, 64, 4))
        total, parts = reconstruction_objective(
            output.reconstruction, x_C, self.config["loss"]
        )
        expected = (
            parts["reconstruction_mse"]
            + parts["reconstruction_l1"]
            + 0.1 * parts["reconstruction_spectral"]
        )
        torch.testing.assert_close(total, expected)

    def test_stage2_trainable_set_and_student_only_inference(self):
        for parameter in self.stage1.parameters():
            parameter.requires_grad_(False)
        model = MemoryAccessAlignment(self.stage1)
        model.assert_gradient_contract()
        self.assertEqual(
            model.trainable_parameter_names(),
            [
                "E_S.bottleneck.depthwise.weight",
                "E_S.bottleneck.pointwise.weight",
                "E_S.bottleneck.pointwise.bias",
                "E_S.bottleneck.norm.weight",
                "E_S.bottleneck.norm.bias",
                "R_S.projection.weight",
            ],
        )
        x_N = torch.randn(2, 1, 64)
        output = model.student_pathway(x_N)
        self.assertEqual(tuple(output.reconstruction.shape), (2, 1, 64))

    def test_stage2_objective(self):
        for parameter in self.stage1.parameters():
            parameter.requires_grad_(False)
        model = MemoryAccessAlignment(self.stage1)
        x_N = torch.randn(2, 1, 64)
        x_C = torch.randn(2, 1, 64)
        total, losses, _ = memory_access_alignment_objective(
            model, x_N, x_C, self.config
        )
        expected = (
            losses["bottleneck_alignment"]
            + losses["memory_readout_alignment"]
        )
        torch.testing.assert_close(total, expected)
        total.backward()
        active = {
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad.detach()).item())
        }
        self.assertEqual(active, set(model.trainable_parameter_names()))


if __name__ == "__main__":
    unittest.main()
