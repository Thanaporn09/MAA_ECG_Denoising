from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.nn import functional as F


def spectral_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Spectral loss shape mismatch: {prediction.shape} vs {target.shape}"
        )
    prediction_spectrum = torch.fft.rfft(prediction.float(), dim=-1).abs()
    target_spectrum = torch.fft.rfft(target.float(), dim=-1).abs()
    scale = target_spectrum.mean(dim=-1, keepdim=True).clamp_min(1e-8)
    return F.l1_loss(prediction_spectrum / scale, target_spectrum / scale)


def reconstruction_objective(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    config: dict,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mse = F.mse_loss(reconstruction, target)
    l1 = F.l1_loss(reconstruction, target)
    spectral = spectral_l1(reconstruction, target)
    total = (
        mse
        + float(config["lambda_l1"]) * l1
        + float(config["lambda_spec"]) * spectral
    )
    return total, {
        "reconstruction_mse": mse,
        "reconstruction_l1": l1,
        "reconstruction_spectral": spectral,
    }
