from __future__ import annotations

import math
from typing import Dict

import torch


METRIC_NAMES = ("RMSE", "PRD", "MaxAE", "DeltaSNR", "CC", "CosSim")


def _snr(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    signal = reference.square().mean(dim=(1, 2)).clamp_min(1e-12)
    error = (estimate - reference).square().mean(dim=(1, 2)).clamp_min(1e-12)
    return 10.0 * torch.log10(signal / error)


def metric_values(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    clean_hat: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if noisy.shape != clean.shape or clean_hat.shape != clean.shape:
        raise ValueError("Metric inputs must have identical [B,1,T] shapes")
    noisy = noisy.double()
    clean = clean.double()
    clean_hat = clean_hat.double()
    mu = mu.double().reshape(-1, 1, 1)
    sigma = sigma.double().reshape(-1, 1, 1).clamp_min(1e-6)
    noisy = noisy * sigma + mu
    clean = clean * sigma + mu
    clean_hat = clean_hat * sigma + mu
    difference = clean_hat - clean
    flat_difference = difference.flatten(1)
    flat_clean = clean.flatten(1)
    flat_hat = clean_hat.flatten(1)
    rmse = flat_difference.square().mean(dim=1).sqrt()
    prd = 100.0 * (
        flat_difference.square().sum(dim=1)
        / flat_clean.square().sum(dim=1).clamp_min(1e-12)
    ).sqrt()
    maxae = flat_difference.abs().max(dim=1).values
    clean_centered = flat_clean - flat_clean.mean(dim=1, keepdim=True)
    hat_centered = flat_hat - flat_hat.mean(dim=1, keepdim=True)
    cc = (clean_centered * hat_centered).mean(dim=1) / (
        clean_centered.square().mean(dim=1).sqrt()
        * hat_centered.square().mean(dim=1).sqrt()
    ).clamp_min(1e-12)
    cosine = (flat_clean * flat_hat).sum(dim=1) / (
        flat_clean.square().sum(dim=1).sqrt()
        * flat_hat.square().sum(dim=1).sqrt()
    ).clamp_min(1e-12)
    delta_snr = _snr(clean_hat, clean) - _snr(noisy, clean)
    return {
        "RMSE": rmse,
        "PRD": prd,
        "MaxAE": maxae,
        "DeltaSNR": delta_snr,
        "CC": cc,
        "CosSim": cosine,
    }


class MetricAccumulator:
    def __init__(self) -> None:
        self.values = {name: [] for name in METRIC_NAMES}

    def update(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        clean_hat: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> None:
        values = metric_values(noisy, clean, clean_hat, mu, sigma)
        for name, value in values.items():
            self.values[name].append(value.detach().cpu())

    def finalize(self) -> Dict[str, Dict[str, float]]:
        if not any(self.values.values()):
            raise RuntimeError("Metric accumulator is empty")
        report = {}
        for name, chunks in self.values.items():
            values = torch.cat(chunks).double()
            report[name] = {
                "mean": float(values.mean()),
                "std": float(values.std(unbiased=False)),
            }
            if not math.isfinite(report[name]["mean"]):
                raise RuntimeError(f"Non-finite metric: {name}")
        return report
