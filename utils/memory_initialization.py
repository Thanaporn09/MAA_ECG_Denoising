from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch.nn import functional as F

from models.stage1 import CleanMemoryCuration


def _validate_settings(settings: Dict[str, Any]) -> None:
    required = {
        "enabled",
        "seed",
        "max_samples",
        "key_fit_tokens",
        "key_iterations",
        "value_iterations",
    }
    if set(settings) != required:
        raise ValueError(f"memory_initialization must contain exactly {sorted(required)}")
    if settings["enabled"] is not True:
        raise ValueError("memory initialization must be enabled")
    for name in required - {"enabled"}:
        value = settings[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"memory_initialization.{name} must be a positive integer")


def _kmeans_plus_plus(
    vectors: torch.Tensor,
    clusters: int,
    seed: int,
    cosine: bool,
) -> torch.Tensor:
    if vectors.ndim != 2 or vectors.shape[0] < clusters:
        raise RuntimeError(
            f"Cannot initialize {clusters} prototypes from {vectors.shape[0]} vectors"
        )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    first = int(torch.randint(vectors.shape[0], (1,), generator=generator).item())
    centers = [vectors[first].clone()]
    for _ in range(1, clusters):
        stacked = torch.stack(centers)
        if cosine:
            distances = 1.0 - (
                vectors @ F.normalize(stacked, dim=-1).t()
            ).max(dim=1).values
        else:
            distances = (
                vectors.square().sum(dim=1, keepdim=True)
                + stacked.square().sum(dim=1).unsqueeze(0)
                - 2.0 * vectors @ stacked.t()
            ).min(dim=1).values
        distances = distances.clamp_min(0.0)
        total = distances.sum()
        if not torch.isfinite(total) or float(total.item()) <= 0.0:
            raise RuntimeError("K-means++ cannot choose distinct prototypes")
        index = int(
            torch.multinomial(distances / total, 1, generator=generator).item()
        )
        centers.append(vectors[index].clone())
    result = torch.stack(centers)
    return F.normalize(result, dim=-1) if cosine else result


@torch.no_grad()
def _deterministic_kmeans(
    vectors: torch.Tensor,
    clusters: int,
    iterations: int,
    seed: int,
    cosine: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vectors = vectors.detach().cpu().float().contiguous()
    if cosine:
        vectors = F.normalize(vectors, dim=-1)
    centers = _kmeans_plus_plus(vectors, clusters, seed, cosine)
    for iteration in range(int(iterations)):
        if cosine:
            assignment = (vectors @ centers.t()).argmax(dim=1)
        else:
            distances = (
                vectors.square().sum(dim=1, keepdim=True)
                + centers.square().sum(dim=1).unsqueeze(0)
                - 2.0 * vectors @ centers.t()
            )
            assignment = distances.argmin(dim=1)
        counts = torch.bincount(assignment, minlength=clusters)
        empty = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        if empty:
            raise RuntimeError(f"Empty clusters at iteration {iteration}: {empty}")
        updated = torch.zeros_like(centers)
        updated.index_add_(0, assignment, vectors)
        updated = updated / counts.to(updated.dtype).unsqueeze(1)
        centers = F.normalize(updated, dim=-1) if cosine else updated
    if cosine:
        assignment = (vectors @ centers.t()).argmax(dim=1)
    else:
        distances = (
            vectors.square().sum(dim=1, keepdim=True)
            + centers.square().sum(dim=1).unsqueeze(0)
            - 2.0 * vectors @ centers.t()
        )
        assignment = distances.argmin(dim=1)
    counts = torch.bincount(assignment, minlength=clusters)
    if bool((counts == 0).any()):
        raise RuntimeError("Final clustering contains empty prototypes")
    return centers, assignment, counts


@torch.no_grad()
def _collect_tokens(
    model: CleanMemoryCuration,
    loader: Iterable[Dict[str, Any]],
    device: torch.device,
    max_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    z_parts: List[torch.Tensor] = []
    q_parts: List[torch.Tensor] = []
    sample_keys: List[str] = []
    collected = 0
    model.eval()
    for batch in loader:
        remaining = int(max_samples) - collected
        if remaining <= 0:
            break
        x_C = batch["clean"][:remaining].to(device, non_blocking=True)
        keys = [str(value) for value in batch["sample_key"][:remaining]]
        z_T, _ = model.encode(x_C)
        q_T = model.R_T(z_T)
        if q_T.shape[:2] != (z_T.shape[0], z_T.shape[-1]):
            raise AssertionError("Teacher query and latent tokens lost alignment")
        z_parts.append(z_T.permute(0, 2, 1).reshape(-1, model.value_dim).cpu())
        q_parts.append(q_T.reshape(-1, model.key_dim).cpu())
        sample_keys.extend(keys)
        collected += int(x_C.shape[0])
    if collected != int(max_samples):
        raise RuntimeError(
            f"Initialization requested {max_samples} training samples, got {collected}"
        )
    z_tokens = torch.cat(z_parts).float().contiguous()
    q_tokens = F.normalize(torch.cat(q_parts).float().contiguous(), dim=-1)
    return z_tokens, q_tokens, sample_keys


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


@torch.no_grad()
def initialize_memory(
    model: CleanMemoryCuration,
    loader: Iterable[Dict[str, Any]],
    settings: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    _validate_settings(settings)
    z_tokens, q_tokens, sample_keys = _collect_tokens(
        model, loader, device, int(settings["max_samples"])
    )
    seed = int(settings["seed"])
    fit_count = min(int(settings["key_fit_tokens"]), q_tokens.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fit_indices = torch.randperm(q_tokens.shape[0], generator=generator)[:fit_count]
    keys, _, _ = _deterministic_kmeans(
        q_tokens[fit_indices],
        model.num_slots,
        int(settings["key_iterations"]),
        seed + 1,
        cosine=True,
    )
    full_assignment = (q_tokens @ keys.t()).argmax(dim=1)
    slot_counts = torch.bincount(full_assignment, minlength=model.num_slots)
    empty_slots = torch.nonzero(slot_counts == 0, as_tuple=False).flatten().tolist()
    if empty_slots:
        raise RuntimeError(f"Empty K slots after full assignment: {empty_slots}")
    values = []
    slot_reports = []
    for slot in range(model.num_slots):
        slot_tokens = z_tokens[full_assignment == slot]
        if slot_tokens.shape[0] < model.num_value_bases:
            raise RuntimeError(
                f"K slot {slot} has {slot_tokens.shape[0]} tokens; "
                f"needs {model.num_value_bases}"
            )
        prototypes, _, value_counts = _deterministic_kmeans(
            slot_tokens,
            model.num_value_bases,
            int(settings["value_iterations"]),
            seed + 2 + slot,
            cosine=False,
        )
        values.append(prototypes)
        slot_reports.append(
            {
                "slot": slot,
                "assigned_tokens": int(slot_tokens.shape[0]),
                "percentage": 100.0 * float(slot_tokens.shape[0]) / z_tokens.shape[0],
                "value_cluster_counts": value_counts.tolist(),
                "six_way_valid": True,
            }
        )
    values_tensor = torch.stack(values)
    if values_tensor.shape != model.V.shape:
        raise AssertionError(
            f"Slot-conditioned V shape {values_tensor.shape} != {model.V.shape}"
        )
    model.K.copy_(keys.to(device=model.K.device, dtype=model.K.dtype))
    model.V.copy_(values_tensor.to(device=model.V.device, dtype=model.V.dtype))
    normalized_keys = F.normalize(keys, dim=-1)
    similarities = q_tokens @ normalized_keys.t()
    probabilities = torch.softmax(similarities / model.temperature, dim=-1)
    usage = probabilities.mean(dim=0)
    usage_entropy = -(usage.clamp_min(1e-12) * usage.clamp_min(1e-12).log()).sum()
    token_entropy = -(
        probabilities.clamp_min(1e-12)
        * probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1).mean()
    between = normalized_keys @ normalized_keys.t()
    between = between[~torch.eye(model.num_slots, dtype=torch.bool)]
    occupancy = slot_counts.float()
    report = {
        "method": "spherical_K_then_slot_conditioned_V_kmeans",
        "split": "train",
        "seed": seed,
        "training_samples": len(sample_keys),
        "paired_tokens": int(z_tokens.shape[0]),
        "key_fit_tokens": fit_count,
        "sample_key_hash": hashlib.sha256("\n".join(sample_keys).encode()).hexdigest(),
        "assignment_hash": _tensor_hash(full_assignment),
        "K_hash": _tensor_hash(model.K),
        "V_hash": _tensor_hash(model.V),
        "K_shape": list(model.K.shape),
        "V_shape": list(model.V.shape),
        "active_slots": int((slot_counts > 0).sum().item()),
        "dead_slots": int((slot_counts == 0).sum().item()),
        "occupancy_min": int(slot_counts.min().item()),
        "occupancy_median": float(occupancy.quantile(0.5).item()),
        "occupancy_max": int(slot_counts.max().item()),
        "top1_histogram": slot_counts.tolist(),
        "routing_entropy": float(token_entropy.item()),
        "effective_prototypes": float(usage_entropy.exp().item()),
        "within_cluster_cosine": float(
            similarities.gather(1, full_assignment.unsqueeze(1)).mean().item()
        ),
        "between_key_cosine_mean": float(between.mean().item()),
        "between_key_cosine_max": float(between.max().item()),
        "slots": slot_reports,
        "memory_content": "clean",
    }
    model.initialization_report = report
    return report
