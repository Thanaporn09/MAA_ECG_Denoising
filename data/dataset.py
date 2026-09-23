from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


_SUFFIXES = ("noisy", "noise", "clean")


def _triplet_stems(root: Path) -> List[str]:
    by_kind = {
        kind: {
            path.name[: -len(f"_{kind}.npy")]
            for path in root.glob(f"*_{kind}.npy")
        }
        for kind in _SUFFIXES
    }
    union = set().union(*by_kind.values())
    for kind, stems in by_kind.items():
        missing = sorted(union - stems)
        if missing:
            raise RuntimeError(
                f"Unpaired files: {len(missing)} samples lack _{kind}.npy; "
                f"examples={missing[:5]}"
            )
    return sorted(union)


class ECGTripletDataset(Dataset):
    def __init__(
        self,
        root: str,
        source: str,
        cache_signals: bool = False,
        relation_atol: float = 1e-5,
        relation_rtol: float = 1e-5,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.source = str(source)
        self.cache_signals = bool(cache_signals)
        self.relation_atol = float(relation_atol)
        self.relation_rtol = float(relation_rtol)
        self._cache: Dict[Path, torch.Tensor] = {}
        if not self.root.is_dir():
            raise RuntimeError(f"Dataset directory does not exist: {self.root}")
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"Required normalization metadata is missing: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as stream:
            entries = json.load(stream)
        metadata_keys = [str(entry["filename"]) for entry in entries if entry.get("filename")]
        duplicate_metadata = sorted(
            key for key, count in Counter(metadata_keys).items() if count > 1
        )
        if duplicate_metadata:
            raise RuntimeError(
                f"Duplicate metadata keys: {len(duplicate_metadata)}; "
                f"examples={duplicate_metadata[:5]}"
            )
        metadata = {
            str(entry["filename"]): entry for entry in entries if entry.get("filename")
        }
        self.samples: List[Dict[str, Any]] = []
        for stem in _triplet_stems(self.root):
            if stem not in metadata:
                raise RuntimeError(f"Missing metadata entry for {stem}")
            entry = metadata[stem]
            if entry.get("normalization") != "instance_noisy_zscore":
                raise RuntimeError(
                    f"Unsupported normalization for {stem}: "
                    f"{entry.get('normalization')!r}"
                )
            parts = stem.split("_")
            if len(parts) < 4:
                raise RuntimeError(f"Cannot parse sample name: {stem}")
            snr = entry.get("snr_db")
            if snr is None:
                raise RuntimeError(f"Missing snr_db metadata for {stem}")
            self.samples.append(
                {
                    "sample_key": stem,
                    "record_id": str(entry.get("record", parts[0])),
                    "segment_id": str(entry.get("segment_start", parts[1])),
                    "lead": str(entry.get("lead_name", parts[2])),
                    "noise_type": str(entry.get("noise_type", parts[3])),
                    "snr": float(snr),
                    "mu": float(entry.get("mu", 0.0)),
                    "sigma": float(entry.get("sigma", 1.0)),
                    **{
                        f"{kind}_path": self.root / f"{stem}_{kind}.npy"
                        for kind in _SUFFIXES
                    },
                }
            )
        if not self.samples:
            raise RuntimeError(f"No ECG triplets found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path) -> torch.Tensor:
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        with path.open("rb") as stream:
            array = np.load(stream, allow_pickle=False).astype(np.float32, copy=False)
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 1:
            raise RuntimeError(f"Expected [T] or [1,T] at {path}, got {array.shape}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"NaN or Inf found in {path}")
        tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
        if self.cache_signals:
            self._cache[path] = tensor
        return tensor

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        noisy = self._load(sample["noisy_path"])
        noise = self._load(sample["noise_path"])
        clean = self._load(sample["clean_path"])
        if noisy.shape != noise.shape or noisy.shape != clean.shape:
            raise RuntimeError(
                f"Triplet shape mismatch for {sample['sample_key']}: "
                f"{noisy.shape}, {noise.shape}, {clean.shape}"
            )
        if not torch.allclose(
            noisy,
            clean + noise,
            atol=self.relation_atol,
            rtol=self.relation_rtol,
        ):
            maximum = float((noisy - clean - noise).abs().max().item())
            raise RuntimeError(
                f"noisy != clean + noise for {sample['sample_key']}; "
                f"max_abs={maximum:.8g}"
            )
        return {
            "noisy": noisy,
            "noise": noise,
            "clean": clean,
            "record_id": sample["record_id"],
            "segment_id": sample["segment_id"],
            "lead": sample["lead"],
            "noise_type": sample["noise_type"],
            "snr": torch.tensor(sample["snr"], dtype=torch.float32),
            "sample_key": sample["sample_key"],
            "mu": torch.tensor(sample["mu"], dtype=torch.float32),
            "sigma": torch.tensor(sample["sigma"], dtype=torch.float32),
        }

    def audit(self, max_samples: Optional[int] = None) -> Dict[str, Any]:
        count = len(self) if max_samples is None else min(len(self), int(max_samples))
        maximum_error = 0.0
        for index in range(count):
            item = self[index]
            error = (item["noisy"] - item["clean"] - item["noise"]).abs().max()
            maximum_error = max(maximum_error, float(error.item()))
        return {
            "root": str(self.root),
            "triplets": len(self),
            "audited_triplets": count,
            "max_abs_noisy_minus_clean_minus_noise": maximum_error,
            "record_count": len({sample["record_id"] for sample in self.samples}),
            "noise_types": sorted({sample["noise_type"] for sample in self.samples}),
        }


def assert_disjoint_sample_keys(named_datasets: Dict[str, ECGTripletDataset]) -> None:
    names = sorted(named_datasets)
    keys = {
        name: {sample["sample_key"] for sample in named_datasets[name].samples}
        for name in names
    }
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = keys[left] & keys[right]
            if overlap:
                raise RuntimeError(
                    f"Split leakage: {left}/{right} share {len(overlap)} keys; "
                    f"examples={sorted(overlap)[:5]}"
                )


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_dataloader(
    dataset: ECGTripletDataset,
    loader_config: dict,
    training_seed: int,
    stream: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(training_seed) + 10_000 * int(stream))
    workers = int(loader_config["num_workers"])
    kwargs = {
        "dataset": dataset,
        "batch_size": int(loader_config["batch_size"]),
        "shuffle": bool(shuffle),
        "num_workers": workers,
        "pin_memory": bool(loader_config["pin_memory"]),
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(loader_config["persistent_workers"])
        kwargs["prefetch_factor"] = int(loader_config["prefetch_factor"])
    return DataLoader(**kwargs)


def datasets_from_config(
    config: dict, split_names: tuple[str, ...] = ("train", "val")
) -> Dict[str, ECGTripletDataset]:
    settings = config["dataset"]
    shared = {
        "cache_signals": settings["cache_signals"],
        "relation_atol": settings["relation_atol"],
        "relation_rtol": settings["relation_rtol"],
    }
    datasets = {
        name: ECGTripletDataset(**settings["splits"][name], **shared)
        for name in split_names
    }
    assert_disjoint_sample_keys(datasets)
    return datasets
