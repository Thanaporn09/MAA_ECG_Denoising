from __future__ import annotations

from pathlib import Path

import torch


def atomic_torch_save(state: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"Checkpoint does not exist: {resolved}")
    return torch.load(resolved, map_location=device)
