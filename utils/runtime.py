from __future__ import annotations

import os
import random
from typing import Any, Mapping

import numpy as np
import torch


def configure_runtime(runtime: Mapping[str, Any]) -> None:
    deterministic = bool(runtime["deterministic"])
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = bool(runtime["cudnn_benchmark"])
    torch.backends.cuda.matmul.allow_tf32 = bool(runtime["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(runtime["allow_tf32"])
    threads = int(runtime.get("num_threads", 1))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(max(1, min(threads, 4)))


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(runtime: Mapping[str, Any]) -> torch.device:
    name = str(runtime["device"])
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)
