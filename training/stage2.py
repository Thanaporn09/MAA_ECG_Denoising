from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional

import torch

from losses import memory_access_alignment_objective
from models import MemoryAccessAlignment, load_stage2_components
from utils import atomic_torch_save, load_checkpoint

from .common import autocast_context, build_stage2_optimizer, evaluate_reconstruction
from .stage1 import load_trained_stage1


@torch.inference_mode()
def evaluate_stage2(
    model: MemoryAccessAlignment,
    loader: Iterable[dict],
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.eval()
    return evaluate_reconstruction(
        lambda x_N, _x_C: model.student_pathway(x_N).reconstruction,
        loader,
        device,
        use_amp,
    )


def stage2_checkpoint(
    model: MemoryAccessAlignment,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    config: dict,
    epoch: int,
    validation: dict,
    best_rmse: float,
    best_epoch: Optional[int],
) -> dict:
    return {
        "architecture": "MemoryAccessAlignment",
        "stage": 2,
        "method": "memory_access_alignment",
        "student_target": "clean",
        "teacher_checkpoint": str(
            Path(config["stage1_checkpoint"]).expanduser().resolve()
        ),
        "E_S": {
            name: getattr(model.E_S, name).state_dict()
            for name in ("encoder1", "encoder2", "encoder3", "bottleneck")
        },
        "R_S": model.R_S.state_dict(),
        "trainable_groups": list(model.trainable_groups()),
        "trainable_parameter_names": model.trainable_parameter_names(),
        "epoch": int(epoch),
        "validation_metrics": validation,
        "config": config,
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_validation_reconstruction_rmse": float(best_rmse),
        "best_validation_reconstruction_rmse_scale": "physical",
        "best_epoch": best_epoch,
    }


class Stage2Trainer:
    def __init__(
        self,
        model: MemoryAccessAlignment,
        optimizer: torch.optim.Optimizer,
        scheduler,
        config: dict,
        device: torch.device,
        output_directory: Path,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.output_directory = output_directory
        self.best_rmse = math.inf
        self.best_epoch = None
        self.start_epoch = 1
        self.use_amp = bool(config["training"]["amp"] and device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.frozen_state = model.frozen_state()
        intended = {
            id(parameter)
            for parameters in model.trainable_groups().values()
            for parameter in parameters
        }
        optimized = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if intended != optimized:
            raise AssertionError("Stage 2 optimizer parameters do not match MAA model")

    def assert_frozen_unchanged(self) -> None:
        current = self.model.frozen_state()
        changed = [
            name
            for name, source in self.frozen_state.items()
            if name not in current or not torch.equal(current[name], source)
        ]
        if changed:
            raise AssertionError(f"Frozen Stage 2 state changed: {changed}")

    def train_epoch(self, loader: Iterable[dict]) -> dict:
        self.model.train()
        sums = {}
        samples = 0
        for batch in loader:
            x_N = batch["noisy"].to(self.device, non_blocking=True)
            x_C = batch["clean"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.device, self.use_amp):
                total, losses, _ = memory_access_alignment_objective(
                    self.model, x_N, x_C, self.config
                )
            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in self.model.parameters()
                    if parameter.requires_grad
                ],
                float(self.config["training"]["grad_clip"]),
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            batch_size = int(x_N.shape[0])
            samples += batch_size
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * batch_size
        if samples == 0:
            raise RuntimeError("Stage 2 training loader is empty")
        self.assert_frozen_unchanged()
        return {name: value / samples for name, value in sums.items()}

    def resume(self, path: str | Path) -> None:
        checkpoint = load_checkpoint(path, self.device)
        load_stage2_components(self.model, checkpoint)
        self.assert_frozen_unchanged()
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint.get("scaler", {}))
        self.best_rmse = float(checkpoint["best_validation_reconstruction_rmse"])
        self.best_epoch = checkpoint.get("best_epoch")
        self.start_epoch = int(checkpoint["epoch"]) + 1

    def fit(self, train_loader, validation_loader) -> Path:
        log_path = self.output_directory / "stage2_training.jsonl"
        epochs = int(self.config["training"]["epochs"])
        for epoch in range(self.start_epoch, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            validation = evaluate_stage2(
                self.model, validation_loader, self.device, self.use_amp
            )
            validation_rmse = float(validation["RMSE"]["mean"])
            selected = validation_rmse < self.best_rmse
            if selected:
                self.best_rmse = validation_rmse
                self.best_epoch = epoch
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation,
                "selected_best": selected,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            if self.scheduler is not None:
                self.scheduler.step()
            checkpoint = stage2_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.config,
                epoch,
                validation,
                self.best_rmse,
                self.best_epoch,
            )
            atomic_torch_save(checkpoint, self.output_directory / "latest.pt")
            if selected:
                atomic_torch_save(checkpoint, self.output_directory / "best_stage2.pt")
            print(
                f"epoch={epoch} train_total={train_metrics['total_loss']:.6f} "
                f"val_RMSE={validation_rmse:.6f} best={selected}",
                flush=True,
            )
        return self.output_directory / "best_stage2.pt"


def build_stage2_training(config: dict, device: torch.device):
    stage1, stage1_checkpoint = load_trained_stage1(
        config, config["stage1_checkpoint"], device
    )
    for parameter in stage1.parameters():
        parameter.requires_grad_(False)
    stage1.eval()
    model = MemoryAccessAlignment(stage1).to(device)
    model.assert_gradient_contract()
    optimizer = build_stage2_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["scheduler"]["T_max"]),
        eta_min=float(config["scheduler"]["eta_min"]),
    )
    return model, optimizer, scheduler, stage1_checkpoint


def load_trained_stage2(
    config: dict, checkpoint_path: str | Path, device: torch.device
) -> tuple[MemoryAccessAlignment, dict]:
    stage1, _ = load_trained_stage1(config, config["stage1_checkpoint"], device)
    for parameter in stage1.parameters():
        parameter.requires_grad_(False)
    stage1.eval()
    model = MemoryAccessAlignment(stage1).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    frozen_state = model.frozen_state()
    load_stage2_components(model, checkpoint)
    current_frozen_state = model.frozen_state()
    changed = [
        name
        for name, source in frozen_state.items()
        if name not in current_frozen_state
        or not torch.equal(current_frozen_state[name], source)
    ]
    if changed:
        raise RuntimeError(f"Stage 2 checkpoint changes frozen state: {changed}")
    model.eval()
    return model, checkpoint
