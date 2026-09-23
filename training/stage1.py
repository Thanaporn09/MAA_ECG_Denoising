from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional

import torch

from losses import reconstruction_objective
from models import CleanMemoryCuration, build_stage1, load_stage1_components
from models.stage1 import split_lunet_state
from utils import atomic_torch_save, load_checkpoint

from .common import autocast_context, evaluate_reconstruction


def stage1_objective(
    model: CleanMemoryCuration,
    x_C: torch.Tensor,
    config: dict,
):
    output = model.teacher_pathway(x_C)
    total, losses = reconstruction_objective(
        output.reconstruction, x_C, config["loss"]
    )
    return total, losses, output


@torch.inference_mode()
def evaluate_stage1(
    model: CleanMemoryCuration,
    loader: Iterable[dict],
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.eval()
    return evaluate_reconstruction(
        lambda _x_N, x_C: model.teacher_pathway(x_C).reconstruction,
        loader,
        device,
        use_amp,
    )


def stage1_checkpoint(
    model: CleanMemoryCuration,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    config: dict,
    epoch: int,
    validation: dict,
    best_rmse: float,
    best_epoch: Optional[int],
) -> dict:
    encoder_state, decoder_state = split_lunet_state(model.backbone)
    return {
        "architecture": "CleanMemoryCuration",
        "stage": 1,
        "method": "clean_memory_curation",
        "teacher_target": "clean",
        "memory_content": "clean",
        "trained_from_scratch": True,
        "E_T": encoder_state,
        "D": decoder_state,
        "R_T": model.R_T.state_dict(),
        "C_T": model.C_T.state_dict(),
        "K": model.K.detach().cpu(),
        "V": model.V.detach().cpu(),
        "epoch": int(epoch),
        "validation_metrics": validation,
        "config": config,
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_validation_reconstruction_rmse": float(best_rmse),
        "best_validation_reconstruction_rmse_scale": "physical",
        "best_epoch": best_epoch,
        "memory_initialization": model.initialization_report,
    }


class Stage1Trainer:
    def __init__(
        self,
        model: CleanMemoryCuration,
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
        expected = {
            id(parameter)
            for parameters in model.trainable_groups().values()
            for parameter in parameters
        }
        actual = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if expected != actual:
            raise AssertionError("Stage 1 optimizer parameters do not match model")

    def train_epoch(self, loader: Iterable[dict]) -> dict:
        self.model.train()
        sums = {}
        samples = 0
        for batch in loader:
            x_C = batch["clean"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.device, self.use_amp):
                total, losses, _ = stage1_objective(self.model, x_C, self.config)
            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float(self.config["training"]["grad_clip"])
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            batch_size = int(x_C.shape[0])
            samples += batch_size
            for name, value in {"total": total, **losses}.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * batch_size
        if samples == 0:
            raise RuntimeError("Stage 1 training loader is empty")
        return {name: value / samples for name, value in sums.items()}

    def resume(self, path: str | Path) -> None:
        checkpoint = load_checkpoint(path, self.device)
        load_stage1_components(self.model, checkpoint, self.device)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint.get("scaler", {}))
        self.best_rmse = float(checkpoint["best_validation_reconstruction_rmse"])
        self.best_epoch = checkpoint.get("best_epoch")
        self.start_epoch = int(checkpoint["epoch"]) + 1

    def fit(self, train_loader, validation_loader) -> Path:
        log_path = self.output_directory / "stage1_training.jsonl"
        epochs = int(self.config["training"]["epochs"])
        for epoch in range(self.start_epoch, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            validation = evaluate_stage1(
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
            checkpoint = stage1_checkpoint(
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
                atomic_torch_save(checkpoint, self.output_directory / "best_stage1.pt")
            print(
                f"epoch={epoch} train_total={train_metrics['total']:.6f} "
                f"val_RMSE={validation_rmse:.6f} best={selected}",
                flush=True,
            )
        return self.output_directory / "best_stage1.pt"


def load_trained_stage1(
    config: dict, checkpoint_path: str | Path, device: torch.device
) -> tuple[CleanMemoryCuration, dict]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_stage1(config).to(device)
    load_stage1_components(model, checkpoint, device)
    model.eval()
    return model, checkpoint
