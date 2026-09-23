from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import datasets_from_config, make_dataloader
from models import build_stage1
from training import Stage1Trainer, stage1_objective
from utils import (
    configure_runtime,
    initialize_memory,
    load_config,
    resolve_device,
    save_config,
    seed_everything,
)


def validate_config(config: dict) -> None:
    if config.get("stage") != 1 or config.get("method") != "clean_memory_curation":
        raise ValueError("Expected Stage 1 clean-memory curation configuration")
    if config["backbone"].get("use_skip") is not False:
        raise ValueError("Stage 1 manuscript LUNet requires use_skip=false")
    if config["memory"] != {
        "num_slots": 48,
        "top_k": 8,
        "key_dim": 32,
        "value_dim": 64,
        "num_value_bases": 6,
        "router_window_size": 13,
        "temperature": 0.1,
    }:
        raise ValueError("Stage 1 memory geometry differs from manuscript experiment")
    if config["loss"] != {"lambda_l1": 1.0, "lambda_spec": 0.1, "eps": 1e-8}:
        raise ValueError("Stage 1 reconstruction loss differs from manuscript experiment")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 1 clean-memory curation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--resume")
    parser.add_argument("--smoke-only", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.out_dir:
        config["out_dir"] = arguments.out_dir
    validate_config(config)
    configure_runtime(config["runtime"])
    seed_everything(config["seed"])
    device = resolve_device(config["runtime"])
    output_directory = Path(config["out_dir"]).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    save_config(config, output_directory / "resolved_stage1_config.yaml")
    datasets = datasets_from_config(config)
    audits = {
        name: dataset.audit(config["dataset"].get("pair_audit_samples"))
        for name, dataset in datasets.items()
    }
    with (output_directory / "dataset_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audits, stream, indent=2, sort_keys=True)
    model = build_stage1(config).to(device)
    model.assert_gradient_contract()
    print("TRAINABLE:", flush=True)
    for name in model.trainable_groups():
        print(f"- {name}", flush=True)
    initialization_loader = make_dataloader(
        datasets["train"], config["loader"], config["seed"], 99, False
    )
    if not arguments.resume and not arguments.smoke_only:
        report = initialize_memory(
            model, initialization_loader, config["memory_initialization"], device
        )
        with (output_directory / "memory_initialization.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameters in model.trainable_groups().values()
            for parameter in parameters
        ],
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["scheduler"]["T_max"]),
        eta_min=float(config["scheduler"]["eta_min"]),
    )
    trainer = Stage1Trainer(
        model, optimizer, scheduler, config, device, output_directory
    )
    if arguments.resume:
        trainer.resume(arguments.resume)
    if arguments.smoke_only:
        batch = next(iter(initialization_loader))
        x_C = batch["clean"][:1, :, :64].to(device)
        total, losses, output = stage1_objective(model, x_C, config)
        total.backward()
        model.assert_gradient_contract()
        print(
            json.dumps(
                {
                    "loss": float(total.detach()),
                    "losses": {
                        name: float(value.detach()) for name, value in losses.items()
                    },
                    "z_T_shape": list(output.target_bottleneck.shape),
                    "memory_readout_shape": list(output.memory.shape),
                    "x_hat_C_T_shape": list(output.reconstruction.shape),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    train_loader = make_dataloader(
        datasets["train"], config["loader"], config["seed"], 0, True
    )
    validation_loader = make_dataloader(
        datasets["val"], config["loader"], config["seed"], 1, False
    )
    print(f"Best Stage 1 checkpoint: {trainer.fit(train_loader, validation_loader)}")


if __name__ == "__main__":
    main()
