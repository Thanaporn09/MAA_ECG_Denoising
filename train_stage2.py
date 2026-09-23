from __future__ import annotations

import argparse
import json
from pathlib import Path

from data import datasets_from_config, make_dataloader
from losses import memory_access_alignment_objective
from training import Stage2Trainer, build_stage2_training
from utils import configure_runtime, load_config, resolve_device, save_config, seed_everything


def validate_config(config: dict) -> None:
    if config.get("stage") != 2 or config.get("method") != "memory_access_alignment":
        raise ValueError("Expected Stage 2 Memory Access Alignment configuration")
    if config["backbone"].get("use_skip") is not False:
        raise ValueError("Stage 2 manuscript LUNet requires use_skip=false")
    if config["trainable"] != {
        "student_encoder_stages": ["bottleneck"],
        "student_router": True,
    }:
        raise ValueError("Stage 2 trainable set must be bottleneck and student router")
    expected = {
        "bottleneck_alignment": {"enabled": True, "weight": 1.0},
        "query_alignment": {"enabled": False, "weight": 0.0},
        "route_alignment": {"enabled": False, "weight": 0.0},
        "memory_readout_alignment": {"enabled": True, "weight": 1.0},
        "student_reconstruction": {"enabled": False, "weight": 0.0},
    }
    if config["maa_losses"] != expected:
        raise ValueError("Stage 2 MAA losses differ from manuscript experiment")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2 Memory Access Alignment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--stage1-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--smoke-only", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.out_dir:
        config["out_dir"] = arguments.out_dir
    if arguments.stage1_checkpoint:
        config["stage1_checkpoint"] = arguments.stage1_checkpoint
    validate_config(config)
    configure_runtime(config["runtime"])
    seed_everything(config["seed"])
    device = resolve_device(config["runtime"])
    output_directory = Path(config["out_dir"]).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    save_config(config, output_directory / "resolved_stage2_config.yaml")
    datasets = datasets_from_config(config)
    model, optimizer, scheduler, _ = build_stage2_training(config, device)
    print("TRAINABLE:", flush=True)
    for name in model.trainable_parameter_names():
        print(f"- {name}", flush=True)
    print("FROZEN:", flush=True)
    print("- teacher LUNet", flush=True)
    print("- key-value memory", flush=True)
    print("- teacher memory router", flush=True)
    print("- value mixing head", flush=True)
    print("- LUNet decoder", flush=True)
    print("- E_S.encoder1", flush=True)
    print("- E_S.encoder2", flush=True)
    print("- E_S.encoder3", flush=True)
    trainer = Stage2Trainer(
        model, optimizer, scheduler, config, device, output_directory
    )
    if arguments.resume:
        trainer.resume(arguments.resume)
    audit_loader = make_dataloader(
        datasets["train"], config["loader"], config["seed"], 99, False
    )
    if arguments.smoke_only:
        batch = next(iter(audit_loader))
        x_N = batch["noisy"][:1, :, :64].to(device)
        x_C = batch["clean"][:1, :, :64].to(device)
        total, losses, output = memory_access_alignment_objective(
            model, x_N, x_C, config
        )
        total.backward()
        model.assert_gradient_contract()
        trainer.assert_frozen_unchanged()
        student_only = model.student_pathway(x_N)
        print(
            json.dumps(
                {
                    "loss": float(total.detach()),
                    "losses": {
                        name: float(value.detach()) for name, value in losses.items()
                    },
                    "z_T_shape": list(output.teacher.bottleneck.shape),
                    "z_S_shape": list(output.student.bottleneck.shape),
                    "teacher_memory_readout_shape": list(output.teacher.memory.shape),
                    "student_memory_readout_shape": list(output.student.memory.shape),
                    "student_only_x_hat_C_S_shape": list(
                        student_only.reconstruction.shape
                    ),
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
    print(f"Best Stage 2 checkpoint: {trainer.fit(train_loader, validation_loader)}")


if __name__ == "__main__":
    main()
