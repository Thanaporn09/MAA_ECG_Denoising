from __future__ import annotations

import argparse
import json
from pathlib import Path

from data import ECGTripletDataset, make_dataloader
from training import evaluate_stage2, load_trained_stage2
from utils import configure_runtime, load_config, resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 Memory Access Alignment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage1-checkpoint")
    parser.add_argument("--test-root")
    parser.add_argument("--source-label", default="test")
    parser.add_argument("--out-json")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.stage1_checkpoint:
        config["stage1_checkpoint"] = arguments.stage1_checkpoint
    configure_runtime(config["runtime"])
    seed_everything(config["seed"])
    device = resolve_device(config["runtime"])
    split = dict(config["dataset"]["splits"]["test"])
    if arguments.test_root:
        split["root"] = arguments.test_root
    dataset = ECGTripletDataset(
        **split,
        cache_signals=config["dataset"]["cache_signals"],
        relation_atol=config["dataset"]["relation_atol"],
        relation_rtol=config["dataset"]["relation_rtol"],
    )
    loader = make_dataloader(dataset, config["loader"], config["seed"], 2, False)
    model, checkpoint = load_trained_stage2(config, arguments.checkpoint, device)
    metrics = evaluate_stage2(
        model,
        loader,
        device,
        bool(config["training"]["amp"] and device.type == "cuda"),
    )
    report = {
        "stage": 2,
        "method": "Memory Access Alignment",
        "source": arguments.source_label,
        "checkpoint": str(Path(arguments.checkpoint).expanduser().resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "samples": len(dataset),
        "inference_input": "noisy ECG only",
        "metrics": metrics,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if arguments.out_json:
        output = Path(arguments.out_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
