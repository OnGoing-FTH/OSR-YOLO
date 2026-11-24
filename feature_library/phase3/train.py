#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from feature_library.phase3.osr_trainer import OSRSegmentationTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11s-seg Phase 3 OSR anomaly branch at 832x832.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epoch count.")
    parser.add_argument("--batch", type=int, default=None, help="Override configured batch size.")
    parser.add_argument("--workers", type=int, default=None, help="Override configured data-loader workers.")
    parser.add_argument("--name", default=None, help="Override run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = config["model"]
    data = config["data"]
    osr = config["osr"]
    train = config["train"]
    augment = config["augment"]
    overrides = {
        "model": model["weights"],
        "data": data["yaml"],
        "imgsz": model["imgsz"],
        "task": "segment",
        **train,
        **augment,
    }
    for key in ("epochs", "batch", "workers", "name"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    trainer = OSRSegmentationTrainer(
        overrides=overrides,
        anomaly_manifest=data["anomaly_manifest"],
        tau_low=osr["tau_low"],
        tau_high=osr["tau_high"],
        lambda_anom=osr["lambda_anom"],
        enable_cls_ignore=osr["enable_cls_ignore"],
    )
    trainer.train()


if __name__ == "__main__":
    main()
