#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics.utils import IterableSimpleNamespace

from feature_library.phase3.osr_dataset import OSRSegDataset
from feature_library.phase3.osr_trainer import OSRSegmentationTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real Phase 3 batch through YOLO11s-seg OSR loss.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_config, data_config, osr, train, augment = (
        config["model"], config["data"], config["osr"], config["train"], config["augment"]
    )
    overrides = {"model": model_config["weights"], "data": data_config["yaml"], "imgsz": model_config["imgsz"], "task": "segment", **train, **augment, "epochs": 1, "batch": 2, "workers": 0, "plots": False, "val": False}
    trainer = OSRSegmentationTrainer(
        overrides=overrides,
        anomaly_manifest=data_config["anomaly_manifest"],
        tau_low=osr["tau_low"],
        tau_high=osr["tau_high"],
        lambda_anom=osr["lambda_anom"],
        enable_cls_ignore=osr["enable_cls_ignore"],
    )
    trainer._setup_train()
    batch = next(iter(trainer.train_loader))
    batch = trainer.preprocess_batch(batch)
    trainer.model.train()
    predictions = trainer.model(batch["img"])
    loss, items = trainer.model.loss(batch, predictions)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite Phase 3 loss: {loss}")
    loss.backward()

    head = trainer.model.model[-1]
    grad = head.anom_head[-1].weight.grad
    if grad is None or not torch.isfinite(grad).all():
        raise RuntimeError("Anomaly head did not receive finite gradients.")
    result = {
        "img": list(batch["img"].shape),
        "masks": list(batch["masks"].shape),
        "anom_prior": list(batch["anom_prior"].shape),
        "geom_valid": list(batch["geom_valid"].shape),
        "pred_anom_logits": list(predictions["anom_logits"].shape),
        "loss": float(loss.detach()),
        "loss_items": {key: float(value) for key, value in items.items()},
        "anom_grad_norm": float(grad.norm()),
    }
    print(result)


if __name__ == "__main__":
    main()
