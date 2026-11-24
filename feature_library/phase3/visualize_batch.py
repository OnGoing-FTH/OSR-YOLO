#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from feature_library.phase3.osr_loss import OSRSegmentationLoss
from feature_library.phase3.osr_trainer import OSRSegmentationTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Phase 3 input/GT/A_208 alignment panel.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--output", default="feature_library/phase3/runs/batch_alignment.png")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    overrides = {
        "model": config["model"]["weights"], "data": config["data"]["yaml"], "imgsz": config["model"]["imgsz"],
        "task": "segment", **config["train"], **config["augment"], "epochs": 1, "batch": 2, "workers": 0, "plots": False, "val": False,
    }
    trainer = OSRSegmentationTrainer(
        overrides=overrides,
        anomaly_manifest=config["data"]["anomaly_manifest"],
        tau_low=config["osr"]["tau_low"], tau_high=config["osr"]["tau_high"],
        lambda_anom=config["osr"]["lambda_anom"], enable_cls_ignore=config["osr"]["enable_cls_ignore"],
    )
    trainer._setup_train()
    batch = trainer.preprocess_batch(next(iter(trainer.train_loader)))
    criterion = trainer.model.init_criterion()
    image = batch["img"][0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
    prior = batch["anom_prior"][0, 0]
    known = criterion._known_mask(batch, prior.shape, batch["img"].shape[0], trainer.device)[0, 0].bool()
    positive = known | (prior > criterion.tau_high)
    negative = (~positive) & (prior < criterion.tau_low)
    ignore = ~(positive | negative)

    panels = [
        (image, "Input (832x832)", None),
        (known.float().cpu(), "Known GT union", "gray"),
        (prior.detach().cpu(), "A_208", "turbo"),
        (ignore.float().cpu(), "Ignore region", "magma"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for axis, (data, title, cmap) in zip(axes, panels):
        axis.imshow(data, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print({"output": str(output.resolve()), "image": batch["im_file"][0], "tau_low": criterion.tau_low, "tau_high": criterion.tau_high})


if __name__ == "__main__":
    main()
