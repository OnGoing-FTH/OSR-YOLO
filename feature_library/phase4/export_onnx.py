#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from feature_library.phase4.osr_inference import OSRInferenceModel


class OSROnnxWrapper(torch.nn.Module):
    """Exports raw known predictions, mask prototypes, and anomaly logits."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        predictions = self.model(images)
        known_predictions, proto = predictions[0]
        anom_logits = predictions[1]["anom_logits"]
        return known_predictions, proto, anom_logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Phase 4 OSR raw-output ONNX model.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--output", default=None)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = config["model"]
    loaded = OSRInferenceModel(model_cfg["checkpoint"], "cpu", model_cfg["input_size"])
    wrapper = OSROnnxWrapper(loaded.model).cpu().eval()
    input_size = loaded.input_size
    output = Path(args.output or f"feature_library/phase4/exports/{Path(model_cfg['checkpoint']).stem}_osr.onnx").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 3, input_size, input_size)
    torch.onnx.export(
        wrapper,
        example,
        output,
        input_names=["images"],
        output_names=["known_predictions", "mask_prototypes", "anom_logits"],
        dynamic_axes={
            "images": {0: "batch"},
            "known_predictions": {0: "batch"},
            "mask_prototypes": {0: "batch"},
            "anom_logits": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print({"output": str(output), "input": [1, 3, input_size, input_size], "outputs": ["known_predictions", "mask_prototypes", "anom_logits"]})


if __name__ == "__main__":
    main()
