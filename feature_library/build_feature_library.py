#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import image_paths, load_config
from src.dataset import ImageDataset, collate_images
from src.io_utils import ensure_dir, write_json
from src.wrn_feature_extractor import WRNSpatialFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1: build the 832-input spatial WRN50 feature bank.")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration path.")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    runtime = config["runtime"]
    model_config = config["model"]
    device = resolve_device(runtime.get("device", "auto"))
    input_size = int(model_config["input_size"])

    normal_paths = image_paths(paths["normal_dir"])
    dataset = ImageDataset(normal_paths, input_size=input_size)
    loader = DataLoader(
        dataset,
        batch_size=int(runtime.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(runtime.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_images,
    )
    extractor = WRNSpatialFeatureExtractor(feature_dim=int(model_config["feature_dim"])).to(device).eval()

    feature_batches: list[torch.Tensor] = []
    for batch_index, batch in enumerate(loader, start=1):
        features = extractor(batch["image"].to(device, non_blocking=True))
        feature_batches.append(features.cpu())
        print(f"[{batch_index}/{len(loader)}] extracted {features.shape[0]} image(s): {features.shape[-2]}x{features.shape[-1]}")

    bank = torch.cat(feature_batches, dim=0)
    expected_map_size = input_size // 8
    if bank.shape[-2:] != (expected_map_size, expected_map_size):
        raise RuntimeError(f"Expected {expected_map_size}x{expected_map_size} features, got {tuple(bank.shape[-2:])}.")

    output_root = ensure_dir(paths["output_dir"])
    bank_dir = ensure_dir(output_root / "bank")
    bank_dtype = getattr(torch, str(model_config.get("bank_dtype", "float16")))
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_size": input_size,
        "feature_map_size": expected_map_size,
        "feature_dim": int(model_config["feature_dim"]),
        "neighbor_radius": int(model_config["spatial_neighbor_radius"]),
        "bank_shape": list(bank.shape),
        "bank_dtype": str(model_config.get("bank_dtype", "float16")),
        "extractor": "torchvision.wide_resnet50_2 / IMAGENET1K_V2 / layer2+layer3",
        "normal_root": str(Path(paths["normal_dir"]).resolve()),
        "normal_images": [str(path.resolve()) for path in normal_paths],
    }
    artifact_path = bank_dir / "bank_spatial.pt"
    torch.save({"bank": bank.to(bank_dtype).contiguous(), "metadata": metadata}, artifact_path)
    write_json(bank_dir / "metadata.json", metadata)
    print(f"Saved {metadata['bank_shape']} spatial bank to: {artifact_path}")


if __name__ == "__main__":
    main()
