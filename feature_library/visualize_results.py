#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.anomaly_scoring import resize_anomaly_map, spatial_anomaly_map
from src.config import image_paths, load_config
from src.dataset import ImageDataset, collate_images
from src.io_utils import ensure_dir, load_bank
from src.visualization import save_anomaly_panel, save_html_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize normal leave-one-out or train anomaly maps.")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration path.")
    parser.add_argument("--source", choices=("normal", "train"), default="normal")
    parser.add_argument("--limit", type=int, default=None, help="Maximum panels to render.")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = ensure_dir(config["paths"]["output_dir"])
    device = resolve_device(config["runtime"].get("device", "auto"))
    bank, metadata = load_bank(output_root / "bank" / "bank_spatial.pt", device)
    import json
    thresholds = json.loads((output_root / "calibration" / "thresholds.json").read_text(encoding="utf-8"))
    source_root = Path(config["paths"][f"{args.source}_dir"])
    source_paths = image_paths(source_root)
    input_size = int(config["model"]["input_size"])
    dataset = ImageDataset(source_paths, input_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_images)
    visual_root = ensure_dir(output_root / "visualizations" / f"{args.source}_recomputed")
    limit = args.limit or int(config["visualization"].get(f"max_{args.source}_samples", 20))
    rows = []

    for index, batch in enumerate(loader):
        if index >= limit:
            break
        # The feature model itself is loaded lazily after the visualization command begins.
        if index == 0:
            from src.wrn_feature_extractor import WRNSpatialFeatureExtractor
            extractor = WRNSpatialFeatureExtractor(int(config["model"]["feature_dim"])).to(device).eval()
        features = extractor(batch["image"].to(device))
        reference = bank
        if args.source == "normal":
            reference = torch.cat((bank[:index], bank[index + 1 :]), dim=0)
        anomaly_104 = spatial_anomaly_map(features, reference, int(metadata["neighbor_radius"]))
        anomaly_208 = resize_anomaly_map(anomaly_104, input_size // 4)
        image_path = Path(batch["path"][0])
        panel = visual_root / image_path.with_suffix(".png").name
        high_ratio = float((anomaly_208[0, 0] >= thresholds["tau_high"]).float().mean())
        save_anomaly_panel(batch["image"][0], anomaly_104[0], anomaly_208[0], thresholds["tau_low"], thresholds["tau_high"], panel, image_path.name)
        rows.append({"name": image_path.name, "visualization": panel, "max_score": float(anomaly_208.max()), "high_ratio": high_ratio})

    save_html_report(rows, visual_root / "index.html", f"{args.source.title()} anomaly-map QA")
    print(f"Saved {len(rows)} visual panels to: {visual_root}")


if __name__ == "__main__":
    main()
