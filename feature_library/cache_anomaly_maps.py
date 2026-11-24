#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.anomaly_scoring import resize_anomaly_map, spatial_anomaly_map
from src.config import image_paths, load_config
from src.dataset import ImageDataset, collate_images
from src.io_utils import ensure_dir, load_bank, relative_output_path, save_anomaly_map
from src.visualization import save_anomaly_panel, save_html_report
from src.wrn_feature_extractor import WRNSpatialFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2.2: cache A_208 anomaly maps and visual QA panels.")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration path.")
    parser.add_argument("--source", choices=("train", "normal"), default="train", help="Image source to score.")
    parser.add_argument("--bank", default=None, help="Override path to bank_spatial.pt.")
    parser.add_argument("--thresholds", default=None, help="Override path to thresholds.json.")
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
    output_root = ensure_dir(paths["output_dir"])
    device = resolve_device(runtime.get("device", "auto"))
    source_root = Path(paths[f"{args.source}_dir"])
    source_paths = image_paths(source_root)

    bank_path = Path(args.bank) if args.bank else output_root / "bank" / "bank_spatial.pt"
    threshold_path = Path(args.thresholds) if args.thresholds else output_root / "calibration" / "thresholds.json"
    bank, metadata = load_bank(bank_path, device)
    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
    input_size = int(model_config["input_size"])
    expected_feature_size = input_size // 8
    if metadata["input_size"] != input_size or metadata["feature_map_size"] != expected_feature_size:
        raise ValueError("Bank metadata does not match config.model.input_size; rebuild the bank before caching maps.")

    dataset = ImageDataset(source_paths, input_size=input_size)
    loader = DataLoader(
        dataset,
        batch_size=int(runtime.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(runtime.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_images,
    )
    extractor = WRNSpatialFeatureExtractor(feature_dim=int(model_config["feature_dim"])).to(device).eval()
    output_size = input_size // 4
    maps_root = ensure_dir(output_root / "anomaly_maps" / args.source)
    visuals_root = ensure_dir(output_root / "visualizations" / args.source)
    manifest: list[dict[str, object]] = []
    report_rows: list[dict[str, object]] = []
    visualization_limit = int(config["visualization"].get(f"max_{args.source}_samples", 32))

    for batch_index, batch in enumerate(loader, start=1):
        features = extractor(batch["image"].to(device, non_blocking=True))
        anomaly_104 = spatial_anomaly_map(
            features, bank, neighbor_radius=int(metadata["neighbor_radius"])
        )
        anomaly_208 = resize_anomaly_map(anomaly_104, output_size=output_size)
        for local_index, path_string in enumerate(batch["path"]):
            image_path = Path(path_string)
            relative = relative_output_path(image_path, source_root)
            map_path = maps_root / relative
            save_anomaly_map(map_path, anomaly_208[local_index, 0])
            map_values = anomaly_208[local_index, 0]
            high_ratio = float((map_values >= thresholds["tau_high"]).float().mean())
            entry = {
                "image": str(image_path.resolve()),
                "relative_image": str(image_path.resolve().relative_to(source_root.resolve())),
                "anomaly_map": str(map_path.resolve()),
                "shape": [output_size, output_size],
                "min_score": float(map_values.min()),
                "mean_score": float(map_values.mean()),
                "max_score": float(map_values.max()),
                "high_ratio": high_ratio,
            }
            manifest.append(entry)
            if len(report_rows) < visualization_limit:
                visualization_path = visuals_root / relative.with_suffix(".png")
                save_anomaly_panel(
                    batch["image"][local_index],
                    anomaly_104[local_index],
                    anomaly_208[local_index],
                    thresholds["tau_low"],
                    thresholds["tau_high"],
                    visualization_path,
                    title=f"{entry['relative_image']} | max={entry['max_score']:.4f}",
                    overlay_alpha=float(config["visualization"].get("overlay_alpha", 0.45)),
                )
                report_rows.append({"name": entry["relative_image"], "visualization": visualization_path, "max_score": entry["max_score"], "high_ratio": high_ratio})
        print(f"[{batch_index}/{len(loader)}] cached {len(batch['path'])} A_{output_size} map(s)")

    (maps_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    save_html_report(report_rows, visuals_root / "index.html", f"{args.source.title()} anomaly-map QA")
    alert_ratio = float(config["visualization"].get("normal_alert_ratio", 0.001))
    if args.source == "normal":
        alerted = [entry for entry in manifest if entry["high_ratio"] > alert_ratio]
        if alerted:
            print(f"WARNING: {len(alerted)}/{len(manifest)} normal image(s) exceed high-score ratio {alert_ratio:.4%}.")
    print(f"Saved {len(manifest)} A_{output_size}.npy files under: {maps_root}")


if __name__ == "__main__":
    main()
