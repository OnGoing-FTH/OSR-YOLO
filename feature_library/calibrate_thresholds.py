#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.anomaly_scoring import spatial_anomaly_map
from src.config import load_config
from src.io_utils import ensure_dir, load_bank, write_json
from src.visualization import save_threshold_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2.1: leave-one-out threshold calibration for a spatial bank.")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration path.")
    parser.add_argument("--bank", default=None, help="Override path to bank_spatial.pt.")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["runtime"].get("device", "auto"))
    output_root = ensure_dir(config["paths"]["output_dir"])
    bank_path = Path(args.bank) if args.bank else output_root / "bank" / "bank_spatial.pt"
    bank, metadata = load_bank(bank_path, device)
    if bank.shape[0] < 2:
        raise ValueError("Leave-one-out calibration needs at least two normal images.")

    scores_per_image: list[torch.Tensor] = []
    image_summaries: list[dict[str, object]] = []
    radius = int(metadata["neighbor_radius"])
    for index in range(bank.shape[0]):
        query = bank[index : index + 1]
        reference = torch.cat((bank[:index], bank[index + 1 :]), dim=0)
        anomaly = spatial_anomaly_map(query, reference, neighbor_radius=radius)
        scores_per_image.append(anomaly.cpu())
        flat = anomaly.flatten()
        image_summaries.append(
            {
                "index": index,
                "image": metadata["normal_images"][index],
                "min": float(flat.min()),
                "mean": float(flat.mean()),
                "max": float(flat.max()),
                "p99": float(torch.quantile(flat, 0.99)),
            }
        )
        print(f"[{index + 1}/{bank.shape[0]}] max={image_summaries[-1]['max']:.5f} p99={image_summaries[-1]['p99']:.5f}")

    scores = torch.cat(scores_per_image, dim=0).numpy().astype(np.float32)
    calibration = config["calibration"]
    percentile_values = {
        str(percentile): float(np.percentile(scores, percentile))
        for percentile in calibration.get("reported_percentiles", [95.0, 99.0, 99.5, 99.9])
    }
    tau_low_base = float(np.percentile(scores, float(calibration["tau_low_percentile"])))
    tau_low = tau_low_base + float(calibration["tau_low_margin"])
    tau_high = tau_low + float(calibration["tau_high_gap"])

    calibration_dir = ensure_dir(output_root / "calibration")
    np.savez_compressed(calibration_dir / "loo_scores.npz", scores=scores)
    save_threshold_distribution(scores.reshape(-1), tau_low, tau_high, calibration_dir / "threshold_distribution.png")
    result = {
        "bank_path": str(bank_path.resolve()),
        "bank_shape": list(bank.shape),
        "input_size": metadata["input_size"],
        "feature_map_size": metadata["feature_map_size"],
        "anomaly_map_size": metadata["input_size"] // 4,
        "tau_low_percentile": calibration["tau_low_percentile"],
        "tau_low_base": tau_low_base,
        "tau_low_margin": calibration["tau_low_margin"],
        "tau_low": tau_low,
        "tau_high_gap": calibration["tau_high_gap"],
        "tau_high": tau_high,
        "reported_percentiles": percentile_values,
        "normal_image_summaries": image_summaries,
    }
    write_json(calibration_dir / "thresholds.json", result)
    print(f"tau_low={tau_low:.6f}, tau_high={tau_high:.6f}")
    print(f"Saved calibration to: {calibration_dir}")


if __name__ == "__main__":
    main()
