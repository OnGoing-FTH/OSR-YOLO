from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_bank(path: str | Path, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if "bank" not in artifact or "metadata" not in artifact:
        raise ValueError(f"Invalid bank artifact: {path}")
    return artifact["bank"].to(device), artifact["metadata"]


def save_anomaly_map(path: str | Path, anomaly_map: torch.Tensor) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    np.save(destination, anomaly_map.detach().float().cpu().numpy().astype(np.float16))


def relative_output_path(image_path: Path, source_root: Path) -> Path:
    try:
        relative = image_path.resolve().relative_to(source_root.resolve())
    except ValueError:
        relative = Path(image_path.name)
    return relative.with_suffix(".npy")
