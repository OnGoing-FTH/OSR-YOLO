from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    base_dir = config_file.parent
    paths = config.setdefault("paths", {})
    for key in ("normal_dir", "train_dir", "output_dir"):
        if key in paths:
            candidate = Path(paths[key])
            paths[key] = str(candidate if candidate.is_absolute() else (base_dir / candidate).resolve())
    return config


def image_paths(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No supported images found below: {root}")
    return paths
