from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics.data.augment import Compose, Format
from ultralytics.data.dataset import YOLODataset


class OSRSegDataset(YOLODataset):
    """YOLO segmentation dataset aligned with precomputed stretched A_208 maps.

    Geometry is intentionally disabled in Phase 3. Images and polygon labels are stretched
    directly to imgsz x imgsz, exactly matching the Phase 2 WRN/A_208 preprocessing.
    """

    def __init__(self, *args, anomaly_manifest: str | Path | None = None, require_anomaly_maps: bool = False, **kwargs):
        self.require_anomaly_maps = require_anomaly_maps
        self.anomaly_manifest_path = Path(anomaly_manifest).resolve() if anomaly_manifest else None
        self.anomaly_maps = self._load_manifest(self.anomaly_manifest_path) if self.anomaly_manifest_path else {}
        super().__init__(*args, **kwargs)
        if self.require_anomaly_maps:
            missing = [path for path in self.im_files if str(Path(path).resolve()) not in self.anomaly_maps]
            if missing:
                preview = "\n".join(missing[:5])
                raise FileNotFoundError(f"Missing A_208 map for {len(missing)} training image(s), e.g.\n{preview}")

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, Path]:
        entries = json.loads(path.read_text(encoding="utf-8"))
        mapping = {str(Path(item["image"]).resolve()): Path(item["anomaly_map"]).resolve() for item in entries}
        missing = [str(value) for value in mapping.values() if not value.is_file()]
        if missing:
            raise FileNotFoundError(f"Manifest references missing A_208 file: {missing[0]}")
        return mapping

    def load_image(self, i: int, rect_mode: bool = True, resize_short: bool = False):
        """Load image with deterministic direct stretch to match cached A_208 coordinates."""
        image_path = self.im_files[i]
        image = cv2.imread(image_path, self.cv2_flag)
        if image is None:
            raise FileNotFoundError(f"Image Not Found {image_path}")
        original_shape = image.shape[:2]
        target = (self.imgsz, self.imgsz) if isinstance(self.imgsz, int) else tuple(self.imgsz)
        if image.shape[:2] != target:
            image = cv2.resize(image, target[::-1], interpolation=cv2.INTER_LINEAR)
        return image, original_shape, image.shape[:2]

    def build_transforms(self, hyp) -> Compose:
        """Do not apply geometry, mosaic, copy-paste or mixup in the aligned first implementation."""
        return Compose(
            [
                Format(
                    bbox_format="xywh",
                    normalize=True,
                    return_mask=True,
                    return_keypoint=False,
                    return_obb=False,
                    batch_idx=True,
                    mask_ratio=hyp.mask_ratio,
                    mask_overlap=hyp.overlap_mask,
                    bgr=0.0,
                )
            ]
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        map_path = self.anomaly_maps.get(str(Path(self.im_files[index]).resolve()))
        if map_path is None:
            if self.require_anomaly_maps:
                raise FileNotFoundError(f"A_208 map missing for {self.im_files[index]}")
            item["anom_prior"] = torch.zeros(1, self.imgsz // 4, self.imgsz // 4, dtype=torch.float32)
        else:
            anomaly = np.load(map_path).astype(np.float32, copy=False)
            expected_shape = (self.imgsz // 4, self.imgsz // 4)
            if anomaly.shape != expected_shape:
                raise ValueError(f"Expected {expected_shape} A_208 map for {self.im_files[index]}, got {anomaly.shape}")
            item["anom_prior"] = torch.from_numpy(np.ascontiguousarray(anomaly)).unsqueeze(0)
        item["geom_valid"] = torch.ones_like(item["anom_prior"])
        item["im_file"] = self.im_files[index]
        return item

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        output = YOLODataset.collate_fn(batch)
        output["anom_prior"] = torch.stack([sample["anom_prior"] for sample in batch], 0)
        output["geom_valid"] = torch.stack([sample["geom_valid"] for sample in batch], 0)
        return output
