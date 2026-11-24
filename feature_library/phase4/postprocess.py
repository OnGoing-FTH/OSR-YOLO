from __future__ import annotations

import cv2
import numpy as np


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def scale_box(box: list[float], source_size: int, original_width: int, original_height: int) -> list[int]:
    scale_x = original_width / source_size
    scale_y = original_height / source_size
    x1, y1, x2, y2 = box
    return [
        int(round(max(0, min(original_width, x1 * scale_x)))),
        int(round(max(0, min(original_height, y1 * scale_y)))),
        int(round(max(0, min(original_width, x2 * scale_x)))),
        int(round(max(0, min(original_height, y2 * scale_y)))),
    ]


def build_known_mask_208(known_instances: list[dict], dilation_px: int) -> np.ndarray:
    known_mask = np.zeros((208, 208), dtype=np.uint8)
    for instance in known_instances:
        known_mask |= cv2.resize(instance["mask_832"], (208, 208), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    if dilation_px > 0 and known_mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
        known_mask = cv2.dilate(known_mask, kernel, iterations=1)
    return known_mask.astype(bool)


def extract_unknown_instances(
    anomaly_probability: np.ndarray,
    known_mask_208: np.ndarray,
    threshold: float,
    min_area: int,
    max_area_ratio: float,
    morph_kernel_size: int,
    open_iterations: int,
    close_iterations: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    residual = anomaly_probability.astype(np.float32).copy()
    residual[known_mask_208] = 0.0
    binary = (residual >= threshold).astype(np.uint8)
    if morph_kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        if open_iterations:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
        if close_iterations:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    unknown_mask = np.zeros_like(binary)
    unknowns = []
    max_area = int(max_area_ratio * binary.size)
    for label in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[label]]
        if area < min_area or area > max_area:
            continue
        component = labels == label
        unknown_mask[component] = 1
        values = residual[component]
        unknowns.append(
            {
                "index": len(unknowns),
                "type": "unknown_anomaly",
                "class_id": None,
                "class_name": "unknown_anomaly",
                "score": float(values.max()),
                "peak_score": float(values.max()),
                "mean_score": float(values.mean()),
                "bbox_xyxy_208": [x, y, x + width, y + height],
                "area_208": int(area),
                "mask_208": component.astype(np.uint8),
            }
        )
    return residual, unknown_mask.astype(bool), unknowns
