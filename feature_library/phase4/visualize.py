from __future__ import annotations

import cv2
import numpy as np

KNOWN_COLORS = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
]
UNKNOWN_COLOR = (255, 0, 255)


def save_gray(path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), np.clip(image * 255.0, 0, 255).astype(np.uint8))


def save_heatmap(path, probability: np.ndarray) -> None:
    normalized = np.clip(probability, 0.0, 1.0)
    heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(path), heatmap)


def save_overlay(path, image: np.ndarray, known_instances: list[dict], unknown_instances: list[dict]) -> None:
    overlay = image.copy()
    height, width = image.shape[:2]
    for instance in known_instances:
        color = KNOWN_COLORS[instance["class_id"] % len(KNOWN_COLORS)]
        mask = instance["mask_original"]
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array(color)).astype(np.uint8)
        x1, y1, x2, y2 = instance["bbox_xyxy_original"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{instance['class_name']} {instance['confidence']:.2f}"
        cv2.putText(overlay, label, (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    for instance in unknown_instances:
        color = UNKNOWN_COLOR
        mask = instance["mask_original"]
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array(color)).astype(np.uint8)
        x1, y1, x2, y2 = instance["bbox_xyxy_original"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, f"unknown {instance['score']:.2f}", (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), overlay)
