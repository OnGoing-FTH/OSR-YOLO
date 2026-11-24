#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from feature_library.phase4.osr_inference import OSRInferenceModel
from feature_library.phase4.postprocess import build_known_mask_208, extract_unknown_instances, resize_mask, scale_box
from feature_library.phase4.visualize import save_gray, save_heatmap, save_overlay

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 known-defect and unknown-anomaly inference.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--source", default=None, help="Image file or directory override.")
    parser.add_argument("--output-dir", default=None, help="Output directory override.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum images to process.")
    return parser.parse_args()


def image_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Source does not exist: {source}")
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def safe_result(instance: dict) -> dict:
    return {key: value for key, value in instance.items() if key not in {"mask_832", "mask_208", "mask_original"}}


def write_gallery(rows: list[dict], output: Path) -> None:
    cards = []
    for row in rows:
        cards.append(
            "<article>"
            f"<a href='{row['relative']}/result.json'><img src='{row['relative']}/overlay.png' alt='{row['name']}'></a>"
            f"<p><strong>{row['name']}</strong><br>known={row['known_count']}, unknown={row['unknown_count']}, max-anom={row['max_anomaly']:.4f}</p>"
            "</article>"
        )
    html = """<!doctype html><html><head><meta charset='utf-8'><style>
body{font-family:Arial,sans-serif;margin:24px;background:#fafafa}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}article{background:white;border:1px solid #ddd;padding:10px}img{width:100%;height:auto}p{line-height:1.45;margin:8px 0 0}
</style></head><body><h1>Phase 4 OSR inference gallery</h1><div class='grid'>""" + "".join(cards) + "</div></body></html>"
    output.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source = Path(args.source or config["paths"]["source"]).resolve()
    output_root = Path(args.output_dir or config["paths"]["output_dir"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    paths = image_paths(source)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No supported images at {source}")

    model_cfg = config["model"]
    infer_cfg = config["inference"]
    model = OSRInferenceModel(model_cfg["checkpoint"], model_cfg.get("device", "auto"), model_cfg["input_size"])
    summary = []
    for image_path in paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        original_height, original_width = image.shape[:2]
        known_predictions, proto, anom_probability = model.forward(image)
        known = model.known_instances(
            known_predictions, proto, infer_cfg["known_conf_threshold"], infer_cfg["known_iou_threshold"], infer_cfg["known_max_det"]
        )
        known_mask_208 = build_known_mask_208(known, infer_cfg["known_mask_dilate_px"])
        anom_208 = anom_probability[0, 0].detach().cpu().numpy().astype(np.float32)
        residual, unknown_mask_208, unknown = extract_unknown_instances(
            anom_208, known_mask_208,
            infer_cfg["unknown_prob_threshold"], infer_cfg["unknown_min_area_208"], infer_cfg["unknown_max_area_ratio"],
            infer_cfg["morph_kernel_size"], infer_cfg["morph_open_iterations"], infer_cfg["morph_close_iterations"],
        )
        output_dir = output_root / image_path.stem
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        for instance in known:
            instance["bbox_xyxy_original"] = scale_box(instance["bbox_xyxy_832"], model.input_size, original_width, original_height)
            instance["mask_original"] = resize_mask(instance["mask_832"], original_width, original_height)
            mask_path = masks_dir / f"known_{instance['index']:02d}_class_{instance['class_id']}.png"
            cv2.imwrite(str(mask_path), instance["mask_original"].astype(np.uint8) * 255)
            instance["mask_path"] = str(mask_path.relative_to(output_dir))
        for instance in unknown:
            x1, y1, x2, y2 = instance["bbox_xyxy_208"]
            instance["bbox_xyxy_original"] = scale_box([x1 * 4, y1 * 4, x2 * 4, y2 * 4], model.input_size, original_width, original_height)
            instance["mask_original"] = resize_mask(instance["mask_208"], original_width, original_height)
            mask_path = masks_dir / f"unknown_{instance['index']:02d}.png"
            cv2.imwrite(str(mask_path), instance["mask_original"].astype(np.uint8) * 255)
            instance["mask_path"] = str(mask_path.relative_to(output_dir))

        save_heatmap(output_dir / "anomaly_heatmap.png", anom_208)
        save_heatmap(output_dir / "anomaly_residual.png", residual)
        save_gray(output_dir / "known_mask_208.png", known_mask_208.astype(np.float32))
        save_gray(output_dir / "unknown_mask_208.png", unknown_mask_208.astype(np.float32))
        save_overlay(output_dir / "overlay.png", image, known, unknown)
        result = {
            "image": str(image_path),
            "input_size": model.input_size,
            "original_size_wh": [original_width, original_height],
            "known_instances": [safe_result(item) for item in known],
            "unknown_instances": [safe_result(item) for item in unknown],
            "anomaly": {
                "max_probability": float(anom_208.max()),
                "mean_probability": float(anom_208.mean()),
                "residual_max_probability": float(residual.max()),
            },
            "thresholds": {
                "known_conf_threshold": infer_cfg["known_conf_threshold"],
                "unknown_prob_threshold": infer_cfg["unknown_prob_threshold"],
            },
        }
        (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append({
            "name": image_path.name, "relative": output_dir.relative_to(output_root).as_posix(), "known_count": len(known),
            "unknown_count": len(unknown), "max_anomaly": result["anomaly"]["max_probability"], "result": str((output_dir / "result.json").relative_to(output_root)),
        })
        print(f"{image_path.name}: known={len(known)}, unknown={len(unknown)}, max_anom={anom_208.max():.4f}")

    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "relative", "known_count", "unknown_count", "max_anomaly", "result"])
        writer.writeheader()
        writer.writerows(summary)
    write_gallery(summary, output_root / "gallery.html")
    print(f"Saved {len(summary)} result(s) under: {output_root}")


if __name__ == "__main__":
    main()
