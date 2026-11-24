from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    return image.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def save_anomaly_panel(
    image: torch.Tensor,
    anomaly_104: torch.Tensor,
    anomaly_208: torch.Tensor,
    tau_low: float,
    tau_high: float,
    output_path: str | Path,
    title: str,
    overlay_alpha: float = 0.45,
) -> None:
    rgb = tensor_to_rgb(image)
    a104 = anomaly_104.detach().float().cpu().numpy()
    a208 = anomaly_208.detach().float().cpu().squeeze().numpy()
    threshold_map = np.zeros((*a208.shape, 3), dtype=np.float32)
    threshold_map[(a208 >= tau_low) & (a208 < tau_high)] = (1.0, 0.75, 0.0)
    threshold_map[a208 >= tau_high] = (0.9, 0.05, 0.05)

    anomaly_resized = np.asarray(
        Image.fromarray(a208).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR)
    )
    heatmap = plt.get_cmap("turbo")(np.clip(anomaly_resized / max(tau_high * 1.35, 1e-6), 0.0, 1.0))[..., :3]
    overlay = (1.0 - overlay_alpha) * rgb + overlay_alpha * heatmap
    threshold_resized = np.asarray(
        Image.fromarray((threshold_map * 255).astype(np.uint8)).resize(
            (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST
        )
    ).astype(np.float32) / 255.0

    figure, axes = plt.subplots(1, 5, figsize=(22, 5), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("Input (832x832)")
    axes[1].imshow(a104, cmap="turbo", vmin=0.0, vmax=max(tau_high * 1.35, float(a104.max())))
    axes[1].set_title("A_104")
    axes[2].imshow(a208, cmap="turbo", vmin=0.0, vmax=max(tau_high * 1.35, float(a208.max())))
    axes[2].set_title("A_208")
    axes[3].imshow(overlay)
    axes[3].set_title("Anomaly overlay")
    axes[4].imshow(rgb)
    axes[4].imshow(threshold_resized, alpha=0.55)
    axes[4].set_title(f"Yellow >= {tau_low:.3f}; red >= {tau_high:.3f}")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title, fontsize=12)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_threshold_distribution(scores: np.ndarray, tau_low: float, tau_high: float, output_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.hist(scores, bins=180, density=True, color="#3274a1", alpha=0.82)
    axis.axvline(tau_low, color="#e69f00", linewidth=2, label=f"tau_low={tau_low:.4f}")
    axis.axvline(tau_high, color="#d62728", linewidth=2, label=f"tau_high={tau_high:.4f}")
    axis.set_xlabel("Normal leave-one-out anomaly score")
    axis.set_ylabel("Density")
    axis.set_title("Normal-score distribution and calibrated thresholds")
    axis.legend()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_html_report(rows: list[dict[str, object]], output_path: str | Path, title: str) -> None:
    cells = []
    for row in rows:
        image = Path(str(row["visualization"])).name
        cells.append(
            "<article>"
            f"<img src='{image}' alt='{row['name']}'>"
            f"<p><strong>{row['name']}</strong><br>max={row['max_score']:.4f}, high-ratio={row['high_ratio']:.4%}</p>"
            "</article>"
        )
    document = """<!doctype html><html><head><meta charset='utf-8'>
<style>body{font-family:Arial,sans-serif;margin:24px;background:#fafafa} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}article{background:white;border:1px solid #ddd;padding:10px}img{width:100%;height:auto}p{margin:8px 0 0;line-height:1.45}</style>
</head><body>""" + f"<h1>{title}</h1><div class='grid'>" + "".join(cells) + "</div></body></html>"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(document, encoding="utf-8")
