from __future__ import annotations

import torch
import torch.nn.functional as F


def spatial_anomaly_map(
    query_features: torch.Tensor,
    bank_features: torch.Tensor,
    neighbor_radius: int = 1,
    bank_chunk_size: int = 4,
    position_chunk_size: int = 1024,
) -> torch.Tensor:
    """Return 1-max cosine similarity using a same-location local spatial neighborhood.

    query_features: [B, C, H, W], L2-normalized
    bank_features: [N, C, H, W], L2-normalized
    result: [B, H, W]
    """
    if query_features.ndim != 4 or bank_features.ndim != 4:
        raise ValueError("Query and bank tensors must use [B/N, C, H, W] shape.")
    if query_features.shape[1:] != bank_features.shape[1:]:
        raise ValueError(f"Feature shapes differ: query={query_features.shape}, bank={bank_features.shape}")

    _, channels, height, width = query_features.shape
    kernel = neighbor_radius * 2 + 1
    query = query_features.float().permute(0, 2, 3, 1).reshape(query_features.shape[0], height * width, channels)
    best_similarity = torch.full(
        (query_features.shape[0], height * width), -1.0, device=query_features.device, dtype=torch.float32
    )

    for start in range(0, bank_features.shape[0], bank_chunk_size):
        bank_chunk = bank_features[start : start + bank_chunk_size].float()
        unfolded = F.unfold(bank_chunk, kernel_size=kernel, padding=neighbor_radius)
        unfolded = unfolded.view(bank_chunk.shape[0], channels, kernel * kernel, height * width)
        for position_start in range(0, height * width, position_chunk_size):
            position_end = min(position_start + position_chunk_size, height * width)
            query_chunk = query[:, position_start:position_end]
            neighborhoods = unfolded[..., position_start:position_end].permute(0, 3, 2, 1)
            similarities = torch.einsum("bpc,npkc->bnpk", query_chunk, neighborhoods)
            best_similarity[:, position_start:position_end] = torch.maximum(
                best_similarity[:, position_start:position_end], similarities.amax(dim=(1, 3))
            )

    return (1.0 - best_similarity).reshape(query_features.shape[0], height, width).clamp_(0.0, 2.0)


def resize_anomaly_map(anomaly_map: torch.Tensor, output_size: int = 208) -> torch.Tensor:
    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map.unsqueeze(1)
    return F.interpolate(anomaly_map, size=(output_size, output_size), mode="bilinear", align_corners=False)
