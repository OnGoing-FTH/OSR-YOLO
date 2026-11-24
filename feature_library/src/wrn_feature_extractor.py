from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


class WRNSpatialFeatureExtractor(nn.Module):
    """Frozen ImageNet WRN50 layer2/layer3 extractor for a spatial feature bank."""

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        backbone = wide_resnet50_2(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.feature_dim = feature_dim
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.mean) / self.std
        x = self.stem(images)
        x = self.layer1(x)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)

        layer2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat([layer2, layer3], dim=1)

        # Adaptive channel pooling is parameter-free and preserves spatial coordinates.
        batch, channels, height, width = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, 1, channels)
        features = F.adaptive_avg_pool1d(features, self.feature_dim).reshape(batch, height, width, self.feature_dim)
        features = features.permute(0, 3, 1, 2).contiguous()
        return F.normalize(features, p=2, dim=1, eps=1e-12)
