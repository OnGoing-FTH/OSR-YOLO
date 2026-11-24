from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.head import Segment


class OSRSegment(Segment):
    """Standard Ultralytics Segment head with a training-time anomaly logits branch."""

    def __init__(self, base_head: Segment):
        super().__init__(
            nc=base_head.nc,
            nm=base_head.nm,
            npr=base_head.npr,
            reg_max=base_head.reg_max,
            end2end=getattr(base_head, "end2end", False),
            ch=[module[0].conv.in_channels for module in base_head.cv4],
        )
        self.load_state_dict(base_head.state_dict(), strict=True)
        proto_channels = self.proto.cv2.conv.out_channels
        self.anom_head = nn.Sequential(Conv(proto_channels, 64, k=3), nn.Conv2d(64, 1, kernel_size=1))

    def _proto_and_anomaly(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.proto.cv1(feature)
        feature = self.proto.upsample(feature)
        feature = self.proto.cv2(feature)
        return self.proto.cv3(feature), self.anom_head(feature)

    def forward(self, x: list[torch.Tensor]):
        outputs = super(Segment, self).forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto, anom_logits = self._proto_and_anomaly(x[0])
        if isinstance(preds, dict):
            if "one2one" in preds:
                preds["one2many"]["proto"] = proto
                preds["one2many"]["anom_logits"] = anom_logits
                preds["one2one"]["proto"] = proto.detach()
                preds["one2one"]["anom_logits"] = anom_logits.detach()
            else:
                preds["proto"] = proto
                preds["anom_logits"] = anom_logits
        if self.training:
            return preds
        # Preserve standard Segment output for Ultralytics validators/predictors.
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


def replace_segment_head(model: torch.nn.Module) -> OSRSegment:
    """Replace exactly the final Segment module while retaining pretrained weights."""
    base_head = model.model[-1]
    if not isinstance(base_head, Segment):
        raise TypeError(f"Expected final Segment head, received {type(base_head).__name__}")
    replacement = OSRSegment(base_head)
    replacement.i = base_head.i
    replacement.f = base_head.f
    replacement.type = base_head.type
    replacement.stride = base_head.stride
    replacement.np = sum(parameter.numel() for parameter in replacement.parameters())
    model.model[-1] = replacement
    model.stride = replacement.stride
    return replacement
