from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

# Required for torch.load() to resolve custom checkpoint classes.
from feature_library.phase3.osr_model import OSRSegment  # noqa: F401
from feature_library.phase3.osr_trainer import OSRSegmentationModel  # noqa: F401
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression


class OSRInferenceModel:
    def __init__(self, checkpoint: str | Path, device: str = "auto", input_size: int = 832):
        self.input_size = int(input_size)
        self.device = self._resolve_device(device)
        checkpoint = Path(checkpoint).resolve()
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = artifact.get("ema") or artifact.get("model")
        if self.model is None:
            raise ValueError(f"Checkpoint contains neither EMA nor model: {checkpoint}")
        if not any("anom_head" in key for key in self.model.state_dict()):
            raise ValueError(f"Checkpoint does not include Phase 3 anomaly head: {checkpoint}")
        self.model = self.model.float().to(self.device).eval()
        self.names = {int(key): value for key, value in self.model.names.items()}
        self.checkpoint = checkpoint

    @staticmethod
    def _resolve_device(value: str) -> torch.device:
        if value == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(value)

    def preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(image_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)
        return tensor.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def forward(self, image_bgr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.preprocess(image_bgr)
        predictions = self.model(inputs)
        known, proto = predictions[0]
        anom_logits = predictions[1]["anom_logits"]
        return known, proto, torch.sigmoid(anom_logits)

    def known_instances(
        self,
        known_predictions: torch.Tensor,
        proto: torch.Tensor,
        conf_threshold: float,
        iou_threshold: float,
        max_det: int,
    ) -> list[dict]:
        predictions = non_max_suppression(
            known_predictions,
            conf_thres=float(conf_threshold),
            iou_thres=float(iou_threshold),
            max_det=int(max_det),
            nc=len(self.names),
        )[0]
        if predictions.numel() == 0:
            return []
        masks = ops.process_mask(proto[0], predictions[:, 6:], predictions[:, :4], (self.input_size, self.input_size), upsample=True)
        masks = masks > 0.5
        results = []
        for index, (prediction, mask) in enumerate(zip(predictions, masks)):
            class_id = int(prediction[5].item())
            results.append(
                {
                    "index": index,
                    "class_id": class_id,
                    "class_name": self.names[class_id],
                    "confidence": float(prediction[4].item()),
                    "bbox_xyxy_832": [float(value) for value in prediction[:4].tolist()],
                    "mask_832": mask.detach().cpu().numpy().astype(np.uint8),
                }
            )
        return results
