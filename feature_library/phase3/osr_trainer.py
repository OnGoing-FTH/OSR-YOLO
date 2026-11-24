from __future__ import annotations

from copy import copy
from pathlib import Path

from ultralytics.models import yolo
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from .osr_dataset import OSRSegDataset
from .osr_loss import OSRSegmentationLoss
from .osr_model import replace_segment_head


class OSRSegmentationModel(SegmentationModel):
    """SegmentationModel that initializes the Phase 3 anomaly criterion."""

    def init_criterion(self):
        return OSRSegmentationLoss(self)


class OSRSegmentationTrainer(SegmentationTrainer):
    """Ultralytics trainer with A_208-aligned dataset and OSR segmentation model."""

    def __init__(self, *args, anomaly_manifest: str | Path, tau_low: float, tau_high: float, lambda_anom: float = 1.0, enable_cls_ignore: bool = True, **kwargs):
        self.anomaly_manifest = Path(anomaly_manifest).resolve()
        self.osr_tau_low = float(tau_low)
        self.osr_tau_high = float(tau_high)
        self.osr_lambda_anom = float(lambda_anom)
        self.osr_enable_cls_ignore = bool(enable_cls_ignore)
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return OSRSegDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache,
            single_cls=self.args.single_cls or self.data["nc"] == 1,
            stride=gs,
            pad=0.0,
            prefix=f"{mode}: ",
            task="segment",
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            anomaly_manifest=self.anomaly_manifest if mode == "train" else None,
            require_anomaly_maps=mode == "train",
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = OSRSegmentationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        replace_segment_head(model)
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.osr_tau_low = self.osr_tau_low
        self.model.osr_tau_high = self.osr_tau_high
        self.model.osr_lambda_anom = self.osr_lambda_anom
        self.model.osr_enable_cls_ignore = self.osr_enable_cls_ignore

    def get_validator(self):
        return yolo.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
