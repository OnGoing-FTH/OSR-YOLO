from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.tal import make_anchors


class OSRSegmentationLoss(v8SegmentationLoss):
    """Ultralytics segmentation loss plus anomaly supervision and cls Ignore masking."""

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.tau_low = float(model.osr_tau_low)
        self.tau_high = float(model.osr_tau_high)
        self.lambda_anom = float(model.osr_lambda_anom)
        self.enable_cls_ignore = bool(model.osr_enable_cls_ignore)
        self.loss_names = ("box_loss", "seg_loss", "cls_loss", "dfl_loss", "anom_bce", "anom_dice", "cls_ignore")

    def _classification_loss(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores_sum: torch.Tensor,
        anomaly_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bce_loss = self.bce(pred_scores, target_scores.to(pred_scores.dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights

        if not self.enable_cls_ignore:
            return bce_loss.sum() / target_scores_sum, pred_scores.new_zeros(())

        prior_208 = anomaly_prior.float()
        priors = [
            F.max_pool2d(prior_208, kernel_size=2, stride=2),
            F.max_pool2d(prior_208, kernel_size=4, stride=4),
            F.max_pool2d(prior_208, kernel_size=8, stride=8),
        ]
        flattened = torch.cat([prior.flatten(1) for prior in priors], dim=1)
        if flattened.shape[1] != pred_scores.shape[1]:
            raise RuntimeError(
                f"A_208 priors ({flattened.shape[1]}) do not match YOLO anchors ({pred_scores.shape[1]})."
            )
        suspicious_negative = (~fg_mask) & (flattened >= self.tau_low)
        cls_weight = (~suspicious_negative).to(bce_loss.dtype).unsqueeze(-1)
        cls_weight = torch.where(fg_mask.unsqueeze(-1), torch.ones_like(cls_weight), cls_weight)
        weighted = (bce_loss * cls_weight).sum() / target_scores_sum
        return weighted, suspicious_negative.float().mean()

    @staticmethod
    def _known_mask(batch: dict[str, torch.Tensor], output_size: tuple[int, int], batch_size: int, device: torch.device) -> torch.Tensor:
        masks = batch["masks"].to(device).float()
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        if masks.shape[-2:] != output_size:
            masks = F.interpolate(masks, size=output_size, mode="nearest")
        if masks.shape[0] == batch_size:  # overlap_mask=True -> one union-ready mask per image
            return (masks > 0).float()
        union = torch.zeros(batch_size, 1, *output_size, device=device)
        batch_idx = batch["batch_idx"].view(-1).long()
        for index in range(batch_size):
            selected = masks[batch_idx == index]
            if selected.numel():
                union[index] = (selected > 0).any(dim=0, keepdim=True).float()
        return union

    def _anomaly_loss(
        self, logits: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        prior = batch["anom_prior"].to(logits.device).float()
        valid_geometry = batch["geom_valid"].to(logits.device).bool()
        if prior.shape[-2:] != logits.shape[-2:]:
            prior = F.interpolate(prior, size=logits.shape[-2:], mode="bilinear", align_corners=False)
            valid_geometry = F.interpolate(valid_geometry.float(), size=logits.shape[-2:], mode="nearest").bool()
        known = self._known_mask(batch, logits.shape[-2:], logits.shape[0], logits.device).bool()
        positive = known | (prior > self.tau_high)
        negative = (~positive) & (prior < self.tau_low)
        valid = (positive | negative) & valid_geometry
        target = positive.float()

        valid_count = valid.float().sum().clamp_min(1.0)
        bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid.float()).sum() / valid_count
        probabilities = torch.sigmoid(logits) * valid.float()
        target_valid = target * valid.float()
        intersection = (probabilities * target_valid).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + target_valid.sum(dim=(1, 2, 3))
        positive_images = target_valid.sum(dim=(1, 2, 3)) > 0
        if positive_images.any():
            dice = (1.0 - (2.0 * intersection[positive_images] + 1e-5) / (denominator[positive_images] + 1e-5)).mean()
        else:
            dice = logits.sum() * 0.0

        stats = {
            "known_ratio": known.float().mean().detach(),
            "teacher_ratio": ((prior > self.tau_high) & ~known).float().mean().detach(),
            "ignore_ratio": ((~positive) & ~negative & valid_geometry).float().mean().detach(),
            "valid_ratio": valid.float().mean().detach(),
        }
        return bce, dice, stats

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        pred_masks = preds["mask_coefficient"].permute(0, 2, 1).contiguous()
        proto = preds["proto"]
        anom_logits = preds["anom_logits"]
        loss = torch.zeros(7, device=self.device)  # box, seg, cls, dfl, anom bce, anom dice, ignore ratio
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = target_scores.sum().clamp_(min=1)
        loss[2], loss[6] = self._classification_loss(pred_scores, target_scores, fg_mask, target_scores_sum, batch["anom_prior"])
        loss[0], loss[3] = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor, target_scores,
            target_scores_sum, fg_mask, imgsz, stride_tensor,
        )
        loss[0] *= self.hyp.box
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl

        _, _, mask_h, mask_w = proto.shape
        if fg_mask.sum():
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                proto_for_seg = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)
            else:
                proto_for_seg = proto
            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch["batch_idx"].view(-1, 1), proto_for_seg,
                pred_masks, imgsz,
            )
        else:
            loss[1] = (proto * 0).sum() + (pred_masks * 0).sum()
        loss[1] *= self.hyp.box

        loss[4], loss[5], anomaly_stats = self._anomaly_loss(anom_logits, batch)
        total = loss[:4].sum() + self.lambda_anom * (loss[4] + loss[5])
        metrics = dict(zip(self.loss_names, loss.detach()))
        metrics.update({f"anom_{key}": value for key, value in anomaly_stats.items()})
        return total * batch_size, metrics
