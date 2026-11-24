# Phase 3: YOLO11s-seg OSR Branch at 832

This extension adds a training-only anomaly branch to the existing YOLO11s-seg model without modifying installed Ultralytics source files.

## Scope

- Base checkpoint: `/home/fth/yolov11segment/yolo11s-seg.pt`
- Input: `832 x 832`
- Existing YOLO11s segmentation head remains responsible for known-class detection and masks.
- A new `OSRSegment.anom_head` reads the Proto intermediate feature map and emits `anom_logits [B, 1, 208, 208]`.
- `A_208.npy` maps come from the manifest generated in Phase 2.
- `cls_loss Ignore` is enabled by default for negative detector locations with pooled `A_208 >= tau_low`.

## Alignment Rule

Phase 2 generated A_208 using direct image stretch to `832 x 832`. Therefore Phase 3 deliberately disables all geometry and uses the same direct stretch for the image and segmentation polygons:

```text
mosaic = mixup = copy_paste = cutmix = 0
random perspective / translate / scale / shear / flips = 0
multi_scale = 0
```

Do not enable geometric augmentation unless `A_208` generation and transformation handling are changed together.

## Loss

Known segmentation losses are retained from Ultralytics. The extension adds:

```text
is_pos = known_gt_mask OR (A_208 > tau_high)
is_neg = NOT is_pos AND (A_208 < tau_low)
valid = is_pos OR is_neg

loss_total = yolo_box + yolo_seg + yolo_cls + yolo_dfl
           + lambda_anom * (masked_BCE + masked_Dice)
```

At the current thresholds, the cached training maps have no `A_208 > tau_high` pixels. This is surfaced as `anom_teacher_ratio=0.0`; only manually labeled defects will be anomaly positives until threshold calibration changes.

## Commands

Run from the repository root:

```bash
source /home/fth/miniconda3/bin/activate
conda activate py11
cd /home/fth/OSR-YOLO

# Required before full training: one real batch, forward, loss, and backward checks.
python -m feature_library.phase3.verify_batch

# Full training after the batch check passes.
python -m feature_library.phase3.train
```

Output checkpoints and metrics are written under `feature_library/phase3/runs/yolo11s_seg_osr_832/`.
