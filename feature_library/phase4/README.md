# Phase 4: Known-Class Segmentation + Unknown Anomaly Inference

This stage runs one Phase 3 OSR checkpoint and produces two result streams:

- Known instances: standard YOLO11s-seg classes 0-7.
- Unknown instances: thresholded connected components from the anomaly branch after known-instance masks are removed.

## Processing

```text
original BGR image
  -> direct stretch to 832x832
  -> YOLO11s-seg known boxes, classes, masks
  -> AnomalyHead probability map at 208x208
  -> dilate known masks at 208x208 and remove them from anomaly map
  -> threshold + morphology + connected components
  -> unknown_anomaly masks/boxes/scores
  -> map both branches back to original image coordinates
```

The direct stretch is deliberate. It matches the Phase 2 `A_208` generation and Phase 3 training alignment.

## Run

```bash
source /home/fth/miniconda3/bin/activate
conda activate py11
cd /home/fth/OSR-YOLO
python -m feature_library.phase4.infer
```

Examples:

```bash
# Single image
python -m feature_library.phase4.infer --source 8v6_front_test_data/images/val/000000_108.png

# First ten source images
python -m feature_library.phase4.infer --limit 10
```

## Outputs

```text
outputs/
├── <image-stem>/
│   ├── result.json
│   ├── overlay.png
│   ├── anomaly_heatmap.png
│   ├── anomaly_residual.png
│   ├── known_mask_208.png
│   ├── unknown_mask_208.png
│   └── masks/
│       ├── known_00_class_<id>.png
│       └── unknown_00.png
├── summary.json
├── summary.csv
└── gallery.html
```

`unknown_anomaly` is not a ninth trained YOLO category. It is a post-processing result with `class_id: null` and only means an anomalous region not already covered by a high-confidence known instance.

## Current Smoke Checkpoint

The default checkpoint was trained for one smoke epoch only. Its output is valid for pipeline verification, not a quality benchmark. Point `model.checkpoint` in `config.yaml` to the later full-training `best.pt` before production evaluation.
