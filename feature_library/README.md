# 832 Spatial Feature Library

This directory implements Phase 1 and Phase 2 of the fixed-station spatial feature-bank design.

- Input image size: `832 x 832`
- WRN50 `layer2`: `104 x 104`
- WRN50 `layer3`: `52 x 52`, bilinearly resized to `104 x 104`
- Spatial bank: `[N_ok, 256, 104, 104]`, FP16 on disk
- Cached training supervision: `A_208.npy`, FP16 on disk
- Matching: each query feature compares to the same spatial location plus a `3 x 3` bank neighborhood

No pseudo boxes, detection priors, YOLO head changes, or training losses are created here.

## Configuration

`config.yaml` is the active configuration. The important inputs are:

```yaml
paths:
  normal_dir: ../Normal_images
  train_dir: ../train_images
  output_dir: ./outputs
```

All images must represent the same aligned fixed station or SKU. `train_dir` can contain nested folders; output maps preserve the relative folder structure.

## Commands

Run commands from this directory with the existing `py11` environment:

```bash
source /home/fth/miniconda3/bin/activate
conda activate py11
cd /home/fth/OSR-YOLO/feature_library

# Phase 1: ImageNet WRN50 features -> Bank_Spatial
python build_feature_library.py --config config.yaml

# Phase 2.1: leave-one-out normal scoring -> tau_low/tau_high
python calibrate_thresholds.py --config config.yaml

# QA normal images. Uses leave-one-out, so each normal image cannot match itself.
python visualize_results.py --config config.yaml --source normal

# Phase 2.2: score the training image set -> A_208.npy + visual panels
python cache_anomaly_maps.py --config config.yaml --source train
```

## Output Layout

```text
outputs/
├── bank/
│   ├── bank_spatial.pt
│   └── metadata.json
├── calibration/
│   ├── thresholds.json
│   ├── loo_scores.npz
│   └── threshold_distribution.png
├── anomaly_maps/train/
│   ├── manifest.json
│   └── <same-relative-path-as-image>.npy
└── visualizations/
    ├── normal_recomputed/index.html
    └── train/index.html
```

`thresholds.json` is authoritative for `tau_low` and `tau_high`. Visual panels show the input, `A_104`, `A_208`, continuous heatmap overlay, and threshold regions. Yellow means `score >= tau_low`; red means `score >= tau_high`.

## Data-loader Integration Later

Load the cached map named in `outputs/anomaly_maps/train/manifest.json` as a single-channel FP16 array, convert it to float, and apply the same geometric transforms as the image and segmentation masks. Do not cache YOLO-grid priors: create them dynamically from transformed `A_208` in the future loss implementation.

## Caveats

- The initial 20 normal images are enough to prove the pipeline but are a narrow normal-distribution estimate. Validate thresholds against separate production normal batches before enabling alarms.
- Thresholds are tuned at the pixel level. Use the visualization report and downstream connected-component filtering to set final image-level alarm criteria.
- `torchvision` downloads ImageNet WRN50 weights automatically on the first run if they are not already cached.
