# Data preparation and experiment pipeline

This document summarizes the workflow used to prepare custom multi-camera data for COLMAP and DUSt3R experiments. Paths are intentionally generic; the original restricted dataset is not part of this repository.

## 1. Expected input layout

The helper scripts assume six synchronized camera folders:

```text
data/
├── cam0/
├── cam1/
├── cam2/
├── cam3/
├── cam4/
└── cam5/
```

Input file names are expected to contain a shared frame identifier, for example:

```text
cam0_frame_0000414.jpg
cam1_frame_0000414.jpg
```

## 2. Prepare synchronized rig images

```bash
python scripts/rig_prepare.py \
  --src_root data \
  --dst_root work/scene/images/rig0
```

Only frame IDs available in all six camera folders are kept.

## 3. Convert camera calibration

```bash
python scripts/rig_config.py \
  --yaml_path calibration.yaml \
  --out_path work/rig_config.json \
  --rig_name rig0
```

The calibration YAML is converted to the JSON format used by COLMAP's rig configurator.

## 4. Run COLMAP

A typical sequence is:

```bash
colmap feature_extractor \
  --database_path work/scene/database.db \
  --image_path work/scene/images \
  --ImageReader.single_camera_per_folder 1

colmap rig_configurator \
  --database_path work/scene/database.db \
  --rig_config_path work/rig_config.json

colmap exhaustive_matcher \
  --database_path work/scene/database.db

colmap mapper \
  --database_path work/scene/database.db \
  --image_path work/scene/images \
  --output_path work/scene/sparse
```

Dense stereo processing then produces undistorted images and depth maps. The sparse model should also be exported to text form so that `cameras.txt` and `images.txt` are available.

## 5. Convert COLMAP outputs

```bash
python scripts/colmap_to_samples.py --dataset work/scene
```

The resulting `samples.json` stores, for each usable image:

- image and depth-map paths;
- camera model and intrinsics;
- camera-to-world / world-to-camera transforms.

## 6. Generate and split image pairs

```bash
python scripts/colmap_to_pairs.py --dataset work/scene
python scripts/split_pairs.py --dataset work/scene
```

Two pair categories are created:

- temporal pairs from nearby frames of the same camera;
- cross-camera pairs from neighboring cameras at the same frame ID.

## 7. Inspect and filter pseudo-reference depth

```bash
python scripts/depth_quality_check.py --dataset work/scene
python scripts/filter_pairs_valid.py --dataset work/scene
```

The filter checks how many valid positive depth values remain after resizing to the 224 px training scale. Pairs with insufficient valid depth in either view are rejected.

## 8. Build 3D target maps

```bash
python scripts/dataset_with_targets.py \
  --pairs_json work/scene/pairs_train_filtered.json
```

Depth values are back-projected using the camera intrinsics to produce per-pixel 3D point maps. The second view is transformed into the first camera's coordinate frame for paired training targets.

## 9. DUSt3R fine-tuning

Fine-tuning was performed in a modified local checkout of the official DUSt3R repository rather than in this standalone code package.

The final experiment:

- started from `DUSt3R_ViTLarge_BaseDecoder_224_linear`;
- froze the encoder, decoder and all other parameters;
- trained only `downstream_head1` and `downstream_head2`;
- used 1121 training pairs and 59 validation pairs;
- used batch size 1 with gradient accumulation 16;
- used learning rate `3e-6` and minimum learning rate `1e-8`;
- evaluated the best checkpoint according to validation loss.

The aggregate final metrics are stored in `results/metrics_summary.csv`.
