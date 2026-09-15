# 3D Reconstruction with DUSt3R and COLMAP

Bachelor's thesis project focused on adapting a pretrained **DUSt3R** model to custom multi-camera outdoor imagery. The practical work combines **COLMAP** reconstruction, custom Python data-preparation scripts, pseudo-reference depth maps and limited fine-tuning of DUSt3R output heads.

The project was developed at the Faculty of Information Technology, Brno University of Technology, in 2026.

## Project overview

The goal was to test whether a pretrained image-based 3D reconstruction model could be adapted to a custom urban multi-camera dataset under limited GPU memory.

The implemented pipeline:

1. prepares synchronized images from a six-camera rig;
2. converts camera calibration to a COLMAP rig configuration;
3. reconstructs camera poses and depth maps with COLMAP;
4. converts COLMAP outputs into a dataset representation containing images, depth maps, intrinsics and poses;
5. creates temporal and cross-camera image pairs;
6. filters pairs using depth-map validity checks;
7. converts depth maps into target 3D point maps;
8. fine-tunes only the DUSt3R output heads while keeping the encoder and decoder frozen;
9. evaluates depth predictions against COLMAP pseudo-reference depth maps.

```text
multi-camera images
        |
        v
  rig preparation
        |
        v
     COLMAP
  SfM + stereo
        |
        v
images + poses + depth maps
        |
        v
 sample conversion
        |
        v
 pair generation + filtering
        |
        v
      DUSt3R
  head-only fine-tuning
        |
        v
 depth evaluation
```

## My implementation

The repository contains the custom Python scripts written for the practical part of the thesis:

- `rig_prepare.py` - synchronizes frames across six camera folders and prepares the COLMAP image layout;
- `rig_config.py` - converts multi-camera calibration from YAML to COLMAP rig configuration;
- `colmap_to_samples.py` - combines COLMAP camera poses, intrinsics, images and depth-map paths into a common representation;
- `colmap_to_pairs.py` - generates temporal and neighboring-camera image pairs;
- `split_pairs.py` - creates training, validation and small overfit subsets;
- `depth_quality_check.py` - computes simple depth-map diagnostics and creates visual previews;
- `filter_pairs_valid.py` - removes pairs with insufficient valid depth after resizing to training resolution;
- `dataset_with_targets.py` - converts depth maps to camera-space 3D point maps and transforms paired targets into a shared coordinate frame.

Fine-tuning itself was performed in a modified local checkout of the official DUSt3R implementation. The upstream-derived training source is not redistributed in this repository; the relevant training strategy and parameters are documented below and in [`docs/pipeline.md`](docs/pipeline.md).

## Fine-tuning strategy

Full-model fine-tuning was first tested but was not practical on the available 8 GB GPU because of memory requirements. The final experiments therefore froze the encoder, decoder and remaining model parameters and optimized only:

- `downstream_head1`
- `downstream_head2`

The final configuration used 224 px input resolution, batch size 1, gradient accumulation 16, learning rate `3e-6`, minimum learning rate `1e-8` and one warm-up epoch.

The final dataset contained **1121 training pairs** and **59 validation pairs** after additional quality filtering.

## Results

Evaluation was performed against **COLMAP-derived pseudo-reference depth maps**, not measured ground-truth depth. The best head-only variant was trained for 24 epochs.

| Model | Epochs | MAE ↓ | RMSE ↓ | AbsRel ↓ | δ1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretrained DUSt3R | 0 | 3.7889 | 6.3441 | 0.4300 | 0.4326 |
| Output heads only | 16 | 3.6185 | 6.2122 | 0.4086 | 0.4642 |
| Output heads only | 20 | 3.5910 | 6.1891 | 0.4054 | 0.4695 |
| **Output heads only** | **24** | **3.5793** | **6.1773** | **0.4043** | **0.4714** |

Compared with the pretrained model, the 24-epoch variant reduced MAE by approximately **5.53%**, RMSE by **2.63%** and AbsRel by **5.96%**, while δ1 increased by approximately **8.95%**. The fine-tuned model was better on 53-55 of the 59 validation pairs depending on the metric.

The aggregate values are available in [`results/metrics_summary.csv`](results/metrics_summary.csv).

![Final training and validation loss](results/final_training_loss.png)

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── docs/
│   └── pipeline.md
├── results/
│   ├── final_training_loss.png
│   └── metrics_summary.csv
└── scripts/
    ├── rig_prepare.py
    ├── rig_config.py
    ├── colmap_to_samples.py
    ├── colmap_to_pairs.py
    ├── split_pairs.py
    ├── depth_quality_check.py
    ├── filter_pairs_valid.py
    └── dataset_with_targets.py
```

## Requirements

The standalone data-preparation scripts use a small Python dependency set:

```bash
pip install -r requirements.txt
```

COLMAP must be installed separately for reconstruction and depth-map generation. DUSt3R must also be installed from its official repository for model inference or fine-tuning.

## Data availability

The original multi-camera image dataset, camera calibration files, raw COLMAP reconstruction, pair metadata and model checkpoints are not included. The source imagery is subject to access restrictions and is not redistributed publicly.

The repository therefore contains the reusable processing code and aggregate experiment results, but not the private source data required to reproduce the exact thesis experiment end-to-end.

## Thesis

**Tatiana Fedorova - 3D Reconstruction from Images with Deep Neural Networks**  
Brno University of Technology, Faculty of Information Technology, 2026

Official university record: https://dspace.vut.cz/items/65252e16-74cb-4ff7-8487-ef00624d8dc0

## External software

DUSt3R is developed by NAVER and distributed under the **CC BY-NC-SA 4.0** license. This repository does not contain the DUSt3R source code or pretrained checkpoints.

Official DUSt3R repository: https://github.com/naver/dust3r

COLMAP is used as the classical structure-from-motion and multi-view stereo component of the pipeline.
