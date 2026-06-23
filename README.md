# KVASIR-Medical-Segmentation

A medical image segmentation project using the Kvasir SEG dataset. The project builds a clean baseline pipeline for polyp segmentation, with dataset preparation, model training, model comparison, and evaluation using Dice and IoU.

The goal is not to create a highly optimized medical system. The goal is to show a reliable computer vision workflow with strong fundamentals: image mask pairing, preprocessing, model baselines, validation metrics, prediction overlays, and reproducible scripts.

## Motivation and design principles

Medical segmentation projects can be difficult to trust when the dataset pipeline is unclear or when only final predictions are shown without validation metrics.

This repo is built around a simple principle: every step should be easy to inspect.

The project emphasizes:

- reliable image and mask pairing
- binary mask preprocessing
- reproducible train and validation splits
- baseline model comparison
- Dice and IoU reporting
- qualitative prediction overlays
- lightweight tests for core utilities

## What the project does

The pipeline trains and compares segmentation models for polyp segmentation on Kvasir SEG.

Current models:

- UNet
- DeepLabV3 MobileNet

Current evaluation metrics:

- Dice score
- Intersection over Union
- validation loss

## Pipeline overview

Kvasir SEG images and masks  
to image mask pair indexing  
to dataset loader and preprocessing  
to train validation split  
to UNet and DeepLabV3 training  
to Dice and IoU evaluation  
to prediction overlays

## Dataset

This project uses Kvasir SEG, a publicly available gastrointestinal polyp segmentation dataset.

The local dataset is expected under `data/raw/kvasir_seg/`.

The generated image mask pair index is saved to `data/processed/pairs.csv`.

Generated dataset files are intentionally not committed to Git.

## Dataset preparation

The dataset preparation step scans the raw dataset folder and creates a CSV file that maps each image to its corresponding segmentation mask.

Main output: `data/processed/pairs.csv`

The pair index contains two columns:

| Column | Meaning |
| --- | --- |
| image_path | Path to the input image |
| mask_path | Path to the binary segmentation mask |

This makes the training pipeline reproducible and avoids relying on hidden folder assumptions.

## Models

### UNet

UNet is used as the classic segmentation baseline. It provides an interpretable encoder decoder architecture and is a strong starting point for binary medical segmentation tasks.

### DeepLabV3 MobileNet

DeepLabV3 MobileNet is used as a stronger modern baseline. It is lightweight enough for practical experimentation while still giving better segmentation quality in the current run.

## Current results

The table below shows validation results from the current baseline comparison.

| Model | Validation Dice | Validation IoU | Validation Loss |
| --- | ---: | ---: | ---: |
| UNet | 0.5786 | 0.4639 | 0.8851 |
| DeepLabV3 MobileNet | 0.7002 | 0.5964 | 0.6254 |

In this run, DeepLabV3 MobileNet performed better than the UNet baseline on both Dice and IoU.

These results should be treated as baseline outputs. A stronger version would include longer training, heavier augmentation, threshold tuning, and a dedicated test split.

## Main generated outputs

The main generated outputs are:

| Output | Purpose |
| --- | --- |
| outputs/checkpoints/unet_best.pt | Best UNet checkpoint |
| outputs/checkpoints/deeplabv3_best.pt | Best DeepLabV3 checkpoint |
| outputs/runs/unet_day3_summary.json | UNet training summary |
| outputs/runs/deeplabv3_day4_summary.json | DeepLabV3 training summary |
| outputs/day4/day4_compare.csv | Model comparison table |
| outputs/day4/day4_compare.json | Model comparison JSON |
| outputs/day4/preds/ | Prediction overlay samples |
| outputs/day5/results_table.md | Markdown result table |

These files are generated locally and are not committed by default.

## Project structure

| Folder | Purpose |
| --- | --- |
| src/data/ | Dataset indexing and PyTorch dataset |
| src/eval/ | Dice and IoU metrics |
| src/infer/ | Mask overlay utilities |
| src/models/ | UNet and DeepLabV3 wrappers |
| src/training/ | Model training loops |
| scripts/ | Runnable project commands |
| tests/ | Lightweight tests that are safe to run in CI |

## Main workflow

The local workflow is:

1. Build image and mask pairs with `python -m scripts.build_pairs`
2. Train UNet with `python -m scripts.train_unet`
3. Train DeepLabV3 with `python -m scripts.train_deeplabv3`
4. Compare models with `python -m scripts.compare_models`
5. Generate prediction overlays with `python -m scripts.predict_samples`

## Testing

The repo includes lightweight tests for core utilities such as:

- overlay generation
- metric computation
- dataset pair file format
- model forward pass shape checks

These tests are designed to run in CI without requiring the full dataset.

## Reproducibility

Generated data, checkpoints, and prediction outputs are intentionally not committed:

| Path | Reason |
| --- | --- |
| data/raw/ | Raw dataset files |
| data/processed/ | Generated pair index and processed artifacts |
| outputs/ | Checkpoints, metrics, and predictions |

The code is organized so local artifacts can be regenerated from scripts when the dataset is available.

## Limitations

This is a baseline medical segmentation project. The current version does not yet include:

- full hyperparameter tuning
- cross validation
- external test set evaluation
- advanced augmentation
- test time augmentation
- model calibration
- deployment dashboard
