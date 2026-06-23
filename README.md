\## KVASIR-Medical-Segmentation



A medical image segmentation project using the Kvasir SEG dataset. The project builds a clean baseline pipeline for polyp segmentation, with dataset preparation, model training, model comparison, and evaluation using Dice and IoU.



The goal is not to create a highly optimized medical system. The goal is to show a reliable computer vision workflow with strong fundamentals: image mask pairing, preprocessing, model baselines, validation metrics, prediction overlays, and reproducible scripts.



\## Motivation and design principles



Medical segmentation projects can be difficult to trust when the dataset pipeline is unclear or when only final predictions are shown without validation metrics.



This repo is built around a simple principle: every step should be easy to inspect.



The project emphasizes:



\- reliable image and mask pairing

\- binary mask preprocessing

\- reproducible train and validation splits

\- baseline model comparison

\- Dice and IoU reporting

\- qualitative prediction overlays

\- lightweight tests for core utilities



\## What the project does



The pipeline trains and compares segmentation models for polyp segmentation on Kvasir SEG.



Current models:



\- UNet

\- DeepLabV3 MobileNet



Current evaluation metrics:



\- Dice score

\- Intersection over Union

\- validation loss



\## Pipeline overview



```text

Kvasir SEG images and masks

&#x20;       |

&#x20;       v

Image mask pair indexing

&#x20;       |

&#x20;       v

Dataset loader and preprocessing

&#x20;       |

&#x20;       v

Train validation split

&#x20;       |

&#x20;       v

UNet and DeepLabV3 training

&#x20;       |

&#x20;       v

Dice and IoU evaluation

&#x20;       |

&#x20;       v

Prediction overlays

```



\## Dataset



This project uses Kvasir SEG, a publicly available gastrointestinal polyp segmentation dataset.



The local dataset is expected under:



```text

data/raw/kvasir\_seg/

```



The generated image mask pair index is saved to:



```text

data/processed/pairs.csv

```



Generated dataset files are intentionally not committed to Git.



\## Dataset preparation



The dataset preparation step scans the raw dataset folder and creates a CSV file that maps each image to its corresponding segmentation mask.



Main output:



```text

data/processed/pairs.csv

```



The pair index contains:



```text

image\_path, mask\_path

```



This makes the training pipeline reproducible and avoids relying on hidden folder assumptions.



\## Models



\### UNet



UNet is used as the classic segmentation baseline. It provides an interpretable encoder decoder architecture and is a strong starting point for binary medical segmentation tasks.



\### DeepLabV3 MobileNet



DeepLabV3 MobileNet is used as a stronger modern baseline. It is lightweight enough for practical experimentation while still giving better segmentation quality in the current run.



\## Current results



The table below shows validation results from the current baseline comparison.



| Model | Validation Dice | Validation IoU | Validation Loss |

| --- | ---: | ---: | ---: |

| UNet | 0.5786 | 0.4639 | 0.8851 |

| DeepLabV3 MobileNet | 0.7002 | 0.5964 | 0.6254 |



In this run, DeepLabV3 MobileNet performed better than the UNet baseline on both Dice and IoU.



These results should be treated as baseline outputs. A stronger version would include longer training, heavier augmentation, threshold tuning, and a dedicated test split.



\## Main generated outputs



The main generated outputs are:



```text

outputs/checkpoints/unet\_best.pt

outputs/checkpoints/deeplabv3\_best.pt

outputs/runs/unet\_day3\_summary.json

outputs/runs/deeplabv3\_day4\_summary.json

outputs/day4/day4\_compare.csv

outputs/day4/day4\_compare.json

outputs/day4/preds/

outputs/day5/results\_table.md

```



These files are generated locally and are not committed by default.



\## Project structure



```text

src/

&#x20; data/          dataset indexing and PyTorch dataset

&#x20; eval/          Dice and IoU metrics

&#x20; infer/         mask overlay utilities

&#x20; models/        UNet and DeepLabV3 wrappers

&#x20; training/      model training loops



scripts/

&#x20; kaggle\_download.py

&#x20; smoke\_dataset.py

&#x20; build\_pairs.py

&#x20; sanity\_grid.py

&#x20; train\_unet.py

&#x20; train\_deeplabv3.py

&#x20; compare\_models.py

&#x20; predict\_samples.py

&#x20; make\_day5\_assets.py



tests/

&#x20; lightweight tests that are safe to run in CI

```



\## Reproducibility



Generated data, checkpoints, and prediction outputs are intentionally not committed:



```text

data/raw/

data/processed/

outputs/

```



The code is organized so local artifacts can be regenerated from scripts when the dataset is available.



Main workflow:



```text

python -m scripts.build\_pairs

python -m scripts.train\_unet

python -m scripts.train\_deeplabv3

python -m scripts.compare\_models

python -m scripts.predict\_samples

```



\## Testing



The repo includes lightweight tests for core utilities such as:



\- overlay generation

\- metric computation

\- dataset pair file format

\- model forward pass shape checks



These tests are designed to run in CI without requiring the full dataset.



\## Limitations



This is a baseline medical segmentation project. The current version does not yet include:



\- full hyperparameter tuning

\- cross validation

\- external test set evaluation

\- advanced augmentation

\- test time augmentation

\- model calibration

\- deployment dashboard

