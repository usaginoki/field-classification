# Image Classification Pipeline

A flexible image classification pipeline supporting multiple vision encoders for feature extraction and various ML classifiers for binary classification tasks.

## Features

- **Multiple Feature Extractors**: CLIP, RemoteCLIP, SigLIP2, and Qwen3-VL
- **Configurable Classifiers**: MLP, SVM, XGBoost, Logistic Regression
- **Automatic Caching**: HDF5-based feature caching to avoid recomputation
- **Hyperparameter Search**: Grid search with cross-validation
- **YAML Configuration**: Full pipeline configuration via config file

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd field-classification-reworked

# Install dependencies with uv
uv sync
```

## Quick Start

```bash
# Run with default configuration
uv run python main.py

# Run with sample dataset
uv run python main.py --use-sample

# Force re-extraction of features (skip cache)
uv run python main.py --no-cache

# Use holdout test set for evaluation
uv run python main.py --holdout
```

## Feature Extraction Modes

Configure via `extraction_mode` in `config.yaml`:

| Mode | Config Field | Description |
|------|--------------|-------------|
| `vision_encoder` | `name` | Qwen3-VL vision encoder only (fast) |
| `full` | `name` | Qwen3-VL full model with LLM hidden states |
| `clip` | `clip_model` | OpenAI CLIP models |
| `remote_clip` | `remote_clip_variant` | RemoteCLIP for remote sensing |
| `siglip2` | `siglip2_model` | Google SigLIP2 models |

### Available Models

#### CLIP
| Model | Feature Dim |
|-------|-------------|
| `openai/clip-vit-base-patch32` (default) | 512 |
| `openai/clip-vit-base-patch16` | 512 |
| `openai/clip-vit-large-patch14` | 768 |
| `openai/clip-vit-large-patch14-336` | 768 |

#### RemoteCLIP
| Variant | Feature Dim |
|---------|-------------|
| `RN50` | 1024 |
| `ViT-B-32` | 512 |
| `ViT-L-14` (default) | 768 |

#### SigLIP2
| Model | Feature Dim |
|-------|-------------|
| `google/siglip2-base-patch16-224` (default) | 768 |
| `google/siglip2-base-patch16-naflex` | 768 |
| `google/siglip2-large-patch16-512` | 1024 |
| `google/siglip2-so400m-patch14-384` | 1152 |

#### Qwen3-VL
| Model | Feature Dim (vision_encoder) |
|-------|------------------------------|
| `Qwen/Qwen3-VL-4B-Instruct` | ~1536 |
| `Qwen/Qwen3-VL-8B-Instruct` (default) | ~1536 |

## Configuration

Example `config.yaml`:

```yaml
model:
  extraction_mode: "siglip2"  # vision_encoder, full, clip, remote_clip, siglip2
  name: "Qwen/Qwen3-VL-8B-Instruct"
  clip_model: "openai/clip-vit-base-patch32"
  remote_clip_variant: "ViT-L-14"
  siglip2_model: "google/siglip2-base-patch16-224"
  pooling: "mean"  # mean, max, cls (for Qwen3-VL modes only)
  dtype: "float16"
  device: "cuda"

data:
  root: "data"
  use_sample: false
  cache_dir: "cache/features"
  output_dir: "output"

training:
  scoring_metric: "f1"
  hp_search_cv_folds: 3
  final_cv_folds: 5
  random_seed: 42
  holdout_fraction: 0.2

classifiers:
  mlp:
    enabled: true
    hidden_layer_sizes: [[128], [256], [128, 64], [256, 128]]
    alpha: [0.0001, 0.001, 0.01]
    learning_rate_init: [0.001, 0.0001]
    activation: ["relu", "tanh"]
  svm:
    enabled: true
    C: [0.1, 1.0, 10.0, 100.0]
    gamma: ["scale", "auto", 0.001, 0.01]
    kernel: ["rbf", "linear"]
  xgboost:
    enabled: true
    n_estimators: [100, 200, 300]
    max_depth: [3, 5, 7]
    learning_rate: [0.01, 0.1, 0.3]
    subsample: [0.8, 1.0]
  logistic_regression:
    enabled: true
    C: [0.01, 0.1, 1.0, 10.0]
    solver: ["lbfgs"]
```

## Data Structure

```
data/
├── class_0/          # Images for class 0
│   ├── image1.tif
│   ├── image2.png
│   └── ...
├── class_1/          # Images for class 1
│   ├── image1.tif
│   ├── image2.jpg
│   └── ...
└── sample/           # Optional sample dataset
    ├── class_0/
    └── class_1/
```

Supported formats: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`

## Output Structure

```
output/
├── MLP/
│   ├── best_model.joblib
│   ├── results.json
│   └── gridsearch_results.csv
├── SVM/
│   └── ...
├── XGBoost/
│   └── ...
└── LogisticRegression/
    └── ...
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--config PATH` | Path to configuration YAML file |
| `--use-sample` | Use sample dataset for testing |
| `--no-cache` | Skip cache, re-extract features |
| `--holdout` | Hold out test set for final evaluation |
| `--log-level LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `--log-file PATH` | Path to log file |

## Project Structure

```
├── main.py                      # Pipeline entry point
├── config.yaml                  # Configuration file
├── pyproject.toml               # Project dependencies
└── src/
    ├── config.py                # Configuration dataclasses
    ├── dataset.py               # Image loading and dataset
    ├── feature_extraction.py    # Qwen3-VL full model extractor + cache
    ├── vision_encoder_extraction.py  # Qwen3-VL vision encoder
    ├── clip_extraction.py       # CLIP extractor
    ├── remote_clip_extraction.py # RemoteCLIP extractor
    ├── siglip2_extraction.py    # SigLIP2 extractor
    ├── classification.py        # Classifier wrappers
    ├── evaluation.py            # HP search and cross-validation
    └── logging_config.py        # Logging setup
```

## Requirements

- Python >= 3.11
- CUDA-capable GPU (recommended)
- See `pyproject.toml` for full dependency list
