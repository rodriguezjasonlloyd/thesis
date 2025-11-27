# PCOS Classification

A Thesis on PCOS (Polycystic Ovary Syndrome) classification supporting multiple CNN architectures including a simple baseline CNN and ConvNeXt V2 with optional Focal Self-Attention (FSA) mechanism.
Features k-fold cross-validation, comprehensive preprocessing options, and interactive Gradio dashboards for inference with Grad-CAM++ visualizations.

## Features

- **Multiple Architectures**: Simple baseline CNN or ConvNeXt V2 with optional pretrained weights
- **Focal Self-Attention**: Enhanced attention mechanism for improved feature learning (ConvNeXt only)
- **K-Fold Cross-Validation**: Robust model evaluation with configurable folds
- **Multiple Preprocessing Methods**: CLAHE, Otsu Thresholding, Deep Contrast, and composite approaches
- **Interactive Dashboards**:
  - Advanced dashboard with full customization options
  - Simplified demo with pre-trained models
- **Dataset Preparation Tools**: Scripts for detecting data leakage, removing duplicates, and balancing classes
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and confusion matrices
- **Descriptive Analysis**: Dataset statistics including class distribution, image dimensions, brightness, and color analysis
- **Training Visualization**: Automatic generation of training graphs using Plotly

## Requirements

### System Requirements

- [uv](https://docs.astral.sh/uv/) package manager
- Python 3.13 or higher (managed by uv)
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for training)

## Project Structure

```text
thesis/
├── data/                     # Dataset directory
│   ├── infected/             # PCOS-positive images
│   └── notinfected/          # PCOS-negative images
├── demo/                     # Pre-trained model checkpoints
│   ├── base_cnn.pt
│   ├── convnext.pt
│   ├── convnext_fsa.pt
│   ├── convnext_fsa_imagenet.pt
│   ├── convnext_fsa_pre.pt
│   ├── convnext_fsa_pre_imagenet.pt
│   ├── convnext_imagenet.pt
│   ├── convnext_pre.pt
│   └── convnext_pre_imagenet.pt
├── experiments/              # Experiment outputs
│   └── experiment_name/
│       ├── graphs/           # Training visualization graphs
│       ├── models/           # Saved model checkpoints
│       ├── config.toml       # Experiment configuration
│       ├── experiment.log    # Experiment logs
│       └── results.json      # Training results
├── modules/
│   ├── analysis.py           # Analysis and plotting
│   ├── dashboard.py          # Gradio interface (advanced)
│   ├── data.py               # Dataset and data loading
│   ├── demo.py               # Gradio interface (simplified)
│   ├── experiment.py         # Experiment management
│   ├── fsa.py                # Focal Self-Attention implementation
│   ├── model.py              # Model architectures
│   ├── preprocessing.py      # Image preprocessing
│   ├── state_machine.py      # CLI state machine
│   ├── trainer.py            # Training loop
│   └── utilities.py          # Helper functions
├── scripts/                  # Dataset preparation utilities
│   ├── balance.py            # Balance class distribution
│   ├── dedupe.py             # Remove duplicate/similar images
│   └── detect.py             # Detect data leakage issues
├── __main__.py               # Main entry point
```

## Setup

### 1. Clone or Download the Project

```bash
git clone https://github.com/rodriguezjasonlloyd/thesis.git
cd thesis
```

### 2. Install Dependencies

This will create a virtual environment and install all required packages.

```bash
uv sync
```

### 3. Prepare Your Dataset

Organize your data in the following structure:

```text
data/
├── infected/          # Class 0: PCOS-positive images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── notinfected/       # Class 1: PCOS-negative images
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Supported image formats: `.jpg`, `.jpeg`, `.png`

### 4. Create an Experiment

Create an experiment directory with a configuration file:

```bash
mkdir -p experiments/my_experiment
```

Create `experiments/my_experiment/config.toml`:

```toml
name = "My First Experiment"
seed = 42

[data]
root = "data"
k_folds = 5
batch_size = 32
num_workers = 2
max_items_per_class = 0
augmented = false
preprocessing = "none"

[model]
architecture = "convnext"
pretrained = false
with_fsa = false

[optimizer]
learning_rate = 0.001
weight_decay = 0.0001

[training]
num_epochs = 30
patience = 5
min_delta = 0.001
```

## Configuration Reference

### Top-Level Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | "Unnamed Experiment" | Experiment name for identification |
| `seed` | integer | 42 | Random seed for reproducibility |

### `[data]` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root` | string | "data" | Path to dataset directory |
| `k_folds` | integer | 5 | Number of cross-validation folds |
| `batch_size` | integer | 32 | Training batch size |
| `num_workers` | integer | 2 | DataLoader worker threads |
| `max_items_per_class` | integer | 0 | Limit items per class (0 = no limit) |
| `augmented` | boolean | false | Enable data augmentation |
| `preprocessing` | string | "none" | Preprocessing mode (see below) |

**Preprocessing Options:**

- `"none"`: No preprocessing
- `"clahe"`: Contrast Limited Adaptive Histogram Equalization
- `"otsu_threshold"`: Otsu's thresholding with Gaussian blur
- `"deep_contrast"`: Bilateral filtering with gamma correction
- `"all"`: Aggressive composite of all methods

### `[model]` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `architecture` | string | "convnext" | Model architecture: "base" or "convnext" |
| `pretrained` | boolean | false | Use ImageNet pretrained weights (ConvNeXt only) |
| `with_fsa` | boolean | false | Enable Focal Self-Attention mechanism (ConvNeXt only) |

**Architecture Options:**

- `"base"`: Simple 5-layer CNN baseline (~6M parameters)
- `"convnext"`: ConvNeXt V2 Atto with optional pretrained weights and FSA support

### `[optimizer]` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 0.001 | Initial learning rate |
| `weight_decay` | float | 0.0001 | L2 regularization weight |

### `[training]` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_epochs` | integer | 30 | Maximum training epochs |
| `patience` | integer | 5 | Early stopping patience |
| `min_delta` | float | 0.001 | Minimum improvement for early stopping |

## Usage

### Running the Interactive CLI

```bash
uv run .
```

The CLI provides the following options:

1. **Analysis Menu**
   - Analyze descriptive statistics (dataset overview, class distribution, image properties)
   - Visualize sample batch with preprocessing options
   - Show training graphs from previous experiments
   - Show training results and metrics

2. **Experiment Menu**
   - Run all experiments in `experiments/` directory
   - Run selected experiments

3. **Launch Dashboard**
   - Start Gradio web interface for inference (advanced mode with full customization)

4. **Launch Demo**
   - Start simplified Gradio interface with pre-trained models from `demo/` folder

### Using the Dashboards

```bash
uv run .
# Select: Launch Dashboard or Launch Demo
```

**Dashboard (Advanced Mode):**

- Select model architecture (Base CNN or ConvNeXt V2)
- Toggle Focal Self-Attention (for ConvNeXt only)
- Choose preprocessing method
- Toggle ImageNet pretrained weights
- Upload trained model checkpoints (`.pt` files)
- Upload images for classification
- View predictions with confidence scores
- Generate Grad-CAM++ visualizations for model interpretability

**Demo (Simplified Mode):**

- Select from pre-trained models in `demo/` folder
- Upload images for instant classification
- View predictions and Grad-CAM++ visualizations
- No need to configure architecture details

## Output Files

After training, each experiment directory contains:

### `experiment.log`

Detailed logging output including:

- Training progress for each fold and epoch
- Metric updates and early stopping events
- Timing information

### `results.json`

Complete training results including:

- Experiment metadata and configuration
- Per-fold metrics: loss, accuracy, precision, recall, F1-score, ROC-AUC
- Epoch-by-epoch training history
- Confusion matrices

### `models/`

Best model checkpoint for each fold:

- `best_model_fold_1.pt`
- `best_model_fold_2.pt`
- etc.

### `graphs/` (if enabled)

Individual training graphs for each fold and metric.

## Training Metrics

The following metrics are computed and tracked:

- **Loss**: Binary cross-entropy with logits
- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: True/False positives and negatives

## Advanced Usage

### Dataset Preparation Scripts

Before training, you can use the included scripts to clean and prepare your dataset:

#### 1. Detect Data Leakage Issues

```bash
uv run scripts/detect.py
```

This script runs comprehensive tests to identify issues that could cause unrealistic performance:

- Exact duplicate images across classes
- Perceptually similar images
- Train/validation overlap in cross-validation folds
- Class distribution imbalances
- Label mapping verification
- Seed reproducibility
- Image integrity checks
- Artificially easy classification patterns

#### 2. Remove Duplicates and Similar Images

```bash
uv run scripts/dedupe.py
```

Two-step deduplication process:

- **Step 1**: Remove exact duplicates (MD5 hash matching)
- **Step 2**: Remove perceptually similar images (perceptual hashing with configurable threshold)
- Automatically backs up removed images
- Prevents data leakage from near-identical images in train/val splits

#### 3. Balance Class Distribution

```bash
uv run scripts/balance.py
```

Undersample majority class to achieve target ratio:

- Default: 60:40 ratio (configurable)
- Randomly removes excess images from majority class
- Automatically backs up removed images
- Helps prevent class imbalance issues during training

**Recommended workflow:**

1. Run `detect.py` to identify issues
2. Run `dedupe.py` to remove duplicates
3. Run `balance.py` to fix class imbalance
4. Run `detect.py` again to verify fixes
5. Begin training

### Custom Preprocessing

Add new preprocessing methods in `modules/preprocessing.py`:

```python
class CustomPreprocessing(Module):
    def forward(self, image: Image.Image) -> Image.Image:
        # Your preprocessing logic
        return processed_image
```

Register in `PreprocessingMode` enum and update the preprocessing logic in `utilities.py`.

### Model Customization

The project supports two architectures:

**Base CNN** (`architecture = "base"`):

- Simple 5-layer convolutional network
- ~6M parameters
- Good baseline for comparison
- No pretrained weights or FSA support

**ConvNeXt V2** (`architecture = "convnext"`):

- ConvNeXt V2 Atto architecture
- Optional ImageNet pretrained weights
- Optional Focal Self-Attention mechanism (see `modules/fsa.py`)
- Adjust FSA parameters (window size, number of heads) in `modules/fsa.py`

You can also modify the model architecture in `modules/model.py`:

- Add new architectures
- Change backbone architecture
- Modify final classification head

### Custom Training Loop

Extend `modules/trainer.py` for custom training behavior:

- Custom loss functions
- Learning rate schedules
- Additional metrics

## Troubleshooting

### Checking Logs

If training fails or behaves unexpectedly, check `experiments/<experiment_name>/experiment.log` for detailed information about what went wrong.

### Architecture-Specific Issues

- **Base architecture**: Does not support `pretrained=true` or `with_fsa=true`
- **ConvNeXt architecture**: Requires more GPU memory; reduce batch size if needed

### Out of Memory Errors

- Reduce `batch_size` in config
- Reduce `num_workers`
- Use CPU instead of GPU (slower but uses system RAM)

### Dataset Not Found

- Verify `data/` directory structure
- Check image file extensions
- Ensure at least 2 classes with images

### CUDA Errors

- Update GPU drivers
- Check CUDA compatibility with PyTorch version
- Try CPU mode if GPU unavailable

### Import Errors

- Reinstall dependencies: `uv sync --reinstall`
- Check Python version: `uv run python --version` (should be 3.13+)
- Try resyncing the virtual environment: `rm -rf .venv && uv sync`
