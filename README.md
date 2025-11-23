# PCOS Classification with ConvNeXt V2

A deep learning project for PCOS (Polycystic Ovary Syndrome) classification using ConvNeXt V2 architecture with optional Focal Self-Attention (FSA) mechanism. Features k-fold cross-validation, comprehensive preprocessing options, and an interactive Gradio dashboard for inference with Grad-CAM++ visualizations.

## Features

- **ConvNeXt V2 Architecture**: State-of-the-art CNN with optional pretrained weights
- **Focal Self-Attention**: Enhanced attention mechanism for improved feature learning
- **K-Fold Cross-Validation**: Robust model evaluation with configurable folds
- **Multiple Preprocessing Methods**: CLAHE, Otsu Thresholding, Deep Contrast, and composite approaches
- **Interactive Dashboard**: Gradio-based interface for model inference and visualization
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and confusion matrices
- **Training Visualization**: Automatic generation of training graphs using Plotly

## Requirements

### System Requirements
- [uv](https://docs.astral.sh/uv/) package manager
- Python 3.13 or higher (managed by uv)
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for training)

## Project Structure

```
thesis/
├── data/                     # Dataset directory
│   ├── class_0/              # Images for class 0
│   └── class_1/              # Images for class 1
├── experiments/              # Experiment outputs
│   └── experiment_name/
│       ├── config.toml       # Experiment configuration
│       ├── results.json      # Training results
│       ├── graphs/           # Training visualization graphs
│       └── models/           # Saved model checkpoints
├── modules/
│   ├── analysis.py           # Analysis and plotting
│   ├── dashboard.py          # Gradio interface
│   ├── data.py               # Dataset and data loading
│   ├── experiment.py         # Experiment management
│   ├── model.py              # Model architecture
│   ├── trainer.py            # Training loop
│   ├── preprocessing.py      # Image preprocessing
│   ├── state_machine.py      # State machine for prompt
│   └── utilities.py          # Helper functions
├── __main__.py               # Main entry point
```

## Setup

### 1. Clone or Download the Project

```bash
git clone <repository-url>
cd thesis
```

### 2. Install Dependencies
This will create a virtual environment and install all required packages.
```bash
uv sync
```

### 3. Prepare Your Dataset

Organize your data in the following structure:

```
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
| `pretrained` | boolean | false | Use ImageNet pretrained weights |
| `with_fsa` | boolean | false | Enable Focal Self-Attention mechanism |

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
   - Analyze descriptive statistics
   - Visualize sample batch
   - Show training graphs from previous experiments

2. **Experiment Menu**
   - Run all experiments in `experiments/` directory
   - Run a specific experiment

3. **Launch Dashboard**
   - Start Gradio web interface for inference

### Using the Dashboard

```bash
uv run .
# Select: Launch Dashboard
```

The dashboard allows you to:
- Upload trained model checkpoints (`.pt` files)
- Upload images for classification
- View predictions with confidence scores
- Generate Grad-CAM++ visualizations for model interpretability

## Output Files

After training, each experiment directory contains:

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

Modify the model architecture in `modules/model.py`:
- Adjust FSA parameters (window size, number of heads)
- Change backbone architecture
- Modify final classification head

### Custom Training Loop

Extend `modules/trainer.py` for custom training behavior:
- Custom loss functions
- Learning rate schedules
- Additional metrics

## Troubleshooting

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
