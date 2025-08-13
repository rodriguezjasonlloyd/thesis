"""
PCOM Classification Package

This package contains:
- config: Global constants (dataset paths, image size, etc.)
- utils: Helper functions (seed setting, logging, etc.)
- dataset: Data loading and splitting logic
- model: Model architecture definition
- train: Training loop
- eval: Evaluation logic
- explain: Explainability methods (Grad-CAM, Score-CAM)
"""

from . import config, dataset, eval, explain, model, train, utils

__all__ = ["config", "utils", "dataset", "train", "model", "eval", "explain"]
