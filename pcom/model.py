import torch.nn as nn
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification


def build_model(model_config: dict) -> tuple[nn.Module, AutoImageProcessor]:
    """
    Builds and returns a ConvNeXt V2 model with processor.

    Args:
        model_config (dict): Example:
            {
                "num_classes": 2,
                "freeze_backbone": False
            }

    Returns:
        (torch.nn.Module, AutoImageProcessor)
    """
    model_name = "facebook/convnextv2-atto-1k-224"
    num_classes = model_config.get("num_classes", 2)
    freeze_backbone = model_config.get("freeze_backbone", False)

    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = ConvNextV2ForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    return model, processor
