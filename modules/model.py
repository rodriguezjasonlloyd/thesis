from pathlib import Path

import timm
import torch
from torch import Tensor
from torch.nn import (
    AdaptiveAvgPool2d,
    Conv2d,
    Flatten,
    LayerNorm,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)

from modules import utilities
from modules.fsa import TransformerBlock


class BaseCNN(Module):
    def __init__(self) -> None:
        super().__init__()

        self.features = Sequential(
            Conv2d(3, 32, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(2, 2),
            Conv2d(32, 64, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(2, 2),
            Conv2d(64, 128, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(2, 2),
            Conv2d(128, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(2, 2),
            Conv2d(256, 512, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(2, 2),
        )

        self.classifier = Sequential(
            AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            Linear(512, 1),
        )

    def forward(self, input: Tensor) -> Tensor:
        output = self.features(input)
        output = self.classifier(output)
        return output


def get_channels_for_stage(model: Module, stage_index: int) -> int:
    stage_path = f"stages.{stage_index}"

    try:
        stage = model.get_submodule(stage_path)
        blocks = stage.get_submodule("blocks")
    except AttributeError as exception:
        raise RuntimeError(f"Stage not found at path: {stage_path}") from exception

    blocks_list: list[Module] = list(blocks.children())

    if len(blocks_list) == 0:
        raise RuntimeError(f"No blocks found under {stage_path}.blocks")

    first_block: Module = blocks_list[0]

    if hasattr(first_block, "conv_dw"):
        conv_dw = first_block.conv_dw

        if isinstance(conv_dw, Conv2d):
            return int(conv_dw.in_channels)

    if hasattr(first_block, "norm"):
        norm = first_block.norm

        if isinstance(norm, LayerNorm) and hasattr(norm, "normalized_shape"):
            normalized_shape = norm.normalized_shape

            if (
                isinstance(normalized_shape, (tuple, list, torch.Size))
                and len(normalized_shape) > 0
            ):
                return int(normalized_shape[0])

    for module in first_block.modules():
        if isinstance(module, Conv2d):
            return int(module.in_channels)

    raise RuntimeError(
        f"Could not determine channel dim for stage {stage_index}. "
        f"First block type: {type(first_block).__name__}"
    )


def get_all_convolutional_layers(model: Module) -> list[tuple[str, str]]:
    convolutional_layers = []
    seen = set()

    for name, module in model.named_modules():
        name: str
        module: Module

        if isinstance(module, Conv2d):
            parts = name.split(".")
            shallow_name = ".".join(parts[:3]) if len(parts) >= 3 else name

            if shallow_name not in seen:
                display_name = shallow_name.replace(".", " > ")
                convolutional_layers.append((display_name, shallow_name))
                seen.add(shallow_name)

    return convolutional_layers


def load_model(
    model_path: Path, architecture: str = "convnext", with_fsa: bool = False
) -> Module:
    device = utilities.get_device()

    if architecture == "base":
        model = BaseCNN()
    elif architecture == "convnext":
        model = build_model(with_fsa=with_fsa)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model = model.to(device)

    if model_path and model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    return model


def build_model(pretrained: bool = False, with_fsa: bool = False) -> Module:
    model = timm.create_model(
        f"convnextv2_atto.fcmae{'_ft_in1k' if pretrained else ''}",
        pretrained=pretrained,
        num_classes=1,
    )

    if pretrained:
        for parameters in model.get_submodule("stem").parameters():
            parameters.requires_grad = False

        for stage_index in [0, 1]:
            for parameters in model.get_submodule(f"stages.{stage_index}").parameters():
                parameters.requires_grad = False

    if with_fsa:
        try:
            stage2_blocks = model.get_submodule("stages.2.blocks")
            stage3_blocks = model.get_submodule("stages.3.blocks")
        except AttributeError as exception:
            raise RuntimeError(
                "Expected stages.2.blocks and stages.3.blocks in model"
            ) from exception

        stage2_block_count = len(list(stage2_blocks.children()))
        stage3_block_count = len(list(stage3_blocks.children()))

        stage2_channels = get_channels_for_stage(model, 2)
        stage3_channels = get_channels_for_stage(model, 3)

        stage2_transformer_blocks = Sequential(
            *[
                TransformerBlock(dim=stage2_channels, num_heads=4, win_size=4)
                for _ in range(stage2_block_count)
            ]
        )

        stage3_transformer_blocks = Sequential(
            *[
                TransformerBlock(dim=stage3_channels, num_heads=8, win_size=4)
                for _ in range(stage3_block_count)
            ]
        )

        model.set_submodule("stages.2.blocks", stage2_transformer_blocks)
        model.set_submodule("stages.3.blocks", stage3_transformer_blocks)

    model.set_submodule("head.fc", Linear(in_features=320, out_features=1, bias=True))

    return model
