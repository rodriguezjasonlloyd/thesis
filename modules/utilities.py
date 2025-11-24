import dataclasses
import logging
import random
from enum import Enum
from pathlib import Path
from typing import Literal, TypeVar

import numpy
import torch
from albumentations import CoarseDropout, HorizontalFlip
from albumentations import Compose as AlbumentationsCompose
from PIL import Image
from torch import Tensor
from torch.nn import Module
from torchvision.transforms import Compose as TorchCompose
from torchvision.transforms import Normalize, Resize, ToTensor

from modules.preprocessing import (
    CLAHE,
    AggressiveComposite,
    DeepContrast,
    OtsuThreshold,
    PreprocessingMode,
)


def image_to_tensor(
    image: Image.Image,
    pretrained: bool = False,
    augmented: bool = False,
    preprocessing: PreprocessingMode = PreprocessingMode.NONE,
) -> Tensor:
    if augmented:
        transforms = AlbumentationsCompose(
            [
                HorizontalFlip(),
                CoarseDropout(num_holes_range=(1, 5)),
            ]
        )

        image = Image.fromarray(transforms(image=numpy.array(image))["image"])

    transformations: list[Module] = []

    if preprocessing == PreprocessingMode.CLAHE:
        transformations.append(CLAHE())
    elif preprocessing == PreprocessingMode.OTSU_THRESHOLD:
        transformations.append(OtsuThreshold())
    elif preprocessing == PreprocessingMode.DEEP_CONTRAST:
        transformations.append(DeepContrast())
    elif preprocessing == PreprocessingMode.ALL:
        transformations.append(AggressiveComposite())

    transformations.extend(
        [
            Resize((224, 224)),
            ToTensor(),
        ]
    )

    if pretrained:
        transformations.append(
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    transform = TorchCompose(transformations)

    return transform(image)


def tensor_to_numpy(
    tensor: Tensor,
) -> numpy.ndarray[
    tuple[Literal[224], Literal[224], Literal[3]], numpy.dtype[numpy.float32]
]:
    return tensor.permute(1, 2, 0).numpy()


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)


def truncate(number: float, decimals: int = 0):
    multiplier = 10**decimals
    return int(number * multiplier) / multiplier


def format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


T = TypeVar("T")


def dataclass_to_dict(object: T) -> dict[str, int | float | str | bool]:
    if not dataclasses.is_dataclass(object):
        raise TypeError(f"Expected dataclass, got '{type(object)}'")

    result: dict[str, int | float | str | bool] = {}

    for field in dataclasses.fields(object):
        value = getattr(object, field.name)

        if isinstance(value, Enum):
            result[field.name] = value.value
        else:
            result[field.name] = value

    return result


def setup_logging(log_directory: Path, level=logging.INFO):
    logging.basicConfig(
        format="%(message)s",
        handlers=[
            logging.FileHandler(
                log_directory / "experiment.log",
                encoding="utf-8",
            )
        ],
        level=level,
    )


def format_file_size(size_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"

        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"
