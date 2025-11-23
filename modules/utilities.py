from typing import Literal

import numpy
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
