import numpy
from albumentations import CoarseDropout, HorizontalFlip
from albumentations import Compose as AlbumentationsCompose
from PIL import Image
from torch import Tensor
from torch.nn import Module
from torchvision.transforms import Compose as TorchCompose
from torchvision.transforms import Normalize, Resize, ToTensor


def transform_image_to_tensor(
    image: Image.Image, pretrained: bool = False, augmented: bool = False
) -> Tensor:
    if augmented:
        transforms = AlbumentationsCompose(
            [
                HorizontalFlip(),
                CoarseDropout(num_holes_range=(1, 5)),
            ]
        )

        image = Image.fromarray(transforms(image=numpy.array(image))["image"])

    transformations: list[Module] = [
        AggressiveComposite(),
        Resize((224, 224)),
        ToTensor(),
    ]

    if pretrained:
        transformations.append(
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    transform = TorchCompose(transformations)

    return transform(image)
