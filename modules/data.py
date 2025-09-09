from pathlib import Path
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
from PIL import Image
from torch import Tensor
from torch.accelerator import is_available as is_accelerator_available
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

EXTENSIONS = (".jpg", ".jpeg", ".png")


class ImageDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], pretrained: bool) -> None:
        self._items = items
        self._pretrained = pretrained

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        path, label = self._items[index]
        return transform_image_to_tensor(load_image(path), self._pretrained), label


def get_data_root_path(path_name: str = "data") -> Path:
    return Path(path_name)


def get_class_names(root: Path = get_data_root_path()) -> List[str]:
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    return [path.name for path in root.iterdir() if path.is_dir()]


def get_data_loaders(
    root: Path = get_data_root_path(),
    pretrained: bool = False,
    ratio: List[float] = [0.7, 0.3],
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 2,
    max_items_per_class: int = 0,
) -> Tuple[DataLoader[ImageDataset], DataLoader[ImageDataset], ImageDataset]:
    class_names = get_class_names(root)
    label_map = {label: index for index, label in enumerate(class_names)}

    items = []

    for class_name in class_names:
        class_dir = root.joinpath(class_name)

        files = [
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        ]

        if max_items_per_class > 0:
            files = files[:max_items_per_class]

        for path in files:
            items.append((path, label_map[class_name]))

    dataset = ImageDataset(items, pretrained)
    train_split, val_split = random_split(dataset, ratio)

    return (
        DataLoader(
            train_split,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=is_accelerator_available(),
        ),
        DataLoader(
            val_split,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=is_accelerator_available(),
        ),
        dataset,
    )


def load_image(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except:
        raise


def transform_image_to_tensor(image: Image.Image, pretrained: bool = False) -> Tensor:
    transformations: List[Any] = [
        Resize((224, 224)),
        ToTensor(),
    ]

    if pretrained:
        transformations.append(
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    transform = Compose(transformations)

    return transform(image)


def visualize_raw_tensor(tensor: Tensor) -> None:
    numpy_image = tensor.permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(numpy_image)
    plt.title("Raw Transformed Tensor (After Normalization)")
    plt.axis("off")
    plt.show()


def visualize_raw_tensors(tensors: List[Tensor], rows: int = 3, cols: int = 3) -> None:
    figure, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for index, tensor in enumerate(tensors[: rows * cols]):
        numpy_image = tensor.permute(1, 2, 0).numpy()
        axes[index].imshow(numpy_image)
        axes[index].set_title(f"Image {index + 1}")
        axes[index].axis("off")

    plt.tight_layout()
    plt.show()
