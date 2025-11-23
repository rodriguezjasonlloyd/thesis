import random
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from modules import utilities

EXTENSIONS = (".jpg", ".jpeg", ".png")


class ImageDataset(Dataset):
    def __init__(
        self, items: list[tuple[Path, int]], pretrained: bool, augmented: bool
    ) -> None:
        self._items = items
        self._pretrained = pretrained
        self._augmented = augmented

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path, label = self._items[index]

        with Image.open(path) as file:
            image = file.convert("RGB")

        return utilities.image_to_tensor(
            image, self._pretrained, self._augmented
        ), label


def get_data_root_path(path_name: str = "data") -> Path:
    return Path(path_name)


def get_class_names(root: Path = get_data_root_path()) -> list[str]:
    if not root.exists():
        raise FileNotFoundError(f"Path '{root}' does not exist.")

    return [path.name for path in root.iterdir() if path.is_dir()]


def get_data_loaders(
    root: Path = get_data_root_path(),
    pretrained: bool = False,
    augmented: bool = False,
    k_folds: int = 5,
    batch_size: int = 32,
    num_workers: int = 2,
    max_items_per_class: int = 0,
) -> list[tuple[DataLoader[ImageDataset], DataLoader[ImageDataset]]]:
    class_names = get_class_names(root)
    label_map = {label: index for index, label in enumerate(class_names)}

    items = []

    for class_name in class_names:
        class_directory = root / class_name

        files = [
            path
            for path in class_directory.iterdir()
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        ]

        if max_items_per_class > 0:
            files = files[:max_items_per_class]

        for path in files:
            items.append((path, label_map[class_name]))

    dataset_size = len(items)

    if dataset_size == 0:
        raise ValueError(
            f"Dataset is empty. Check that {root} contains valid image files."
        )

    if dataset_size < k_folds:
        raise ValueError(
            f"Dataset has {dataset_size} samples but {k_folds} k-folds. Need at least {k_folds} samples for {k_folds}-fold cross validation."
        )

    indices = list(range(dataset_size))
    random.shuffle(indices)

    fold_size = dataset_size // k_folds
    fold_loaders = []

    for fold in range(k_folds):
        validation_start = fold * fold_size
        validation_end = (fold + 1) * fold_size if fold < k_folds - 1 else dataset_size
        validation_indices = indices[validation_start:validation_end]
        train_indices = indices[:validation_start] + indices[validation_end:]

        train_dataset = ImageDataset(
            items=[items[i] for i in train_indices],
            pretrained=pretrained,
            augmented=augmented,
        )

        validation_dataset = ImageDataset(
            items=[items[i] for i in validation_indices],
            pretrained=pretrained,
            augmented=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.accelerator.is_available(),
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.accelerator.is_available(),
        )

        fold_loaders.append((train_loader, validation_loader))

    return fold_loaders
