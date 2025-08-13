import json
import random
from typing import Dict, List, Tuple

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader

from pcom import config
from pcom.utils import set_seed

ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_or_load_splits() -> Dict[str, List[Tuple[str, str]]]:
    """
    Creates a train/val split if it doesn't exist, otherwise loads it from file.
    """
    if config.SPLIT_FILE.exists():
        with open(config.SPLIT_FILE, "r") as f:
            return json.load(f)

    set_seed(config.SEED)

    all_images = []
    for label in ["infected", "noninfected"]:
        label_dir = config.DATA_DIR / label
        for img_path in label_dir.glob("*.*"):
            all_images.append((str(img_path), label))

    random.shuffle(all_images)

    train_size = int(config.TRAIN_RATIO * len(all_images))
    train_split = all_images[:train_size]
    val_split = all_images[train_size:]

    splits = {"train": train_split, "val": val_split}

    with open(config.SPLIT_FILE, "w") as f:
        json.dump(splits, f, indent=2)

    return splits


class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor
        self.class_to_idx = {"infected": 0, "noninfected": 1}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, torch.tensor(self.class_to_idx[label])


def get_data_loaders(batch_size: int, processor) -> Tuple[DataLoader, DataLoader]:
    """
    Returns DataLoaders for train and val datasets using Hugging Face processor.
    """
    splits = create_or_load_splits()

    if config.MAX_SAMPLES:
        splits["train"] = splits["train"][: config.MAX_SAMPLES]
        splits["val"] = splits["val"][: config.MAX_SAMPLES]

    train_dataset = SplitDataset(splits["train"], processor)
    val_dataset = SplitDataset(splits["val"], processor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader
