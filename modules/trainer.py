import warnings
from random import seed as random_seed
from typing import Any, Optional, Tuple

from numpy.random import seed as np_seed
from torch import Tensor, device, load, manual_seed, no_grad, save
from torch import max as torch_max
from torch.cuda import is_available as is_cuda_available
from torch.cuda import manual_seed_all
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm.rich import tqdm
from tqdm.std import TqdmExperimentalWarning

from modules.data import ImageDataset


def seed_all(seed: int) -> None:
    manual_seed(seed)
    manual_seed_all(seed)
    np_seed(seed)
    random_seed(seed)


def get_device() -> device:
    if is_cuda_available():
        return device("cuda")
    else:
        return device("cpu")


def save_checkpoint(
    model: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    loss: float,
    path: str,
) -> None:
    save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
        },
        path,
    )


def load_checkpoint(
    model: Module, optimizer: Optimizer, scheduler: LRScheduler, path: str
) -> Tuple[int, float]:
    checkpoint = load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"], checkpoint["loss"]


def compute_accuracy(output: Tensor, target: Tensor) -> float:
    with no_grad():
        _, predicted = torch_max(output, 1)
        correct = (predicted == target).sum().item()
        total = target.size(0)

        return 100.0 * correct / total


def train_model(
    model: Module,
    optimizer: Optimizer,
    criterion: Module,
    train_loader: DataLoader[ImageDataset],
    val_loader: DataLoader[ImageDataset],
    num_epochs: int = 30,
    patience: int = 5,
    min_delta: float = 1e-3,
) -> None:
    device = get_device()
    model = model.to(device)

    scheduler = CosineAnnealingLR(optimizer, num_epochs, 1e-6)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state_dict: Optional[dict[str, Any]] = None

    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for input, label in tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}", leave=False
        ):
            input = input.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output_tensor: Tensor = model(input)
            loss_tensor: Tensor = criterion(output_tensor, label)
            loss_tensor.backward()
            optimizer.step()

            train_loss += loss_tensor.item()
            train_acc += compute_accuracy(output_tensor, label)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with no_grad():
            for input, label in tqdm(
                val_loader, desc=f"Validation Epoch {epoch + 1}", leave=False
            ):
                input = input.to(device)
                label = label.to(device)

                output_tensor: Tensor = model(input)
                loss_tensor: Tensor = criterion(output_tensor, label)

                val_loss += loss_tensor.item()
                val_acc += compute_accuracy(output_tensor, label)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state_dict = model.state_dict().copy()
        else:
            epochs_without_improvement += 1

        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Train Accuracy: {avg_train_acc:.2f}%, "
            f"Validation Loss: {avg_val_loss:.4f}, "
            f"Validation Accuracy: {avg_val_acc:.2f}%"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")

            if best_model_state_dict is not None:
                model.load_state_dict(best_model_state_dict)

            break
