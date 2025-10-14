import warnings
from random import seed as random_seed
from typing import Any, Callable, Iterator, Optional

from numpy.random import seed as np_seed
from torch import Tensor, device, load, manual_seed, no_grad, save
from torch import max as torch_max
from torch.cuda import is_available as is_cuda_available
from torch.cuda import manual_seed_all
from torch.nn import Module, Parameter
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
) -> tuple[int, float]:
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
    get_model: Callable[[], Module],
    get_optimizer: Callable[[Iterator[Parameter]], Optimizer],
    criterion: Module,
    fold_loaders: list[tuple[DataLoader[ImageDataset], DataLoader[ImageDataset]]],
    num_epochs: int = 30,
    patience: int = 5,
    min_delta: float = 1e-3,
) -> dict[str, Any]:
    device = get_device()
    criterion = criterion.to(device)

    k_folds = len(fold_loaders)
    fold_results = []

    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    for fold_index, (train_loader, validation_loader) in enumerate(fold_loaders):
        print(f"Training Fold {fold_index + 1}/{k_folds}")

        model = get_model()
        model = model.to(device)
        optimizer = get_optimizer(model.parameters())
        scheduler = CosineAnnealingLR(optimizer, num_epochs, 1e-6)

        best_validation_loss = float("inf")
        best_validation_accuracy = 0.0
        epochs_without_improvement = 0
        best_model_state_dict: Optional[dict[str, Any]] = None

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_accuracy = 0.0

            for input, label in tqdm(
                train_loader,
                desc=f"Fold {fold_index + 1} - Training Epoch {epoch + 1}",
                leave=False,
            ):
                input = input.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                output_tensor: Tensor = model(input)
                loss_tensor: Tensor = criterion(output_tensor, label)
                loss_tensor.backward()
                optimizer.step()

                train_loss += loss_tensor.item()
                train_accuracy += compute_accuracy(output_tensor, label)

            model.eval()
            validation_loss = 0.0
            validation_accuracy = 0.0

            with no_grad():
                for input, label in tqdm(
                    validation_loader,
                    desc=f"Fold {fold_index + 1} - Validation Epoch {epoch + 1}",
                    leave=False,
                ):
                    input = input.to(device)
                    label = label.to(device)

                    output_tensor: Tensor = model(input)
                    loss_tensor: Tensor = criterion(output_tensor, label)

                    validation_loss += loss_tensor.item()
                    validation_accuracy += compute_accuracy(output_tensor, label)

            if len(train_loader) == 0:
                raise ValueError(f"Fold {fold_index + 1}: Training loader is empty")

            if len(validation_loader) == 0:
                raise ValueError(f"Fold {fold_index + 1}: Validation loader is empty")

            average_train_loss = train_loss / len(train_loader)
            average_train_accuracy = train_accuracy / len(train_loader)
            average_validation_loss = validation_loss / len(validation_loader)
            average_validation_accuracy = validation_accuracy / len(validation_loader)

            if average_validation_loss < best_validation_loss - min_delta:
                best_validation_loss = average_validation_loss
                best_validation_accuracy = average_validation_accuracy
                epochs_without_improvement = 0
                best_model_state_dict = model.state_dict().copy()
            else:
                epochs_without_improvement += 1

            scheduler.step()

            print(
                f"Fold {fold_index + 1} - Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss: {average_train_loss:.4f}, "
                f"Train Accuracy: {average_train_accuracy:.2f}%, "
                f"Validation Loss: {average_validation_loss:.4f}, "
                f"Validation Accuracy: {average_validation_accuracy:.2f}%"
            )

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")

                if best_model_state_dict is not None:
                    model.load_state_dict(best_model_state_dict)

                break

        fold_results.append(
            {
                "fold": fold_index + 1,
                "best_validation_loss": best_validation_loss,
                "best_validation_accuracy": best_validation_accuracy,
                "model_state_dict": best_model_state_dict,
            }
        )

        print(
            f"\nFold {fold_index + 1} completed - Best Validation Loss: {best_validation_loss:.4f}, Best Validation Accuracy: {best_validation_accuracy:.2f}%"
        )

    average_validation_loss = (
        sum(fold["best_validation_loss"] for fold in fold_results) / k_folds
    )
    average_validation_accuracy = (
        sum(fold["best_validation_accuracy"] for fold in fold_results) / k_folds
    )

    print("K-Fold Cross Validation Results")
    print(f"Average Validation Loss: {average_validation_loss:.4f}")
    print(f"Average Validation Accuracy: {average_validation_accuracy:.2f}%")

    return {
        "fold_results": fold_results,
        "average_validation_loss": average_validation_loss,
        "average_validation_accuracy": average_validation_accuracy,
        "k_folds": k_folds,
    }
