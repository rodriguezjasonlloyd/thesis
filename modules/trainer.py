import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, TypedDict

import torch
from rich.console import Console
from rich.table import Table
from sklearn import metrics
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryConfusionMatrix
from tqdm.rich import tqdm
from tqdm.std import TqdmExperimentalWarning

from modules import utilities
from modules.data import ImageDataset

console = Console()


class EpochMetrics(TypedDict):
    epoch: int
    train_loss: float
    train_accuracy: float
    train_precision: float
    train_recall: float
    train_f1_score: float
    train_roc_auc: float
    validation_loss: float
    validation_accuracy: float
    validation_precision: float
    validation_recall: float
    validation_f1_score: float
    validation_roc_auc: float


class FoldResult(TypedDict):
    fold: int
    best_validation_loss: float
    best_validation_accuracy: float
    confusion_matrix: list[list[int]]
    epoch_history: list[EpochMetrics]


class TrainingResults(TypedDict):
    fold_results: list[FoldResult]
    average_validation_loss: float
    average_validation_accuracy: float
    k_folds: int


def compute_accuracy(output: Tensor, target: Tensor) -> float:
    with torch.no_grad():
        predicted = (torch.sigmoid(output.squeeze()) > 0.5).long()
        target_squeezed = target.squeeze().long()
        correct = (predicted == target_squeezed).sum().item()
        total = target.size(0)

        return 100.0 * correct / total


def compute_metrics(
    all_outputs: list[Tensor], all_targets: list[Tensor]
) -> dict[str, float]:
    with torch.no_grad():
        outputs = torch.cat(all_outputs, dim=0)
        targets = torch.cat(all_targets, dim=0)

        probabilities = torch.sigmoid(outputs.squeeze())
        predicted = (probabilities > 0.5).long()

        predicted_numpy = predicted.cpu().numpy()
        targets_numpy = targets.squeeze().cpu().numpy()
        probabilities_numpy = probabilities.cpu().numpy()

        precision = metrics.precision_score(
            targets_numpy, predicted_numpy, zero_division=0
        )
        recall = metrics.recall_score(targets_numpy, predicted_numpy, zero_division=0)
        f1 = metrics.f1_score(targets_numpy, predicted_numpy, zero_division=0)

        unique_classes = len(set(targets_numpy.tolist()))

        if unique_classes < 2:
            roc_auc = 0.0
        else:
            roc_auc = metrics.roc_auc_score(targets_numpy, probabilities_numpy)

        return {
            "precision": precision * 100.0,
            "recall": recall * 100.0,
            "f1_score": f1 * 100.0,
            "roc_auc": roc_auc * 100.0,
        }


def train_model(
    experiment_directory: Path,
    get_model: Callable[[], Module],
    get_optimizer: Callable[[Iterator[Parameter]], Optimizer],
    criterion: Module,
    fold_loaders: list[tuple[DataLoader[ImageDataset], DataLoader[ImageDataset]]],
    num_epochs: int = 30,
    patience: int = 5,
    min_delta: float = 1e-3,
) -> TrainingResults:
    models_directory = experiment_directory / "models"
    models_directory.mkdir(exist_ok=True)

    experiment_start_time = datetime.now()
    device = utilities.get_device()
    criterion = criterion.to(device)

    k_folds = len(fold_loaders)
    fold_results = []

    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    for fold_index, (train_loader, validation_loader) in enumerate(fold_loaders):
        fold_start_time = datetime.now()

        console.print(
            f"\n[bold cyan]Training Fold {fold_index + 1}/{k_folds}[/bold cyan]"
        )

        model = get_model()
        model = model.to(device)
        optimizer = get_optimizer(model.parameters())
        scheduler = CosineAnnealingLR(optimizer, num_epochs, 1e-6)

        confusion_matrix = BinaryConfusionMatrix().to(device)

        best_validation_loss = float("inf")
        best_validation_accuracy = 0.0
        epochs_without_improvement = 0
        epoch_history = []

        for epoch in range(num_epochs):
            epoch_start_time = datetime.now()

            model.train()
            train_loss = 0.0
            train_accuracy = 0.0
            train_outputs = []
            train_targets = []

            for input, label in tqdm(
                train_loader,
                desc=f"Fold {fold_index + 1} - Training Epoch {epoch + 1}",
                leave=False,
            ):
                input: Tensor = input
                label: Tensor = label

                input = input.to(device)
                label = label.float().unsqueeze(1).to(device)

                optimizer.zero_grad()
                output_tensor: Tensor = model(input)
                loss_tensor: Tensor = criterion(output_tensor, label)
                loss_tensor.backward()
                optimizer.step()

                train_loss += loss_tensor.item()
                train_accuracy += compute_accuracy(output_tensor, label)
                train_outputs.append(output_tensor.detach())
                train_targets.append(label.detach())

            model.eval()
            validation_loss = 0.0
            validation_accuracy = 0.0
            validation_outputs = []
            validation_targets = []

            with torch.no_grad():
                for input, label in tqdm(
                    validation_loader,
                    desc=f"Fold {fold_index + 1} - Validation Epoch {epoch + 1}",
                    leave=False,
                ):
                    input: Tensor = input
                    label: Tensor = label

                    input = input.to(device)
                    label = label.float().unsqueeze(1).to(device)

                    output_tensor: Tensor = model(input)
                    loss_tensor: Tensor = criterion(output_tensor, label)

                    validation_loss += loss_tensor.item()
                    validation_accuracy += compute_accuracy(output_tensor, label)
                    validation_outputs.append(output_tensor.detach())
                    validation_targets.append(label.detach())

            validation_metrics = compute_metrics(validation_outputs, validation_targets)

            with torch.no_grad():
                all_validation_outputs = torch.cat(validation_outputs, dim=0)
                all_validation_targets = torch.cat(validation_targets, dim=0)
                validation_preds = (
                    torch.sigmoid(all_validation_outputs.squeeze()) > 0.5
                ).long()
                confusion_matrix.update(
                    validation_preds, all_validation_targets.squeeze().long()
                )

            if len(train_loader) == 0:
                raise ValueError(f"Fold {fold_index + 1}: Training loader is empty")

            if len(validation_loader) == 0:
                raise ValueError(f"Fold {fold_index + 1}: Validation loader is empty")

            average_train_loss = train_loss / len(train_loader)
            average_train_accuracy = train_accuracy / len(train_loader)
            average_validation_loss = validation_loss / len(validation_loader)
            average_validation_accuracy = validation_accuracy / len(validation_loader)

            train_metrics = compute_metrics(train_outputs, train_targets)
            validation_metrics = compute_metrics(validation_outputs, validation_targets)

            epoch_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": average_train_loss,
                    "train_accuracy": average_train_accuracy,
                    "train_precision": train_metrics["precision"],
                    "train_recall": train_metrics["recall"],
                    "train_f1_score": train_metrics["f1_score"],
                    "train_roc_auc": train_metrics["roc_auc"],
                    "validation_loss": average_validation_loss,
                    "validation_accuracy": average_validation_accuracy,
                    "validation_precision": validation_metrics["precision"],
                    "validation_recall": validation_metrics["recall"],
                    "validation_f1_score": validation_metrics["f1_score"],
                    "validation_roc_auc": validation_metrics["roc_auc"],
                }
            )

            if average_validation_loss < best_validation_loss - min_delta:
                best_validation_loss = average_validation_loss
                best_validation_accuracy = average_validation_accuracy
                epochs_without_improvement = 0
                torch.save(
                    model.state_dict(),
                    models_directory / f"best_model_fold_{fold_index + 1}.pt",
                )
            else:
                epochs_without_improvement += 1

            scheduler.step()

            epoch_duration = (datetime.now() - epoch_start_time).total_seconds()

            table = Table(
                title=f"Fold {fold_index + 1} - Epoch {epoch + 1}/{num_epochs}"
            )
            table.add_column("Split", style="cyan")
            table.add_column("Loss", justify="right", style="magenta")
            table.add_column("Accuracy", justify="right", style="green")
            table.add_column("Precision", justify="right", style="yellow")
            table.add_column("Recall", justify="right", style="blue")
            table.add_column("F1", justify="right", style="red")
            table.add_column("ROC-AUC", justify="right", style="white")

            table.add_row(
                "Train",
                f"{utilities.truncate(average_train_loss, 4):.4f}",
                f"{utilities.truncate(average_train_accuracy, 2):.2f}%",
                f"{utilities.truncate(train_metrics['precision'], 2):.2f}%",
                f"{utilities.truncate(train_metrics['recall'], 2):.2f}%",
                f"{utilities.truncate(train_metrics['f1_score'], 2):.2f}%",
                f"{utilities.truncate(train_metrics['roc_auc'], 2):.2f}%",
            )
            table.add_row(
                "Validation",
                f"{utilities.truncate(average_validation_loss, 4):.4f}",
                f"{utilities.truncate(average_validation_accuracy, 2):.2f}%",
                f"{utilities.truncate(validation_metrics['precision'], 2):.2f}%",
                f"{utilities.truncate(validation_metrics['recall'], 2):.2f}%",
                f"{utilities.truncate(validation_metrics['f1_score'], 2):.2f}%",
                f"{utilities.truncate(validation_metrics['roc_auc'], 2):.2f}%",
            )

            console.print(table)
            console.print(
                f"[dim]Took {utilities.format_duration(epoch_duration)}[/dim]"
            )

            if epochs_without_improvement >= patience:
                console.print(
                    f"[yellow]Early stopping triggered at epoch {epoch + 1}[/yellow]"
                )

                break

        fold_duration = (datetime.now() - fold_start_time).total_seconds()

        fold_confusion_matrix = confusion_matrix.compute()

        fold_results.append(
            {
                "fold": fold_index + 1,
                "best_validation_loss": best_validation_loss,
                "best_validation_accuracy": best_validation_accuracy,
                "confusion_matrix": fold_confusion_matrix.cpu().tolist(),
                "epoch_history": epoch_history,
            }
        )

        console.print(
            f"[bold green]Fold {fold_index + 1} completed[/bold green] - "
            f"Best Validation Loss: {utilities.truncate(best_validation_loss, 4):.4f}, "
            f"Best Validation Accuracy: {utilities.truncate(best_validation_accuracy, 2):.2f}%"
        )
        console.print(
            f"[dim]Fold took {utilities.format_duration(fold_duration)}[/dim]\n"
        )

        confusion_matrix_table = Table(
            title=f"Confusion Matrix - Fold {fold_index + 1}"
        )
        confusion_matrix_table.add_column("", style="cyan")
        confusion_matrix_table.add_column(
            "Predicted: 0", justify="center", style="magenta"
        )
        confusion_matrix_table.add_column(
            "Predicted: 1", justify="center", style="magenta"
        )

        true_negative, false_positive = fold_confusion_matrix[0].tolist()
        false_negative, true_positive = fold_confusion_matrix[1].tolist()

        confusion_matrix_table.add_row(
            "Actual: 0", str(true_negative), str(false_positive)
        )
        confusion_matrix_table.add_row(
            "Actual: 1", str(false_negative), str(true_positive)
        )

        console.print(confusion_matrix_table)

    average_validation_loss = (
        sum(fold["best_validation_loss"] for fold in fold_results) / k_folds
    )
    average_validation_accuracy = (
        sum(fold["best_validation_accuracy"] for fold in fold_results) / k_folds
    )

    experiment_duration = (datetime.now() - experiment_start_time).total_seconds()

    console.print("[bold cyan]K-Fold Cross Validation Results[/bold cyan]")
    console.print(
        f"Average Validation Loss: [magenta]{utilities.truncate(average_validation_loss, 4):.4f}[/magenta]"
    )
    console.print(
        f"Average Validation Accuracy: [green]{utilities.truncate(average_validation_accuracy, 2):.2f}%[/green]"
    )
    console.print(
        f"[bold]Total experiment time: {utilities.format_duration(experiment_duration)}[/bold]\n"
    )

    return {
        "fold_results": fold_results,
        "average_validation_loss": average_validation_loss,
        "average_validation_accuracy": average_validation_accuracy,
        "k_folds": k_folds,
    }
