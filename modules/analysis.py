import json
import logging
from collections import Counter
from pathlib import Path

import numpy
from PIL import Image
from plotly import express, subplots
from plotly.graph_objects import Figure, Heatmap, Scatter
from rich.console import Console
from rich.table import Table
from torch import Tensor

from modules import data, utilities
from modules.trainer import EpochMetrics, TrainingResults


def analyze_descriptive(root: Path = data.get_data_root_path()) -> None:
    console = Console()

    if not root.exists():
        raise FileNotFoundError(f"Root '{root}' does not exist.")

    class_names = data.get_class_names(root)

    if len(class_names) == 0:
        raise ValueError(f"No class directories found in '{root}'.")

    console.print("[cyan]Analyzing Dataset[/cyan]\n")

    class_images: dict[str, list[dict]] = {cls: [] for cls in class_names}

    for class_name in class_names:
        class_directory = root / class_name
        image_files = [
            path
            for path in class_directory.iterdir()
            if path.is_file() and path.suffix.lower() in data.EXTENSIONS
        ]

        console.print(f"Processing class '{class_name}': {len(image_files)} images")

        for image_path in image_files:
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    file_size = image_path.stat().st_size

                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    image_array = numpy.array(image)
                    mean_rgb = tuple(
                        float(image_array[:, :, i].mean()) for i in range(3)
                    )

                    brightness = float(image_array.mean())

                class_images[class_name].append(
                    {
                        "width": width,
                        "height": height,
                        "file_size": file_size,
                        "mean_rgb": mean_rgb,
                        "brightness": brightness,
                        "aspect_ratio": width / height if height > 0 else 0.0,
                        "resolution": (width, height),
                    }
                )
            except Exception as exception:
                console.print(
                    f"[yellow]Warning: Could not process {image_path.name}: {exception}[/yellow]"
                )

    total_images = sum(len(images) for images in class_images.values())

    if total_images == 0:
        raise ValueError("No valid images found in dataset.")

    console.print("\n[bold cyan]Dataset Descriptive Statistics[/bold cyan]\n")

    overall_table = Table(
        title="Overall Statistics", show_header=False, title_justify="left"
    )
    overall_table.add_column("Metric", style="cyan")
    overall_table.add_column("Value", style="green")
    overall_table.add_row("Total Images", str(total_images))
    overall_table.add_row("Number of Classes", str(len(class_names)))
    console.print(overall_table)
    console.print()

    class_table = Table(title="Class Distribution", title_justify="left")
    class_table.add_column("Class", style="cyan")
    class_table.add_column("Count", justify="right", style="green")
    class_table.add_column("Percentage", justify="right", style="yellow")

    for class_name, images in class_images.items():
        count = len(images)
        percentage = (count / total_images) * 100.0
        class_table.add_row(class_name, str(count), f"{percentage:.1f}%")

    console.print(class_table)
    console.print()

    all_images = [image for images in class_images.values() for image in images]
    widths = [image["width"] for image in all_images]
    heights = [image["height"] for image in all_images]
    file_sizes = [image["file_size"] for image in all_images]
    aspect_ratios = [image["aspect_ratio"] for image in all_images]
    brightnesses = [image["brightness"] for image in all_images]
    resolutions = [image["resolution"] for image in all_images]

    dimensions_table = Table(title="Image Dimensions (pixels)", title_justify="left")
    dimensions_table.add_column("Statistic", style="cyan")
    dimensions_table.add_column("Width", justify="right", style="green")
    dimensions_table.add_column("Height", justify="right", style="green")

    for stat_name, stat_function in [
        ("Min", numpy.min),
        ("Max", numpy.max),
        ("Mean", numpy.mean),
        ("Median", numpy.median),
        ("Std", numpy.std),
    ]:
        dimensions_table.add_row(
            stat_name,
            f"{stat_function(widths):.1f}",
            f"{stat_function(heights):.1f}",
        )

    console.print(dimensions_table)
    console.print()

    resolution_counts = Counter(resolutions)
    most_common = resolution_counts.most_common(1)[0]
    resolution, count = most_common
    percentage = (count / total_images) * 100
    console.print(
        f"[cyan]Most Common Resolution:[/cyan] [green]{resolution[0]}x{resolution[1]}[/green] "
        f"[yellow]({count} images, {percentage:.1f}%)[/yellow]"
    )
    console.print()

    aspect_table = Table(title="Aspect Ratio Statistics", title_justify="left")
    aspect_table.add_column("Statistic", style="cyan")
    aspect_table.add_column("Value", justify="right", style="green")

    for stat_name, stat_function in [
        ("Min", numpy.min),
        ("Max", numpy.max),
        ("Mean", numpy.mean),
        ("Median", numpy.median),
        ("Std", numpy.std),
    ]:
        aspect_table.add_row(stat_name, f"{stat_function(aspect_ratios):.3f}")

    console.print(aspect_table)
    console.print()

    brightness_table = Table(
        title="Brightness Statistics (0-255)", title_justify="left"
    )
    brightness_table.add_column("Statistic", style="cyan")
    brightness_table.add_column("Value", justify="right", style="green")

    for stat_name, stat_function in [
        ("Min", numpy.min),
        ("Max", numpy.max),
        ("Mean", numpy.mean),
        ("Median", numpy.median),
        ("Std", numpy.std),
    ]:
        brightness_table.add_row(stat_name, f"{stat_function(brightnesses):.1f}")

    console.print(brightness_table)
    console.print()

    file_size_table = Table(title="File Size Statistics", title_justify="left")
    file_size_table.add_column("Statistic", style="cyan")
    file_size_table.add_column("Size", justify="right", style="green")

    for stat_name, stat_function in [
        ("Min", numpy.min),
        ("Max", numpy.max),
        ("Mean", numpy.mean),
        ("Median", numpy.median),
        ("Std", numpy.std),
    ]:
        file_size_table.add_row(
            stat_name, utilities.format_file_size(stat_function(file_sizes))
        )

    console.print(file_size_table)
    console.print()

    color_table = Table(title="Mean RGB Values by Class", title_justify="left")
    color_table.add_column("Class", style="cyan")
    color_table.add_column("Red", justify="right", style="red")
    color_table.add_column("Green", justify="right", style="green")
    color_table.add_column("Blue", justify="right", style="blue")

    for class_name, images in class_images.items():
        if len(images) == 0:
            continue
        rgb_values = [image["mean_rgb"] for image in images]
        mean_rgb = tuple(numpy.mean([rgb[i] for rgb in rgb_values]) for i in range(3))
        r, g, b = mean_rgb
        color_table.add_row(class_name, f"{r:.1f}", f"{g:.1f}", f"{b:.1f}")

    console.print(color_table)
    console.print()


def analyze_sample_batch(pretrained: bool = False) -> None:
    rows = 3
    cols = 3

    try:
        data_loaders = data.get_data_loaders(pretrained=pretrained)
        train_loader, _ = data_loaders[0]

        batch_images, _ = next(iter(train_loader))
        batch_images: Tensor

        tensors = [batch_images[index] for index in range(len(batch_images))]

        figure = subplots.make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[
                f"Image {i + 1}" for i in range(min(len(tensors), rows * cols))
            ],
        )

        for index, tensor in enumerate(tensors[: rows * cols]):
            numpy_image = utilities.tensor_to_numpy(tensor)
            row = index // cols + 1
            col = index % cols + 1

            figure.add_trace(express.imshow(numpy_image).data[0], row=row, col=col)

        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        figure.update_layout(
            title={"text": "Sample PCOS Batch", "x": 0.5, "xanchor": "center"},
            height=rows * 300,
            width=cols * 300,
        )
        figure.show()
    except Exception as exception:
        print(f"Error during visualization: {exception}")


def analyze_training_graphs(
    experiment_directory: Path, save_graphs: bool = False
) -> None:
    results_path = experiment_directory / "results.json"

    with open(results_path, "r") as file:
        results: TrainingResults = json.load(file)

    metric_keys = ["loss", "accuracy", "precision", "recall", "f1_score", "roc_auc"]

    fold_data = []

    for fold_result in results["fold_results"]:
        fold_dict = {
            "fold": fold_result["fold"],
            "epochs": [epoch["epoch"] for epoch in fold_result["epoch_history"]],
            "confusion_matrix": fold_result["confusion_matrix"],
        }

        for metric in metric_keys:
            fold_dict[f"train_{metric}"] = [
                epoch[f"train_{metric}"] for epoch in fold_result["epoch_history"]
            ]
            fold_dict[f"validation_{metric}"] = [
                epoch[f"validation_{metric}"] for epoch in fold_result["epoch_history"]
            ]

        fold_data.append(fold_dict)

    graphs_directory = experiment_directory / "graphs"
    graphs_directory.mkdir(exist_ok=True)

    metrics_info = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy (%)"),
        ("precision", "Precision (%)"),
        ("recall", "Recall (%)"),
        ("f1_score", "F1 Score (%)"),
        ("roc_auc", "ROC-AUC (%)"),
    ]

    if save_graphs:
        for fold in fold_data:
            fold_number = fold["fold"]

            for metric_name, y_label in metrics_info:
                figure = Figure()

                figure.add_trace(
                    Scatter(
                        x=fold["epochs"],
                        y=fold[f"train_{metric_name}"],
                        name=f"Train {metric_name.capitalize()}",
                        mode="lines+markers",
                    )
                )

                figure.add_trace(
                    Scatter(
                        x=fold["epochs"],
                        y=fold[f"validation_{metric_name}"],
                        name=f"Validation {metric_name.capitalize()}",
                        mode="lines+markers",
                        line=dict(dash="dash"),
                    )
                )

                figure.update_layout(
                    title={
                        "text": f"Fold {fold_number} - {metric_name.replace('_', ' ').title()}",
                        "x": 0.5,
                        "xanchor": "center",
                    },
                    xaxis_title="Epoch",
                    yaxis_title=y_label,
                    height=400,
                    width=600,
                    showlegend=True,
                )

                output_path = graphs_directory / f"fold_{fold_number}_{metric_name}.png"
                figure.write_image(str(output_path))

            confusion_matrix = fold["confusion_matrix"]
            confusion_matrix_figure = Figure(
                data=Heatmap(
                    z=confusion_matrix,
                    x=["Predicted: 0", "Predicted: 1"],
                    y=["Actual: 0", "Actual: 1"],
                    text=confusion_matrix,
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=True,
                )
            )

            confusion_matrix_figure.update_layout(
                title={
                    "text": f"Fold {fold_number} - Confusion Matrix",
                    "x": 0.5,
                    "xanchor": "center",
                },
                height=400,
                width=500,
            )

            confusion_matrix_output_path = (
                graphs_directory / f"fold_{fold_number}_confusion_matrix.png"
            )

            confusion_matrix_figure.write_image(str(confusion_matrix_output_path))

    num_folds = len(fold_data)
    num_metrics = len(metrics_info)

    subplot_titles = []

    for fold in fold_data:
        for metric_name, _ in metrics_info:
            subplot_titles.append(
                f"Fold {fold['fold']} {metric_name.replace('_', ' ').title()}"
            )
        subplot_titles.append(f"Fold {fold['fold']} Confusion Matrix")
        specs = [
            [{"type": "xy"}] * num_metrics + [{"type": "heatmap"}]
            for _ in range(num_folds)
        ]

    figure = subplots.make_subplots(
        rows=num_folds,
        cols=num_metrics + 1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
        horizontal_spacing=0.025,
        specs=specs,
    )

    line_styles = {
        "train": {},
        "validation": {"dash": "dash"},
    }

    for fold_index, fold in enumerate(fold_data):
        row = fold_index + 1
        fold_number = fold["fold"]

        for col_index, (metric_name, y_label) in enumerate(metrics_info, start=1):
            figure.add_trace(
                Scatter(
                    x=fold["epochs"],
                    y=fold[f"train_{metric_name}"],
                    name=f"Fold {fold_number} Train {metric_name.title()}",
                    mode="lines+markers",
                    legendgroup=f"fold{fold_number}",
                    line=line_styles["train"],
                ),
                row=row,
                col=col_index,
            )

            figure.add_trace(
                Scatter(
                    x=fold["epochs"],
                    y=fold[f"validation_{metric_name}"],
                    name=f"Fold {fold_number} Validation {metric_name.title()}",
                    mode="lines+markers",
                    legendgroup=f"fold{fold_number}",
                    line=line_styles["validation"],
                ),
                row=row,
                col=col_index,
            )

            figure.update_xaxes(title_text="Epoch", row=row, col=col_index)
            figure.update_yaxes(title_text=y_label, row=row, col=col_index)

        confusion_matrix = fold["confusion_matrix"]

        figure.add_trace(
            Heatmap(
                z=confusion_matrix,
                x=["Predicted: 0", "Predicted: 1"],
                y=["Actual: 0", "Actual: 1"],
                text=confusion_matrix,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            ),
            row=row,
            col=num_metrics + 1,
        )

    figure.update_layout(
        title={
            "text": f"Training Metrics: {results['experiment_name']}",
            "x": 0.5,
            "xanchor": "center",
        },
        height=300 * num_folds,
        width=450 * num_metrics + 500,
        showlegend=True,
    )

    figure.show()


def analyze_results(experiment_directory: Path) -> None:
    console = Console(record=True)

    results_file = experiment_directory / "results.json"

    if not results_file.exists():
        console.print(f"[red]Error: results.json not found at {results_file}[/red]")
        return

    with open(results_file, "r") as file:
        results: TrainingResults = json.load(file)

    console.print("[bold cyan]Experiment Analysis[/bold cyan]\n")

    best_epochs: list[tuple[int, EpochMetrics]] = []

    for fold_result in results["fold_results"]:
        fold_number = fold_result["fold"]
        epoch_history = fold_result["epoch_history"]

        best_epoch = min(epoch_history, key=lambda epoch: epoch["validation_loss"])
        best_epochs.append((fold_number, best_epoch))

    for fold_number, best_epoch in best_epochs:
        table = Table(
            title=f"Fold {fold_number} - Best Epoch (Epoch {best_epoch['epoch']})",
            title_justify="left",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Train", justify="right", style="magenta")
        table.add_column("Validation", justify="right", style="green")

        table.add_row(
            "Loss",
            f"{best_epoch['train_loss']:.6f}",
            f"{best_epoch['validation_loss']:.6f}",
        )
        table.add_row(
            "Accuracy",
            f"{best_epoch['train_accuracy']:.2f}%",
            f"{best_epoch['validation_accuracy']:.2f}%",
        )
        table.add_row(
            "Precision",
            f"{best_epoch['train_precision']:.2f}%",
            f"{best_epoch['validation_precision']:.2f}%",
        )
        table.add_row(
            "Recall",
            f"{best_epoch['train_recall']:.2f}%",
            f"{best_epoch['validation_recall']:.2f}%",
        )
        table.add_row(
            "F1 Score",
            f"{best_epoch['train_f1_score']:.2f}%",
            f"{best_epoch['validation_f1_score']:.2f}%",
        )
        table.add_row(
            "ROC-AUC",
            f"{best_epoch['train_roc_auc']:.2f}%",
            f"{best_epoch['validation_roc_auc']:.2f}%",
        )

        console.print(table)
        console.print()

        fold_result = results["fold_results"][fold_number - 1]
        confusion_matrix = fold_result["confusion_matrix"]
        true_negative, false_positive = confusion_matrix[0]
        false_negative, true_positive = confusion_matrix[1]

        confusion_matrix_table = Table(
            title=f"Confusion Matrix - Fold {fold_number}",
            title_justify="left",
        )
        confusion_matrix_table.add_column("", style="cyan")
        confusion_matrix_table.add_column(
            "Predicted: 0", justify="center", style="magenta"
        )
        confusion_matrix_table.add_column(
            "Predicted: 1", justify="center", style="magenta"
        )
        confusion_matrix_table.add_row(
            "Actual: 0", str(true_negative), str(false_positive)
        )
        confusion_matrix_table.add_row(
            "Actual: 1", str(false_negative), str(true_positive)
        )

        console.print(confusion_matrix_table)
        console.print()

    best_fold_number, best_fold_epoch = min(
        best_epochs, key=lambda epoch: epoch[1]["validation_loss"]
    )

    summary_table = Table(
        title=f"[bold yellow]Best Overall: Fold {best_fold_number} - Epoch {best_fold_epoch['epoch']}[/bold yellow]",
        title_justify="left",
    )
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Train", justify="right", style="magenta")
    summary_table.add_column("Validation", justify="right", style="green")

    summary_table.add_row(
        "Loss",
        f"{best_fold_epoch['train_loss']:.6f}",
        f"{best_fold_epoch['validation_loss']:.6f}",
    )
    summary_table.add_row(
        "Accuracy",
        f"{best_fold_epoch['train_accuracy']:.2f}%",
        f"{best_fold_epoch['validation_accuracy']:.2f}%",
    )
    summary_table.add_row(
        "Precision",
        f"{best_fold_epoch['train_precision']:.2f}%",
        f"{best_fold_epoch['validation_precision']:.2f}%",
    )
    summary_table.add_row(
        "Recall",
        f"{best_fold_epoch['train_recall']:.2f}%",
        f"{best_fold_epoch['validation_recall']:.2f}%",
    )
    summary_table.add_row(
        "F1 Score",
        f"{best_fold_epoch['train_f1_score']:.2f}%",
        f"{best_fold_epoch['validation_f1_score']:.2f}%",
    )
    summary_table.add_row(
        "ROC-AUC",
        f"{best_fold_epoch['train_roc_auc']:.2f}%",
        f"{best_fold_epoch['validation_roc_auc']:.2f}%",
    )

    console.print(summary_table)
    console.print()

    best_fold_result = results["fold_results"][best_fold_number - 1]
    best_confusion_matrix = best_fold_result["confusion_matrix"]
    best_true_negative, best_false_positive = best_confusion_matrix[0]
    best_false_negative, best_true_positive = best_confusion_matrix[1]

    best_confusion_matrix_table = Table(
        title=f"[bold yellow]Confusion Matrix - Best Fold {best_fold_number}[/bold yellow]",
        title_justify="left",
    )
    best_confusion_matrix_table.add_column("", style="cyan")
    best_confusion_matrix_table.add_column(
        "Predicted: 0", justify="center", style="magenta"
    )
    best_confusion_matrix_table.add_column(
        "Predicted: 1", justify="center", style="magenta"
    )
    best_confusion_matrix_table.add_row(
        "Actual: 0", str(best_true_negative), str(best_false_positive)
    )
    best_confusion_matrix_table.add_row(
        "Actual: 1", str(best_false_negative), str(best_true_positive)
    )

    console.print(best_confusion_matrix_table)

    utilities.setup_logging(experiment_directory, "analysis")
    logging.info(console.export_text())
