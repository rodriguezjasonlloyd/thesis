import json
from collections import Counter
from pathlib import Path

import numpy
from PIL import Image
from plotly import express, subplots
from plotly.graph_objects import Figure, Scatter
from rich.console import Console
from rich.table import Table
from torch import Tensor

from modules import data, utilities

console = Console()


def analyze_descriptive(root: Path = data.get_data_root_path()) -> None:
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

        batch_images, batch_labels = next(iter(train_loader))
        batch_images: Tensor
        batch_labels: Tensor

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


def show_training_graphs(experiment_directory: Path, save_graphs: bool = False) -> None:
    results_path = experiment_directory / "results.json"

    with open(results_path, "r") as file:
        results = json.load(file)

    metric_keys = ["loss", "accuracy", "precision", "recall", "f1_score", "roc_auc"]

    fold_data = []

    for fold_result in results["fold_results"]:
        fold_dict = {
            "fold": fold_result["fold"],
            "epochs": [epoch["epoch"] for epoch in fold_result["epoch_history"]],
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

    num_folds = len(fold_data)
    num_metrics = len(metrics_info)

    subplot_titles = [
        f"Fold {fold['fold']} {metric_name.replace('_', ' ').title()}"
        for fold in fold_data
        for metric_name, _ in metrics_info
    ]

    figure = subplots.make_subplots(
        rows=num_folds,
        cols=num_metrics,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
        horizontal_spacing=0.025,
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

    figure.update_layout(
        title={
            "text": f"Training Metrics: {results['experiment_name']}",
            "x": 0.5,
            "xanchor": "center",
        },
        height=300 * num_folds,
        width=450 * num_metrics,
        showlegend=True,
    )

    figure.show()
