import json
from pathlib import Path

from plotly import express, subplots
from plotly.graph_objects import Figure, Scatter
from torch import Tensor

from modules import data, utilities


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
