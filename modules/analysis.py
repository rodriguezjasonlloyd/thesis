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

    fold_data = []

    for fold_result in results["fold_results"]:
        epochs = [epoch["epoch"] for epoch in fold_result["epoch_history"]]
        train_loss = [epoch["train_loss"] for epoch in fold_result["epoch_history"]]
        validation_loss = [
            epoch["validation_loss"] for epoch in fold_result["epoch_history"]
        ]
        train_accuracy = [
            epoch["train_accuracy"] for epoch in fold_result["epoch_history"]
        ]
        validation_accuracy = [
            epoch["validation_accuracy"] for epoch in fold_result["epoch_history"]
        ]
        train_precision = [
            epoch["train_precision"] for epoch in fold_result["epoch_history"]
        ]
        validation_precision = [
            epoch["validation_precision"] for epoch in fold_result["epoch_history"]
        ]
        train_recall = [epoch["train_recall"] for epoch in fold_result["epoch_history"]]
        validation_recall = [
            epoch["validation_recall"] for epoch in fold_result["epoch_history"]
        ]
        train_f1_score = [
            epoch["train_f1_score"] for epoch in fold_result["epoch_history"]
        ]
        validation_f1_score = [
            epoch["validation_f1_score"] for epoch in fold_result["epoch_history"]
        ]
        train_roc_auc = [
            epoch["train_roc_auc"] for epoch in fold_result["epoch_history"]
        ]
        validation_roc_auc = [
            epoch["validation_roc_auc"] for epoch in fold_result["epoch_history"]
        ]
        fold_data.append(
            {
                "fold": fold_result["fold"],
                "epochs": epochs,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
                "train_precision": train_precision,
                "validation_precision": validation_precision,
                "train_recall": train_recall,
                "validation_recall": validation_recall,
                "train_f1_score": train_f1_score,
                "validation_f1_score": validation_f1_score,
                "train_roc_auc": train_roc_auc,
                "validation_roc_auc": validation_roc_auc,
            }
        )

    graphs_directory = experiment_directory / "graphs"
    graphs_directory.mkdir(exist_ok=True)

    metrics = [
        ("loss", "Loss", "train_loss", "validation_loss"),
        ("accuracy", "Accuracy (%)", "train_accuracy", "validation_accuracy"),
        ("precision", "Precision (%)", "train_precision", "validation_precision"),
        ("recall", "Recall (%)", "train_recall", "validation_recall"),
        ("f1_score", "F1 Score (%)", "train_f1_score", "validation_f1_score"),
        ("roc_auc", "ROC-AUC (%)", "train_roc_auc", "validation_roc_auc"),
    ]

    if save_graphs:
        for fold in fold_data:
            fold_number = fold["fold"]

            for metric_info in metrics:
                metric_name, y_label, train_key, validation_key = metric_info

                figure = Figure()

                figure.add_trace(
                    Scatter(
                        x=fold["epochs"],
                        y=fold[train_key],
                        name=f"Train {metric_name.capitalize()}",
                        mode="lines+markers",
                    )
                )

                figure.add_trace(
                    Scatter(
                        x=fold["epochs"],
                        y=fold[validation_key],
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

    figure = subplots.make_subplots(
        rows=num_folds,
        cols=6,
        subplot_titles=[
            title
            for fold in fold_data
            for title in (
                f"Fold {fold['fold']} Loss",
                f"Fold {fold['fold']} Accuracy",
                f"Fold {fold['fold']} Precision",
                f"Fold {fold['fold']} Recall",
                f"Fold {fold['fold']} F1",
                f"Fold {fold['fold']} ROC-AUC",
            )
        ],
        vertical_spacing=0.05,
        horizontal_spacing=0.025,
    )

    for index, fold in enumerate(fold_data):
        row = index + 1
        fold_number = fold["fold"]

        figure.add_trace(
            Scatter(
                x=fold["epochs"],
                y=fold["train_loss"],
                name=f"Fold {fold_number} Train Loss",
                mode="lines+markers",
                legendgroup=f"fold{fold_number}",
            ),
            row=row,
            col=1,
        )

        figure.add_trace(
            Scatter(
                x=fold["epochs"],
                y=fold["validation_loss"],
                name=f"Fold {fold_number} Validation Loss",
                mode="lines+markers",
                line=dict(dash="dash"),
                legendgroup=f"fold{fold_number}",
            ),
            row=row,
            col=1,
        )

        figure.add_trace(
            Scatter(
                x=fold["epochs"],
                y=fold["train_accuracy"],
                name=f"Fold {fold_number} Train Accuracy",
                mode="lines+markers",
                legendgroup=f"fold{fold_number}",
            ),
            row=row,
            col=2,
        )

        figure.add_trace(
            Scatter(
                x=fold["epochs"],
                y=fold["validation_accuracy"],
                name=f"Fold {fold_number} Validation Accuracy",
                mode="lines+markers",
                line=dict(dash="dash"),
                legendgroup=f"fold{fold_number}",
            ),
            row=row,
            col=2,
        )

        figure.add_trace(
            Scatter(
                x=fold["epochs"],
                y=fold["train_precision"],
                name=f"Fold {fold_number} Train Precision",
                mode="lines+markers",
                legendgroup=f"fold{fold_number}",
            ),
            row=row,
            col=3,
        )

        figure.add_trace(
            Scatter(
                x=fold["epochs"],
                y=fold["validation_precision"],
                name=f"Fold {fold_number} Validation Precision",
                mode="lines+markers",
                line=dict(dash="dash"),
                legendgroup=f"fold{fold_number}",
            ),
            row=row,
            col=3,
        )

        figure.add_trace(
            Scatter(
                x=fold["epochs"],
                y=fold["train_recall"],
                name=f"Fold {fold_number} Train Recall",
                mode="lines+markers",
                legendgroup=f"fold{fold_number}",
            ),
            row=row,
            col=4,
        )

        figure.add_trace(
            Scatter(
                x=fold["epochs"],
                y=fold["validation_recall"],
                name=f"Fold {fold_number} Validation Recall",
                mode="lines+markers",
                line=dict(dash="dash"),
                legendgroup=f"fold{fold_number}",
            ),
            row=row,
            col=4,
        )

        figure.add_trace(
            Scatter(
                x=fold["epochs"],
                y=fold["train_f1_score"],
                name=f"Fold {fold_number} Train F1",
                mode="lines+markers",
                legendgroup=f"fold{fold_number}",
            ),
            row=row,
            col=5,
        )

        figure.add_trace(
            Scatter(
                x=fold["epochs"],
                y=fold["validation_f1_score"],
                name=f"Fold {fold_number} Validation F1",
                mode="lines+markers",
                line=dict(dash="dash"),
                legendgroup=f"fold{fold_number}",
            ),
            row=row,
            col=5,
        )

        figure.add_trace(
            Scatter(
                x=fold["epochs"],
                y=fold["train_roc_auc"],
                name=f"Fold {fold_number} Train ROC-AUC",
                mode="lines+markers",
                legendgroup=f"fold{fold_number}",
                line=dict(dash="dot"),
            ),
            row=row,
            col=6,
        )

        figure.add_trace(
            Scatter(
                x=fold["epochs"],
                y=fold["validation_roc_auc"],
                name=f"Fold {fold_number} Validation ROC-AUC",
                mode="lines+markers",
                line=dict(dash="dashdot"),
                legendgroup=f"fold{fold_number}",
            ),
            row=row,
            col=6,
        )

        figure.update_xaxes(title_text="Epoch", row=row, col=1)
        figure.update_xaxes(title_text="Epoch", row=row, col=2)
        figure.update_xaxes(title_text="Epoch", row=row, col=3)
        figure.update_xaxes(title_text="Epoch", row=row, col=4)
        figure.update_xaxes(title_text="Epoch", row=row, col=5)
        figure.update_xaxes(title_text="Epoch", row=row, col=6)
        figure.update_yaxes(title_text="Loss", row=row, col=1)
        figure.update_yaxes(title_text="Accuracy (%)", row=row, col=2)
        figure.update_yaxes(title_text="Precision (%)", row=row, col=3)
        figure.update_yaxes(title_text="Recall (%)", row=row, col=4)
        figure.update_yaxes(title_text="Score (%)", row=row, col=5)
        figure.update_yaxes(title_text="Score (%)", row=row, col=6)

    figure.update_layout(
        title={
            "text": f"Training Metrics: {results['experiment_name']}",
            "x": 0.5,
            "xanchor": "center",
        },
        height=300 * num_folds,
        width=450 * len(metrics),
        showlegend=True,
    )

    figure.show()
