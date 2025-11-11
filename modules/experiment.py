from dataclasses import dataclass, field
from json import dump as json_dump
from pathlib import Path
from tomllib import load as toml_load
from typing import Any, Callable, Iterator

from torch.nn import CrossEntropyLoss, Module, Parameter
from torch.optim import AdamW, Optimizer

from modules.data import get_data_loaders, get_data_root_path
from modules.model import build_model
from modules.trainer import seed_all, train_model


@dataclass
class DataConfig:
    root: str = "data"
    k_folds: int = 5
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 2
    max_items_per_class: int = 0


@dataclass
class ModelConfig:
    pretrained: bool = False
    with_fsa: bool = False


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class TrainingConfig:
    num_epochs: int = 30
    patience: int = 5
    min_delta: float = 1e-3


@dataclass
class ExperimentConfig:
    name: str
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def parse_config(config_path: Path) -> ExperimentConfig:
    with open(config_path, "rb") as file:
        raw_config = toml_load(file)

    name = raw_config.get("name", "unnamed_experiment")
    seed = raw_config.get("seed", 42)

    data_dict = raw_config.get("data", {})
    data_config = DataConfig(
        root=data_dict.get("root", "data"),
        k_folds=data_dict.get("k_folds", 5),
        batch_size=data_dict.get("batch_size", 32),
        shuffle=data_dict.get("shuffle", True),
        num_workers=data_dict.get("num_workers", 2),
        max_items_per_class=data_dict.get("max_items_per_class", 0),
    )

    model_dict = raw_config.get("model", {})
    model_config = ModelConfig(
        pretrained=model_dict.get("pretrained", False),
        with_fsa=model_dict.get("with_fsa", False),
    )

    optimizer_dict = raw_config.get("optimizer", {})
    optimizer_config = OptimizerConfig(
        learning_rate=optimizer_dict.get("learning_rate", 1e-3),
        weight_decay=optimizer_dict.get("weight_decay", 1e-4),
    )

    training_dict = raw_config.get("training", {})
    training_config = TrainingConfig(
        num_epochs=training_dict.get("num_epochs", 30),
        patience=training_dict.get("patience", 5),
        min_delta=training_dict.get("min_delta", 1e-3),
    )

    return ExperimentConfig(
        name=name,
        seed=seed,
        data=data_config,
        model=model_config,
        optimizer=optimizer_config,
        training=training_config,
    )


def create_get_model(model_config: ModelConfig) -> Callable[[], Module]:
    def get_model() -> Module:
        return build_model(
            pretrained=model_config.pretrained, with_fsa=model_config.with_fsa
        )

    return get_model


def get_criterion() -> Module:
    return CrossEntropyLoss()


def create_get_optimizer(
    optimizer_config: OptimizerConfig,
) -> Callable[[Iterator[Parameter]], Optimizer]:
    def get_optimizer(model_parameters: Iterator[Parameter]) -> Optimizer:
        return AdamW(
            model_parameters,
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
        )

    return get_optimizer


def run_experiment(experiment_directory: Path) -> dict[str, Any]:
    if not experiment_directory.exists():
        raise FileNotFoundError(
            f"Experiment directory not found: {experiment_directory}"
        )

    if not experiment_directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {experiment_directory}")

    config_path = experiment_directory / "config.toml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file 'config.toml' not found in {experiment_directory}"
        )

    config = parse_config(config_path)

    print(f"Running experiment: {experiment_directory} - {config.name}")

    seed_all(config.seed)

    fold_loaders = get_data_loaders(
        root=get_data_root_path(config.data.root),
        pretrained=config.model.pretrained,
        k_folds=config.data.k_folds,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        max_items_per_class=config.data.max_items_per_class,
    )

    get_model = create_get_model(config.model)
    get_optimizer = create_get_optimizer(config.optimizer)
    criterion = CrossEntropyLoss()

    results = train_model(
        experiment_directory=experiment_directory,
        get_model=get_model,
        get_optimizer=get_optimizer,
        criterion=criterion,
        fold_loaders=fold_loaders,
        num_epochs=config.training.num_epochs,
        patience=config.training.patience,
        min_delta=config.training.min_delta,
    )

    results["experiment_name"] = config.name
    results["experiment_directory"] = str(experiment_directory)
    results["config"] = {
        "seed": config.seed,
        "data": config.data.__dict__,
        "model": config.model.__dict__,
        "optimizer": config.optimizer.__dict__,
        "training": config.training.__dict__,
    }

    results_path = experiment_directory / "results.json"
    results_to_save = {
        "experiment_name": results["experiment_name"],
        "experiment_directory": results["experiment_directory"],
        "config": results["config"],
        "average_validation_loss": results["average_validation_loss"],
        "average_validation_accuracy": results["average_validation_accuracy"],
        "k_folds": results["k_folds"],
        "fold_results": [
            {
                "fold": fold["fold"],
                "best_validation_loss": fold["best_validation_loss"],
                "best_validation_accuracy": fold["best_validation_accuracy"],
                "epoch_history": fold["epoch_history"],
            }
            for fold in results["fold_results"]
        ],
    }

    with open(results_path, "w") as f:
        json_dump(results_to_save, f, indent=2)

    print(f"Results saved to {results_path}")

    return results
