import dataclasses
import json
import logging
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from torch.nn import BCEWithLogitsLoss, Module, Parameter
from torch.optim import AdamW, Optimizer

from modules import data, model, trainer, utilities
from modules.preprocessing import PreprocessingMode


@dataclass
class DataConfig:
    root: str = "data"
    k_folds: int = 5
    batch_size: int = 32
    num_workers: int = 2
    max_items_per_class: int = 0
    augmented: bool = False
    preprocessing: PreprocessingMode = PreprocessingMode.NONE


@dataclass
class ModelConfig:
    architecture: str = "convnext"
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
    name: str = "Unnammed Experiment"
    seed: int = 42
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)


def parse_config(config_path: Path) -> ExperimentConfig:
    with open(config_path, "rb") as file:
        raw_config = tomllib.load(file)

    config_kwargs = {}

    if "name" in raw_config:
        config_kwargs["name"] = raw_config["name"]
    if "seed" in raw_config:
        config_kwargs["seed"] = raw_config["seed"]

    if "data" in raw_config:
        data_dict = raw_config["data"].copy()

        if "preprocessing" in data_dict:
            data_dict["preprocessing"] = PreprocessingMode(data_dict["preprocessing"])

        config_kwargs["data"] = DataConfig(**data_dict)

    if "model" in raw_config:
        config_kwargs["model"] = ModelConfig(**raw_config["model"])

    if "optimizer" in raw_config:
        config_kwargs["optimizer"] = OptimizerConfig(**raw_config["optimizer"])

    if "training" in raw_config:
        config_kwargs["training"] = TrainingConfig(**raw_config["training"])

    return ExperimentConfig(**config_kwargs)


def create_get_model(model_config: ModelConfig) -> Callable[[], Module]:
    from modules.model import BaseCNN

    def get_model() -> Module:
        if model_config.architecture == "base":
            return BaseCNN()
        elif model_config.architecture == "convnext":
            return model.build_model(
                pretrained=model_config.pretrained, with_fsa=model_config.with_fsa
            )
        else:
            raise ValueError(f"Unknown architecture: {model_config.architecture}")

    return get_model


def get_criterion() -> Module:
    return BCEWithLogitsLoss()


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


def run_experiment(experiment_directory: Path) -> None:
    utilities.setup_logging(experiment_directory)

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

    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Experiment: {config.name}\n"
    )

    utilities.seed_all(config.seed)

    fold_loaders = data.get_data_loaders(
        root=data.get_data_root_path(config.data.root),
        pretrained=config.model.pretrained,
        augmented=config.data.augmented,
        preprocessing=config.data.preprocessing,
        k_folds=config.data.k_folds,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        max_items_per_class=config.data.max_items_per_class,
    )

    get_model = create_get_model(config.model)
    get_optimizer = create_get_optimizer(config.optimizer)
    criterion = get_criterion()

    try:
        train_results = trainer.train_model(
            experiment_directory=experiment_directory,
            get_model=get_model,
            get_optimizer=get_optimizer,
            criterion=criterion,
            fold_loaders=fold_loaders,
            num_epochs=config.training.num_epochs,
            patience=config.training.patience,
            min_delta=config.training.min_delta,
        )
    except KeyboardInterrupt:
        logging.info(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Experiment Interrupted\n"
        )
        return None

    results = {
        "experiment_name": config.name,
        "experiment_directory": str(experiment_directory),
        "config": {
            "seed": config.seed,
            "data": utilities.dataclass_to_dict(config.data),
            "model": utilities.dataclass_to_dict(config.model),
            "optimizer": utilities.dataclass_to_dict(config.optimizer),
            "training": utilities.dataclass_to_dict(config.training),
        },
        "k_folds": train_results["k_folds"],
        "average_validation_loss": train_results["average_validation_loss"],
        "average_validation_accuracy": train_results["average_validation_accuracy"],
        "fold_results": train_results["fold_results"],
    }

    results_path = experiment_directory / "results.json"

    with open(results_path, "w") as file:
        json.dump(results, file, indent=2)

    logging.info(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Experiment Finished\n"
    )
