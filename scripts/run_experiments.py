import argparse
import json
from pathlib import Path

from pcom import config as global_config
from pcom.train import train_model
from pcom.utils import set_seed


def run_experiment(exp_dir: Path):
    """Run a single experiment given its folder."""
    config_path = exp_dir / "config.json"

    if not config_path.exists():
        print(f"No config.json found in {exp_dir}")
        return

    with open(config_path, "r") as f:
        exp_config = json.load(f)

    exp_config["dataset"] = {
        "train_dir": str(global_config.DATA_DIR / "infected"),
        "val_dir": str(global_config.DATA_DIR / "noninfected"),
        "image_size": global_config.IMAGE_SIZE,
        "normalize_mean": global_config.NORMALIZE_MEAN,
        "normalize_std": global_config.NORMALIZE_STD,
    }
    exp_config["output_dir"] = str(exp_dir / global_config.RESULTS_DIR_NAME)

    Path(exp_config["output_dir"]).mkdir(parents=True, exist_ok=True)

    set_seed(global_config.SEED)

    try:
        train_model(exp_config)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")


def main():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument(
        "--exp", type=str, help="Run a specific experiment by folder name"
    )
    args = parser.parse_args()

    exp_root = Path("experiments")

    if args.all:
        for exp_dir in sorted(exp_root.iterdir()):
            if (exp_dir / "config.json").exists():
                run_experiment(exp_dir)
    elif args.exp:
        exp_dir = exp_root / args.exp
        if not exp_dir.exists():
            print(f"Experiment folder {args.exp} does not exist.")
            return
        run_experiment(exp_dir)
    else:
        print("Please provide either --all or --exp <name>")


if __name__ == "__main__":
    main()
