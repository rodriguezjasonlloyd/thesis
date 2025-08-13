from pathlib import Path

SEED = 42

DATA_DIR = Path("data")

IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.3

MAX_SAMPLES = 10

SPLIT_FILE = Path("data_splits.json")

RESULTS_DIR_NAME = "results"
