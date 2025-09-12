import os
from pathlib import Path
from loguru import logger

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT_PATH = Path(os.getenv("ODYSSEY_STORAGE_PATH") or PROJ_ROOT).resolve()

DATA_DIR = STORAGE_ROOT_PATH / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = STORAGE_ROOT_PATH / "models"
REPORTS_DIR = STORAGE_ROOT_PATH / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
