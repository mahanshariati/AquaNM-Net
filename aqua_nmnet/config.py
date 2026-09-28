from pathlib import Path

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------
# LSUI dataset
# ---------------------------------------------------------------------
LSUI_DIR = DATA_DIR / "LSUI"
LSUI_INPUT_DIR = LSUI_DIR / "input"
LSUI_TARGET_DIR = LSUI_DIR / "GT"

# ---------------------------------------------------------------------
# Dataset split files
# ---------------------------------------------------------------------
SPLIT_DIR = DATA_DIR / "splits" / "lsui"
TRAIN_SPLIT_FILE = SPLIT_DIR / "train.txt"
VAL_SPLIT_FILE = SPLIT_DIR / "val.txt"
TEST_SPLIT_FILE = SPLIT_DIR / "test.txt"

# ---------------------------------------------------------------------
# NAS search subset
# ---------------------------------------------------------------------
SEARCH_DATA_DIR = DATA_DIR / "search_subset"
SEARCH_INPUT_DIR = SEARCH_DATA_DIR / "input"
SEARCH_TARGET_DIR = SEARCH_DATA_DIR / "target"