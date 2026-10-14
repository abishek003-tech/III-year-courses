"""
src/config.py
=============
Loads and exposes all project configuration from configs/params.yaml.
Every other module imports settings from here – single source of truth.

Author : Abi (Abhishek S.)
College: KGiSL Institute of Technology
Dept   : Artificial Intelligence and Machine Learning
"""

import os
import yaml
from pathlib import Path

# ── Locate the project root (parent of src/) ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH  = PROJECT_ROOT / "configs" / "params.yaml"


def load_params() -> dict:
    """Read params.yaml and return the full config dictionary."""
    with open(PARAMS_PATH, "r") as fh:
        return yaml.safe_load(fh)


# Load once at import time
_cfg = load_params()

# ── Data settings ────────────────────────────────────────────────────────────
DATA_URL       = _cfg["data"]["url"]
RAW_ZIP        = PROJECT_ROOT / _cfg["data"]["raw_zip"]
CSV_FILE       = PROJECT_ROOT / _cfg["data"]["csv_file"]
CSV_SEPARATOR  = _cfg["data"]["separator"]

# ── Feature settings ─────────────────────────────────────────────────────────
DROP_COLUMNS   = _cfg["features"]["drop_columns"]
TARGET_COL     = _cfg["features"]["target"]
TEST_SIZE      = _cfg["features"]["test_size"]
RANDOM_STATE   = _cfg["features"]["random_state"]

# ── Model hyperparameters ────────────────────────────────────────────────────
MODEL_PARAMS   = _cfg["models"]

# ── Selection criterion ──────────────────────────────────────────────────────
SELECTION_METRIC    = _cfg["selection"]["metric"]
SELECTION_DIRECTION = _cfg["selection"]["direction"]

# ── Artifact paths ───────────────────────────────────────────────────────────
MODEL_DIR           = PROJECT_ROOT / _cfg["paths"]["model_dir"]
BEST_MODEL_PATH     = PROJECT_ROOT / _cfg["paths"]["best_model"]
PREPROCESSOR_PATH   = PROJECT_ROOT / _cfg["paths"]["preprocessor"]
METRICS_JSON        = PROJECT_ROOT / _cfg["paths"]["metrics_json"]
METRICS_CSV         = PROJECT_ROOT / _cfg["paths"]["metrics_csv"]
CHARTS_DIR          = PROJECT_ROOT / _cfg["paths"]["charts_dir"]
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / _cfg["paths"]["mlflow_tracking_uri"])
MLFLOW_EXPERIMENT   = _cfg["paths"]["mlflow_experiment"]

# ── Ensure output directories exist ─────────────────────────────────────────
for _dir in [MODEL_DIR, CHARTS_DIR, METRICS_JSON.parent]:
    _dir.mkdir(parents=True, exist_ok=True)
