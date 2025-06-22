from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE_PATH = PROJECT_ROOT / "config.yaml"
if not CONFIG_FILE_PATH.exists():
    # Create a default config if it doesn't exist
    default_config = {
        "polygon_api_key": "YOUR_POLYGON_API_KEY_HERE",
        "data_directory": "data",
        "artifacts_directory": "artifacts",
    }
    with open(CONFIG_FILE_PATH, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False)
    print(f"Created default config file at: {CONFIG_FILE_PATH}")

with open(CONFIG_FILE_PATH) as f:
    _config = yaml.safe_load(f)

# API Key Management
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", _config.get("polygon_api_key"))

# Directory Management
DATA_DIR = PROJECT_ROOT / _config.get("data_directory", "data")
ARTIFACTS_DIR = PROJECT_ROOT / _config.get("artifacts_directory", "artifacts")

# Input Data Directories
MARKET_SNAPSHOT_DIR = DATA_DIR / "market_data_snapshots"
HISTORICAL_DIR = DATA_DIR / "historical_data"

# Output Artifact Directories
BACKTEST_LOGS_DIR = ARTIFACTS_DIR / "backtest_logs"
CALIBRATED_PARAMS_DIR = ARTIFACTS_DIR / "calibrated_params"
CALIBRATION_LOGS_DIR = ARTIFACTS_DIR / "calibration_logs"

# Ensure all directories exist
for path in [
    DATA_DIR,
    ARTIFACTS_DIR,
    MARKET_SNAPSHOT_DIR,
    HISTORICAL_DIR,
    BACKTEST_LOGS_DIR,
    CALIBRATED_PARAMS_DIR,
    CALIBRATION_LOGS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
