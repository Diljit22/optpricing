from __future__ import annotations

from pathlib import Path

import yaml

__doc__ = """
Load or create the project's config.yaml and ensure all data/artifact directories exist.
"""


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE_PATH = PROJECT_ROOT / "config.yaml"

if not CONFIG_FILE_PATH.exists():
    print(f"Config file not found. Creating default config.yaml at {CONFIG_FILE_PATH}")
    default_config = {
        "live_data_provider": "yfinance",
        "data_directory": "data",
        "artifacts_directory": "artifacts",
    }
    with open(CONFIG_FILE_PATH, "w") as f:
        yaml.dump(
            default_config,
            f,
            default_flow_style=False,
            sort_keys=False,
        )

with open(CONFIG_FILE_PATH) as f:
    _config = yaml.safe_load(f) or {}

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
ALL_DIRS = [
    DATA_DIR,
    ARTIFACTS_DIR,
    MARKET_SNAPSHOT_DIR,
    HISTORICAL_DIR,
    BACKTEST_LOGS_DIR,
    CALIBRATED_PARAMS_DIR,
    CALIBRATION_LOGS_DIR,
]
for path in ALL_DIRS:
    path.mkdir(parents=True, exist_ok=True)
