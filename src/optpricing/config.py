from __future__ import annotations

import os
from pathlib import Path

import yaml

__doc__ = """
Resolve the project root, read the optional ``config.yaml`` and expose the data
and artifact directories.

The root is ``$OPTPRICING_HOME`` if set, otherwise the current working directory.
Running from a source checkout therefore behaves as before, while an installed
package no longer writes ``config.yaml`` or data folders next to its own modules
in ``site-packages``. Nothing is created at import time; writers create the
directory they need (see :func:`ensure_directories`).
"""

DEFAULT_CONFIG: dict[str, object] = {
    "live_data_provider": "yfinance",
    "data_directory": "data",
    "artifacts_directory": "artifacts",
}

PROJECT_ROOT = Path(os.environ.get("OPTPRICING_HOME", Path.cwd())).resolve()
CONFIG_FILE_PATH = PROJECT_ROOT / "config.yaml"


def _load_config(path: Path) -> dict:
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    with open(path) as f:
        loaded = yaml.safe_load(f) or {}
    return {**DEFAULT_CONFIG, **loaded}


_config = _load_config(CONFIG_FILE_PATH)

# Directory Management
DATA_DIR = PROJECT_ROOT / str(_config.get("data_directory", "data"))
ARTIFACTS_DIR = PROJECT_ROOT / str(_config.get("artifacts_directory", "artifacts"))

# Input Data Directories
MARKET_SNAPSHOT_DIR = DATA_DIR / "market_data_snapshots"
HISTORICAL_DIR = DATA_DIR / "historical_data"
LIVE_QUOTES_DIR = DATA_DIR / "live_quotes"

# Output Artifact Directories
BACKTEST_LOGS_DIR = ARTIFACTS_DIR / "backtest_logs"
CALIBRATED_PARAMS_DIR = ARTIFACTS_DIR / "calibrated_params"
CALIBRATION_LOGS_DIR = ARTIFACTS_DIR / "calibration_logs"
TUNING_DIR = ARTIFACTS_DIR / "tuning"

ALL_DIRS = [
    DATA_DIR,
    ARTIFACTS_DIR,
    MARKET_SNAPSHOT_DIR,
    HISTORICAL_DIR,
    LIVE_QUOTES_DIR,
    BACKTEST_LOGS_DIR,
    CALIBRATED_PARAMS_DIR,
    CALIBRATION_LOGS_DIR,
    TUNING_DIR,
]


def ensure_directories() -> None:
    """Create all data and artifact directories (idempotent)."""
    for path in ALL_DIRS:
        path.mkdir(parents=True, exist_ok=True)
