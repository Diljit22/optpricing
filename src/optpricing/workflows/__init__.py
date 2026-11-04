from __future__ import annotations

__doc__ = """
The `workflows` package provides high-level classes that orchestrate
multi-step processes like daily model calibration, historical backtesting and
per-ticker tuning of the calibration settings.
"""

from .backtest_workflow import BacktestWorkflow
from .daily_workflow import DailyWorkflow
from .tuning import (
    CANDIDATE_SETTINGS,
    TuningResult,
    load_tuned_settings,
    load_tuning,
    tune_ticker,
    tuning_path,
)

__all__ = [
    "BacktestWorkflow",
    "CANDIDATE_SETTINGS",
    "DailyWorkflow",
    "TuningResult",
    "load_tuned_settings",
    "load_tuning",
    "tune_ticker",
    "tuning_path",
]
