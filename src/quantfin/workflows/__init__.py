from __future__ import annotations

__doc__ = """
The `workflows` package provides high-level classes that orchestrate
multi-step processes like daily model calibration and historical backtesting.
"""

from .backtest_workflow import BacktestWorkflow
from .daily_workflow import DailyWorkflow

__all__ = [
    "BacktestWorkflow",
    "DailyWorkflow",
]
