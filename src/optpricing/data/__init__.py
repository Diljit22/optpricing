from __future__ import annotations

__doc__ = """
The `data` package provides a clean, unified interface for fetching, loading,
and saving all market and historical data required by the optpricing library.
"""

from .historical_manager import (
    load_historical_returns,
    save_historical_returns,
)
from .market_data_manager import (
    get_available_snapshot_dates,
    get_live_dividend_yield,
    get_live_option_chain,
    load_market_snapshot,
    save_market_snapshot,
)

__all__ = [
    "get_available_snapshot_dates",
    "get_live_dividend_yield",
    "get_live_option_chain",
    "load_historical_returns",
    "load_market_snapshot",
    "save_historical_returns",
    "save_market_snapshot",
]
