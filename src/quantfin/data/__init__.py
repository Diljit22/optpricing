# src/quantfin/data/__init__.py

__doc__ = """
The `data` package provides a clean, unified interface for fetching, loading,
and saving all market and historical data required by the quantfin library.
"""

# --- Functions for option chain snapshots ---
from .market_data_manager import (
    get_live_option_chain,
    save_market_snapshot,
    load_market_snapshot,
    get_available_snapshot_dates,
)

# --- Functions for historical time-series data ---
from .historical_manager import (
    save_historical_returns,
    load_historical_returns,
)

# --- Define the public API for the data package ---
__all__ = [
    "get_live_option_chain",
    "save_market_snapshot",
    "load_market_snapshot",
    "get_available_snapshot_dates",
    "save_historical_returns",
    "load_historical_returns",
]