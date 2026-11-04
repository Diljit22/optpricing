from __future__ import annotations

__doc__ = """
The `data` package provides a clean, unified interface for fetching, loading,
and saving all market and historical data required by the optpricing library.
"""

from .historical_manager import (
    get_live_risk_free_rate,
    historical_returns_if_available,
    load_historical_prices,
    load_historical_returns,
    load_risk_free_rate,
    save_historical_returns,
    save_risk_free_history,
    trailing_dividend_yield,
)
from .market_data_manager import (
    compact_snapshot,
    get_available_snapshot_dates,
    get_live_dividend_yield,
    get_live_option_chain,
    is_v2_snapshot,
    liquid_option_mask,
    load_market_snapshot,
    save_market_snapshot,
    snapshot_market_params,
    upgrade_snapshot,
)
from .stream import option_contract_symbols, record_live_quotes

__all__ = [
    "compact_snapshot",
    "get_available_snapshot_dates",
    "get_live_dividend_yield",
    "get_live_option_chain",
    "get_live_risk_free_rate",
    "historical_returns_if_available",
    "is_v2_snapshot",
    "liquid_option_mask",
    "load_historical_prices",
    "load_historical_returns",
    "load_market_snapshot",
    "load_risk_free_rate",
    "option_contract_symbols",
    "record_live_quotes",
    "save_historical_returns",
    "save_market_snapshot",
    "save_risk_free_history",
    "snapshot_market_params",
    "trailing_dividend_yield",
    "upgrade_snapshot",
]
