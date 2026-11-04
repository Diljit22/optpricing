from __future__ import annotations

__doc__ = """
The `calibration` package provides tools for fitting financial models to
market data, estimating parameters from historical data, and calculating
implied volatility surfaces.
"""

from .calibrator import CalibrationResult, Calibrator
from .fit_jump_parameters import fit_jump_params_from_history
from .fit_market_params import (
    find_atm_options,
    fit_rate_and_dividend,
    implied_dividend_yield,
    implied_forward_curve,
)
from .iv_surface import VolatilitySurface
from .market_filters import (
    DEFAULT_SETTINGS,
    CalibrationSettings,
    LiquidityFilter,
    annotate_chain,
    filter_options,
)
from .metrics import bucket_metrics, evaluate_model, residual_table, summarize_residuals
from .technique_selector import select_fastest_technique
from .vectorized_bsm_iv import BSMIVSolver
from .vectorized_integration_iv import VectorizedIntegrationIVSolver
from .vectorized_pricer import price_options_vectorized, supports_vectorized_pricing

__all__ = [
    "BSMIVSolver",
    "CalibrationResult",
    "CalibrationSettings",
    "Calibrator",
    "DEFAULT_SETTINGS",
    "LiquidityFilter",
    "VectorizedIntegrationIVSolver",
    "VolatilitySurface",
    "annotate_chain",
    "bucket_metrics",
    "evaluate_model",
    "filter_options",
    "fit_jump_params_from_history",
    "fit_rate_and_dividend",
    "find_atm_options",
    "implied_dividend_yield",
    "implied_forward_curve",
    "price_options_vectorized",
    "residual_table",
    "select_fastest_technique",
    "summarize_residuals",
    "supports_vectorized_pricing",
]
