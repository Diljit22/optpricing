from __future__ import annotations

__doc__ = """
The `calibration` package provides tools for fitting financial models to
market data, estimating parameters from historical data, and calculating
implied volatility surfaces.
"""

from .calibrator import Calibrator
from .fit_jump_parameters import fit_jump_params_from_history
from .fit_market_params import find_atm_options, fit_rate_and_dividend
from .iv_surface import VolatilitySurface
from .technique_selector import select_fastest_technique
from .vectorized_bsm_iv import BSMIVSolver
from .vectorized_integration_iv import VectorizedIntegrationIVSolver
from .vectorized_pricer import price_options_vectorized

__all__ = [
    "BSMIVSolver",
    "Calibrator",
    "VectorizedIntegrationIVSolver",
    "VolatilitySurface",
    "fit_jump_params_from_history",
    "fit_rate_and_dividend",
    "find_atm_options",
    "select_fastest_technique",
    "price_options_vectorized",
]
