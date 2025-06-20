__doc__ = """
The `calibration` package provides tools for fitting model parameters to market data.

This includes high-performance, vectorized implied volatility solvers, tools for
estimating historical parameters, and the core `Calibrator` class that ties
these components together to perform model calibration against option chains.
"""

# Core calibration class
from .calibrator import Calibrator

# Parameter fitting utilities
from .fit_jump_parameters import fit_jump_params_from_history
from .fit_market_params import fit_rate_and_dividend

# Volatility surface representation
from .iv_surface import VolatilitySurface

# Helper for selecting the best pricing technique
from .technique_selector import select_fastest_technique

# --- High-performance, vectorized implied volatility solvers ---
# The BSMIVSolver was moved to its own file, let's assume 'vectorized_bsm_iv.py'
# or similar. If it's in 'vectorized_iv_solver.py', adjust the import.
from .bsm_iv_solver import BSMIVSolver
from .vectorized_integration_iv import VectorizedIntegrationIVSolver


# Define the public API for the calibration package
__all__ = [
    "Calibrator",
    "fit_jump_params_from_history",
    "fit_rate_and_dividend",
    "VolatilitySurface",
    "select_fastest_technique",
    "BSMIVSolver",
    "VectorizedIntegrationIVSolver",
]