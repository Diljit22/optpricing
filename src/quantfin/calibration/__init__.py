__doc__ = "Tools for fitting model parameters to market data."
from .calibrator import Calibrator
from .fit_jump_parameters import fit_jump_params_from_history
from .fit_market_params import fit_rate_and_dividend
from .iv_surface import VolatilitySurface
from .technique_selector import select_fastest_technique
from .vectorized_bsm_iv import BSMIVSolver
from .vectorized_integration_iv import VectorizedIntegrationIVSolver

__all__ = [
    "Calibrator", "fit_jump_params_from_history", "fit_rate_and_dividend",
    "VolatilitySurface", "select_fastest_technique", "BSMIVSolver",
    "VectorizedIntegrationIVSolver",
]