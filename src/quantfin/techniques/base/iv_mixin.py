from __future__ import annotations
from scipy.optimize import brentq
from typing import Any, TYPE_CHECKING
import numpy as np

from quantfin.models import BSMModel

if TYPE_CHECKING:
    from quantfin.atoms import Option, Stock, Rate
    from quantfin.models import BaseModel

class IVMixin:
    """
    Calculates Black-Scholes implied volatility for a given price.

    This mixin uses a root-finding algorithm (Brent's method) to find the
    volatility ('sigma') in a BSM model that matches a given target price.
    It is a generic, model-agnostic way to compute IV.
    """
    def implied_volatility(self, option: Option, stock: Stock, model: BaseModel, rate: Rate, target_price: float, low: float = 1e-6, high: float = 5.0, tol: float = 1e-8, **kwargs: Any) -> float:
        """
        Finds the implied volatility that matches the target_price.
        """
        # We create a temporary BSM model for the solver.
        bsm_solver_model = BSMModel(params={"sigma": 0.3})

        def bsm_price_minus_target(vol: float) -> float:
            """Objective function for the root finder."""
            if vol <= 0: return -target_price
            
            # Create a new BSM model with the current volatility guess.
            current_bsm_model = bsm_solver_model.with_params(sigma=vol)
            
            # Price using the same technique but with the BSM model.
            price = self.price(option, stock, current_bsm_model, rate).price
            return price - target_price

        try:
            iv = brentq(bsm_price_minus_target, low, high, xtol=tol, disp=False)
        except (ValueError, RuntimeError):
            iv = np.nan
        
        return iv