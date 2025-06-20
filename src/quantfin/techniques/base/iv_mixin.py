
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
    Calculates Black-Scholes implied volatility for a given price using a
    root-finding algorithm.
    """
    def implied_volatility(self, option: Option, stock: Stock, model: BaseModel, rate: Rate, target_price: float, low: float = 1e-6, high: float = 5.0, tol: float = 1e-6, **kwargs: Any) -> float:
        # ... (bsm_price_minus_target function is the same) ...
        bsm_solver_model = BSMModel(params={"sigma": 0.3})
        def bsm_price_minus_target(vol: float) -> float:
            current_bsm_model = bsm_solver_model.with_params(sigma=vol)
            try:
                with np.errstate(all='ignore'):
                    price = self.price(option, stock, current_bsm_model, rate).price
                if not np.isfinite(price): return 1e6 
                return price - target_price
            except (ZeroDivisionError, OverflowError):
                return 1e6

        try:
            # First, try the fast and precise Brent's method
            iv = brentq(bsm_price_minus_target, low, high, xtol=tol, disp=False)
        except (ValueError, RuntimeError):
            try:
                # If brentq fails, fall back to the slower but more robust Secant method
                iv = self._secant_iv(bsm_price_minus_target, 0.2, tol, 100)
            except (ValueError, RuntimeError):
                iv = np.nan
        
        return iv

    @staticmethod
    def _secant_iv(fn: Any, x0: float, tol: float, max_iter: int) -> float:
        # Your original Secant method implementation
        x1 = x0 * 1.1
        fx0 = fn(x0)
        for _ in range(max_iter):
            fx1 = fn(x1)
            if abs(fx1) < tol: return x1
            denom = fx1 - fx0
            if abs(denom) < 1e-14: break
            x2 = x1 - fx1 * (x1 - x0) / denom
            x0, x1, fx0 = x1, x2, fx1
        if abs(fn(x1)) < tol * 10: # Looser check for final result
             return x1
        raise RuntimeError("Secant method failed to converge.")