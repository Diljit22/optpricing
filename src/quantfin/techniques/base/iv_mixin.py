from scipy.optimize import brentq
from typing import Any
import numpy as np
from quantfin.models.bsm import BSMModel

class IVMixin:
    """
    Calculates Black-Scholes implied volatility for a given price.
    It works by creating a temporary BSM model and finding the 'sigma'
    that matches the target price, regardless of which model generated that price.
    """
    def implied_volatility(
        self,
        option: Any,
        stock:  Any,
        model:  Any, # The original model (e.g., Heston) - will NOT be modified.
        rate:   Any,
        target_price: float,
        low:    float = 1e-6,
        high:   float = 5.0,
        tol:    float = 1e-8,
        max_iter: int = 200,
        initial_guess: float = 0.3,
        **kwargs: Any # Captures v0, etc., but we will NOT use it for the BSM price.
    ) -> float:
        
        bsm_solver_model = BSMModel(params={"sigma": initial_guess})

        def bsm_price_for_iv(vol: float) -> float:
            if vol <= 0: return -target_price
            bsm_solver_model.params["sigma"] = vol
            
            # --- THIS IS THE FIX ---
            # When pricing with the BSM model for the solver, we do NOT pass
            # the extra kwargs (like v0) that were meant for the original model.
            # The BSM model doesn't need them and will crash if it gets them.
            p = self.price(option, stock, bsm_solver_model, rate).price
            # Note: No **kwargs in the line above.
            
            return p - target_price

        iv = np.nan
        try:
            iv = brentq(bsm_price_for_iv, low, high, xtol=tol, disp=False)
        except ValueError:
            try:
                iv = self._secant_iv(bsm_price_for_iv, initial_guess, tol, max_iter)
            except (ValueError, RuntimeError):
                iv = np.nan
        
        return iv

    @staticmethod
    def _secant_iv(fn: Any, x0: float, tol: float, max_iter: int) -> float:
        x1 = x0 * 1.1
        fx0 = fn(x0)
        for _ in range(max_iter):
            fx1 = fn(x1)
            if abs(fx1) < tol: return x1
            denom = fx1 - fx0
            if abs(denom) < 1e-14: break
            x2 = x1 - fx1 * (x1 - x0) / denom
            x0, x1, fx0 = x1, x2, fx1
        return x1