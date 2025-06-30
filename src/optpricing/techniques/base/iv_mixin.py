from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import brentq

from optpricing.models import BSMModel

if TYPE_CHECKING:
    from optpricing.atoms import Option, Rate, Stock
    from optpricing.models import BaseModel

__doc__ = """
Provides a mixin class for calculating implied volatility.
"""


class IVMixin:
    """
    Calculates Black-Scholes implied volatility for a given price using a
    root-finding algorithm.

    This implementation uses Brent's method for speed and precision, with a
    fallback to a more robust Secant method if the initial search fails.
    """

    def implied_volatility(
        self,
        option: Option,
        stock: Stock,
        model: BaseModel,
        rate: Rate,
        target_price: float,
        low: float = 1e-6,
        high: float = 5.0,
        tol: float = 1e-6,
        **kwargs: Any,
    ) -> float:
        """
        Calculates the implied volatility for a given option price.

        Parameters
        ----------
        option : Option
            The option contract.
        stock : Stock
            The underlying asset's properties.
        model : BaseModel
            The model to use for pricing. Note: IV is always calculated
            relative to the Black-Scholes-Merton model.
        rate : Rate
            The risk-free rate structure.
        target_price : float
            The market price of the option for which to find the IV.
        low : float, optional
            The lower bound for the volatility search, by default 1e-6.
        high : float, optional
            The upper bound for the volatility search, by default 5.0.
        tol : float, optional
            The tolerance for the root-finding algorithm, by default 1e-6.

        Returns
        -------
        float
            The implied volatility, or `np.nan` if the search fails.
        """
        bsm_solver_model = BSMModel(params={"sigma": 0.3})

        def bsm_price_minus_target(vol: float) -> float:
            current_bsm_model = bsm_solver_model.with_params(sigma=vol)
            try:
                with np.errstate(all="ignore"):
                    price = self.price(option, stock, current_bsm_model, rate).price
                if not np.isfinite(price):
                    return 1e6
                return price - target_price
            except (ZeroDivisionError, OverflowError):
                return 1e6

        try:
            # First, try the fast and precise Brent's method
            iv = brentq(bsm_price_minus_target, low, high, xtol=tol, disp=False)
        except (ValueError, RuntimeError):
            try:
                # If brentq fails, fall back to the slower Secant method
                iv = self._secant_iv(bsm_price_minus_target, 0.2, tol, 100)
            except (ValueError, RuntimeError):
                iv = np.nan

        return iv

    @staticmethod
    def _secant_iv(
        fn: Any,
        x0: float,
        tol: float,
        max_iter: int,
    ) -> float:
        """
        A simple Secant method implementation as a fallback for root finding.
        """
        x1 = x0 * 1.1
        fx0 = fn(x0)
        for _ in range(max_iter):
            fx1 = fn(x1)
            if abs(fx1) < tol:
                return x1
            denom = fx1 - fx0
            if abs(denom) < 1e-14:
                break
            x2 = x1 - fx1 * (x1 - x0) / denom
            x0, x1, fx0 = x1, x2, fx1
        if abs(fn(x1)) < tol * 10:  # Looser check for final result
            return x1
        raise RuntimeError("Secant method failed to converge.")
