from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from optpricing.atoms import Rate, Stock

__doc__ = """
Defines a high-performance, vectorized solver for Black-Scholes-Merton
implied volatility.
"""


class BSMIVSolver:
    """
    High-performance, vectorized Newton-Raphson solver for BSM implied volatility.

    This solver is designed to calculate the implied volatility for a large
    number of options simultaneously, leveraging NumPy for vectorized operations.
    """

    def __init__(
        self,
        max_iter: int = 20,
        tolerance: float = 1e-6,
    ):
        """
        Initializes the BSM implied volatility solver.

        Parameters
        ----------
        max_iter : int, optional
            The maximum number of iterations for the Newton-Raphson method,
            by default 20.
        tolerance : float, optional
            The error tolerance to determine convergence, by default 1e-6.
        """
        self.max_iter = max_iter
        self.tolerance = tolerance

    def solve(
        self,
        target_prices: np.ndarray,
        options: pd.DataFrame,
        stock: Stock,
        rate: Rate,
    ) -> np.ndarray:
        """
        Calculates implied volatility for an array of options.

        Parameters
        ----------
        target_prices : np.ndarray
            An array of market prices for which to find the implied volatility.
        options : pd.DataFrame
            A DataFrame of option contracts, must include 'strike', 'maturity',
            and 'optionType' columns.
        stock : Stock
            The underlying asset's properties.
        rate : Rate
            The risk-free rate structure.

        Returns
        -------
        np.ndarray
            An array of calculated implied volatilities corresponding to the
            target prices.
        """
        S, q = stock.spot, stock.dividend
        K, T = options["strike"].values, options["maturity"].values
        r = rate.get_rate(T)  # Use get_rate for term structure
        is_call = options["optionType"].values == "call"
        iv = np.full_like(target_prices, 0.20)

        for _ in range(self.max_iter):
            with np.errstate(all="ignore"):
                sqrt_T = np.sqrt(T)
                d1 = (np.log(S / K) + (r - q + 0.5 * iv**2) * T) / (iv * sqrt_T)
                d2 = d1 - iv * sqrt_T
                vega = S * np.exp(-q * T) * sqrt_T * norm.pdf(d1)
                call_prices = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(
                    -r * T
                ) * norm.cdf(d2)
                put_prices = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(
                    -q * T
                ) * norm.cdf(-d1)
            model_prices = np.where(is_call, call_prices, put_prices)
            error = model_prices - target_prices
            if np.all(np.abs(error) < self.tolerance):
                break
            iv = iv - error / np.maximum(vega, 1e-8)
        return iv
