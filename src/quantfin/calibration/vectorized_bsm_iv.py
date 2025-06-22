import numpy as np
import pandas as pd
from scipy.stats import norm

from quantfin.atoms import Rate, Stock


class BSMIVSolver:
    """High-performance, vectorized Newton-Raphson solver for BSM implied volatility."""

    def __init__(self, max_iter: int = 20, tolerance: float = 1e-6):
        self.max_iter = max_iter
        self.tolerance = tolerance

    def solve(
        self, target_prices: np.ndarray, options: pd.DataFrame, stock: Stock, rate: Rate
    ) -> np.ndarray:
        """Calculates implied volatility for an array of options."""
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
