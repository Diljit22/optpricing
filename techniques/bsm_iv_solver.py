import numpy as np
from scipy.stats import norm

from atoms.option import Option, OptionType
from atoms.stock import Stock
from atoms.rate import Rate
import pandas as pd

class BSMIVSolver:
    """
    A high-performance, vectorized Newton-Raphson solver for Black-Scholes
    implied volatility.
    """
    def __init__(self, max_iter: int = 20, tolerance: float = 1e-6):
        self.max_iter = max_iter
        self.tolerance = tolerance

    def solve(self, target_prices: np.ndarray, options: pd.DataFrame, stock: Stock, rate: Rate) -> np.ndarray:
        """
        Calculates implied volatility for a whole array of options at once.
        """
        S = stock.spot
        r = rate.rate
        q = stock.dividend
        
        K = options['strike'].values
        T = options['maturity'].values
        is_call = (options['optionType'].values == 'call')

        # Initial guess for volatility
        iv = np.full_like(target_prices, 0.20)

        for _ in range(self.max_iter):
            # --- Vectorized BSM Price and Vega ---
            sqrt_T = np.sqrt(T)
            d1 = (np.log(S / K) + (r - q + 0.5 * iv**2) * T) / (iv * sqrt_T)
            d2 = d1 - iv * sqrt_T
            
            # Price
            call_prices = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            put_prices = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
            model_prices = np.where(is_call, call_prices, put_prices)
            
            # Vega
            vega = S * np.exp(-q * T) * sqrt_T * norm.pdf(d1)
            
            # --- Newton-Raphson Step ---
            error = model_prices - target_prices
            
            # If error is small enough, we're done
            if np.all(np.abs(error) < self.tolerance):
                break
            
            # Update guess (avoid division by zero)
            iv = iv - error / np.maximum(vega, 1e-8)

        return iv