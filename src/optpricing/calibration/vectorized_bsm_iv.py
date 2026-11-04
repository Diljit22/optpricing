from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from optpricing.atoms import Rate, Stock

__doc__ = """
Defines a high-performance, vectorized solver for Black-Scholes-Merton
implied volatility.
"""

_INV_SQRT_2PI = 0.3989422804014327


def bsm_prices_and_vegas(
    iv: np.ndarray,
    S: float,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray | float,
    q: float,
    is_call: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised BSM prices and vegas for arrays of options."""
    sqrt_T = np.sqrt(T)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    with np.errstate(all="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * iv**2) * T) / (iv * sqrt_T)
        d2 = d1 - iv * sqrt_T
        vega = S * df_q * sqrt_T * _INV_SQRT_2PI * np.exp(-0.5 * d1 * d1)
        calls = S * df_q * ndtr(d1) - K * df_r * ndtr(d2)
        puts = K * df_r * ndtr(-d2) - S * df_q * ndtr(-d1)
    return np.where(is_call, calls, puts), vega


class BSMIVSolver:
    """
    High-performance, vectorized implied-volatility solver.

    Safeguarded Newton-Raphson (vega floor, step clipping, bracket tracking) on
    every option at once, followed by bisection on the ones Newton did not
    converge. Prices outside the no-arbitrage band (below the discounted
    intrinsic value or above the forward/strike bound) get ``NaN``.
    """

    def __init__(
        self,
        max_iter: int = 20,
        tolerance: float = 1e-6,
        *,
        bisection_iter: int = 60,
        iv_bounds: tuple[float, float] = (1e-4, 5.0),
    ):
        """
        Initializes the BSM implied volatility solver.

        Parameters
        ----------
        max_iter : int, optional
            Newton iterations, by default 20.
        tolerance : float, optional
            Absolute price tolerance for convergence, by default 1e-6.
        bisection_iter : int, optional
            Bisection iterations for options Newton could not solve, default 60.
        iv_bounds : tuple[float, float], optional
            Search interval for the volatility, by default (1e-4, 5.0).
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.bisection_iter = bisection_iter
        self.iv_bounds = iv_bounds

    def solve(
        self,
        target_prices: np.ndarray,
        options: pd.DataFrame,
        stock: Stock,
        rate: Rate,
        *,
        initial_guess: np.ndarray | None = None,
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
        initial_guess : np.ndarray | None, optional
            Starting volatilities (e.g. yesterday's or the market's implied
            vols); defaults to 0.2 everywhere.

        Returns
        -------
        np.ndarray
            Implied volatilities aligned with ``target_prices``; ``NaN`` where
            the price admits no Black-Scholes volatility.
        """
        target = np.asarray(target_prices, dtype=float)
        S, q = stock.spot, stock.dividend
        K = options["strike"].to_numpy(dtype=float)
        T = options["maturity"].to_numpy(dtype=float)
        if callable(rate.rate):
            r = np.asarray([rate.get_rate(t) for t in T], dtype=float)
        else:
            r = np.full(T.shape, float(rate.get_rate()))
        is_call = options["optionType"].to_numpy() == "call"
        lo, hi = self.iv_bounds

        # No-arbitrage band: intrinsic (discounted) <= price <= S e^{-qT} or K e^{-rT}
        fwd_leg, strike_leg = S * np.exp(-q * T), K * np.exp(-r * T)
        intrinsic = np.where(is_call, fwd_leg - strike_leg, strike_leg - fwd_leg)
        upper = np.where(is_call, fwd_leg, strike_leg)
        solvable = (target > np.maximum(intrinsic, 0.0)) & (target < upper)
        solvable &= np.isfinite(target) & (T > 0) & (K > 0)

        iv = (
            np.full(target.shape, 0.2)
            if initial_guess is None
            else np.array(initial_guess, dtype=float)
        )
        iv = np.where(np.isfinite(iv), np.clip(iv, lo, hi), 0.2)
        active = solvable.copy()
        low_b, high_b = np.full(target.shape, lo), np.full(target.shape, hi)

        for _ in range(self.max_iter):
            if not active.any():
                break
            prices, vega = bsm_prices_and_vegas(iv, S, K, T, r, q, is_call)
            error = prices - target
            converged = np.abs(error) < self.tolerance
            active &= ~converged
            # Price is increasing in vol: tighten the bracket on the way.
            low_b = np.where(active & (error < 0), np.maximum(low_b, iv), low_b)
            high_b = np.where(active & (error > 0), np.minimum(high_b, iv), high_b)
            step = error / np.maximum(vega, 1e-12)
            new_iv = iv - np.clip(step, -1.0, 1.0)
            inside = (new_iv > low_b) & (new_iv < high_b)
            new_iv = np.where(inside, new_iv, 0.5 * (low_b + high_b))
            iv = np.where(active, new_iv, iv)

        # Bisection for whatever Newton left over (deep ITM/OTM, tiny vegas).
        if active.any():
            idx = np.flatnonzero(active)
            a, b = low_b[idx], high_b[idx]
            for _ in range(self.bisection_iter):
                mid = 0.5 * (a + b)
                prices, _ = bsm_prices_and_vegas(
                    mid, S, K[idx], T[idx], r[idx], q, is_call[idx]
                )
                too_high = prices > target[idx]
                a, b = np.where(too_high, a, mid), np.where(too_high, mid, b)
            iv[idx] = 0.5 * (a + b)
            prices, _ = bsm_prices_and_vegas(
                iv[idx], S, K[idx], T[idx], r[idx], q, is_call[idx]
            )
            solvable[idx] &= np.abs(prices - target[idx]) < 1e-3 * np.maximum(
                target[idx], 1.0
            )

        return np.where(solvable, iv, np.nan)
