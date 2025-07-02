from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import integrate

from optpricing.models import BSMModel

if TYPE_CHECKING:
    from optpricing.atoms import Rate
    from optpricing.models import BaseModel


__doc__ = """
Defines a high-performance, vectorized solver for implied volatility using
numerical integration of the characteristic function.
"""


class VectorizedIntegrationIVSolver:
    """
    A high-performance, vectorized Secant method solver for implied volatility
    for any model that supports a characteristic function.
    """

    def __init__(
        self,
        max_iter: int = 20,
        tolerance: float = 1e-7,
        upper_bound: float = 200.0,
    ):
        """
        Initializes the vectorized IV solver.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations for the Secant method, by default 20.
        tolerance : float, optional
            Error tolerance for convergence, by default 1e-7.
        upper_bound : float, optional
            The upper limit for the numerical integration, by default 200.0.
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.upper_bound = upper_bound

    def solve(
        self,
        target_prices: np.ndarray,
        options: pd.DataFrame,
        model: BaseModel,
        rate: Rate,
    ) -> np.ndarray:
        """
        Calculates implied volatility for an array of options and prices.

        This method uses a vectorized Secant root-finding algorithm. The pricing
        at each step is performed using a vectorized version of the Gil-Pelaez
        inversion formula.

        Parameters
        ----------
        target_prices : np.ndarray
            An array of market prices for which to find the implied volatility.
        options : pd.DataFrame
            A DataFrame of option contracts.
        model : BaseModel
            The financial model to use for pricing.
        rate : Rate
            The risk-free rate structure.

        Returns
        -------
        np.ndarray
            An array of calculated implied volatilities.
        """
        iv0 = np.full_like(target_prices, 0.20)
        iv1 = np.full_like(target_prices, 0.25)

        p0 = self._price_vectorized(iv0, options, model, rate)
        p1 = self._price_vectorized(iv1, options, model, rate)

        f0 = p0 - target_prices
        f1 = p1 - target_prices

        for _ in range(self.max_iter):
            if np.all(np.abs(f1) < self.tolerance):
                break
            denom = f1 - f0
            denom[np.abs(denom) < 1e-12] = 1e-12
            iv_next = iv1 - f1 * (iv1 - iv0) / denom
            iv_next = np.clip(iv_next, 1e-4, 5.0)
            iv0, iv1 = iv1, iv_next
            f0 = f1
            p1 = self._price_vectorized(iv1, options, model, rate)
            f1 = p1 - target_prices

        return iv1

    def _price_vectorized(
        self,
        iv_vector: np.ndarray,
        options: pd.DataFrame,
        model: BaseModel,
        rate: Rate,
    ) -> np.ndarray:
        """
        Internal method to price a vector of options for a given vector of volatilities.
        """
        bsm_model = BSMModel(params={"sigma": 0.2})
        prices = np.zeros_like(iv_vector)

        for T, group in options.groupby("maturity"):
            idx = group.index
            S, K = group["spot"].values, group["strike"].values
            q, r = group["dividend"].values, rate.get_rate(T)
            is_call = group["optionType"].values == "call"

            bsm_model.params["sigma"] = iv_vector[idx]
            phi = bsm_model.cf(t=T, spot=S, r=r, q=q)
            k_log = np.log(K)

            def integrand_p2(u):
                """Calculates the integrand for P2."""
                return (np.exp(-1j * u * k_log) * phi(u)).imag / u

            def integrand_p1(u):
                """Calculates the integrand for P1."""
                return (np.exp(-1j * u * k_log) * phi(u - 1j)).imag / u

            integral_p2, _ = integrate.quad_vec(integrand_p2, 1e-15, self.upper_bound)

            phi_minus_i = phi(-1j)
            phi_minus_i[np.abs(phi_minus_i) < 1e-12] = 1.0
            integral_p1, _ = integrate.quad_vec(integrand_p1, 1e-15, self.upper_bound)

            P1 = 0.5 + integral_p1 / (np.pi * np.real(phi_minus_i))
            P2 = 0.5 + integral_p2 / np.pi

            call_prices = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
            put_prices = K * np.exp(-r * T) * (1 - P2) - S * np.exp(-q * T) * (1 - P1)

            prices[idx] = np.where(is_call, call_prices, put_prices)

        return prices
