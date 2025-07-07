from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import integrate

if TYPE_CHECKING:
    import pandas as pd

    from optpricing.atoms import Rate, Stock
    from optpricing.models import BaseModel


def price_options_vectorized(
    options_df: pd.DataFrame,
    stock: Stock,
    model: BaseModel,
    rate: Rate,
    upper_bound: float = 200.0,
) -> np.ndarray:
    """
    Prices a DataFrame of options using a vectorized characteristic function approach.
    """
    prices = np.zeros(len(options_df))

    # Group by maturity to handle term structure of rates
    for T, group in options_df.groupby("maturity"):
        idx = group.index
        S, K = stock.spot, group["strike"].values
        q, r = stock.dividend, rate.get_rate(T)
        is_call = group["optionType"].values == "call"

        # Get the characteristic function for this maturity group
        phi = model.cf(t=T, spot=S, r=r, q=q)
        k_log = np.log(K)

        def integrand_p2(u):
            return (np.exp(-1j * u * k_log) * phi(u)).imag / u

        def integrand_p1(u):
            return (np.exp(-1j * u * k_log) * phi(u - 1j)).imag / u

        integral_p2, _ = integrate.quad_vec(integrand_p2, 1e-15, upper_bound)

        phi_minus_i = phi(-1j)
        # Handle potential division by zero for phi(-1j)
        phi_minus_i_real = np.real(phi_minus_i)
        safe_denom = np.where(np.abs(phi_minus_i_real) < 1e-12, 1.0, phi_minus_i_real)

        integral_p1, _ = integrate.quad_vec(integrand_p1, 1e-15, upper_bound)

        P1 = 0.5 + integral_p1 / (np.pi * safe_denom)
        P2 = 0.5 + integral_p2 / np.pi

        call_prices = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
        put_prices = K * np.exp(-r * T) * (1 - P2) - S * np.exp(-q * T) * (1 - P1)

        prices[idx] = np.where(is_call, call_prices, put_prices)

    return prices
