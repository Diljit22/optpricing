from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import integrate

if TYPE_CHECKING:
    import pandas as pd

    from optpricing.atoms import Rate, Stock
    from optpricing.models import BaseModel

__doc__ = """
Defines a high-performance, vectorized pricer for models with CF.
"""


def price_options_vectorized(
    options_df: pd.DataFrame,
    stock: Stock,
    model: BaseModel,
    rate: Rate,
    *,
    upper_bound: float = 200.0,
    **kwargs: Any,
) -> np.ndarray:
    """
    Vectorised integral pricer (Carr-Madan representation).

    Parameters
    ----------
    options_df : pd.DataFrame
        Must contain `strike`, `maturity` and `optionType`.
        The index is reset internally to guarantee safe positional writes.
    stock : Stock
        Underlying description.
    model : BaseModel
        Any model exposing a `cf(t, spot, r, q)` callable.
    rate : Rate
        Continuous zero-curve.
    upper_bound : float, default 200
        Integration truncation limit (works for double precision).

    Returns
    -------
    np.ndarray
        Model prices - aligned with the row order of options_df.
    """
    options_df = options_df.reset_index(drop=True)

    n_opts = len(options_df)
    prices = np.empty(n_opts)

    S = stock.spot
    q = stock.dividend

    for T, grp in options_df.groupby("maturity", sort=False):
        loc = grp.index.to_numpy()
        K = grp["strike"].to_numpy()
        is_call = grp["optionType"].to_numpy() == "call"

        r = rate.get_rate(T)
        phi = model.cf(t=T, spot=S, r=r, q=q, **kwargs)
        lnK = np.log(K)

        def _integrand_p2(u: float) -> np.ndarray:
            return (np.exp(-1j * u * lnK) * phi(u)).imag / u

        def _integrand_p1(u: float) -> np.ndarray:
            return (np.exp(-1j * u * lnK) * phi(u - 1j)).imag / u

        # Vectorised quad once per maturity
        p2, _ = integrate.quad_vec(_integrand_p2, 1e-15, upper_bound)
        p1, _ = integrate.quad_vec(_integrand_p1, 1e-15, upper_bound)

        denom = np.real(phi(-1j))
        denom = 1.0 if abs(denom) < 1e-12 else denom

        P1 = 0.5 + p1 / (np.pi * denom)
        P2 = 0.5 + p2 / np.pi

        call_vals = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
        put_vals = K * np.exp(-r * T) * (1.0 - P2) - S * np.exp(-q * T) * (1.0 - P1)

        prices[loc] = np.where(is_call, call_vals, put_vals)

    return prices
