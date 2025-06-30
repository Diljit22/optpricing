from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__doc__ = """
Provides utility functions to fit market-implied parameters like the
risk-free rate and dividend yield from option prices using put-call parity.
"""


def find_atm_options(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
) -> pd.DataFrame:
    """
    Finds the closest at-the-money (ATM) call-put pair for each expiry.

    Parameters
    ----------
    calls : pd.DataFrame
        A DataFrame of call options.
    puts : pd.DataFrame
        A DataFrame of put options.
    spot : float
        The current spot price of the underlying.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the merged ATM call-put pairs.
    """
    merged = pd.merge(
        calls, puts, on=["strike", "maturity"], suffixes=("_call", "_put"), how="inner"
    )
    if merged.empty:
        return pd.DataFrame()
    merged["moneyness_dist"] = abs(merged["strike"] - spot)
    atm_indices = merged.groupby("maturity")["moneyness_dist"].idxmin()
    return merged.loc[atm_indices]


def fit_rate_and_dividend(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    r_fixed: float | None = None,
    q_fixed: float | None = None,
) -> tuple[float, float]:
    """
    Fits the risk-free rate (r) and dividend yield (q) from put-call parity.

    This function uses the prices of at-the-money (ATM) call-put pairs to
    solve for the `r` and `q` that minimize the parity pricing error.
    Parameters can be held fixed or fitted.

    Parameters
    ----------
    calls : pd.DataFrame
        A DataFrame of call options with 'strike', 'maturity', 'marketPrice'.
    puts : pd.DataFrame
        A DataFrame of put options with 'strike', 'maturity', 'marketPrice'.
    spot : float
        The current spot price of the underlying.
    r_fixed : float | None, optional
        If provided, the risk-free rate is held fixed at this value.
        Defaults to None.
    q_fixed : float | None, optional
        If provided, the dividend yield is held fixed at this value.
        Defaults to None.

    Returns
    -------
    tuple[float, float]
        A tuple containing the estimated (or fixed) risk-free rate and dividend yield.
    """
    atm_pairs = find_atm_options(calls, puts, spot)
    if atm_pairs.empty:
        print("Warning: No ATM pairs found. Using default r=0.05, q=0.0.")
        return 0.05, 0.0

    free_param_indices, initial_guess, bounds = [], [], []
    if r_fixed is None:
        free_param_indices.append(0)
        initial_guess.append(0.05)
        bounds.append((0.0, 0.15))
    if q_fixed is None:
        free_param_indices.append(1)
        initial_guess.append(0.01)
        bounds.append((0.0, 0.10))

    if not free_param_indices:
        return r_fixed, q_fixed

    def parity_error(x: np.ndarray) -> float:
        r = r_fixed if 0 not in free_param_indices else x[free_param_indices.index(0)]
        q = q_fixed if 1 not in free_param_indices else x[free_param_indices.index(1)]
        parity_rhs = spot * np.exp(-q * atm_pairs["maturity"]) - atm_pairs[
            "strike"
        ] * np.exp(-r * atm_pairs["maturity"])
        parity_lhs = atm_pairs["marketPrice_call"] - atm_pairs["marketPrice_put"]
        return np.sum((parity_lhs - parity_rhs) ** 2)

    solution = minimize(
        fun=parity_error, x0=initial_guess, bounds=bounds, method="L-BFGS-B"
    )
    r_est = (
        r_fixed
        if 0 not in free_param_indices
        else solution.x[free_param_indices.index(0)]
    )
    q_est = (
        q_fixed
        if 1 not in free_param_indices
        else solution.x[free_param_indices.index(1)]
    )
    print(f"Fitted market params -> r: {r_est:.4f}, q: {q_est:.4f}")
    return float(r_est), float(q_est)
