from __future__ import annotations

import numba
import numpy as np

__doc__ = """
JIT-compiled Longstaff-Schwartz pricing algorithm.
"""

# Docstring for longstaff_schwartz_pricer
"""
Prices an American option using the Longstaff-Schwartz algorithm.

Parameters
----------
stock_paths : np.ndarray
    A (n_paths, n_steps + 1) array of simulated spot prices, including t=0.
K : float
    Strike price.
dt : float
    Time step size.
r : float
    Risk-free rate.
is_call : bool
    True for a call, False for a put.
degree : int
    Polynomial degree for the regression basis.

Returns
-------
float
    The estimated American option price.
"""


@numba.jit(nopython=True, fastmath=True, cache=True)
def longstaff_schwartz_pricer(
    stock_paths: np.ndarray,
    K: float,
    dt: float,
    r: float,
    is_call: bool,
    degree: int,
) -> float:
    n_paths, n_steps_plus_1 = stock_paths.shape
    n_steps = n_steps_plus_1 - 1

    if is_call:
        cashflow = np.maximum(stock_paths[:, -1] - K, 0.0)
    else:
        cashflow = np.maximum(K - stock_paths[:, -1], 0.0)

    for i in range(n_steps - 1, 0, -1):
        cashflow = cashflow * np.exp(-r * dt)
        stock_price_at_t = stock_paths[:, i]

        if is_call:
            intrinsic_value = np.maximum(stock_price_at_t - K, 0.0)
        else:
            intrinsic_value = np.maximum(K - stock_price_at_t, 0.0)

        in_the_money_mask = intrinsic_value > 0
        if np.any(in_the_money_mask):
            x = stock_price_at_t[in_the_money_mask]
            y = cashflow[in_the_money_mask]

            X = np.ones((len(x), degree + 1))
            for d in range(1, degree + 1):
                X[:, d] = x**d

            try:
                beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
                continuation_value = X @ beta

                exercise_mask = intrinsic_value[in_the_money_mask] > continuation_value
                cashflow[in_the_money_mask] = np.where(
                    exercise_mask,
                    intrinsic_value[in_the_money_mask],
                    cashflow[in_the_money_mask],
                )
            except:
                pass

    return np.mean(cashflow * np.exp(-r * dt))
