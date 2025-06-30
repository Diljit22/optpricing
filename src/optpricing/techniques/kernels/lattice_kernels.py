from __future__ import annotations

import math
from typing import Any

import numpy as np

__doc__ = """
This module contains the low-level implementations for various lattice-based
option pricing algorithms. These functions are designed to be pure and operate
on numerical inputs.
"""


def _crr_pricer(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    N: int,
    is_call: bool,
    is_am: bool,
) -> dict[str, Any]:
    """
    Cox-Ross-Rubinstein binomial tree pricer.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate.
    q : float
        Dividend yield.
    sigma : float
        Volatility of the asset.
    N : int
        Number of steps in the tree.
    is_call : bool
        True for a call option, False for a put.
    is_am : bool
        True for an American option, False for European.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the option price and node values for Greek calcs.
    """
    if sigma < 1e-6:
        sigma = 1e-6
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    j = np.arange(N + 1)
    payoff = np.where(
        is_call,
        np.maximum(S0 * u**j * d ** (N - j) - K, 0.0),
        np.maximum(K - S0 * u**j * d ** (N - j), 0.0),
    )
    for i in range(N - 1, -1, -1):
        if i == 1:
            price_uu, price_ud, price_dd = payoff[2], payoff[1], payoff[0]
        elif i == 0:
            price_up, price_down = payoff[1], payoff[0]
        payoff = disc * (p * payoff[1:] + (1 - p) * payoff[:-1])
        if is_am:
            stock_prices = S0 * u ** np.arange(i + 1) * d ** (i - np.arange(i + 1))
            early_exercise = np.where(
                is_call,
                np.maximum(stock_prices - K, 0.0),
                np.maximum(K - stock_prices, 0.0),
            )
            payoff = np.maximum(payoff, early_exercise)
    return {
        "price": payoff[0],
        "price_up": price_up,
        "price_down": price_down,
        "price_uu": price_uu,
        "price_ud": price_ud,
        "price_dd": price_dd,
        "spot_up": S0 * u,
        "spot_down": S0 * d,
        "spot_uu": S0 * u * u,
        "spot_ud": S0 * u * d,
        "spot_dd": S0 * d * d,
    }


def _lr_pricer(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    N: int,
    is_call: bool,
    is_am: bool,
) -> dict[str, Any]:
    """
    Leisen-Reimer binomial tree pricer with Peizer-Pratt inversion.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate.
    q : float
        Dividend yield.
    sigma : float
        Volatility of the asset.
    N : int
        Number of steps in the tree.
    is_call : bool
        True for a call option, False for a put.
    is_am : bool
        True for an American option, False for European.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the option price and node values for Greek calcs.
    """
    if sigma < 1e-6:
        price = max(0.0, S0 - K) if is_call else max(0.0, K - S0)
        return {
            "price": price,
            "price_up": price,
            "price_down": price,
            "price_uu": price,
            "price_ud": price,
            "price_dd": price,
            "spot_up": S0,
            "spot_down": S0,
            "spot_uu": S0,
            "spot_ud": S0,
            "spot_dd": S0,
        }
    if N % 2 == 0:
        N += 1
    dt = T / N
    disc = math.exp(-r * dt)
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    p_d1 = _peizer_pratt(d1, N)
    p_d2 = _peizer_pratt(d2, N)
    u = math.exp((r - q) * dt) * (p_d1 / p_d2)
    d = (math.exp((r - q) * dt) - p_d2 * u) / (1 - p_d2)
    p = p_d2
    j = np.arange(N + 1)
    payoff = np.where(
        is_call,
        np.maximum(S0 * u**j * d ** (N - j) - K, 0.0),
        np.maximum(K - S0 * u**j * d ** (N - j), 0.0),
    )
    for i in range(N - 1, -1, -1):
        if i == 1:
            price_uu, price_ud, price_dd = payoff[2], payoff[1], payoff[0]
        elif i == 0:
            price_up, price_down = payoff[1], payoff[0]
        payoff = disc * (p * payoff[1:] + (1 - p) * payoff[:-1])
        if is_am:
            stock_prices = S0 * u ** np.arange(i + 1) * d ** (i - np.arange(i + 1))
            early_exercise = np.where(
                is_call,
                np.maximum(stock_prices - K, 0.0),
                np.maximum(K - stock_prices, 0.0),
            )
            payoff = np.maximum(payoff, early_exercise)
    return {
        "price": payoff[0],
        "price_up": price_up,
        "price_down": price_down,
        "price_uu": price_uu,
        "price_ud": price_ud,
        "price_dd": price_dd,
        "spot_up": S0 * u,
        "spot_down": S0 * d,
        "spot_uu": S0 * u * u,
        "spot_ud": S0 * u * d,
        "spot_dd": S0 * d * d,
    }


def _peizer_pratt(z: float, N: int) -> float:
    """
    Peizer-Pratt inversion method for Leisen-Reimer tree.
    """
    if abs(z) > 50:
        return 1.0 if z > 0 else 0.0
    term = z / (N + 1 / 3 + 0.1 / (N + 1))
    return 0.5 + math.copysign(
        0.5 * math.sqrt(1 - math.exp(-(term**2) * (N + 1 / 6))), z
    )


def _topm_pricer(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    N: int,
    is_call: bool,
    is_am: bool,
) -> dict[str, Any]:
    """
    Kamrad-Ritchken trinomial tree pricer.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate.
    q : float
        Dividend yield.
    vol : float
        Volatility of the asset.
    N : int
        Number of steps in the tree.
    is_call : bool
        True for a call option, False for a put.
    is_am : bool
        True for an American option, False for European.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the option price and node values for Greek calcs.
    """
    if vol < 1e-6:
        vol = 1e-6
    dt = T / N
    disc = math.exp(-r * dt)
    dx = vol * math.sqrt(2 * dt)
    drift_term = (r - q - 0.5 * vol**2) * dt
    pu = 0.5 * ((vol**2 * dt + drift_term**2) / dx**2 + drift_term / dx)
    pd = 0.5 * ((vol**2 * dt + drift_term**2) / dx**2 - drift_term / dx)
    pm = 1.0 - pu - pd
    j = np.arange(-N, N + 1)
    payoff = np.where(
        is_call,
        np.maximum(S0 * np.exp(j * dx) - K, 0.0),
        np.maximum(K - S0 * np.exp(j * dx), 0.0),
    )
    for i in range(N - 1, -1, -1):
        if i == 0:
            price_up, price_mid, price_down = payoff[2], payoff[1], payoff[0]
        payoff = disc * (pu * payoff[2:] + pm * payoff[1:-1] + pd * payoff[:-2])
        if is_am:
            j_inner = np.arange(-i, i + 1)
            stock_prices = S0 * np.exp(j_inner * dx)
            early_exercise = np.where(
                is_call,
                np.maximum(stock_prices - K, 0.0),
                np.maximum(K - stock_prices, 0.0),
            )
            payoff = np.maximum(payoff, early_exercise)
    return {
        "price": payoff[0],
        "price_up": price_up,
        "price_mid": price_mid,
        "price_down": price_down,
        "spot_up": S0 * math.exp(dx),
        "spot_mid": S0,
        "spot_down": S0 * math.exp(-dx),
    }
