from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__doc__ = """
Provides utility functions to fit market-implied parameters like the
risk-free rate and dividend yield from option prices using put-call parity.
"""

logger = logging.getLogger(__name__)

# Sanity bounds for the implied flat dividend yield (continuous compounding).
DIVIDEND_BOUNDS = (-0.05, 0.25)
DEFAULT_RATE = 0.04
DEFAULT_DIVIDEND = 0.0


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


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[min(np.searchsorted(cum, 0.5 * cum[-1]), len(v) - 1)])


def implied_forward_curve(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    rate: float,
    *,
    moneyness_window: float = 0.03,
    min_pairs: int = 3,
) -> pd.DataFrame:
    """
    Implied forward (and dividend yield) per maturity from put-call parity.

    Each call-put pair gives ``F_i = K_i + (C_i - P_i) e^{rT}``; the maturity's
    forward is the bid-ask-weighted median over near-the-money pairs. Only
    pairs close to the money are used because listed US equity and ETF options
    are American: deep in-the-money puts carry an early-exercise premium that
    biases parity (it is also why ``r`` cannot be inferred from the strike
    slope of ``C - P`` and must be supplied).

    Parameters
    ----------
    calls, puts : pd.DataFrame
        Option quotes with ``strike``, ``maturity``, ``marketPrice`` and
        optionally ``bid``/``ask`` columns.
    spot : float
        Spot price used to pick near-the-money pairs and to convert the
        forward into a dividend yield.
    rate : float
        Continuously compounded risk-free rate.
    moneyness_window : float, optional
        Pairs with ``|K/S - 1|`` above this are ignored; the window is widened
        to the nearest ``min_pairs`` strikes when too few qualify. Default 0.03.
    min_pairs : int, optional
        Minimum number of pairs per maturity, by default 3.

    Returns
    -------
    pd.DataFrame
        One row per maturity: ``maturity``, ``forward``, ``implied_dividend``
        and ``n_pairs``; empty when no maturity has a usable pair.
    """
    cols = ["strike", "maturity", "marketPrice"]
    has_quotes = all(c in calls.columns and c in puts.columns for c in ("bid", "ask"))
    quote_cols = ["bid", "ask"] if has_quotes else []
    merged = pd.merge(
        calls[cols + quote_cols],
        puts[cols + quote_cols],
        on=["strike", "maturity"],
        suffixes=("_call", "_put"),
        how="inner",
    ).dropna(subset=["marketPrice_call", "marketPrice_put"])
    if merged.empty:
        return pd.DataFrame()

    rows = []
    for T, grp in merged.groupby("maturity"):
        if T <= 0:
            continue
        grp = grp.drop_duplicates("strike")
        dist = (grp["strike"] / spot - 1.0).abs()
        near = grp[dist <= moneyness_window]
        if len(near) < min_pairs:
            near = grp.loc[dist.nsmallest(min_pairs).index]
        if near.empty:
            continue
        K = near["strike"].to_numpy(dtype=float)
        cmp_ = (near["marketPrice_call"] - near["marketPrice_put"]).to_numpy(float)
        forwards = K + cmp_ * math.exp(rate * T)
        if has_quotes:
            spread = (
                near["ask_call"] - near["bid_call"] + near["ask_put"] - near["bid_put"]
            ).to_numpy(dtype=float)
            w = 1.0 / np.maximum(spread, 1e-3)
        else:
            w = np.ones_like(K)
        forward = _weighted_median(forwards, w)
        if not forward > 0:
            continue
        rows.append(
            {
                "maturity": float(T),
                "forward": float(forward),
                "implied_dividend": float(rate - math.log(forward / spot) / T),
                "n_pairs": int(len(near)),
            }
        )
    return pd.DataFrame(rows)


def implied_dividend_yield(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    rate: float,
    *,
    min_maturity_days: float = 14.0,
) -> float:
    """
    Flat continuous dividend yield implied by the option chain for a given rate.

    Aggregates :func:`implied_forward_curve` with a weighted median (weights
    ``T * n_pairs``, maturities shorter than ``min_maturity_days`` excluded when
    longer ones exist) and falls back to zero when the result is implausible.

    Parameters
    ----------
    calls, puts : pd.DataFrame
        Option quotes (see :func:`implied_forward_curve`).
    spot : float
        Spot price.
    rate : float
        Continuously compounded risk-free rate.
    min_maturity_days : float, optional
        Maturities below this are ignored if longer ones exist, by default 14.

    Returns
    -------
    float
        The implied continuous dividend yield.
    """
    curve = implied_forward_curve(calls, puts, spot, rate)
    if curve.empty:
        return DEFAULT_DIVIDEND
    T = curve["maturity"].to_numpy()
    w = T * curve["n_pairs"].to_numpy()
    long_enough = T >= min_maturity_days / 365.25
    if long_enough.any():
        w = np.where(long_enough, w, 0.0)
    q = _weighted_median(curve["implied_dividend"].to_numpy(), w)
    if not np.isfinite(q) or not (DIVIDEND_BOUNDS[0] <= q <= DIVIDEND_BOUNDS[1]):
        logger.warning("Implied dividend yield %.4f implausible; using 0.", q)
        return DEFAULT_DIVIDEND
    return float(q)


def fit_rate_and_dividend(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    r_fixed: float | None = None,
    q_fixed: float | None = None,
) -> tuple[float, float]:
    """
    Fits the risk-free rate (r) and dividend yield (q) from put-call parity.

    With ``r_fixed`` given, the dividend yield comes from the implied forward
    curve (:func:`implied_dividend_yield`). Otherwise the parity error of the
    ATM call-put pair of every maturity is minimised over the free parameters
    (note that parity identifies ``r - q`` well but ``r`` and ``q`` separately
    only weakly, so supply one of them whenever possible).

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
        The estimated (or fixed) continuously compounded rate and dividend yield.
    """
    if r_fixed is not None and q_fixed is not None:
        return float(r_fixed), float(q_fixed)
    if r_fixed is not None:
        return float(r_fixed), implied_dividend_yield(calls, puts, spot, r_fixed)

    atm_pairs = find_atm_options(calls, puts, spot)
    if atm_pairs.empty:
        logger.warning(
            "No call-put pairs found; using r=%.2f, q=%.2f.",
            DEFAULT_RATE,
            DEFAULT_DIVIDEND,
        )
        return DEFAULT_RATE, DEFAULT_DIVIDEND if q_fixed is None else float(q_fixed)

    free_param_indices, initial_guess, bounds = [], [], []
    free_param_indices.append(0)
    initial_guess.append(0.05)
    bounds.append((0.0, 0.15))
    if q_fixed is None:
        free_param_indices.append(1)
        initial_guess.append(0.01)
        bounds.append((0.0, 0.10))

    def parity_error(x: np.ndarray) -> float:
        r = x[0]
        q = q_fixed if 1 not in free_param_indices else x[1]
        parity_rhs = spot * np.exp(-q * atm_pairs["maturity"]) - atm_pairs[
            "strike"
        ] * np.exp(-r * atm_pairs["maturity"])
        parity_lhs = atm_pairs["marketPrice_call"] - atm_pairs["marketPrice_put"]
        return np.sum((parity_lhs - parity_rhs) ** 2)

    solution = minimize(
        fun=parity_error, x0=initial_guess, bounds=bounds, method="L-BFGS-B"
    )
    r_est = float(solution.x[0])
    q_est = float(q_fixed) if q_fixed is not None else float(solution.x[1])
    logger.debug("Fitted market params -> r: %.4f, q: %.4f", r_est, q_est)
    return r_est, q_est
