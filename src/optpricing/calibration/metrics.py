from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from optpricing.atoms import Rate, Stock
from optpricing.models import BaseModel

from .market_filters import annotate_chain, maturity_bucket, moneyness_bucket
from .vectorized_bsm_iv import BSMIVSolver
from .vectorized_pricer import price_options_vectorized

__doc__ = """
Fit-quality metrics for a model against an option chain: price and
implied-volatility errors overall and by maturity / moneyness bucket.
"""


def _rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def residual_table(
    model: BaseModel,
    data: pd.DataFrame,
    stock: Stock,
    rate: Rate,
    *,
    allow_mc: bool = False,
    iv_solver: BSMIVSolver | None = None,
    **pricing_kwargs: Any,
) -> pd.DataFrame:
    """
    Prices every option in ``data`` with ``model`` and returns the residuals.

    Parameters
    ----------
    model : BaseModel
        The (calibrated) model.
    data : pd.DataFrame
        Options with ``strike``, ``maturity``, ``optionType``, ``marketPrice``;
        annotated columns (``iv_market``, ``moneyness``) are computed if absent.
    stock, rate : Stock, Rate
        Market parameters.
    allow_mc : bool, optional
        Permit the Monte Carlo pricing route, by default False.
    iv_solver : BSMIVSolver | None, optional
        Solver for the model implied volatilities.
    **pricing_kwargs : Any
        Forwarded to :func:`price_options_vectorized`.

    Returns
    -------
    pd.DataFrame
        ``data`` plus ``model_price``, ``iv_model``, ``price_error``
        (model - market), ``rel_price_error`` and ``iv_error``.
    """
    table = data if "iv_market" in data.columns else annotate_chain(data, stock, rate)
    table = table.reset_index(drop=True).copy()
    if table.empty:
        for col in ("model_price", "iv_model", "price_error", "rel_price_error"):
            table[col] = np.nan
        table["iv_error"] = np.nan
        return table
    prices = price_options_vectorized(
        table, stock, model, rate, allow_mc=allow_mc, **pricing_kwargs
    )
    solver = iv_solver or BSMIVSolver()
    iv_model = solver.solve(
        prices, table, stock, rate, initial_guess=table["iv_market"].to_numpy()
    )
    market = table["marketPrice"].to_numpy(dtype=float)
    table["model_price"] = prices
    table["iv_model"] = iv_model
    table["price_error"] = prices - market
    with np.errstate(divide="ignore", invalid="ignore"):
        table["rel_price_error"] = (prices - market) / market
    table["iv_error"] = iv_model - table["iv_market"].to_numpy()
    return table


def summarize_residuals(table: pd.DataFrame) -> dict[str, float]:
    """
    Scalar metrics from a :func:`residual_table`.

    Returns
    -------
    dict[str, float]
        ``n_options``, ``rmse_price``, ``mae_price``, ``rmse_rel_price``,
        ``rmse_iv`` (vol points as a fraction), ``mae_iv``, ``n_iv_failed``
        (options whose model price has no Black-Scholes volatility).
    """
    if table.empty:
        return {
            "n_options": 0,
            "rmse_price": float("nan"),
            "mae_price": float("nan"),
            "rmse_rel_price": float("nan"),
            "rmse_iv": float("nan"),
            "mae_iv": float("nan"),
            "n_iv_failed": 0,
        }
    iv_err = table["iv_error"].to_numpy(dtype=float)
    return {
        "n_options": int(len(table)),
        "rmse_price": _rmse(table["price_error"]),
        "mae_price": float(np.nanmean(np.abs(table["price_error"]))),
        "rmse_rel_price": _rmse(table["rel_price_error"]),
        "rmse_iv": _rmse(iv_err),
        "mae_iv": float(np.nanmean(np.abs(iv_err)))
        if np.isfinite(iv_err).any()
        else float("nan"),
        "n_iv_failed": int(np.sum(~np.isfinite(iv_err))),
    }


def bucket_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """
    RMSE of the IV and price errors by maturity and by moneyness bucket.

    Returns
    -------
    pd.DataFrame
        Columns ``bucket_type`` ("maturity" / "moneyness"), ``bucket``,
        ``n_options``, ``rmse_iv``, ``rmse_price``.
    """
    if table.empty:
        return pd.DataFrame(
            columns=["bucket_type", "bucket", "n_options", "rmse_iv", "rmse_price"]
        )
    rows = []
    groups = {
        "maturity": maturity_bucket(table["maturity"]),
        "moneyness": moneyness_bucket(table["moneyness"]),
    }
    for kind, labels in groups.items():
        for label, grp in table.groupby(labels, observed=True):
            rows.append(
                {
                    "bucket_type": kind,
                    "bucket": str(label),
                    "n_options": int(len(grp)),
                    "rmse_iv": _rmse(grp["iv_error"]),
                    "rmse_price": _rmse(grp["price_error"]),
                }
            )
    return pd.DataFrame(rows)


def evaluate_model(
    model: BaseModel,
    data: pd.DataFrame,
    stock: Stock,
    rate: Rate,
    *,
    allow_mc: bool = False,
    **pricing_kwargs: Any,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    One-call evaluation: scalar metrics, bucket table and residual table.

    Parameters
    ----------
    model, data, stock, rate
        See :func:`residual_table`.
    allow_mc : bool, optional
        Permit the Monte Carlo pricing route.

    Returns
    -------
    tuple[dict[str, float], pd.DataFrame, pd.DataFrame]
        ``(summary, buckets, residuals)``.
    """
    residuals = residual_table(
        model, data, stock, rate, allow_mc=allow_mc, **pricing_kwargs
    )
    return summarize_residuals(residuals), bucket_metrics(residuals), residuals
