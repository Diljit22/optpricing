from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np
import pandas as pd

from optpricing.atoms import Rate, Stock

from .vectorized_bsm_iv import BSMIVSolver

__doc__ = """
Liquidity filters and calibration settings.

:class:`LiquidityFilter` turns a raw option chain into a calibration set:
quotes with a bid, a tight spread, some open interest, a maturity inside a
window and a forward moneyness inside a band, with a Black-Scholes implied
volatility that exists. :class:`CalibrationSettings` bundles such a filter
with the objective, weighting and optimiser options; a settings object is
what the walk-forward tuner searches over and caches per ticker.
"""

MATURITY_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("<1m", 0.0, 1 / 12),
    ("1-3m", 1 / 12, 0.25),
    ("3-6m", 0.25, 0.5),
    ("6-12m", 0.5, 1.0),
    (">12m", 1.0, np.inf),
)
MONEYNESS_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("deep OTM put", 0.0, 0.85),
    ("OTM put", 0.85, 0.97),
    ("ATM", 0.97, 1.03),
    ("OTM call", 1.03, 1.15),
    ("deep OTM call", 1.15, np.inf),
)


@dataclass(frozen=True)
class LiquidityFilter:
    """
    Rules selecting the options a model is fitted to (or evaluated on).

    Parameters
    ----------
    min_maturity_days, max_maturity_years : float
        Maturity window; default 7 days to 2 years.
    moneyness : tuple[float, float]
        Allowed forward moneyness ``K / F`` with ``F = S e^{(r-q)T}``;
        default (0.7, 1.3).
    max_relative_spread : float
        Maximum ``(ask - bid) / mid``; default 0.25. Ignored without quotes.
    min_open_interest : int
        Minimum open interest (or volume when open interest is missing); 1.
    require_bid : bool
        Drop quotes with a zero bid; default True.
    otm_only : bool
        Keep only out-of-the-money options (puts below the forward, calls above),
        the most liquid side of the chain; default True.
    iv_bounds : tuple[float, float]
        Market implied volatilities outside this range are treated as data
        errors; default (0.02, 3.0).
    """

    min_maturity_days: float = 7.0
    max_maturity_years: float = 2.0
    moneyness: tuple[float, float] = (0.7, 1.3)
    max_relative_spread: float = 0.25
    min_open_interest: int = 1
    require_bid: bool = True
    otm_only: bool = True
    iv_bounds: tuple[float, float] = (0.02, 3.0)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (tuples become lists, as JSON would store them)."""
        data = asdict(self)
        for key in ("moneyness", "iv_bounds"):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LiquidityFilter:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        for key in ("moneyness", "iv_bounds"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)

    def evaluation(self) -> LiquidityFilter:
        """
        The wider companion filter used to *evaluate* a fit: moneyness widened
        by 0.1 on each side and twice the spread tolerance, same maturity
        window, liquidity floor and OTM rule (in-the-money American puts carry
        an early-exercise premium no European model should be judged on).
        """
        lo, hi = self.moneyness
        return LiquidityFilter(
            min_maturity_days=self.min_maturity_days,
            max_maturity_years=self.max_maturity_years,
            moneyness=(max(lo - 0.1, 0.05), hi + 0.1),
            max_relative_spread=min(2.0 * self.max_relative_spread, 1.0),
            min_open_interest=self.min_open_interest,
            require_bid=self.require_bid,
            otm_only=self.otm_only,
            iv_bounds=self.iv_bounds,
        )


def annotate_chain(
    df: pd.DataFrame,
    stock: Stock,
    rate: Rate,
    *,
    iv_solver: BSMIVSolver | None = None,
) -> pd.DataFrame:
    """
    Adds pricing helpers to a chain: ``forward``, ``moneyness`` (``K/F``),
    ``log_moneyness``, ``relative_spread`` and ``iv_market`` (Black-Scholes
    implied volatility of the mid, NaN where none exists).

    Parameters
    ----------
    df : pd.DataFrame
        Options with ``strike``, ``maturity``, ``optionType``, ``marketPrice``
        and optionally ``bid``/``ask``.
    stock, rate : Stock, Rate
        Market parameters used for the forward and the implied volatilities.
    iv_solver : BSMIVSolver | None, optional
        Solver instance (default settings if omitted).

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with the extra columns and a fresh integer index.
    """
    out = df.reset_index(drop=True).copy()
    T = out["maturity"].to_numpy(dtype=float)
    if callable(rate.rate):
        r = np.asarray([rate.get_rate(t) for t in T], dtype=float)
    else:
        r = np.full(T.shape, float(rate.get_rate()))
    forward = stock.spot * np.exp((r - stock.dividend) * T)
    out["forward"] = forward
    out["moneyness"] = out["strike"].to_numpy(dtype=float) / forward
    out["log_moneyness"] = np.log(out["moneyness"].to_numpy())
    if {"bid", "ask"} <= set(out.columns):
        mid = out["marketPrice"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["relative_spread"] = (out["ask"] - out["bid"]).to_numpy() / mid
    else:
        out["relative_spread"] = np.nan
    solver = iv_solver or BSMIVSolver()
    guess = (
        out["impliedVolatility"].to_numpy(dtype=float)
        if "impliedVolatility" in out.columns
        else None
    )
    out["iv_market"] = solver.solve(
        out["marketPrice"].to_numpy(dtype=float), out, stock, rate, initial_guess=guess
    )
    return out


def filter_options(
    df: pd.DataFrame,
    stock: Stock,
    rate: Rate,
    liquidity: LiquidityFilter | None = None,
) -> pd.DataFrame:
    """
    Applies a :class:`LiquidityFilter` to a chain.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or annotated chain (see :func:`annotate_chain`).
    stock, rate : Stock, Rate
        Market parameters (forward, implied volatilities).
    liquidity : LiquidityFilter | None, optional
        Rules to apply; defaults to ``LiquidityFilter()``.

    Returns
    -------
    pd.DataFrame
        The surviving options, annotated, with a fresh integer index.
    """
    liq = liquidity or LiquidityFilter()
    data = df if "iv_market" in df.columns else annotate_chain(df, stock, rate)
    T = data["maturity"]
    mask = (T >= liq.min_maturity_days / 365.25) & (T <= liq.max_maturity_years)
    mask &= data["moneyness"].between(*liq.moneyness)
    mask &= data["marketPrice"] > 0
    if liq.require_bid and "bid" in data.columns:
        mask &= data["bid"] > 0
    if "relative_spread" in data.columns:
        spread = data["relative_spread"]
        mask &= spread.isna() | (spread <= liq.max_relative_spread)
    if liq.min_open_interest > 0:
        if "openInterest" in data.columns:
            activity = data["openInterest"].fillna(0)
            if "volume" in data.columns:
                activity = activity.where(activity > 0, data["volume"].fillna(0))
            mask &= activity >= liq.min_open_interest
        elif "volume" in data.columns:
            mask &= data["volume"].fillna(0) >= liq.min_open_interest
    if liq.otm_only:
        is_call = data["optionType"] == "call"
        mask &= np.where(is_call, data["moneyness"] >= 1.0, data["moneyness"] < 1.0)
    iv = data["iv_market"]
    mask &= iv.notna() & iv.between(*liq.iv_bounds)
    return data[mask.fillna(False)].reset_index(drop=True)


def maturity_bucket(maturity: pd.Series | np.ndarray) -> pd.Categorical:
    """Labels maturities with :data:`MATURITY_BUCKETS`."""
    edges = [b[1] for b in MATURITY_BUCKETS] + [np.inf]
    labels = [b[0] for b in MATURITY_BUCKETS]
    return pd.cut(np.asarray(maturity, dtype=float), edges, labels=labels, right=False)


def moneyness_bucket(moneyness: pd.Series | np.ndarray) -> pd.Categorical:
    """Labels forward moneyness ``K/F`` with :data:`MONEYNESS_BUCKETS`."""
    edges = [b[1] for b in MONEYNESS_BUCKETS] + [np.inf]
    labels = [b[0] for b in MONEYNESS_BUCKETS]
    return pd.cut(np.asarray(moneyness, dtype=float), edges, labels=labels, right=False)


@dataclass(frozen=True)
class CalibrationSettings:
    """
    Everything a calibration run needs besides the data and the model.

    Parameters
    ----------
    liquidity : LiquidityFilter
        Selection of the calibration set.
    objective : {"iv", "price"}
        Residuals in Black-Scholes implied-volatility or in price space.
    weighting : {"spread", "vega", "none"}
        Residual weights: inverse bid-ask spread, inverse vega (price space
        only; approximates the IV objective) or equal.
    n_starts : int
        Number of starting points (Latin hypercube in the bounds) besides the
        recipe's initial guess and the warm start; 1 = local optimisation only.
    warm_start : bool
        Seed the optimiser with the previous calibration (when one is given).
    historical_prior : bool
        Use historical-return estimates for jump parameters as a start point.
    feller_penalty : float
        Soft penalty weight on ``vol_of_vol^2 - 2 kappa theta`` for Heston-type
        models (0 disables it).
    time_budget : float | None
        Wall-clock seconds per model (None = unlimited).
    name : str
        Label used in tuning reports.
    """

    liquidity: LiquidityFilter = field(default_factory=LiquidityFilter)
    objective: str = "iv"
    weighting: str = "spread"
    n_starts: int = 4
    warm_start: bool = True
    historical_prior: bool = True
    feller_penalty: float = 0.0
    time_budget: float | None = None
    name: str = "default"

    def __post_init__(self) -> None:
        if self.objective not in ("iv", "price"):
            raise ValueError("objective must be 'iv' or 'price'.")
        if self.weighting not in ("spread", "vega", "none"):
            raise ValueError("weighting must be 'spread', 'vega' or 'none'.")
        if self.n_starts < 1:
            raise ValueError("n_starts must be at least 1.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["liquidity"] = self.liquidity.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationSettings:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        liq = kwargs.get("liquidity")
        if isinstance(liq, dict):
            kwargs["liquidity"] = LiquidityFilter.from_dict(liq)
        elif liq is None:
            kwargs.pop("liquidity", None)
        return cls(**kwargs)

    def replace(self, **changes: Any) -> CalibrationSettings:
        """A copy with some fields changed (``liquidity`` may be a dict of
        filter fields to update)."""
        liq = changes.pop("liquidity", None)
        data = self.to_dict()
        data.update(changes)
        if isinstance(liq, LiquidityFilter):
            data["liquidity"] = liq.to_dict()
        elif isinstance(liq, dict):
            data["liquidity"] = {**data["liquidity"], **liq}
        return CalibrationSettings.from_dict(data)


DEFAULT_SETTINGS = CalibrationSettings()
