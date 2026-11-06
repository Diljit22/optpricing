"""
Shared fixtures for the whole test suite.

`synthetic_snapshot` builds a market snapshot (schema v2) whose mid prices
come from a known model, so calibration and backtest tests can check that the
generating parameters are recovered - and it keeps every workflow test
offline: any attempt to reach yfinance fails loudly through `no_network`.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from hypothesis import settings

from optpricing.atoms import Rate, Stock
from optpricing.calibration.vectorized_pricer import price_options_vectorized
from optpricing.models import BaseModel, BSMModel

# Property tests: a fixed example set by default (reproducible CI, no flakes);
# HYPOTHESIS_PROFILE=thorough searches randomly and much harder.
settings.register_profile("default", derandomize=True, deadline=None, max_examples=60)
settings.register_profile("thorough", deadline=None, max_examples=2000)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


def make_snapshot(
    model: BaseModel,
    *,
    ticker: str = "TEST",
    snapshot_date: str = "2025-06-16",
    spot: float = 100.0,
    r: float = 0.04,
    q: float = 0.01,
    maturities: tuple[float, ...] = (30 / 365.25, 91 / 365.25, 182 / 365.25, 1.0),
    moneyness: np.ndarray | None = None,
    relative_spread: float = 0.02,
    noise_std: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """A schema-v2 snapshot priced by ``model`` with OTM calls and puts."""
    rng = np.random.default_rng(seed)
    if moneyness is None:
        moneyness = np.linspace(0.8, 1.2, 17)
    rows = []
    for T in maturities:
        F = spot * np.exp((r - q) * T)
        for m in moneyness:
            K = round(F * m, 1)
            rows.append({"strike": K, "maturity": T, "optionType": "call"})
            rows.append({"strike": K, "maturity": T, "optionType": "put"})
    df = pd.DataFrame(rows)
    stock, rate = Stock(spot=spot, dividend=q), Rate(rate=r)
    prices = price_options_vectorized(df, stock, model, rate)
    prices = prices * (1.0 + noise_std * rng.standard_normal(len(prices)))
    half_spread = np.maximum(0.5 * relative_spread * prices, 0.005)
    df["marketPrice"] = prices
    df["bid"] = prices - half_spread
    df["ask"] = prices + half_spread
    df["lastPrice"] = prices
    df["volume"] = 50.0
    df["openInterest"] = 500.0
    df["impliedVolatility"] = 0.2
    expiry = pd.Timestamp(snapshot_date) + pd.to_timedelta(
        np.round(df["maturity"] * 365.25), unit="D"
    )
    df["expiry"] = expiry
    df["contractSymbol"] = [
        f"{ticker}{e.strftime('%y%m%d')}{t[0].upper()}{int(k * 1000):08d}"
        for e, t, k in zip(df["expiry"], df["optionType"], df["strike"])
    ]
    df["lastTradeDate"] = pd.Timestamp(snapshot_date + " 15:00", tz="UTC")
    df["spot_price"] = spot
    df["ticker"] = ticker
    df["snapshot_date"] = snapshot_date
    df["as_of"] = pd.Timestamp(snapshot_date + " 20:00", tz="UTC")
    df["risk_free_rate"] = r
    df["dividend_yield"] = q
    df["yf_dividend_yield"] = q
    return df[df["marketPrice"] > 0.02].reset_index(drop=True)


@pytest.fixture
def synthetic_snapshot():
    """Factory fixture: ``synthetic_snapshot(model, **kwargs) -> DataFrame``."""
    return make_snapshot


@pytest.fixture
def bsm_snapshot():
    return make_snapshot(BSMModel(params={"sigma": 0.23}))


@pytest.fixture
def no_network(monkeypatch):
    """Any yfinance call inside the test raises."""

    def _boom(*args, **kwargs):
        raise AssertionError("network access attempted during an offline workflow")

    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", _boom)
    monkeypatch.setattr(yfinance, "download", _boom)
    monkeypatch.setattr("optpricing.data.market_data_manager.yf.Ticker", _boom)
    monkeypatch.setattr("optpricing.data.historical_manager.yf.Ticker", _boom)
    return _boom
