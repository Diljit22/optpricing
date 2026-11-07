from __future__ import annotations

import logging
import math
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from optpricing.config import HISTORICAL_DIR

__doc__ = """
Provides utility functions to download and locally load historical prices,
dividends, log returns and a risk-free rate history.

File formats (all parquet, indexed by date):

* ``{ticker}_{period}_returns.parquet`` - v1 files hold a single ``log_return``
  column; v2 files (written since Phase 4) add ``close`` (dividend- and
  split-adjusted) and ``dividends`` (cash dividend per share on the ex-date).
* ``{RATE_FILE}`` - continuously compounded risk-free rate derived from the
  13-week T-bill yield index ``^IRX`` (column ``rate``).
"""

logger = logging.getLogger(__name__)

RISK_FREE_TICKER = "^IRX"
RATE_FILE = "IRX_risk_free_rates.parquet"
_TBILL_DAYS = 91


def tbill_discount_to_continuous(discount_yield_pct: float) -> float:
    """
    Convert a 13-week T-bill discount yield (percent, as quoted by ``^IRX``)
    into a continuously compounded annual rate.

    ``price = 1 - d * 91/360``; the continuous rate solves
    ``exp(-r * 91/365) = price``.

    Parameters
    ----------
    discount_yield_pct : float
        The quoted discount yield in percent (e.g. ``4.25``).

    Returns
    -------
    float
        Continuously compounded rate as a decimal.
    """
    d = float(discount_yield_pct) / 100.0
    price = 1.0 - d * _TBILL_DAYS / 360.0
    if price <= 0:
        raise ValueError("Discount yield implies a non-positive bill price.")
    return -math.log(price) * 365.0 / _TBILL_DAYS


def save_historical_returns(
    tickers: list[str],
    period: str = "10y",
):
    """
    Fetches and saves historical prices, dividends and log returns for tickers.

    For each ticker the adjusted close history is downloaded from yfinance,
    daily log returns are computed and the frame (``close``, ``dividends``,
    ``log_return``) is written to the historical data directory.

    Parameters
    ----------
    tickers : list[str]
        A list of stock tickers to process, e.g., ['SPY', 'AAPL'].
    period : str, optional
        The time period for which to fetch data, e.g., "10y", "5y", "1mo".
        Defaults to "10y".
    """
    logger.info("Saving %s historical returns for %s", period, tickers)
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if data.empty:
                logger.warning("No historical data found for %s. Skipping.", ticker)
                continue

            frame = pd.DataFrame(index=data.index)
            frame["close"] = data["Close"].astype(float)
            dividends = data["Dividends"] if "Dividends" in data else 0.0
            frame["dividends"] = pd.Series(dividends, index=data.index).fillna(0.0)
            frame["log_return"] = np.log(frame["close"] / frame["close"].shift(1))
            frame = frame.dropna(subset=["log_return"])

            filename = HISTORICAL_DIR / f"{ticker}_{period}_returns.parquet"
            filename.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(filename)
            logger.info("Saved %d rows to %s", len(frame), filename)
        except Exception as e:
            logger.error("FAILED to save data for %s: %s", ticker, e)


def load_historical_prices(
    ticker: str,
    period: str = "10y",
) -> pd.DataFrame:
    """
    Loads the historical frame for a ticker, downloading it if missing.

    Parameters
    ----------
    ticker : str
        The stock ticker, e.g., 'SPY'.
    period : str, optional
        The history period of the file, by default "10y".

    Returns
    -------
    pd.DataFrame
        Frame indexed by date with ``log_return`` and, for v2 files, ``close``
        and ``dividends``.

    Raises
    ------
    FileNotFoundError
        If the data file cannot be found and also fails to be downloaded.
    """
    filename = HISTORICAL_DIR / f"{ticker}_{period}_returns.parquet"
    if not filename.exists():
        logger.info("No historical data for %s; fetching and saving now.", ticker)
        save_historical_returns([ticker], period)
    if not filename.exists():
        raise FileNotFoundError(f"Could not find or save historical data for {ticker}.")
    return pd.read_parquet(filename)


def historical_returns_if_available(
    ticker: str,
    period: str = "10y",
) -> pd.Series | None:
    """
    Historical log returns from the local file, or None if there is no file.

    Unlike :func:`load_historical_returns` this never downloads anything, so
    it is safe inside offline workflows.
    """
    filename = HISTORICAL_DIR / f"{ticker}_{period}_returns.parquet"
    if not filename.exists():
        return None
    return pd.read_parquet(filename)["log_return"]


def load_historical_returns(
    ticker: str,
    period: str = "10y",
) -> pd.Series:
    """
    Loads historical log returns, fetching and saving them if not found.

    Parameters
    ----------
    ticker : str
        The stock ticker for which to load returns, e.g., 'SPY'.
    period : str, optional
        The time period for which to fetch data, e.g., "10y", "5y", "1mo".
        Defaults to "10y".

    Returns
    -------
    pd.Series
        A pandas Series containing the historical log returns.

    Raises
    ------
    FileNotFoundError
        If the data file cannot be found and also fails to be downloaded.
    """
    return load_historical_prices(ticker, period)["log_return"]


def trailing_dividend_yield(
    ticker: str,
    as_of: date | str | pd.Timestamp,
    *,
    period: str = "10y",
    spot: float | None = None,
) -> float | None:
    """
    Trailing twelve-month cash dividend yield from the historical file.

    Parameters
    ----------
    ticker : str
        The stock ticker.
    as_of : date | str | pd.Timestamp
        Valuation date; dividends with ex-dates in the preceding 365 days count.
    period : str, optional
        History period of the file, by default "10y".
    spot : float | None, optional
        Price to divide by; defaults to the last close on or before ``as_of``.

    Returns
    -------
    float | None
        Continuously compounded yield ``log(1 + D/S)``, or None if the file is
        a v1 file without dividends or has no data before ``as_of``.
    """
    filename = HISTORICAL_DIR / f"{ticker}_{period}_returns.parquet"
    if not filename.exists():
        return None
    frame = pd.read_parquet(filename)
    if "dividends" not in frame.columns:
        return None
    idx = pd.DatetimeIndex(frame.index)
    as_of_ts = pd.Timestamp(as_of)
    if idx.tz is not None:
        as_of_ts = as_of_ts.tz_localize(idx.tz) if as_of_ts.tzinfo is None else as_of_ts
    window = frame[(idx > as_of_ts - pd.Timedelta(days=365)) & (idx <= as_of_ts)]
    if window.empty:
        return None
    price = float(window["close"].iloc[-1]) if spot is None else float(spot)
    if price <= 0:
        return None
    return float(math.log1p(float(window["dividends"].sum()) / price))


def save_risk_free_history(period: str = "10y") -> None:
    """
    Downloads the ``^IRX`` (13-week T-bill) history and stores it as a
    continuously compounded rate series in the historical data directory.

    Parameters
    ----------
    period : str, optional
        History period, by default "10y".
    """
    data = yf.Ticker(RISK_FREE_TICKER).history(period=period)
    if data.empty:
        logger.warning("No history returned for %s.", RISK_FREE_TICKER)
        return
    rates = data["Close"].dropna().map(tbill_discount_to_continuous)
    frame = rates.to_frame(name="rate")
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    filename = HISTORICAL_DIR / RATE_FILE
    filename.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(filename)
    logger.info("Saved %d risk-free observations to %s", len(frame), filename)


def load_risk_free_rate(
    as_of: date | str | pd.Timestamp,
    *,
    max_staleness_days: int = 10,
) -> float | None:
    """
    Risk-free rate on (or shortly before) a date from the stored history.

    Parameters
    ----------
    as_of : date | str | pd.Timestamp
        Valuation date.
    max_staleness_days : int, optional
        Oldest acceptable observation, by default 10 days before ``as_of``.

    Returns
    -------
    float | None
        Continuously compounded rate, or None when no history is stored or
        the nearest observation is too old.
    """
    filename = HISTORICAL_DIR / RATE_FILE
    if not filename.exists():
        return None
    frame = pd.read_parquet(filename)
    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    as_of_ts = pd.Timestamp(as_of).normalize()
    mask = (idx <= as_of_ts) & (idx >= as_of_ts - pd.Timedelta(days=max_staleness_days))
    if not mask.any():
        return None
    return float(frame["rate"].to_numpy()[np.flatnonzero(mask)[-1]])


def history_covers_session(
    session: date | str | pd.Timestamp,
    *,
    period: str = "10y",
) -> bool:
    """
    True when a stored returns file contains this New York session.

    Used offline, after the history refresh, to tell a market holiday (no
    close) from a failed option-chain fetch (the close is present).

    Parameters
    ----------
    session : date | str | pd.Timestamp
        Session date.
    period : str, optional
        History period of the files, by default "10y".

    Returns
    -------
    bool
        True if any ``{ticker}_{period}_returns.parquet`` includes the date.
    """
    if not HISTORICAL_DIR.exists():
        return False
    session_day = pd.Timestamp(session).date()
    for path in HISTORICAL_DIR.glob(f"*_{period}_returns.parquet"):
        idx = pd.DatetimeIndex(pd.read_parquet(path).index)
        if idx.tz is not None:
            days = idx.tz_convert("America/New_York").date
        else:
            days = idx.date
        if session_day in set(days):
            return True
    return False


def get_live_risk_free_rate() -> float | None:
    """
    Current risk-free rate from the live ``^IRX`` quote.

    Returns
    -------
    float | None
        Continuously compounded rate, or None if the quote is unavailable.
    """
    try:
        info = yf.Ticker(RISK_FREE_TICKER).fast_info
        quote = info.get("lastPrice", None)
        if quote is None:
            quote = info.get("last_price", None)
        if quote is None or not np.isfinite(quote):
            return None
        return tbill_discount_to_continuous(float(quote))
    except Exception as e:
        logger.warning("Could not fetch %s: %s", RISK_FREE_TICKER, e)
        return None
