from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from optpricing.atoms import Rate, Stock
from optpricing.config import MARKET_SNAPSHOT_DIR, _config

from .historical_manager import (
    get_live_risk_free_rate,
    load_risk_free_rate,
    trailing_dividend_yield,
)

__doc__ = """
Utility functions to download and load market snapshots, live option chains,
and dividend yields.

Snapshot schema
---------------
A snapshot is one parquet file per ticker and day holding the full option
chain as returned by yfinance (``contractSymbol``, ``strike``, ``bid``, ``ask``,
``volume``, ``openInterest``, ``impliedVolatility``, ``expiry`` ...) plus the
derived columns ``maturity`` (years), ``marketPrice`` and ``spot_price``.

``spot_price`` is the underlying print at capture. Calibration and the
backtest read it from the chain. The adjusted close in the returns file is a
different series, used only as a jump-model prior, and is not this spot.

``price_source`` is ``mid`` when the bid and ask were quoted and ``last`` when
the book was empty and the price is that session's last trade.

Version 2 (Phase 4) adds the market parameters needed to price the chain
without any network access, constant over the file:

* ``ticker``, ``snapshot_date`` (``YYYY-MM-DD``), ``as_of`` (UTC capture time)
* ``risk_free_rate`` - continuously compounded, from the 13-week T-bill quote
* ``dividend_yield`` - continuous yield implied by put-call parity given the rate
* ``yf_dividend_yield`` - the trailing yield reported by yfinance (informational)

:func:`load_market_snapshot` upgrades v1 files on the fly (rate from the stored
``^IRX`` history when available, dividend yield re-implied from the chain).
"""

logger = logging.getLogger(__name__)

_NY = ZoneInfo("America/New_York")
SNAPSHOT_SCHEMA_VERSION = 2
V2_COLUMNS = ("ticker", "snapshot_date", "as_of", "risk_free_rate", "dividend_yield")
DEFAULT_RISK_FREE_RATE = 0.04
# Columns dropped by `compact_snapshot`: constant per file, informational, or
# reconstructible on load (`marketPrice` = mid, `contractSymbol` = OCC code).
_COMPACT_DROP = (
    "contractSize",
    "currency",
    "change",
    "percentChange",
    "inTheMoney",
    "lastPrice",
    "lastTradeDate",
    "marketPrice",
    "contractSymbol",
)


def _spot_from_ticker(ticker_obj: yf.Ticker) -> float:
    """Last traded price from `fast_info` (yfinance >= 0.2.x keys), else history."""
    try:
        fast = ticker_obj.fast_info
        price = getattr(fast, "last_price", None)
        if price is None and hasattr(fast, "get"):
            price = fast.get("lastPrice", fast.get("last_price"))
        if price is not None and np.isfinite(price) and price > 0:
            return float(price)
    except Exception:  # pragma: no cover - network/library specific
        pass
    history = ticker_obj.history(period="1d")
    if history.empty:
        raise ValueError("No price history available for spot.")
    return float(history["Close"].iloc[-1])


def _normalise_dividend_yield(value: float | None) -> float:
    """
    yfinance reports ``dividendYield`` as a fraction in old versions and as a
    percentage (e.g. ``1.23``) since 0.2.5x. Values above 0.5 are treated as
    percentages; no listed stock yields more than 50 %.
    """
    if value is None:
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value) or value < 0:
        return 0.0
    return value / 100.0 if value > 0.5 else value


def session_date(now: datetime | None = None) -> date:
    """
    US cash-session date of ``now`` in America/New_York.

    GitHub's runners are UTC. A job that starts just after midnight UTC is
    still the previous evening in New York, so filenames and the freshness
    check follow the exchange date rather than the runner clock.

    Parameters
    ----------
    now : datetime | None, optional
        Instant to classify. Naive values are read as UTC. Defaults to now.

    Returns
    -------
    date
        The New York calendar date.
    """
    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(_NY).date()


def apply_quote_prices(chain: pd.DataFrame, session: date) -> pd.DataFrame:
    """
    Price each row from the live bid/ask mid, or that session's last trade.

    Yahoo clears the book after the close, and a delayed snapshot job then
    sees ``bid == ask == 0`` on contracts that did trade. Those rows are kept
    with ``price_source == "last"`` and ``bid == ask == lastPrice``, so the
    liquidity filters (positive bid, finite spread) still accept them. A
    quoted book is left as ``price_source == "mid"``. Rows with neither a
    book nor a last trade on ``session`` are dropped.

    Parameters
    ----------
    chain : pd.DataFrame
        Raw chain, one row per contract.
    session : date
        New York session the snapshot belongs to.

    Returns
    -------
    pd.DataFrame
        The priced rows.
    """
    out = chain.copy()
    index = out.index
    if "bid" in out.columns:
        bid = pd.to_numeric(out["bid"], errors="coerce")
    else:
        bid = pd.Series(np.nan, index=index)
    if "ask" in out.columns:
        ask = pd.to_numeric(out["ask"], errors="coerce")
    else:
        ask = pd.Series(np.nan, index=index)
    mid = (bid + ask) / 2.0
    has_book = (bid > 0) & (ask >= bid) & (mid > 0.01)

    if "lastPrice" in out.columns:
        last = pd.to_numeric(out["lastPrice"], errors="coerce")
    else:
        last = pd.Series(np.nan, index=index)
    if "lastTradeDate" in out.columns:
        traded = pd.to_datetime(out["lastTradeDate"], utc=True, errors="coerce")
        traded_ny = traded.dt.tz_convert(_NY)
        traded_on_session = pd.Series(traded_ny.dt.date == session, index=index)
        traded_on_session = traded_on_session.fillna(False)
    else:
        traded_on_session = pd.Series(False, index=index)
    has_last = ~has_book & traded_on_session & (last > 0.01)

    keep = has_book | has_last
    out = out.loc[keep].copy()
    has_book = has_book.loc[keep]
    has_last = has_last.loc[keep]
    mid = mid.loc[keep]
    last = last.loc[keep]
    out["marketPrice"] = np.where(has_book.to_numpy(), mid.to_numpy(), last.to_numpy())
    out["price_source"] = np.where(has_book.to_numpy(), "mid", "last")
    out.loc[has_last, "bid"] = last.loc[has_last].to_numpy()
    out.loc[has_last, "ask"] = last.loc[has_last].to_numpy()
    return out


def _fetch_from_yfinance(ticker: str) -> pd.DataFrame | None:
    """Fetches a live option chain (schema v2) using yfinance."""
    logger.info("Fetching live option chain for %s from yfinance...", ticker)
    try:
        ticker_obj = yf.Ticker(ticker)
        expirations = ticker_obj.options
        if not expirations:
            return None

        all_options = []
        for expiry in expirations:
            opt = ticker_obj.option_chain(expiry)
            if not opt.calls.empty:
                calls = opt.calls.copy()
                calls["optionType"] = "call"
                calls["expiry"] = pd.to_datetime(expiry)
                all_options.append(calls)
            if not opt.puts.empty:
                puts = opt.puts.copy()
                puts["optionType"] = "put"
                puts["expiry"] = pd.to_datetime(expiry)
                all_options.append(puts)

        if not all_options:
            return None

        chain_df = pd.concat(all_options, ignore_index=True)
        if "contractSymbol" in chain_df.columns:
            chain_df = chain_df.drop_duplicates(subset=["contractSymbol"])
        chain_df = chain_df.drop_duplicates(subset=["strike", "expiry", "optionType"])

        session = session_date()
        session_ts = datetime.combine(session, datetime.min.time())
        chain_df["maturity"] = (chain_df["expiry"] - session_ts).dt.days / 365.25
        chain_df["spot_price"] = _spot_from_ticker(ticker_obj)

        n_raw = len(chain_df)
        chain_df = apply_quote_prices(chain_df, session)
        chain_df = chain_df.dropna(
            subset=["marketPrice", "strike", "maturity", "impliedVolatility"],
        )
        chain_df = chain_df[chain_df["maturity"] > 1 / 365.25].copy()
        if chain_df.empty:
            return None
        # Share of the raw chain that still has a usable price. A cleared book
        # with no last trade from this session falls through `is_fresh_snapshot`.
        chain_df.attrs["n_raw_quotes"] = n_raw
        n_last = int((chain_df["price_source"] == "last").sum())
        if n_last:
            logger.info(
                "%s: %d contract(s) priced from the session's last trade",
                ticker,
                n_last,
            )

        rate = get_live_risk_free_rate()
        if rate is None:
            rate = load_risk_free_rate(session)
        if rate is None:
            rate = DEFAULT_RISK_FREE_RATE
        yf_yield = np.nan
        try:
            yf_yield = _normalise_dividend_yield(ticker_obj.info.get("dividendYield"))
        except Exception:  # pragma: no cover - network specific
            pass
        return _attach_market_params(
            chain_df,
            ticker=ticker,
            snapshot_date=session.strftime("%Y-%m-%d"),
            rate=rate,
            yf_dividend_yield=yf_yield,
            as_of=pd.Timestamp.now(tz="UTC"),
        )

    except Exception as e:
        logger.error("FAILED to fetch live yfinance data for %s. Error: %s", ticker, e)
        return None


def _attach_market_params(
    chain_df: pd.DataFrame,
    *,
    ticker: str,
    snapshot_date: str,
    rate: float,
    yf_dividend_yield: float = np.nan,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Adds the constant v2 columns (ticker, date, r, parity-implied q) to a chain."""
    from optpricing.calibration.fit_market_params import implied_dividend_yield

    chain_df = chain_df.copy()
    spot = float(chain_df["spot_price"].iloc[0])
    calls = chain_df[chain_df["optionType"] == "call"]
    puts = chain_df[chain_df["optionType"] == "put"]
    q = implied_dividend_yield(calls, puts, spot, rate)
    chain_df["ticker"] = ticker.upper()
    chain_df["snapshot_date"] = snapshot_date
    chain_df["as_of"] = (
        as_of
        if as_of is not None
        else pd.Timestamp(snapshot_date).tz_localize("UTC") + pd.Timedelta(hours=20)
    )
    chain_df["risk_free_rate"] = float(rate)
    chain_df["dividend_yield"] = float(q)
    chain_df["yf_dividend_yield"] = float(yf_dividend_yield)
    return chain_df


def is_v2_snapshot(df: pd.DataFrame) -> bool:
    """True if the frame carries the schema-v2 market-parameter columns."""
    return all(col in df.columns for col in V2_COLUMNS)


def upgrade_snapshot(
    df: pd.DataFrame,
    ticker: str,
    snapshot_date: str,
    *,
    rate: float | None = None,
) -> pd.DataFrame:
    """
    Upgrades a v1 snapshot to schema v2 without network access.

    The risk-free rate is taken from the stored ``^IRX`` history for the
    snapshot date (``rate`` overrides it); when neither is available a flat
    ``DEFAULT_RISK_FREE_RATE`` is used. The dividend yield is implied from
    put-call parity given that rate.

    Parameters
    ----------
    df : pd.DataFrame
        The v1 snapshot frame.
    ticker : str
        Ticker symbol of the snapshot.
    snapshot_date : str
        ``YYYY-MM-DD`` date of the snapshot.
    rate : float | None, optional
        Risk-free rate to use instead of the stored history.

    Returns
    -------
    pd.DataFrame
        The frame with the v2 columns added (already-v2 frames are returned
        unchanged).
    """
    if is_v2_snapshot(df):
        return df
    if rate is None:
        rate = load_risk_free_rate(snapshot_date)
    if rate is None:
        logger.warning(
            "No risk-free history for %s; using %.2f. Run `optpricing data "
            "download --rates` to store the ^IRX history.",
            snapshot_date,
            DEFAULT_RISK_FREE_RATE,
        )
        rate = DEFAULT_RISK_FREE_RATE
    yf_yield = trailing_dividend_yield(ticker, snapshot_date)
    return _attach_market_params(
        df,
        ticker=ticker,
        snapshot_date=snapshot_date,
        rate=rate,
        yf_dividend_yield=np.nan if yf_yield is None else yf_yield,
    )


def snapshot_market_params(df: pd.DataFrame) -> tuple[Stock, Rate]:
    """
    Builds the ``Stock`` and ``Rate`` atoms described by a snapshot.

    Uses the stored v2 columns; for v1 frames (or frames without them, e.g.
    synthetic test data) the rate defaults to ``DEFAULT_RISK_FREE_RATE`` and the
    dividend yield is implied from the chain, all offline.

    Parameters
    ----------
    df : pd.DataFrame
        A market snapshot with at least ``spot_price``, ``strike``, ``maturity``,
        ``marketPrice`` and ``optionType``.

    Returns
    -------
    tuple[Stock, Rate]
        The underlying (spot and continuous dividend yield) and the flat rate.
    """
    spot = float(df["spot_price"].iloc[0])
    if is_v2_snapshot(df):
        return (
            Stock(spot=spot, dividend=float(df["dividend_yield"].iloc[0])),
            Rate(rate=float(df["risk_free_rate"].iloc[0])),
        )
    from optpricing.calibration.fit_market_params import implied_dividend_yield

    rate = DEFAULT_RISK_FREE_RATE
    calls = df[df["optionType"] == "call"]
    puts = df[df["optionType"] == "put"]
    q = (
        implied_dividend_yield(calls, puts, spot, rate)
        if len(calls) and len(puts)
        else 0
    )
    return Stock(spot=spot, dividend=float(q)), Rate(rate=rate)


def get_live_option_chain(ticker: str) -> pd.DataFrame | None:
    """
    Fetches a live option chain from the configured data provider.

    The data provider is determined by the `live_data_provider` key in the
    `config.yaml` file. Supported providers are "yfinance".

    Parameters
    ----------
    ticker : str
        The stock ticker for which to fetch the option chain, e.g., 'SPY'.

    Returns
    -------
    pd.DataFrame | None
        A DataFrame containing the formatted option chain data (schema v2),
        or None if the fetch fails or no data is returned.
    """
    provider = _config.get("live_data_provider", "yfinance").lower()
    if provider != "yfinance":
        logger.warning(
            "Unknown live_data_provider '%s'. Defaulting to yfinance.", provider
        )
    return _fetch_from_yfinance(ticker)


def liquid_option_mask(
    df: pd.DataFrame,
    *,
    max_relative_spread: float = 0.3,
    min_maturity_days: float = 3.0,
    max_maturity_years: float = 1.5,
    moneyness: tuple[float, float] = (0.7, 1.3),
    min_open_interest: int = 5,
    min_volume: int = 1,
) -> pd.Series:
    """
    Boolean mask of options worth keeping in a compact snapshot.

    A row passes when it has a positive bid, a bid-ask spread no wider than
    ``max_relative_spread`` of the mid, open interest of at least
    ``min_open_interest`` or volume of at least ``min_volume``, a maturity
    inside the window and a strike within the moneyness band.

    Parameters
    ----------
    df : pd.DataFrame
        A snapshot frame.
    max_relative_spread : float, optional
        Maximum ``(ask - bid) / mid``, by default 0.3.
    min_maturity_days, max_maturity_years : float, optional
        Maturity window, by default 3 days to 1.5 years.
    moneyness : tuple[float, float], optional
        Allowed ``strike / spot`` range, by default (0.7, 1.3).
    min_open_interest, min_volume : int, optional
        Activity floor (either one suffices), by default 5 contracts / 1 trade.

    Returns
    -------
    pd.Series
        Boolean mask aligned with ``df``.
    """
    spot = df["spot_price"]
    mid = df["marketPrice"]
    spread = (df["ask"] - df["bid"]) / mid.where(mid > 0, np.nan)
    zeros = pd.Series(0.0, index=df.index)
    open_interest = df["openInterest"] if "openInterest" in df.columns else zeros
    volume = df["volume"] if "volume" in df.columns else zeros
    active = (open_interest.fillna(0) >= min_open_interest) | (
        volume.fillna(0) >= min_volume
    )
    mask = (
        (df["bid"] > 0)
        & (spread <= max_relative_spread)
        & active
        & (df["maturity"] >= min_maturity_days / 365.25)
        & (df["maturity"] <= max_maturity_years)
        & (df["strike"] / spot >= moneyness[0])
        & (df["strike"] / spot <= moneyness[1])
    )
    return mask.fillna(False)


def compact_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Liquid-only, slimmed snapshot for compressed storage (e.g. the scheduled
    GitHub Action that grows the corpus): about 25 KB per ticker and day with
    zstd, versus ~130 KB for a full snapshot.

    Keeps the rows selected by :func:`liquid_option_mask` and ``price_source``,
    drops columns that are constant, informational or reconstructible
    (``marketPrice`` is the bid-ask mid, ``contractSymbol`` the OCC code; both
    are restored by :func:`load_market_snapshot`) and stores quotes in float32
    (cent precision for strikes below ~80,000) and counts in int32.

    Parameters
    ----------
    df : pd.DataFrame
        A full snapshot frame.

    Returns
    -------
    pd.DataFrame
        The compacted frame.
    """
    out = df[liquid_option_mask(df)].copy()
    out = out.drop(columns=[c for c in _COMPACT_DROP if c in out.columns])
    for col in ("bid", "ask", "impliedVolatility", "strike", "maturity"):
        if col in out.columns:
            out[col] = out[col].astype("float32")
    for col in ("volume", "openInterest"):
        if col in out.columns:
            out[col] = out[col].fillna(0).astype("int32")
    return out.reset_index(drop=True)


def occ_contract_symbol(
    ticker: str, expiry: pd.Timestamp, option_type: str, strike: float
) -> str:
    """OCC option symbol, e.g. ``SPY250618C00500000``."""
    letter = "C" if str(option_type).lower().startswith("c") else "P"
    strike_code = f"{int(round(strike * 1000)):08d}"
    return f"{ticker.upper()}{pd.Timestamp(expiry):%y%m%d}{letter}{strike_code}"


def restore_compact_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Re-creates ``marketPrice`` and ``contractSymbol`` dropped by compaction
    and widens the quote columns back to float64."""
    out = df.copy()
    for col in ("bid", "ask", "impliedVolatility", "strike", "maturity"):
        if col in out.columns and out[col].dtype != np.float64:
            out[col] = out[col].astype("float64")
    if "marketPrice" not in out.columns and {"bid", "ask"} <= set(out.columns):
        out["marketPrice"] = (out["bid"] + out["ask"]) / 2.0
    if "contractSymbol" not in out.columns and {
        "expiry",
        "optionType",
        "strike",
    } <= set(out.columns):
        out["contractSymbol"] = [
            occ_contract_symbol(ticker, e, t, k)
            for e, t, k in zip(out["expiry"], out["optionType"], out["strike"])
        ]
    return out


def is_fresh_snapshot(
    df: pd.DataFrame,
    snapshot_date: date | None = None,
    *,
    min_quoted_share: float = 0.2,
) -> bool:
    """
    True if the chain looks live: trades on the snapshot date (the market was
    open) and a reasonable share of contracts carrying a quote (outside market
    hours Yahoo zeroes most bids and asks).

    Parameters
    ----------
    df : pd.DataFrame
        A snapshot frame with a ``lastTradeDate`` column; ``df.attrs`` may hold
        ``n_raw_quotes`` (row count before quote filtering) set by the fetcher.
    snapshot_date : date | None, optional
        Date to compare with, by default today.
    min_quoted_share : float, optional
        Minimum fraction of raw contracts with a usable quote, by default 0.2.
    """
    if df.empty:
        return False
    n_raw = df.attrs.get("n_raw_quotes")
    if n_raw and len(df) / n_raw < min_quoted_share:
        return False
    if "lastTradeDate" not in df.columns:
        return True
    last_trade = pd.to_datetime(df["lastTradeDate"], utc=True, errors="coerce").max()
    if pd.isna(last_trade):
        return False
    check_date = snapshot_date or date.today()
    # Compare in US/Eastern, the listing exchanges' time zone.
    return last_trade.tz_convert("America/New_York").date() >= check_date


def save_market_snapshot(
    tickers: list[str],
    *,
    compact: bool = False,
    only_fresh: bool = False,
) -> list[Path]:
    """
    Saves a snapshot of the current market option chain for given tickers.

    For each ticker, it fetches the live option chain using
    `get_live_option_chain` and saves it to a parquet file named with the
    ticker and the New York session date (see :func:`session_date`).

    Parameters
    ----------
    tickers : list[str]
        A list of stock tickers to process, e.g., ['SPY', 'AAPL'].
    compact : bool, optional
        Store only liquid options with zstd compression (see
        :func:`compact_snapshot`), by default False.
    only_fresh : bool, optional
        Skip tickers whose chain shows no trade today (holidays, stale feeds),
        by default False.

    Returns
    -------
    list[Path]
        Paths of the files written.
    """
    session = session_date()
    session_str = session.strftime("%Y-%m-%d")
    logger.info("Saving market data snapshot for %s", session_str)
    written: list[Path] = []
    for ticker in tickers:
        chain_df = get_live_option_chain(ticker)

        if chain_df is None or chain_df.empty:
            logger.warning("No valid option data found for %s. Skipping.", ticker)
            continue
        if only_fresh and not is_fresh_snapshot(chain_df, session):
            logger.warning(
                "Chain for %s is not live (no trades today or quotes missing). "
                "Skipping.",
                ticker,
            )
            continue
        if compact:
            chain_df = compact_snapshot(chain_df)
            if chain_df.empty:
                logger.warning("No liquid options for %s. Skipping.", ticker)
                continue

        filename = MARKET_SNAPSHOT_DIR / f"{ticker}_{session_str}.parquet"
        filename.parent.mkdir(parents=True, exist_ok=True)
        if compact:
            chain_df.to_parquet(filename, compression="zstd", index=False)
        else:
            chain_df.to_parquet(filename)
        logger.info("Saved %d options to %s", len(chain_df), filename)
        written.append(filename)
    return written


def load_market_snapshot(ticker: str, snapshot_date: str) -> pd.DataFrame | None:
    """
    Loads a previously saved market data snapshot for a specific date.

    Version-1 files are upgraded to schema v2 in memory (see
    :func:`upgrade_snapshot`); the file on disk is not modified.

    Parameters
    ----------
    ticker : str
        The stock ticker of the desired snapshot, e.g., 'SPY'.
    snapshot_date : str
        The date of the snapshot in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame | None
        A DataFrame containing the snapshot data, or None if the file
        is not found.
    """
    filename = MARKET_SNAPSHOT_DIR / f"{ticker}_{snapshot_date}.parquet"
    if not filename.exists():
        logger.error("Snapshot file not found: %s", filename)
        return None

    logger.info("Loading data from %s...", filename)
    df = pd.read_parquet(filename)
    if "marketPrice" not in df.columns or "contractSymbol" not in df.columns:
        df = restore_compact_columns(df, ticker)
    if not is_v2_snapshot(df):
        df = upgrade_snapshot(df, ticker, snapshot_date)
    return df


def get_available_snapshot_dates(ticker: str, *, ascending: bool = False) -> list[str]:
    """
    Lists all available snapshot dates for a given ticker.

    Scans the market data directory for saved parquet files corresponding
    to the ticker and extracts the date from the filenames.

    Parameters
    ----------
    ticker : str
        The stock ticker to search for, e.g., 'SPY'.
    ascending : bool, optional
        Return oldest first (what a backtest needs); by default newest first.

    Returns
    -------
    list[str]
        A sorted list of available dates in 'YYYY-MM-DD' format.
    """
    try:
        files = [
            f.name
            for f in MARKET_SNAPSHOT_DIR.iterdir()
            if f.name.startswith(f"{ticker}_") and f.name.endswith(".parquet")
        ]
        return sorted(
            [f.replace(f"{ticker}_", "").replace(".parquet", "") for f in files],
            reverse=not ascending,
        )

    except FileNotFoundError:
        return []


def get_live_dividend_yield(ticker: str) -> float:
    """
    Fetches the live trailing dividend yield for a ticker using yfinance.

    Handles the unit change in yfinance (fraction before 0.2.5x, percent since).

    Parameters
    ----------
    ticker : str
        The stock ticker to search for, e.g., 'SPY'.

    Returns
    -------
    float
        The continuous dividend yield as a decimal, or zero.
    """
    logger.info("Fetching live dividend yield for %s...", ticker)
    try:
        info = yf.Ticker(ticker).info
        value = info.get("dividendYield")
        if value is None:
            value = info.get("trailingAnnualDividendYield")
        return _normalise_dividend_yield(value)
    except Exception as e:
        logger.error("FAILED to fetch dividend yield for %s. Error: %s", ticker, e)
        return 0.0
