from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import yfinance as yf

from optpricing.config import MARKET_SNAPSHOT_DIR, _config

__doc__ = """
Utility functions to download and load market snapshots, live option chains,
and forward dividend yields.
"""


def _fetch_from_yfinance(ticker: str) -> pd.DataFrame | None:
    """Fetches a live option chain using yfinance."""
    print(f"Fetching live option chain for {ticker} from yfinance...")
    try:
        ticker_obj = yf.Ticker(ticker)
        expirations = ticker_obj.options
        if not expirations:
            return None

        all_options = []
        for expiry in expirations:
            opt = ticker_obj.option_chain(expiry)
            if not opt.calls.empty:
                opt.calls["optionType"] = "call"
                opt.calls["expiry"] = pd.to_datetime(expiry)
                all_options.append(opt.calls)
            if not opt.puts.empty:
                opt.puts["optionType"] = "put"
                opt.puts["expiry"] = pd.to_datetime(expiry)
                all_options.append(opt.puts)

        if not all_options:
            return None

        chain_df = pd.concat(all_options, ignore_index=True)
        today_date = datetime.combine(date.today(), datetime.min.time())
        chain_df["maturity"] = (chain_df["expiry"] - today_date).dt.days / 365.25
        chain_df["marketPrice"] = (chain_df["bid"] + chain_df["ask"]) / 2.0
        chain_df["spot_price"] = ticker_obj.fast_info.get(
            "last_price", ticker_obj.history("1d")["Close"].iloc[0]
        )

        chain_df.dropna(
            subset=["marketPrice", "strike", "maturity", "impliedVolatility"],
            inplace=True,
        )
        chain_df = chain_df[
            (chain_df["marketPrice"] > 0.01) & (chain_df["maturity"] > 1 / 365.25)
        ].copy()
        return chain_df

    except Exception as e:
        print(f"  -> FAILED to fetch live yfinance data for {ticker}. Error: {e}")
        return None


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
        A DataFrame containing the formatted option chain data, or None if
        the fetch fails or no data is returned.
    """
    provider = _config.get("live_data_provider", "yfinance").lower()
    if provider == "yfinance":
        return _fetch_from_yfinance(ticker)

    else:
        print(
            f"Warning: Unknown live_data_provider '{provider}'. Defaulting to yfinance."
        )
        return _fetch_from_yfinance(ticker)


def save_market_snapshot(tickers: list[str]):
    """
    Saves a snapshot of the current market option chain for given tickers.

    For each ticker, it fetches the live option chain using
    `get_live_option_chain` and saves it to a parquet file named with the
    ticker and the current date.

    Parameters
    ----------
    tickers : list[str]
        A list of stock tickers to process, e.g., ['SPY', 'AAPL'].
    """
    today_str = date.today().strftime("%Y-%m-%d")

    print(f"--- Saving Market Data Snapshot for {today_str} ---")
    for ticker in tickers:
        chain_df = get_live_option_chain(ticker)

        if chain_df is None or chain_df.empty:
            print(f"  -> No valid option data found for {ticker}. Skipping.")
            continue

        filename = MARKET_SNAPSHOT_DIR / f"{ticker}_{today_str}.parquet"
        chain_df.to_parquet(filename)
        print(f"  -> Successfully saved {len(chain_df)} options to {filename}")


def load_market_snapshot(ticker: str, snapshot_date: str) -> pd.DataFrame | None:
    """
    Loads a previously saved market data snapshot for a specific date.

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
        print(f"Error: Snapshot file not found: {filename}")
        return None

    print(f"Loading data from {filename}...")
    return pd.read_parquet(filename)


def get_available_snapshot_dates(ticker: str) -> list[str]:
    """
    Lists all available snapshot dates for a given ticker.

    Scans the market data directory for saved parquet files corresponding
    to the ticker and extracts the date from the filenames.

    Parameters
    ----------
    ticker : str
        The stock ticker to search for, e.g., 'SPY'.

    Returns
    -------
    list[str]
        A sorted list of available dates in 'YYYY-MM-DD' format, from
        most recent to oldest.
    """
    try:
        files = [
            f.name
            for f in MARKET_SNAPSHOT_DIR.iterdir()
            if f.name.startswith(f"{ticker}_") and f.name.endswith(".parquet")
        ]
        return sorted(
            [f.replace(f"{ticker}_", "").replace(".parquet", "") for f in files],
            reverse=True,
        )

    except FileNotFoundError:
        return []


def get_live_dividend_yield(ticker: str) -> float:
    """
    Fetches the live forward dividend yield for a ticker using yfinance.

    Parameters
    ----------
    ticker : str
        The stock ticker to search for, e.g., 'SPY'.

    Returns
    -------
    float
        The associated div or zero.
    """
    print(f"Fetching live dividend yield for {ticker}...")
    try:
        t = yf.Ticker(ticker)
        dividend_yield = t.info.get("dividendYield")
        return float(dividend_yield / 100 or 0.0)
    except Exception as e:
        # Handle cases where the ticker is invalid or yfinance fails
        print(f"  -> FAILED to fetch dividend yield for {ticker}. Error: {e}")
        return 0.0
