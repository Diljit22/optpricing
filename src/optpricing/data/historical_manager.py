from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from optpricing.config import HISTORICAL_DIR

__doc__ = """
Provides utility functions to download and locally load historical returns.
"""


def save_historical_returns(
    tickers: list[str],
    period: str = "10y",
):
    """
    Fetches and saves historical log returns for a list of tickers.

    This function iterates through a list of stock tickers, fetches historical
    price data from yfinance for the specified period, calculates the daily
    log returns, and saves them to a parquet file in the historical data directory.

    Parameters
    ----------
    tickers : list[str]
        A list of stock tickers to process, e.g., ['SPY', 'AAPL'].
    period : str, optional
        The time period for which to fetch data, e.g., "10y", "5y", "1mo".
        Defaults to "10y".
    """
    print(f"--- Saving {period} Historical Returns ---")
    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}...")
            data = yf.Ticker(ticker).history(period=period)

            if data.empty:
                print(f"  -> No data found for {ticker}. Skipping.")
                continue

            log_returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()
            filename = HISTORICAL_DIR / f"{ticker}_{period}_returns.parquet"
            log_returns.to_frame(name="log_return").to_parquet(filename)

            print(f"  -> Saved to {filename}")

        except Exception as e:
            print(f"  -> FAILED to save data for {ticker}. Error: {e}")


def load_historical_returns(
    ticker: str,
    period: str = "10y",
) -> pd.Series:
    """
    Loads historical log returns, fetching and saving them if not found.

    Checks for a pre-saved parquet file for the given ticker and period. If
    the file does not exist, it calls `save_historical_returns` to download
    and save it first.

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
    filename = HISTORICAL_DIR / f"{ticker}_{period}_returns.parquet"
    if not filename.exists():
        print(f"No historical data found for {ticker}. Fetching and saving now...")
        save_historical_returns([ticker], period)

    if not filename.exists():
        raise FileNotFoundError(f"Could not find or save historical data for {ticker}.")

    return pd.read_parquet(filename)["log_return"]
