import numpy as np
import pandas as pd
import yfinance as yf

from quantfin.config import HISTORICAL_DIR


def save_historical_returns(tickers: list[str], period: str = "10y"):
    """Fetches and saves historical log returns for a list of tickers."""
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


def load_historical_returns(ticker: str, period: str = "10y") -> pd.Series:
    """Loads historical log returns, fetching and saving them if not found."""
    filename = HISTORICAL_DIR / f"{ticker}_{period}_returns.parquet"
    if not filename.exists():
        print(f"No historical data found for {ticker}. Fetching and saving now...")
        save_historical_returns([ticker], period)

    if not filename.exists():
        raise FileNotFoundError(f"Could not find or save historical data for {ticker}.")

    return pd.read_parquet(filename)["log_return"]
