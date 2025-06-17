import os
import yfinance as yf
import numpy as np
import pandas as pd

DATA_DIR = "historical_data"

def save_historical_returns(tickers: list[str], period: str = "5y"):
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"--- Saving {period} Historical Returns ---")
    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}...")
            data = yf.Ticker(ticker).history(period=period)
            log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
            filename = os.path.join(DATA_DIR, f"{ticker}_{period}_returns.parquet")
            log_returns.to_frame(name='log_return').to_parquet(filename)
            print(f"  -> Saved to {filename}")
        except Exception as e:
            print(f"  -> FAILED to save data for {ticker}. Error: {e}")

def load_historical_returns(ticker: str, period: str = "5y") -> pd.Series:
    filename = os.path.join(DATA_DIR, f"{ticker}_{period}_returns.parquet")
    if not os.path.exists(filename):
        print(f"No historical data found for {ticker}. Fetching and saving now...")
        save_historical_returns([ticker], period)
    
    if not os.path.exists(filename):
         raise FileNotFoundError(f"Could not find or save historical data for {ticker}.")

    return pd.read_parquet(filename)['log_return']