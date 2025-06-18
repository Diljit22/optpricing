import yfinance as yf
import numpy as np
import pandas as pd

def get_historical_log_returns(ticker: str, period: str = "5y") -> pd.Series:
    """
    Fetches historical daily prices for a ticker and calculates log returns.
    """
    print(f"Fetching {period} of historical data for {ticker}...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if hist.empty:
        raise ValueError(f"Could not fetch historical data for {ticker}.")
    
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
    return log_returns.dropna()