from quantfin.data.historical_manager import save_historical_returns

TICKERS_TO_SAVE = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "JPM",
    "GS",
    "BAC",
    "JNJ",
    "UNH",
    "PFE",
    "WMT",
    "HD",
    "NKE",
    "XOM",
    "CVX",
    "BA",
    "CAT",
    "AMC",
    "GME",
    "AMD",
]
PERIOD = "10y"

if __name__ == "__main__":
    print("--- Downloading all required historical return data ---")
    save_historical_returns(TICKERS_TO_SAVE, period=PERIOD)
    print("--- Download complete ---")
