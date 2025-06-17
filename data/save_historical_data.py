from data.historical_manager import save_historical_returns

# Define the tickers and period for which to save historical data
TICKERS_TO_SAVE = ['SPY', 'AAPL', 'META', 'GOOGL', 'TSLA']
PERIOD = "10y"

if __name__ == "__main__":
    save_historical_returns(TICKERS_TO_SAVE, period=PERIOD)