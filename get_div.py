import yfinance as yf

tickers = [
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

dividend_yields = {}
for symbol in tickers:
    t = yf.Ticker(symbol)
    # .info['dividendYield'] is given as a decimal (e.g. 0.015 for 1.5%)
    dy = t.info.get("dividendYield")
    # If the stock/ETF doesn’t pay a dividend, info['dividendYield'] may be None
    dividend_yields[symbol] = dy or 0.0

print(dividend_yields)  # as precent yield
