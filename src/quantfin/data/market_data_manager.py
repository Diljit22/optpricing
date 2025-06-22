from datetime import date, datetime

import pandas as pd
import yfinance as yf
from polygon import RESTClient

from quantfin.config import MARKET_SNAPSHOT_DIR, POLYGON_API_KEY, _config


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
        chain_df["last_price"] = (chain_df["bid"] + chain_df["ask"]) / 2.0
        chain_df["spot_price"] = ticker_obj.fast_info.get(
            "last_price", ticker_obj.history("1d")["Close"].iloc[0]
        )

        chain_df.dropna(
            subset=["last_price", "strike", "maturity", "impliedVolatility"],
            inplace=True,
        )
        chain_df = chain_df[
            (chain_df["last_price"] > 0.01) & (chain_df["maturity"] > 1 / 365.25)
        ].copy()
        return chain_df
    except Exception as e:
        print(f"  -> FAILED to fetch live yfinance data for {ticker}. Error: {e}")
        return None


def _fetch_from_polygon(ticker: str) -> pd.DataFrame | None:
    """Fetches a live option chain using Polygon.io."""
    print(f"Fetching live option chain for {ticker} from Polygon.io...")
    if not POLYGON_API_KEY or "YOUR_KEY" in POLYGON_API_KEY:
        print(
            "Error: Polygon API key not configured in config.yaml or environment variables."
        )
        return None

    client = RESTClient(POLYGON_API_KEY)
    try:
        # limited implementation for a free account
        # gets the chain for the next expiration date.
        expirations = list(
            client.list_options_contracts(
                underlying_ticker=ticker, expired=False, limit=1
            )
        )
        if not expirations:
            return None

        expiry_date = expirations[0].expiration_date
        contracts = list(
            client.list_options_contracts(
                underlying_ticker=ticker, expiration_date=expiry_date, limit=1000
            )
        )
        if not contracts:
            return None

        # A single snapshot call
        snapshot = client.get_options_chain(
            underlying_ticker=ticker, expiration_date=expiry_date
        )

        data = []
        for contract_details in snapshot.results:
            greeks = contract_details.greeks
            data.append(
                {
                    "ticker": contract_details.ticker,
                    "underlying": ticker,
                    "strike": contract_details.details.strike_price,
                    "expiry": pd.to_datetime(contract_details.details.expiration_date),
                    "optionType": "call"
                    if contract_details.details.contract_type == "call"
                    else "put",
                    "spot_price": greeks.underlying_price,
                    "last_price": contract_details.last_quote.bid,
                    "impliedVolatility": greeks.implied_volatility,
                    "volume": contract_details.day.volume,
                    "open_interest": contract_details.open_interest,
                }
            )

        df = pd.DataFrame(data)
        df["maturity"] = (df["expiry"] - pd.Timestamp.now()).dt.days / 365.25
        return df
    except Exception as e:
        print(f"An error occurred while fetching live Polygon data: {e}")
        return None


# Public API Functions


def get_live_option_chain(ticker: str) -> pd.DataFrame | None:
    """
    Fetches a live option chain from the provider specified in config.yaml.
    """
    provider = _config.get("live_data_provider", "yfinance").lower()
    if provider == "polygon":
        return _fetch_from_polygon(ticker)
    elif provider == "yfinance":
        return _fetch_from_yfinance(ticker)
    else:
        print(
            f"Warning: Unknown live_data_provider '{provider}'. Defaulting to yfinance."
        )
        return _fetch_from_yfinance(ticker)


def save_market_snapshot(tickers: list[str]):
    """Saves a snapshot of the current market option chain for a list of tickers."""
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
    """Loads a previously saved market data snapshot."""
    filename = MARKET_SNAPSHOT_DIR / f"{ticker}_{snapshot_date}.parquet"
    if not filename.exists():
        print(f"Error: Snapshot file not found: {filename}")
        return None
    print(f"Loading data from {filename}...")
    return pd.read_parquet(filename)


def get_available_snapshot_dates(ticker: str) -> list[str]:
    """lists all available snapshot dates for a given ticker."""
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
