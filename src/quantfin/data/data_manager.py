import os
import pandas as pd
from datetime import date, datetime
from typing import List, Dict
import yfinance as yf

from pathlib import Path

# Locate project root (three levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Snapshots folder under <project_root>/data/market_data_snapshots
DATA_DIR = PROJECT_ROOT / "data" / "market_data_snapshots"


def get_option_chain(ticker: str) -> pd.DataFrame:
    ticker_obj = yf.Ticker(ticker)
    expirations = ticker_obj.options
    if not expirations: return pd.DataFrame()

    all_options = []
    for expiry in expirations:
        opt = ticker_obj.option_chain(expiry)
        if not opt.calls.empty:
            opt.calls['optionType'] = 'call'
            opt.calls['expiry'] = pd.to_datetime(expiry)
            all_options.append(opt.calls)
        if not opt.puts.empty:
            opt.puts['optionType'] = 'put'
            opt.puts['expiry'] = pd.to_datetime(expiry)
            all_options.append(opt.puts)
    
    if not all_options: return pd.DataFrame()

    chain_df = pd.concat(all_options, ignore_index=True)
    
    today_date = datetime.combine(date.today(), datetime.min.time())
    chain_df['maturity'] = (chain_df['expiry'] - today_date).dt.days / 365.25
    chain_df['marketPrice'] = (chain_df['bid'] + chain_df['ask']) / 2.0
    
    chain_df.dropna(subset=['marketPrice', 'strike', 'maturity', 'impliedVolatility', 'bid', 'ask'], inplace=True)
    chain_df = chain_df[(chain_df['marketPrice'] > 0.01) & (chain_df['bid'] > 0) & (chain_df['ask'] > 0)].copy()
    chain_df = chain_df[chain_df['maturity'] > 1/365.25].copy()
    return chain_df

def get_current_price(ticker: str) -> float:
    stock = yf.Ticker(ticker)
    price = stock.fast_info.get('last_price')
    if price is None: price = stock.history(period='1d')['Close'].iloc[0]
    return price

def save_market_snapshot(tickers: List[str]):
    os.makedirs(DATA_DIR, exist_ok=True)
    today_str = date.today().strftime("%Y-%m-%d")
    print(f"--- Saving Market Data Snapshot for {today_str} ---")
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        try:
            spot = get_current_price(ticker)
            chain_df = get_option_chain(ticker)
            if chain_df.empty:
                print(f"  -> No valid option data found for {ticker}. Skipping.")
                continue
            chain_df['spot_price'] = spot
            filename = os.path.join(DATA_DIR, f"{ticker}_{today_str}.parquet")
            chain_df.to_parquet(filename)
            print(f"  -> Successfully saved {len(chain_df)} options to {filename}")
        except Exception as e:
            print(f"  -> FAILED to fetch or save data for {ticker}. Error: {e}")

def get_available_snapshot_dates(ticker: str) -> List[str]:
    try:
        files = [f for f in os.listdir(DATA_DIR) if f.startswith(f"{ticker}_") and f.endswith(".parquet")]
        return sorted([f.replace(f"{ticker}_", "").replace(".parquet", "") for f in files])
    except FileNotFoundError:
        return []

def load_market_snapshot(ticker: str, snapshot_date: str) -> pd.DataFrame | None:
    filename = os.path.join(DATA_DIR, f"{ticker}_{snapshot_date}.parquet")
    if not os.path.exists(filename):
        print(f"Error: Snapshot file not found: {filename}")
        return None
    print(f"Loading data from {filename}...")
    return pd.read_parquet(filename)