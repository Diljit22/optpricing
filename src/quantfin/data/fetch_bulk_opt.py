import os
import pandas as pd
import time
import random
from quantfin.data.data_manager import save_market_snapshot

TICKERS_TO_SAVE = [
    # Index / Broad Market
    'SPY', 'QQQ', 
    # Mega-Cap Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    # Finance
    'JPM', 'GS', 'BAC',
    # Healthcare
    'JNJ', 'UNH', 'PFE',
    # Consumer / Retail
    'WMT', 'HD', 'NKE',
    # Energy
    'XOM', 'CVX',
    # Industrial / Other
    'BA', 'CAT',
    # High Volatility / "Meme"
    'AMC', 'GME',
    # Semiconductor
    'AMD'
]

def main():
    """
    Main function to download and save market data for a large list of tickers,
    with built-in pauses to avoid rate limiting.
    """
    # Get the list of tickers
    tickers_to_save = TICKERS_TO_SAVE
    
    # Pause for 2 to 5 seconds between each ticker
    MIN_PAUSE_S = 2.0
    MAX_PAUSE_S = 5.0
    
    print("\n" + "="*60)
    print("--- Starting Bulk Market Data Downloader ---")
    print(f"Attempting to download data for {len(tickers_to_save)} tickers.")
    print("This will take a significant amount of time.")
    print("="*60)

    for i, ticker in enumerate(tickers_to_save):
        print(f"\n--- Processing Ticker {i+1}/{len(tickers_to_save)}: {ticker} ---")
        save_market_snapshot([ticker])
        
        if i < len(tickers_to_save) - 1:
            pause_duration = random.uniform(MIN_PAUSE_S, MAX_PAUSE_S)
            print(f"  -> Pausing for {pause_duration:.1f} seconds to avoid rate limiting...")
            time.sleep(pause_duration)

    print("\n" + "="*60)
    print("--- Bulk Download Complete ---")
    print("="*60)

if __name__ == "__main__":
    main()