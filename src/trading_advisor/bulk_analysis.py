import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from trading_advisor.data import download_stock_data
from trading_advisor.analysis import calculate_technical_indicators, calculate_score_history

FEATURES_DIR = Path("features")
FEATURES_DIR.mkdir(exist_ok=True)


def get_features_path(ticker, features_dir=FEATURES_DIR):
    return Path(features_dir) / f"{ticker}_features.parquet"


def get_bulk_score_histories(tickers, start_date, end_date, features_dir="features"):
    """
    For each ticker, download/update OHLCV data, calculate indicators and score history,
    and save/load the resulting DataFrame to/from disk as Parquet.
    Returns a dict: {ticker: DataFrame}
    """
    features_dir = Path(features_dir)
    features_dir.mkdir(exist_ok=True)
    results = {}
    for ticker in tqdm(tickers, desc="Processing tickers"):
        features_path = get_features_path(ticker, features_dir)
        up_to_date = False
        if features_path.exists():
            try:
                df = pd.read_parquet(features_path)
                # Check if we have data up to end_date
                if not df.empty and pd.to_datetime(df.index[-1]) >= pd.to_datetime(end_date):
                    # Filter to requested date range
                    df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
                    results[ticker] = df
                    up_to_date = True
            except Exception as e:
                print(f"Error reading {features_path}: {e}")
        if not up_to_date:
            # Download and process
            raw_df = download_stock_data(ticker)
            if raw_df.empty:
                print(f"No data for {ticker}")
                continue
            try:
                df = calculate_technical_indicators(raw_df)
                df = calculate_score_history(df)
                # Save all features to disk
                df.to_parquet(features_path)
                # Filter to requested date range
                df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
                results[ticker] = df
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
    return results

# Example usage (in script or notebook):
# tickers = ["AAPL", "MSFT", ...]
# start_date = "2022-01-01"
# end_date = "2023-01-01"
# features = get_bulk_score_histories(tickers, start_date, end_date)
# features["AAPL"].head() 