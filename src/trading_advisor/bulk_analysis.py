import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from trading_advisor.data import download_stock_data
from trading_advisor.analysis import calculate_score_history, get_analyst_targets
import json

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
    # Convert string dates to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    for ticker in tqdm(tickers, desc="Processing tickers"):
        features_path = get_features_path(ticker, features_dir)
        up_to_date = False
        if features_path.exists():
            try:
                df = pd.read_parquet(features_path)
                # Check if we have all business days in the requested range
                expected_days = pd.bdate_range(start=start_date, end=end_date)
                missing_days = [d for d in expected_days if d not in df.index]
                if not df.empty and len(missing_days) == 0:
                    # Filter to requested date range
                    df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
                    results[ticker] = df
                    up_to_date = True
                else:
                    if len(missing_days) > 0:
                        print(f"Features for {ticker} missing {len(missing_days)} business days in range {start_date.date()} to {end_date.date()}.")
            except Exception as e:
                print(f"Error reading {features_path}: {e}")
        if not up_to_date:
            # Download and process
            raw_df = download_stock_data(ticker, start_date=start_date, end_date=end_date)
            if raw_df.empty:
                print(f"No data for {ticker}")
                continue
            # Merge with existing data BEFORE calculating indicators
            if features_path.exists():
                old_df = pd.read_parquet(features_path)
                raw_df = pd.concat([old_df, raw_df])
                raw_df = raw_df[~raw_df.index.duplicated(keep='last')]
                raw_df = raw_df.sort_index()
            try:
                df = calculate_score_history(raw_df)
                # Log current analyst targets in the latest row
                analyst_targets = get_analyst_targets(ticker)
                if analyst_targets and not df.empty:
                    df.at[df.index[-1], 'analyst_targets'] = json.dumps(analyst_targets)
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