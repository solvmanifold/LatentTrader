#!/usr/bin/env python
"""Script to remove the analyst_targets column from all parquet files."""

import pandas as pd
from pathlib import Path
import glob

def remove_analyst_targets(file_path):
    """Remove the analyst_targets column from a parquet file and save it back."""
    try:
        # Read the parquet file
        df = pd.read_parquet(file_path)
        
        # Check if analyst_targets column exists
        if 'analyst_targets' in df.columns:
            # Remove the column
            df = df.drop(columns=['analyst_targets'])
            # Save back to the same file
            df.to_parquet(file_path)
            print(f"Removed analyst_targets column from {file_path}")
        else:
            print(f"No analyst_targets column in {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    # Process ticker features
    ticker_files = glob.glob('data/ticker_features/*_features.parquet')
    for file in ticker_files:
        remove_analyst_targets(file)
    
    # Process market features
    market_files = [
        'data/market_features/market_volatility.parquet',
        'data/market_features/market_sentiment.parquet',
        'data/market_features/sp500.parquet',
        'data/market_features/vix.parquet',
        'data/market_features/daily_breadth.parquet',
        'data/market_features/gdelt_raw.parquet'
    ]
    for file in market_files:
        remove_analyst_targets(file)
    
    # Process sector files
    sector_files = glob.glob('data/market_features/sectors/*.parquet')
    for file in sector_files:
        remove_analyst_targets(file)

if __name__ == '__main__':
    main() 