#!/usr/bin/env python
"""Script to validate data files in the LatentTrader project."""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from trading_advisor.validation import validate_parquet_file

# Define expected types for market features
MARKET_FEATURE_TYPES = {
    'date': np.datetime64,
    'market_volatility_daily_volatility': np.floating,
    'market_volatility_weekly_volatility': np.floating,
    'market_volatility_monthly_volatility': np.floating,
    'market_volatility_avg_correlation': np.floating,
    'market_volatility_ticker': str
}

# Define required columns for market features
MARKET_FEATURE_REQUIRED = ['date', 'market_volatility_daily_volatility']

# Define expected types for daily breadth
BREADTH_FEATURE_TYPES = {
    'date': np.datetime64,
    'daily_breadth_adv_dec_line': np.floating,
    'daily_breadth_new_highs': np.integer,
    'daily_breadth_new_lows': np.integer,
    'daily_breadth_above_ma20': np.floating,
    'daily_breadth_above_ma50': np.floating,
    'daily_breadth_rsi_bullish': np.floating,
    'daily_breadth_rsi_oversold': np.floating,
    'daily_breadth_rsi_overbought': np.floating,
    'daily_breadth_macd_bullish': np.floating
}

# Define required columns for daily breadth
BREADTH_FEATURE_REQUIRED = ['date', 'daily_breadth_adv_dec_line']

# Define expected types for market sentiment
SENTIMENT_FEATURE_TYPES = {
    'date': np.datetime64,
    'market_sentiment_ma5': np.floating,
    'market_sentiment_ma20': np.floating,
    'market_sentiment_momentum': np.floating,
    'market_sentiment_volatility': np.floating,
    'market_sentiment_zscore': np.floating
}

# Define required columns for market sentiment
SENTIMENT_FEATURE_REQUIRED = ['date', 'market_sentiment_ma5']

# Define expected types for ticker features
TICKER_FEATURE_TYPES = {
    'date': np.datetime64,
    'open': np.floating,
    'high': np.floating,
    'low': np.floating,
    'close': np.floating,
    'adj_close': np.floating,
    'volume': (np.float64, np.int64),
    'sma_20': np.floating,
    'sma_50': np.floating,
    'sma_100': np.floating,
    'sma_200': np.floating,
    'ema_100': np.floating,
    'ema_200': np.floating,
    'rsi': np.floating,
    'macd': np.floating,
    'macd_signal': np.floating,
    'macd_hist': np.floating,
    'bb_upper': np.floating,
    'bb_middle': np.floating,
    'bb_lower': np.floating,
    'bb_pband': np.floating
}

# Define required columns for ticker features
TICKER_FEATURE_REQUIRED = ['date', 'open', 'high', 'low', 'close', 'volume']

def validate_ticker_features():
    """Validate all ticker feature files."""
    data_dir = Path('data/ticker_features')
    all_valid = True
    
    # Get all ticker feature files
    ticker_files = list(data_dir.glob('*_features.parquet'))
    if not ticker_files:
        print("No ticker feature files found")
        return False
    
    print(f"\nValidating {len(ticker_files)} ticker feature files:")
    for ticker_file in ticker_files:
        print(f"  {ticker_file.name}...", end=' ')
        try:
            validate_parquet_file(
                str(ticker_file),
                expected_types=TICKER_FEATURE_TYPES,
                required_columns=TICKER_FEATURE_REQUIRED
            )
            print("✅")
        except Exception as e:
            print("❌")
            print(f"    Error: {str(e)}")
            all_valid = False
    
    return all_valid

def validate_market_features():
    """Validate all market feature files."""
    data_dir = Path('data/market_features')
    all_valid = True
    
    # Validate market volatility
    print("Validating market_volatility.parquet...", end=' ')
    try:
        validate_parquet_file(
            data_dir / 'market_volatility.parquet',
            expected_types=MARKET_FEATURE_TYPES,
            required_columns=MARKET_FEATURE_REQUIRED
        )
        print("✅")
    except Exception as e:
        print("❌")
        print(f"  Error: {str(e)}")
        all_valid = False
    
    # Validate daily breadth
    print("Validating daily_breadth.parquet...", end=' ')
    try:
        validate_parquet_file(
            data_dir / 'daily_breadth.parquet',
            expected_types=BREADTH_FEATURE_TYPES,
            required_columns=BREADTH_FEATURE_REQUIRED
        )
        print("✅")
    except Exception as e:
        print("❌")
        print(f"  Error: {str(e)}")
        all_valid = False
    
    # Validate market sentiment
    print("Validating market_sentiment.parquet...", end=' ')
    try:
        validate_parquet_file(
            data_dir / 'market_sentiment.parquet',
            expected_types=SENTIMENT_FEATURE_TYPES,
            required_columns=SENTIMENT_FEATURE_REQUIRED
        )
        print("✅")
    except Exception as e:
        print("❌")
        print(f"  Error: {str(e)}")
        all_valid = False
    
    # Validate GDELT raw data
    print("Validating gdelt_raw.parquet...", end=' ')
    try:
        validate_parquet_file(
            data_dir / 'gdelt_raw.parquet',
            expected_types={'date': np.datetime64},
            required_columns=['date']
        )
        print("✅")
    except Exception as e:
        print("❌")
        print(f"  Error: {str(e)}")
        all_valid = False
    
    # Validate sector files
    sectors_dir = data_dir / 'sectors'
    if sectors_dir.exists():
        print("\nValidating sector files:")
        for sector_file in sectors_dir.glob('*.parquet'):
            print(f"  {sector_file.name}...", end=' ')
            try:
                validate_parquet_file(
                    str(sector_file),
                    expected_types={
                        'date': np.datetime64,
                        'sector_price': np.floating,
                        'sector_volatility': np.floating,
                        'sector_volume': np.floating,
                        'sector_returns_1d': np.floating,
                        'sector_returns_5d': np.floating,
                        'sector_returns_20d': np.floating,
                        'sector_momentum_5d': np.floating,
                        'sector_momentum_20d': np.floating
                    },
                    required_columns=['date', 'sector_price', 'sector_volume']
                )
                print("✅")
            except Exception as e:
                print("❌")
                print(f"    Error: {str(e)}")
                all_valid = False
    
    return all_valid

def validate_data_consistency():
    """Validate consistency across all data files."""
    all_valid = True
    
    # Get all parquet files
    market_files = list(Path('data/market_features').glob('**/*.parquet'))
    ticker_files = list(Path('data/ticker_features').glob('*_features.parquet'))
    all_files = market_files + ticker_files
    
    if not all_files:
        print("No data files found")
        return False
    
    # Get date ranges for all files
    date_ranges = {}
    for file in all_files:
        try:
            df = pd.read_parquet(file)
            if 'date' in df.columns:
                date_ranges[file] = (df['date'].min(), df['date'].max())
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
            all_valid = False
    
    # Check for overlapping date ranges
    if date_ranges:
        print("\nChecking date ranges:")
        min_date = max(d[0] for d in date_ranges.values())
        max_date = min(d[1] for d in date_ranges.values())
        print(f"  Common date range: {min_date} to {max_date}")
        
        # Check for missing dates in each file
        for file, (start, end) in date_ranges.items():
            if start > min_date or end < max_date:
                print(f"  Warning: {file} has incomplete date range ({start} to {end})")
                all_valid = False
    
    return all_valid

def validate_all():
    """Run all validations."""
    print("Running data validation...")
    
    # Validate ticker features
    ticker_valid = validate_ticker_features()
    
    # Validate market features
    market_valid = validate_market_features()
    
    # Validate data consistency
    consistency_valid = validate_data_consistency()
    
    # Print summary
    print("\nValidation Summary:")
    print(f"  Ticker Features: {'✅' if ticker_valid else '❌'}")
    print(f"  Market Features: {'✅' if market_valid else '❌'}")
    print(f"  Data Consistency: {'✅' if consistency_valid else '❌'}")
    
    return all([ticker_valid, market_valid, consistency_valid])

if __name__ == '__main__':
    validate_all() 