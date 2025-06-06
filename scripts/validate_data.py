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

if __name__ == '__main__':
    validate_market_features() 