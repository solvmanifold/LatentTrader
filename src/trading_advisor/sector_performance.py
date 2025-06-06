"""
Sector performance calculation module.

This module handles the calculation of sector performance metrics including:
- Price levels
- Returns
- Volatility
- Volume
- Momentum
- Relative Strength (vs S&P 500)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from .sector_mapping import load_sector_mapping
from tqdm import tqdm
from trading_advisor.data import fill_missing_trading_days

logger = logging.getLogger(__name__)

def get_sp500_data(start_date: Optional[str] = None) -> pd.DataFrame:
    """
    Get S&P 500 data for relative strength calculations.
    Saves and updates data/market_features/sp500.parquet.
    Args:
        start_date: Optional start date for data collection
    Returns:
        DataFrame with S&P 500 price data
    """
    import os
    market_features_dir = Path("data/market_features")
    market_features_dir.mkdir(parents=True, exist_ok=True)
    sp500_path = market_features_dir / "sp500.parquet"
    
    # Try to load existing data
    if sp500_path.exists():
        sp500 = pd.read_parquet(sp500_path)
        sp500.index = pd.to_datetime(sp500.index)
    else:
        sp500 = pd.DataFrame()
    
    # Determine if we need to download new data
    if sp500.empty:
        # No data, download from start_date (or default)
        sp500_new = yf.download('^GSPC', start=start_date, progress=False)
    else:
        # Download only missing dates
        last_date = sp500.index.max()
        # If start_date is after last_date, use start_date
        if start_date is not None:
            start_dt = pd.to_datetime(start_date)
            if start_dt > last_date:
                fetch_start = start_date
            else:
                fetch_start = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            fetch_start = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        sp500_new = yf.download('^GSPC', start=fetch_start, progress=False)
    
    # Process new data if any
    if not sp500_new.empty:
        sp500_new = sp500_new[['Close']].rename(columns={'Close': 'sp500_price'})
        sp500_new['sp500_returns_20d'] = sp500_new['sp500_price'].pct_change(periods=20)
        sp500_new.index = pd.to_datetime(sp500_new.index)
        # Append new data
        sp500 = pd.concat([sp500, sp500_new])
        sp500 = sp500[~sp500.index.duplicated(keep='last')]
        sp500 = sp500.sort_index()
        # Recompute returns for the whole set (in case of new data)
        sp500['sp500_returns_20d'] = sp500['sp500_price'].pct_change(periods=20)
        sp500.to_parquet(sp500_path)
    # If still empty, try to download from scratch
    if sp500.empty:
        try:
            sp500 = yf.download('^GSPC', start=start_date, progress=False)
            sp500 = sp500[['Close']].rename(columns={'Close': 'sp500_price'})
            sp500['sp500_returns_20d'] = sp500['sp500_price'].pct_change(periods=20)
            sp500.index = pd.to_datetime(sp500.index)
            sp500.to_parquet(sp500_path)
        except Exception as e:
            logger.error(f"Error downloading S&P 500 data: {e}")
            return pd.DataFrame()
    # Ensure index is only Date (not MultiIndex)
    if isinstance(sp500.index, pd.MultiIndex):
        sp500 = sp500.reset_index()
    if 'Date' in sp500.columns:
        sp500 = sp500.set_index('Date')
    sp500.index = pd.to_datetime(sp500.index)
    # Filter to start_date if provided
    if start_date is not None:
        sp500 = sp500[sp500.index >= pd.to_datetime(start_date)]
    return sp500

def calculate_sector_performance(ticker_df: pd.DataFrame, market_features_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Calculate sector performance metrics.
    
    Args:
        ticker_df: DataFrame with ticker data including a 'ticker' column.
        market_features_dir: Directory to store market feature files.
        
    Returns:
        Dictionary of DataFrames with sector performance metrics.
    """
    if ticker_df.empty:
        logger.warning("No data provided for sector performance calculation")
        return {}
    
    # Get S&P 500 data for relative strength calculation
    start_date = ticker_df.index.min().strftime('%Y-%m-%d')
    sp500_df = get_sp500_data(start_date)
    
    if sp500_df.empty:
        logger.warning("Could not get S&P 500 data for relative strength calculation")
    
    # Group by sector and calculate metrics
    sector_dfs = {}
    for sector, group in ticker_df.groupby('sector'):
        # Calculate sector metrics
        sector_df = pd.DataFrame()
        sector_df['price'] = group.groupby(level=0)['close'].mean()
        sector_df['volatility'] = group.groupby(level=0)['close'].std()
        sector_df['volume'] = group.groupby(level=0)['volume'].sum()
        
        # Calculate returns
        sector_df['returns_1d'] = sector_df['price'].pct_change()
        sector_df['returns_5d'] = sector_df['price'].pct_change(periods=5)
        sector_df['returns_20d'] = sector_df['price'].pct_change(periods=20)
        
        # Calculate momentum
        sector_df['momentum_5d'] = sector_df['returns_1d'].rolling(window=5).mean()
        sector_df['momentum_20d'] = sector_df['returns_1d'].rolling(window=20).mean()
        
        # Calculate relative strength (vs S&P 500)
        if not sp500_df.empty:
            # Merge with S&P 500 data
            sector_df = sector_df.join(sp500_df[['sp500_price', 'sp500_returns_20d']])
            # Calculate relative strength as ratio of sector to S&P 500 20-day returns
            sector_df['relative_strength'] = sector_df['returns_20d'] / sector_df['sp500_returns_20d']
            # Drop S&P 500 columns
            sector_df = sector_df.drop(['sp500_price', 'sp500_returns_20d'], axis=1)
        else:
            # If no S&P 500 data, set relative strength to NaN
            sector_df['relative_strength'] = np.nan
        
        # Fill in missing trading days
        sector_df = fill_missing_trading_days(sector_df, ticker_df)
        
        sector_dfs[sector] = sector_df
    
    # Create a wide-format combined table
    all_sectors_df = pd.concat(sector_dfs, axis=1)
    all_sectors_df.columns = [f"{sector}_{col}" for sector, col in all_sectors_df.columns]
    
    # Fill in missing trading days for the combined table
    all_sectors_df = fill_missing_trading_days(all_sectors_df, ticker_df)
    
    sector_dfs['all_sectors'] = all_sectors_df
    
    return sector_dfs 