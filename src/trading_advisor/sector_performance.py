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
from trading_advisor.data import fill_missing_trading_days, download_stock_data

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
    from trading_advisor.data import download_stock_data
    
    market_features_dir = Path("data/market_features")
    market_features_dir.mkdir(parents=True, exist_ok=True)
    sp500_path = market_features_dir / "sp500.parquet"
    
    # Try to load existing data
    if sp500_path.exists():
        sp500 = pd.read_parquet(sp500_path)
        sp500.index = pd.to_datetime(sp500.index)
    else:
        sp500 = pd.DataFrame()
    
    # Download new data using our standard function
    sp500_new = download_stock_data(
        '^GSPC',
        history_days=2200,  # Use a large number to ensure we get all data
        features_dir="data/market_features",
        start_date=pd.to_datetime(start_date) if start_date else None,
        features_filename="sp500.parquet"
    )
    
    if not sp500_new.empty:
        # Ensure we have the required columns
        if 'close' in sp500_new.columns:
            sp500_new = sp500_new[['close']].rename(columns={'close': 'sp500_price'})
        elif 'adj_close' in sp500_new.columns:
            sp500_new = sp500_new[['adj_close']].rename(columns={'adj_close': 'sp500_price'})
        
        # Calculate returns
        sp500_new['sp500_returns_1d'] = sp500_new['sp500_price'].pct_change()
        sp500_new['sp500_returns_5d'] = sp500_new['sp500_price'].pct_change(periods=5)
        sp500_new['sp500_returns_20d'] = sp500_new['sp500_price'].pct_change(periods=20)
        
        # Ensure index is datetime
        sp500_new.index = pd.to_datetime(sp500_new.index)
        
        # Save to parquet
        sp500_new.to_parquet(sp500_path)
        return sp500_new
    
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
    # Handle MultiIndex (date, ticker) by extracting the date part
    min_index = ticker_df.index.min()
    if isinstance(min_index, tuple):
        min_date = min_index[0]
    else:
        min_date = min_index
    start_date = pd.to_datetime(min_date).strftime('%Y-%m-%d')
    sp500_df = get_sp500_data(start_date)
    
    if sp500_df.empty:
        logger.warning("Could not get S&P 500 data for relative strength calculation")
    
    # Group by sector and calculate metrics
    sector_dfs = {}
    for sector, group in ticker_df.groupby('sector'):
        # Calculate sector metrics
        sector_df = pd.DataFrame()
        sector_df['sector_performance_price'] = group.groupby(level=0)['close'].mean()
        sector_df['sector_performance_volatility'] = group.groupby(level=0)['close'].std()
        sector_df['sector_performance_volume'] = group.groupby(level=0)['volume'].sum()
        
        # Calculate returns
        sector_df['sector_performance_returns_1d'] = sector_df['sector_performance_price'].pct_change()
        sector_df['sector_performance_returns_5d'] = sector_df['sector_performance_price'].pct_change(periods=5)
        sector_df['sector_performance_returns_20d'] = sector_df['sector_performance_price'].pct_change(periods=20)
        
        # Calculate momentum
        sector_df['sector_performance_momentum_5d'] = sector_df['sector_performance_returns_1d'].rolling(window=5).mean()
        sector_df['sector_performance_momentum_20d'] = sector_df['sector_performance_returns_1d'].rolling(window=20).mean()
        
        # Calculate relative strength (vs S&P 500)
        if not sp500_df.empty:
            # Merge with S&P 500 data
            sector_df = sector_df.join(sp500_df[['sp500_price', 'sp500_returns_20d']])
            
            # Calculate daily returns for S&P 500
            sp500_df['sp500_returns_1d'] = sp500_df['sp500_price'].pct_change()
            sp500_df['sp500_returns_5d'] = sp500_df['sp500_price'].pct_change(periods=5)
            
            # Calculate relative strength using cumulative returns
            sector_df['sector_performance_cumulative_returns'] = (1 + sector_df['sector_performance_returns_1d']).cumprod()
            sector_df['sp500_cumulative_returns'] = (1 + sp500_df['sp500_returns_1d']).cumprod()
            
            # Calculate relative strength as the ratio of cumulative returns
            sector_df['sector_performance_relative_strength'] = (sector_df['sector_performance_cumulative_returns'] / 
                                            sector_df['sp500_cumulative_returns'])
            
            # Add relative strength ratio (RSR) - alternative measure using 5-day returns
            sector_df['sector_performance_relative_strength_ratio'] = (sector_df['sector_performance_returns_5d'] - 
                                                  sp500_df['sp500_returns_5d'])
            
            # Drop intermediate columns
            sector_df = sector_df.drop(['sector_performance_cumulative_returns', 
                                      'sp500_cumulative_returns',
                                      'sp500_price', 
                                      'sp500_returns_20d'], axis=1)
        else:
            # If no S&P 500 data, set relative strength to NaN
            sector_df['sector_performance_relative_strength'] = np.nan
            sector_df['sector_performance_relative_strength_ratio'] = np.nan
        
        # Ensure index is DatetimeIndex
        if not isinstance(sector_df.index, pd.DatetimeIndex):
            sector_df.index = pd.to_datetime(sector_df.index)
        
        # Use a date-only index for reference
        date_ref = ticker_df.index.get_level_values(0).unique()
        date_ref = pd.DataFrame(index=pd.DatetimeIndex(date_ref))
        
        # Fill in missing trading days
        sector_df = fill_missing_trading_days(sector_df, date_ref, data_type='volume')
        
        # Ensure index is DatetimeIndex after filling
        if not isinstance(sector_df.index, pd.DatetimeIndex):
            sector_df.index = pd.to_datetime(sector_df.index)
        
        sector_dfs[sector] = sector_df
    
    # Create a wide-format combined table
    all_sectors_df = pd.concat(sector_dfs, axis=1)
    all_sectors_df.columns = [f"{sector}_{col}" for sector, col in all_sectors_df.columns]
    
    # Ensure index is DatetimeIndex
    if not isinstance(all_sectors_df.index, pd.DatetimeIndex):
        all_sectors_df.index = pd.to_datetime(all_sectors_df.index)
    
    # Use a date-only index for reference
    date_ref = ticker_df.index.get_level_values(0).unique()
    date_ref = pd.DataFrame(index=pd.DatetimeIndex(date_ref))
    
    # Fill in missing trading days for the combined table
    all_sectors_df = fill_missing_trading_days(all_sectors_df, date_ref, data_type='volume')
    
    # Ensure index is DatetimeIndex after filling
    if not isinstance(all_sectors_df.index, pd.DatetimeIndex):
        all_sectors_df.index = pd.to_datetime(all_sectors_df.index)
    
    sector_dfs['all_sectors'] = all_sectors_df
    
    return sector_dfs 