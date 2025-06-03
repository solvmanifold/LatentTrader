"""
Sector performance calculation module.

This module handles the calculation of sector performance metrics including:
- Price levels
- Returns
- Volatility
- Volume
- Momentum
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
from .sector_mapping import load_sector_mapping
from tqdm import tqdm
from trading_advisor.data import fill_missing_trading_days

logger = logging.getLogger(__name__)

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
    
    # Group by sector and calculate metrics
    sector_dfs = {}
    for sector, group in ticker_df.groupby('sector'):
        # Calculate sector metrics
        sector_df = pd.DataFrame()
        sector_df['price'] = group.groupby(level=0)['Close'].mean()
        sector_df['volatility'] = group.groupby(level=0)['Close'].std()
        sector_df['volume'] = group.groupby(level=0)['Volume'].sum()
        
        # Calculate returns
        sector_df['returns_1d'] = sector_df['price'].pct_change()
        sector_df['returns_5d'] = sector_df['price'].pct_change(periods=5)
        sector_df['returns_20d'] = sector_df['price'].pct_change(periods=20)
        
        # Calculate momentum
        sector_df['momentum_5d'] = sector_df['returns_1d'].rolling(window=5).mean()
        sector_df['momentum_20d'] = sector_df['returns_1d'].rolling(window=20).mean()
        
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