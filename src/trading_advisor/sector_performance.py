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

logger = logging.getLogger(__name__)

def calculate_sector_performance(ticker_df: pd.DataFrame, features_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Calculate sector performance metrics.
    
    Args:
        ticker_df: DataFrame with ticker data, must have 'ticker' column
        features_dir: Directory containing feature files
        
    Returns:
        Dictionary containing:
        - Individual sector DataFrames (e.g., 'Technology', 'Healthcare', etc.)
        - 'all_sectors': Wide-format DataFrame with all sector metrics
    """
    if ticker_df.empty:
        logger.warning("Empty ticker DataFrame provided")
        return {'all_sectors': pd.DataFrame()}
    
    # Ensure DataFrame is indexed by date
    if not isinstance(ticker_df.index, pd.DatetimeIndex):
        if 'Date' in ticker_df.columns:
            ticker_df = ticker_df.set_index('Date')
        elif 'date' in ticker_df.columns:
            ticker_df = ticker_df.set_index('date')
        else:
            raise ValueError("DataFrame must have a 'Date' or 'date' column")
    
    # Convert index to datetime and sort
    ticker_df.index = pd.to_datetime(ticker_df.index)
    ticker_df = ticker_df.sort_index()
    
    # Load sector mapping
    sector_mapping_path = Path(features_dir) / "sector_mapping.parquet"
    if not sector_mapping_path.exists():
        logger.warning(f"No sector mapping found at {sector_mapping_path}")
        return {'all_sectors': pd.DataFrame()}
    
    sector_mapping = pd.read_parquet(sector_mapping_path)
    
    # Merge with sector data
    ticker_df = ticker_df.reset_index().merge(
        sector_mapping[['ticker', 'sector', 'subsector']],
        on='ticker',
        how='left'
    )
    # Set index back to the date column (try 'Date', then 'date', then the original index name)
    for date_col in ['Date', 'date', ticker_df.columns[0]]:
        if date_col in ticker_df.columns:
            ticker_df[date_col] = pd.to_datetime(ticker_df[date_col])
            ticker_df = ticker_df.set_index(date_col)
            break
    ticker_df = ticker_df.sort_index()
    
    # Get unique dates from the ticker data
    unique_dates = pd.DatetimeIndex(sorted(ticker_df.index.unique()))
    sector_dfs = {}
    
    # Calculate metrics for each sector
    for sector in ticker_df['sector'].unique():
        if pd.isna(sector):
            continue
            
        # Filter data for this sector
        sector_data = ticker_df[ticker_df['sector'] == sector]
        
        # Calculate metrics
        metrics = sector_data.groupby(sector_data.index).agg({
            'Close': ['mean', 'std'],
            'Volume': 'sum'
        })
        
        # Flatten column names
        metrics.columns = ['price', 'volatility', 'volume']
        
        # Calculate returns
        metrics['returns_1d'] = metrics['price'].pct_change()
        metrics['returns_5d'] = metrics['price'].pct_change(5)
        metrics['returns_20d'] = metrics['price'].pct_change(20)
        
        # Calculate momentum
        metrics['momentum_5d'] = metrics['returns_5d'].rolling(5).mean()
        metrics['momentum_20d'] = metrics['returns_20d'].rolling(20).mean()
        
        # Forward fill missing values (up to 5 days) to handle holidays
        metrics = metrics.ffill(limit=5)
        
        # Ensure index is datetime and sort
        metrics.index = pd.to_datetime(metrics.index)
        metrics = metrics.sort_index()
        
        # Store in dictionary
        sector_dfs[sector] = metrics
    
    # Create wide-format DataFrame with all sectors
    all_sectors = pd.DataFrame(index=unique_dates)
    
    # Add each sector's metrics with sector prefix
    for sector, metrics in sector_dfs.items():
        for col in metrics.columns:
            all_sectors[f'{sector}_{col}'] = metrics[col]
    
    # Forward fill missing values
    all_sectors = all_sectors.ffill(limit=5)
    
    # Ensure index is datetime and sort
    all_sectors.index = pd.to_datetime(all_sectors.index)
    all_sectors = all_sectors.sort_index()
    
    # Add the wide-format table to the dictionary
    sector_dfs['all_sectors'] = all_sectors
    
    return sector_dfs 