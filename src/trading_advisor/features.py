"""Feature loading and management."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from trading_advisor.data import download_stock_data, normalize_ticker
from trading_advisor.analysis import calculate_technical_indicators

logger = logging.getLogger(__name__)

def load_features(ticker: str, features_dir: str = "features") -> pd.DataFrame:
    """
    Load features for a ticker from parquet file.
    
    Args:
        ticker: Stock ticker symbol
        features_dir: Directory containing feature files
        
    Returns:
        DataFrame with features, or empty DataFrame if file not found
    """
    features_path = Path(features_dir) / f"{ticker}_features.parquet"
    
    if not features_path.exists():
        logger.warning(f"Features file not found: {features_path}")
        return pd.DataFrame()
        
    try:
        df = pd.read_parquet(features_path)
        return df
    except Exception as e:
        logger.error(f"Error loading features for {ticker}: {e}")
        return pd.DataFrame()

def update_features(ticker: str, features_dir: str = "features") -> pd.DataFrame:
    """
    Download latest data and compute features for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        features_dir: Directory to save feature files
        
    Returns:
        DataFrame with updated features
    """
    # Normalize ticker name
    norm_ticker = normalize_ticker(ticker)
    
    # Download latest data
    df = download_stock_data(norm_ticker)
    if df.empty:
        logger.error(f"No data downloaded for {ticker}")
        return pd.DataFrame()
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Save to parquet
    features_path = Path(features_dir) / f"{ticker}_features.parquet"
    features_path.parent.mkdir(exist_ok=True)
    df.to_parquet(features_path)
    
    logger.info(f"Updated features for {ticker} and saved to {features_path}")
    return df 