"""Feature loading and management."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

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