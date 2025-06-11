"""Feature loading and management."""

import logging
from pathlib import Path
from typing import Optional, Union, List

import pandas as pd
from trading_advisor.data import download_stock_data, normalize_ticker, standardize_columns_and_date
from trading_advisor.analysis import calculate_technical_indicators
import numpy as np

logger = logging.getLogger(__name__)

def load_features(ticker: str, features_dir: str = "data/ticker_features") -> pd.DataFrame:
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
        # Ensure date is the index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df.set_index('date', inplace=True)
            elif 'Date' in df.columns:
                df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
        # Add ticker column as string
        df['ticker'] = ticker
        return df
    except Exception as e:
        logger.error(f"Error loading features for {ticker}: {e}")
        return pd.DataFrame()

def update_features(ticker: str, features_dir: str = "data/ticker_features") -> pd.DataFrame:
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
    
    # Standardize columns and date before saving
    df = standardize_columns_and_date(df)
    
    # Save to parquet
    features_path = Path(features_dir) / f"{ticker}_features.parquet"
    features_path.parent.mkdir(exist_ok=True)
    df.to_parquet(features_path)
    
    logger.info(f"Updated features for {ticker} and saved to {features_path}")
    return df 

class MarketFeatureCollector:
    """Collects and manages market-wide features."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the market feature collector.
        
        Args:
            data_dir: Directory containing feature files
        """
        self.data_dir = Path(data_dir)
        
    def get_market_features(
        self,
        feature_types: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get market-wide features.
        
        Args:
            feature_types: Optional list of feature types to include
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            DataFrame with market features
        """
        if feature_types is None:
            feature_types = ['daily_breadth', 'market_volatility', 'market_sentiment']
            
        dfs = []
        for feature_type in feature_types:
            feature_path = self.data_dir / 'market_features' / f"{feature_type}.parquet"
            if not feature_path.exists():
                logger.warning(f"No features found for {feature_type}")
                continue
                
            df = pd.read_parquet(feature_path)
            dfs.append(df)
            
        if not dfs:
            raise FileNotFoundError("No market features found")
            
        market_features = pd.concat(dfs, axis=1)
        
        if start_date:
            market_features = market_features[market_features.index >= start_date]
        if end_date:
            market_features = market_features[market_features.index <= end_date]
            
        return market_features 