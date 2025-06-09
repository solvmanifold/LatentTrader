"""
Market sentiment analysis module.

This module handles the collection and analysis of market sentiment indicators including:
- Put/Call ratios
- Short interest trends
- Analyst sentiment aggregation
- News sentiment (GDELT)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging
from .gdelt import GDELTClient

logger = logging.getLogger(__name__)

class MarketSentiment:
    """Market sentiment analysis and feature generation."""
    
    def __init__(self, data_dir: Path):
        """Initialize the market sentiment analyzer.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = data_dir
        # Removed sentiment_dir and its directory creation
        # Initialize GDELT client
        self.gdelt_client = GDELTClient(data_dir)
        
    def collect_put_call_ratios(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Collect and process put/call ratio data.
        
        Args:
            start_date: Optional start date for data collection
            
        Returns:
            DataFrame with put/call ratio metrics
        """
        # TODO: Implement put/call ratio collection
        return pd.DataFrame()
        
    def collect_short_interest(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Collect and process short interest data.
        
        Args:
            start_date: Optional start date for data collection
            
        Returns:
            DataFrame with short interest metrics
        """
        # TODO: Implement short interest collection
        return pd.DataFrame()
        
    def collect_analyst_sentiment(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Collect and aggregate analyst sentiment data.
        
        Args:
            start_date: Optional start date for data collection
            
        Returns:
            DataFrame with analyst sentiment metrics
        """
        # TODO: Implement analyst sentiment collection
        return pd.DataFrame()
        
    def collect_news_sentiment(self, start_date: Optional[str] = None, years: int = 5) -> pd.DataFrame:
        """Collect and process news sentiment data from GDELT.
        
        Args:
            start_date: Optional start date for data collection
            years: Number of years of historical data to download
            
        Returns:
            DataFrame with news sentiment metrics
        """
        return self.gdelt_client.collect_sentiment_data(start_date, years)
        
    def calculate_sentiment_features(self, gdelt_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate sentiment features from GDELT data.
        
        Args:
            gdelt_data: DataFrame with GDELT sentiment data
            
        Returns:
            DataFrame with sentiment features
        """
        if gdelt_data.empty:
            return pd.DataFrame()
            
        # Calculate rolling statistics
        sentiment_features = pd.DataFrame(index=gdelt_data.index)
        
        # 5-day moving average of sentiment
        sentiment_features['market_sentiment_ma5'] = gdelt_data['avg_tone'].rolling(window=5).mean()
        
        # 20-day moving average of sentiment
        sentiment_features['market_sentiment_ma20'] = gdelt_data['avg_tone'].rolling(window=20).mean()
        
        # Sentiment momentum (5-day change)
        sentiment_features['market_sentiment_momentum'] = gdelt_data['avg_tone'].diff(5)
        
        # Sentiment volatility (20-day standard deviation)
        sentiment_features['market_sentiment_volatility'] = gdelt_data['avg_tone'].rolling(window=20).std()
        
        # Sentiment z-score (20-day)
        sentiment_features['market_sentiment_zscore'] = (
            (gdelt_data['avg_tone'] - sentiment_features['market_sentiment_ma20']) / 
            sentiment_features['market_sentiment_volatility']
        )
        
        return sentiment_features
        
    def generate_sentiment_features(self, start_date: Optional[str] = None, days: int = 60) -> pd.DataFrame:
        """Generate sentiment features.
        
        Args:
            start_date: Optional start date in YYYY-MM-DD format
            days: Number of days of historical data to download (default: 60)
            
        Returns:
            DataFrame with sentiment features
        """
        # Always use the raw file for feature calculation
        raw_path = self.data_dir / "market_features" / "gdelt_raw.parquet"
        gdelt_data = self.gdelt_client.collect_sentiment_data(start_date, days)
        
        if gdelt_data.empty:
            return pd.DataFrame()
            
        # Save raw data (already handled in collect_sentiment_data)
        # Calculate sentiment features
        sentiment_features = self.calculate_sentiment_features(gdelt_data)
        
        # Save processed features
        sentiment_path = self.data_dir / "market_features" / "market_sentiment.parquet"
        sentiment_features.to_parquet(sentiment_path)
        
        return sentiment_features 