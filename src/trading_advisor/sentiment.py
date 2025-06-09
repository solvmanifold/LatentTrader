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
from datetime import datetime
from .sentiment.gdelt import GDELTClient
from trading_advisor.data import fill_missing_trading_days, standardize_columns_and_date

logger = logging.getLogger(__name__)

class MarketSentiment:
    """Market sentiment analysis and feature generation."""
    
    def __init__(self, data_dir: Path):
        """Initialize the market sentiment analyzer.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = data_dir
        self.sentiment_dir = data_dir / "market_features"
        self.sentiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GDELT client with the base data_dir
        self.gdelt_client = GDELTClient(data_dir)
        
    def get_latest_sentiment_date(self) -> Optional[datetime]:
        """Get the most recent date in the sentiment data.
        
        Returns:
            Latest date in the sentiment data, or None if no data exists
        """
        sentiment_path = self.sentiment_dir / "market_sentiment.parquet"
        if not sentiment_path.exists():
            return None
            
        try:
            df = pd.read_parquet(sentiment_path)
            if df.empty:
                return None
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                elif 'date' in df.columns:
                    df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
            return df.index.max()
        except Exception as e:
            logger.error(f"Error reading sentiment data: {e}")
            return None
        
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
        
    def collect_news_sentiment(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Collect and process news sentiment data from GDELT.
        
        Args:
            start_date: Optional start date for data collection
            
        Returns:
            DataFrame with news sentiment metrics
        """
        return self.gdelt_client.collect_sentiment_data(start_date)
        
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
        
    def generate_sentiment_features(self, days: int = 60) -> pd.DataFrame:
        """Generate sentiment features.
        
        Args:
            days: Number of days of historical data to download (default: 60)
            
        Returns:
            DataFrame with sentiment features
        """
        # Check latest date in existing sentiment data
        latest_date = self.get_latest_sentiment_date()
        if latest_date is not None:
            update_start = latest_date + pd.Timedelta(days=1)
            logger.info(f"Found existing sentiment data through {latest_date.date()}. Will update from next day.")
        else:
            # No existing data, use days parameter
            update_start = pd.Timestamp.today() - pd.Timedelta(days=days)
            logger.info(f"No existing sentiment data found. Will generate {days} days of data.")
            
        # Get GDELT data for update period
        gdelt_data = self.gdelt_client.collect_sentiment_data(update_start.strftime('%Y-%m-%d'))
        if gdelt_data.empty:
            logger.warning("No GDELT data available for the update period.")
            return pd.DataFrame()
            
        # Calculate sentiment features
        sentiment_features = self.calculate_sentiment_features(gdelt_data)
        
        # Fill in missing trading days
        sentiment_features = fill_missing_trading_days(sentiment_features, gdelt_data)
        
        # Merge with existing sentiment data
        sentiment_path = self.sentiment_dir / "market_sentiment.parquet"
        if sentiment_path.exists():
            existing_sentiment = pd.read_parquet(sentiment_path)
            sentiment_features = pd.concat([existing_sentiment, sentiment_features])
            sentiment_features = sentiment_features[~sentiment_features.index.duplicated(keep='last')]
            sentiment_features = sentiment_features.sort_index()
            
        return sentiment_features 