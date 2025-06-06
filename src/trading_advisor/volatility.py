"""Market volatility calculation module."""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from trading_advisor.data import fill_missing_trading_days

logger = logging.getLogger(__name__)

def calculate_market_volatility(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market volatility metrics.
    
    Args:
        combined_df: DataFrame with combined ticker data
        
    Returns:
        DataFrame with market volatility metrics
    """
    if combined_df.empty:
        logger.warning("No data provided for market volatility calculation")
        return pd.DataFrame()

    # If multi-ticker, group by ticker; else, treat as single ticker
    if 'ticker' in combined_df.columns:
        tickers = combined_df['ticker'].unique()
        results = []
        for ticker in tickers:
            df = combined_df[combined_df['ticker'] == ticker].copy()
            df = df.sort_index()
            daily_returns = df['close'].pct_change()
            out = pd.DataFrame(index=df.index)
            out['daily_volatility'] = daily_returns.rolling(window=2).std()
            out['weekly_volatility'] = daily_returns.rolling(window=5).std()
            out['monthly_volatility'] = daily_returns.rolling(window=20).std()
            out['avg_correlation'] = daily_returns.rolling(window=5).corr()
            out['ticker'] = ticker
            results.append(out)
        volatility_df = pd.concat(results)
    else:
        df = combined_df.sort_index()
        daily_returns = df['close'].pct_change()
        volatility_df = pd.DataFrame(index=df.index)
        volatility_df['daily_volatility'] = daily_returns.rolling(window=2).std()
        volatility_df['weekly_volatility'] = daily_returns.rolling(window=5).std()
        volatility_df['monthly_volatility'] = daily_returns.rolling(window=20).std()
        volatility_df['avg_correlation'] = daily_returns.rolling(window=5).corr()

    # Fill in missing trading days
    volatility_df = fill_missing_trading_days(volatility_df, combined_df)
    return volatility_df

class MarketVolatility:
    """Market volatility analysis and feature generation."""
    
    def __init__(self, data_dir: Path):
        """Initialize the market volatility analyzer.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = data_dir
        self.volatility_dir = data_dir / "market_features"
        self.volatility_dir.mkdir(parents=True, exist_ok=True)
        
    def get_latest_volatility_date(self) -> Optional[pd.Timestamp]:
        """Get the most recent date in the volatility data.
        
        Returns:
            Latest date in the volatility data, or None if no data exists
        """
        volatility_path = self.volatility_dir / "market_volatility.parquet"
        if not volatility_path.exists():
            return None
            
        try:
            df = pd.read_parquet(volatility_path)
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
            logger.error(f"Error reading volatility data: {e}")
            return None
        
    def generate_volatility_features(self, ticker_df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility features.
        
        Args:
            ticker_df: DataFrame with ticker data
            
        Returns:
            DataFrame with volatility features
        """
        # Calculate market volatility
        volatility_df = calculate_market_volatility(ticker_df)
        
        # Save volatility data
        volatility_path = self.volatility_dir / "market_volatility.parquet"
        volatility_df.to_parquet(volatility_path)
        
        return volatility_df 