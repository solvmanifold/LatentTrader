"""Market volatility indicators calculation."""

import logging
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

def calculate_market_volatility(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market volatility indicators.
    
    Args:
        combined_df: DataFrame with combined ticker data
        
    Returns:
        DataFrame with market volatility indicators
    """
    if combined_df.empty:
        logger.warning("No data provided for market volatility calculation")
        return pd.DataFrame()
        
    # Calculate market volatility indicators
    volatility_df = pd.DataFrame()
    
    # VIX data
    try:
        vix = yf.download('^VIX', start=combined_df.index.min(), end=combined_df.index.max())
        volatility_df['vix'] = vix['Close']
        volatility_df['vix_ma20'] = vix['Close'].rolling(window=20).mean()
        volatility_df['vix_std20'] = vix['Close'].rolling(window=20).std()
    except Exception as e:
        logger.error(f"Error fetching VIX data: {e}")
    
    # Market-wide volatility (using S&P 500)
    try:
        spy = yf.download('^GSPC', start=combined_df.index.min(), end=combined_df.index.max())
        # Daily returns
        spy_returns = spy['Close'].pct_change()
        # Rolling volatility (20-day)
        volatility_df['market_volatility'] = spy_returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
        # Volatility of volatility
        volatility_df['vol_of_vol'] = volatility_df['market_volatility'].rolling(window=20).std()
    except Exception as e:
        logger.error(f"Error calculating market volatility: {e}")
    
    # Cross-sectional volatility (dispersion of returns)
    daily_returns = combined_df.groupby(level=0)['Close'].apply(
        lambda x: x.pct_change().std() * np.sqrt(252)  # Annualized
    )
    volatility_df['cross_sectional_vol'] = daily_returns
    
    # Correlation matrix (using 20-day rolling window)
    def calculate_correlation(group):
        if len(group) < 2:  # Need at least 2 stocks for correlation
            return np.nan
        returns = group.pct_change()
        return returns.corr().mean().mean()  # Average correlation
    
    volatility_df['avg_correlation'] = combined_df.groupby(level=0)['Close'].apply(calculate_correlation)
    
    return volatility_df

class MarketVolatility:
    """Market volatility analysis and feature generation."""
    
    def __init__(self, data_dir: Path):
        """Initialize the market volatility analyzer.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = data_dir
        self.market_features_dir = data_dir / "market_features"
        self.market_features_dir.mkdir(parents=True, exist_ok=True)
        
    def get_latest_volatility_date(self) -> Optional[pd.Timestamp]:
        """Get the most recent date in the volatility data.
        
        Returns:
            Latest date in the volatility data, or None if no data exists
        """
        volatility_path = self.market_features_dir / "market_volatility.parquet"
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
        
    def generate_volatility_features(self, combined_df: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
        """Generate volatility features.
        
        Args:
            combined_df: DataFrame with combined ticker data
            start_date: Optional start date in YYYY-MM-DD format
            
        Returns:
            DataFrame with volatility features
        """
        # Determine start date for updates
        update_start = None
        if start_date is None:
            # Check latest date in existing volatility data
            latest_date = self.get_latest_volatility_date()
            if latest_date is not None:
                update_start = latest_date + pd.Timedelta(days=1)
                logger.info(f"Found existing volatility data through {latest_date.date()}. Will update from next day.")
            else:
                # No existing data, use all available data
                update_start = combined_df.index.min()
                logger.info("No existing volatility data found. Will generate for all available dates.")
        else:
            update_start = pd.to_datetime(start_date)
            logger.info(f"Using provided start date: {update_start.date()}")
            
        # Filter data for update period
        update_df = combined_df[combined_df.index >= update_start]
        if update_df.empty:
            logger.warning("No ticker data available for the update period.")
            return pd.DataFrame()
            
        # Calculate volatility features
        volatility_df = calculate_market_volatility(update_df)
        
        # Merge with existing volatility data
        volatility_path = self.market_features_dir / "market_volatility.parquet"
        if volatility_path.exists():
            existing_volatility = pd.read_parquet(volatility_path)
            existing_volatility.index = pd.to_datetime(existing_volatility.index)
            # Remove any overlapping dates from existing data
            existing_volatility = existing_volatility[existing_volatility.index < volatility_df.index.min()]
            # Combine old and new data
            volatility_df = pd.concat([existing_volatility, volatility_df])
            volatility_df = volatility_df[~volatility_df.index.duplicated(keep='last')]
            volatility_df = volatility_df.sort_index()
            
        # Save updated volatility data
        volatility_df.to_parquet(volatility_path)
        logger.info(f"Saved volatility features to {volatility_path}")
        
        return volatility_df 