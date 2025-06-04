"""Market volatility indicators calculation."""

import logging
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import numpy as np
import yfinance as yf
from trading_advisor.data import fill_missing_trading_days

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
    except Exception as e:
        logger.error(f"Error fetching VIX data: {e}")
    
    # Market-wide volatility (using S&P 500)
    try:
        spy = yf.download('^GSPC', start=combined_df.index.min(), end=combined_df.index.max())
        # Daily returns
        spy_returns = spy['Close'].pct_change()
        volatility_df['market_volatility'] = spy_returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
    except Exception as e:
        logger.error(f"Error calculating market volatility: {e}")
    
    # Cross-sectional volatility (dispersion of returns)
    daily_returns = combined_df.groupby(level=0)['Close'].apply(
        lambda x: x.pct_change().std() * np.sqrt(252)  # Annualized
    )
    volatility_df['cross_sectional_vol'] = daily_returns

    # Fill in missing trading days
    volatility_df = fill_missing_trading_days(volatility_df, combined_df)
    
    # Calculate rolling windows after filling missing days
    volatility_df['vix_ma20'] = volatility_df['vix'].rolling(window=20).mean()
    volatility_df['vix_std20'] = volatility_df['vix'].rolling(window=20).std()
    volatility_df['vol_of_vol'] = volatility_df['market_volatility'].rolling(window=20).std()

    # --- Average correlation calculation (refactored, manual rolling) ---
    # Pivot to get a matrix of Close prices: rows=Date, columns=Ticker
    close_matrix = combined_df.reset_index().pivot(index='Date', columns='ticker', values='Close')
    returns_matrix = close_matrix.pct_change()
    window = 20
    min_periods = 2
    avg_corr = []
    idx = returns_matrix.index
    for i in range(len(returns_matrix)):
        if i < window - 1:
            avg_corr.append(np.nan)
            continue
        window_df = returns_matrix.iloc[i - window + 1:i + 1]
        if window_df.shape[1] < 2 or window_df.dropna(axis=1, how='all').shape[1] < 2:
            avg_corr.append(np.nan)
            continue
        corr_matrix = window_df.corr()
        if corr_matrix.shape[0] < 2:
            avg_corr.append(np.nan)
            continue
        upper_triangle = np.triu_indices_from(corr_matrix, k=1)
        avg_corr.append(corr_matrix.values[upper_triangle].mean())
    volatility_df['avg_correlation'] = pd.Series(avg_corr, index=idx)
    
    # Check if the last row has any NaNs in key columns
    if not volatility_df.empty:
        last_row = volatility_df.iloc[-1]
        key_columns = ['vix', 'market_volatility', 'cross_sectional_vol']
        if last_row[key_columns].isna().any():
            # Remove the last row if any key columns have NaNs
            volatility_df = volatility_df.iloc[:-1]
            logger.info("Removed last row due to NaNs in key columns")
    
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
            start_date: Optional start date in YYYY-MM-DD format (ignored, kept for API compatibility)
            
        Returns:
            DataFrame with volatility features
        """
        # Calculate volatility features for all available data
        volatility_df = calculate_market_volatility(combined_df)
        
        # Save volatility data
        volatility_path = self.market_features_dir / "market_volatility.parquet"
        if not volatility_df.empty:
            volatility_df.to_parquet(volatility_path)
            logger.info(f"Saved volatility features to {volatility_path}")
        else:
            logger.warning(f"Did NOT overwrite {volatility_path} because new volatility DataFrame is empty.")
        
        return volatility_df 