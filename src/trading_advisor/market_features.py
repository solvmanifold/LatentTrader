"""
Market features module.

This module handles the generation of various market features including:
- Market breadth
- Sector performance
- Market sentiment
- Market volatility
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import json

import pandas as pd
import yfinance as yf
from tqdm import tqdm
from trading_advisor.data import normalize_ticker
from .market_breadth import calculate_market_breadth
from .sector_performance import calculate_sector_performance
from .sentiment import MarketSentiment
from .volatility import MarketVolatility
from .sector_mapping import update_sector_mapping, load_sector_mapping

logger = logging.getLogger(__name__)

def get_latest_feature_date(file_path: Path) -> Optional[datetime]:
    """Get the most recent date in a market feature file.
    
    Args:
        file_path: Path to the feature file
        
    Returns:
        Latest date in the file, or None if file doesn't exist
    """
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_parquet(file_path)
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
        logger.error(f"Error reading {file_path}: {e}")
        return None

def load_ticker_data(ticker_list: List[str], features_dir: str, start_date: Optional[datetime] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load ticker data for market features.
    
    Args:
        ticker_list: List of tickers to load
        features_dir: Directory containing feature files
        start_date: Optional start date to filter data
        
    Returns:
        Tuple of (combined_df, ticker_df) where:
        - combined_df is the raw data for market breadth
        - ticker_df has ticker column added for sector performance
    """
    # Load all ticker data
    all_data = []
    ticker_data = []
    for ticker in tqdm(ticker_list, desc="Loading ticker data"):
        feature_file = Path(features_dir) / f"{ticker}_features.parquet"
        if feature_file.exists():
            df = pd.read_parquet(feature_file)
            # Ensure index is date
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df = df.set_index('Date')
                elif 'date' in df.columns:
                    df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
            
            # Filter by start date if provided
            if start_date is not None:
                df = df[df.index >= start_date]
                
            all_data.append(df)
            df_with_ticker = df.copy()
            df_with_ticker['ticker'] = ticker
            ticker_data.append(df_with_ticker)
            
    if not all_data:
        logger.warning("No feature data found. Please run init-features first.")
        return pd.DataFrame(), pd.DataFrame()
        
    # Combine all data
    combined_df = pd.concat(all_data)
    ticker_df = pd.concat(ticker_data)
    
    return combined_df, ticker_df

class MarketFeatures:
    """Market feature generation and management."""
    
    def __init__(self, data_dir: Path):
        """Initialize the market features generator.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = data_dir
        self.market_features_dir = data_dir / "market_features"
        self.market_features_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_market_features(self, start_date: Optional[str] = None, days: int = 60, update_sector_mapping: bool = False):
        """
        Generate and save market features as separate files in data/market_features.
        Handles incremental updates by only processing new dates.
        
        Args:
            start_date: Optional start date for data collection
            days: Number of days of historical data to download
            update_sector_mapping: Whether to force update sector mapping (default: False)
        """
        # Get list of tickers and generate sector mapping if needed
        ticker_list = self.get_ticker_list()
        
        # Load or generate sector mapping
        if update_sector_mapping:
            logger.info("Generating fresh sector mapping...")
            sector_mapping = update_sector_mapping(ticker_list, str(self.market_features_dir))
        else:
            logger.info("Loading existing sector mapping...")
            sector_mapping = load_sector_mapping(str(self.market_features_dir))
            if not sector_mapping:
                logger.info("No existing sector mapping found, generating new one...")
                sector_mapping = update_sector_mapping(ticker_list, str(self.market_features_dir))

        # Load ticker data
        ticker_df = pd.DataFrame()
        for ticker in tqdm(ticker_list, desc="Loading ticker data"):
            features_path = self.data_dir / "features" / f"{ticker}_features.parquet"
            if not features_path.exists():
                continue
            df = pd.read_parquet(features_path)
            df['ticker'] = ticker
            df['sector'] = sector_mapping.get(ticker, 'Unknown')
            ticker_df = pd.concat([ticker_df, df])

        if ticker_df.empty:
            logger.warning("No ticker data found")
            return

        # Generate market breadth
        logger.info("Starting market breadth generation...")
        breadth_df = calculate_market_breadth(ticker_df)
        breadth_path = self.market_features_dir / "daily_breadth.parquet"
        
        # Update breadth data
        if breadth_path.exists():
            existing_breadth = pd.read_parquet(breadth_path)
            latest_dates = breadth_df.index.difference(existing_breadth.index)
            if not latest_dates.empty:
                combined_df = pd.concat([existing_breadth, breadth_df.loc[latest_dates]])
                combined_df = combined_df.sort_index()
                combined_df.to_parquet(breadth_path)
                logger.info(f"Updated market breadth with {len(latest_dates)} new dates")
        else:
            breadth_df.to_parquet(breadth_path)
            logger.info("Created new market breadth file")

        # Generate sector performance
        logger.info("Starting sector performance generation...")
        sector_dfs = calculate_sector_performance(ticker_df, str(self.market_features_dir))

        # Save individual sector files
        sectors_dir = self.market_features_dir / "sectors"
        sectors_dir.mkdir(exist_ok=True)
        for sector, df in sector_dfs.items():
            if sector != 'all_sectors':  # Skip the combined file
                sector_path = sectors_dir / f"{sector}.parquet"
                df.to_parquet(sector_path)
                logger.info(f"Saved sector performance for {sector}")

        # Generate market sentiment
        logger.info("Starting market sentiment generation...")
        sentiment_df = MarketSentiment(self.data_dir).generate_sentiment_features(start_date, days)
        sentiment_path = self.market_features_dir / "market_sentiment.parquet"
        sentiment_df.to_parquet(sentiment_path)
        logger.info("Saved market sentiment")

        # Generate market volatility
        logger.info("Starting market volatility generation...")
        volatility_df = MarketVolatility(self.data_dir).generate_volatility_features(ticker_df, start_date)
        volatility_path = self.market_features_dir / "market_volatility.parquet"
        volatility_df.to_parquet(volatility_path)
        logger.info("Saved market volatility")

        return {
            "breadth": breadth_df,
            "sector": sector_dfs['all_sectors'],  # Return the combined DataFrame but don't save it
            "sentiment": sentiment_df,
            "volatility": volatility_df
        }

    def get_ticker_list(self) -> List[str]:
        """Get list of tickers from the features directory.
        
        Returns:
            List of ticker symbols
        """
        return [f.stem.replace('_features', '') for f in (self.data_dir / "features").glob('*_features.parquet')] 