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

import pandas as pd
import yfinance as yf
from tqdm import tqdm
from trading_advisor.data import normalize_ticker
from .market_breadth import calculate_market_breadth
from .sector_performance import calculate_sector_performance
from .sentiment import MarketSentiment
from .volatility import MarketVolatility
from .sector_mapping import update_sector_mapping

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
        
    def generate_market_features(self, start_date: Optional[str] = None, update_sector_mapping: bool = False, days: int = 60):
        """
        Generate and save market features as separate files in data/market_features.
        Handles incremental updates by only processing new dates.
        """
        # Get list of tickers
        ticker_list = self.get_ticker_list()
        
        # Update sector mapping if requested
        if update_sector_mapping:
            logger.info("Updating sector mapping...")
            from .sector_mapping import update_sector_mapping as update_sector_mapping_fn
            update_sector_mapping_fn(ticker_list, str(self.market_features_dir))
        
        # Determine start date for updates
        update_start = None
        if start_date is None:
            # Check latest dates in existing files
            breadth_path = self.market_features_dir / "daily_breadth.parquet"
            all_sectors_path = self.market_features_dir / "all_sectors.parquet"
            sentiment_path = self.market_features_dir / "market_sentiment.parquet"
            volatility_path = self.market_features_dir / "market_volatility.parquet"
            
            latest_dates = [
                get_latest_feature_date(breadth_path),
                get_latest_feature_date(all_sectors_path),
                get_latest_feature_date(sentiment_path),
                get_latest_feature_date(volatility_path)
            ]
            latest_dates = [d for d in latest_dates if d is not None]
            
            if latest_dates:
                # Use the earliest latest date to ensure consistency
                update_start = min(latest_dates)
                logger.info(f"Found existing data through {update_start.date()}. Will update from next day.")
            else:
                # No existing data, use days parameter
                update_start = pd.Timestamp.today() - pd.Timedelta(days=days)
                logger.info(f"No existing data found. Will generate {days} days of data.")
        else:
            update_start = pd.to_datetime(start_date)
            logger.info(f"Using provided start date: {update_start.date()}")
        
        # Load ticker data for update period
        combined_df, ticker_df = load_ticker_data(ticker_list, str(self.data_dir / "features"), update_start)
        if combined_df.empty:
            logger.warning("No ticker data available for the update period.")
            return {}
        
        # Generate market breadth
        logger.info("Starting market breadth generation...")
        breadth_df = calculate_market_breadth(combined_df)
        breadth_path = self.market_features_dir / "daily_breadth.parquet"
        
        # Merge with existing breadth data
        if breadth_path.exists():
            existing_breadth = pd.read_parquet(breadth_path)
            existing_breadth.index = pd.to_datetime(existing_breadth.index)
            # Remove any overlapping dates from existing data
            existing_breadth = existing_breadth[existing_breadth.index < breadth_df.index.min()]
            # Combine old and new data
            breadth_df = pd.concat([existing_breadth, breadth_df])
            breadth_df = breadth_df[~breadth_df.index.duplicated(keep='last')]
            breadth_df = breadth_df.sort_index()
        
        breadth_df.to_parquet(breadth_path)
        logger.info(f"Saved market breadth to {breadth_path}")
        
        # Generate sector performance
        logger.info("Starting sector performance generation...")
        sector_dfs = calculate_sector_performance(ticker_df, str(self.market_features_dir))
        
        # Save individual sector files
        sectors_dir = self.market_features_dir / "sectors"
        sectors_dir.mkdir(exist_ok=True)
        for sector, df in sector_dfs.items():
            if sector == 'all_sectors':
                continue
            sector_path = sectors_dir / f"{sector}.parquet"
            
            # Merge with existing sector data
            if sector_path.exists():
                existing_sector = pd.read_parquet(sector_path)
                existing_sector.index = pd.to_datetime(existing_sector.index)
                # Remove any overlapping dates from existing data
                existing_sector = existing_sector[existing_sector.index < df.index.min()]
                # Combine old and new data
                df = pd.concat([existing_sector, df])
                df = df[~df.index.duplicated(keep='last')]
                df = df.sort_index()
            
            df.to_parquet(sector_path)
            logger.info(f"Saved {sector} performance to {sector_path}")
        
        # Save the wide-format combined table
        all_sectors_path = self.market_features_dir / "all_sectors.parquet"
        all_sectors_df = sector_dfs['all_sectors']
        
        # Merge with existing all sectors data
        if all_sectors_path.exists():
            existing_all = pd.read_parquet(all_sectors_path)
            existing_all.index = pd.to_datetime(existing_all.index)
            # Remove any overlapping dates from existing data
            existing_all = existing_all[existing_all.index < all_sectors_df.index.min()]
            # Combine old and new data
            all_sectors_df = pd.concat([existing_all, all_sectors_df])
            all_sectors_df = all_sectors_df[~all_sectors_df.index.duplicated(keep='last')]
            all_sectors_df = all_sectors_df.sort_index()
        
        all_sectors_df.to_parquet(all_sectors_path)
        logger.info(f"Saved all sector performance to {all_sectors_path}")
        
        # Generate sentiment features
        logger.info("Starting sentiment feature generation...")
        sentiment_df = MarketSentiment(self.data_dir).generate_sentiment_features(start_date, days)
        sentiment_path = self.market_features_dir / "market_sentiment.parquet"
        
        # Merge with existing sentiment data
        if sentiment_path.exists():
            existing_sentiment = pd.read_parquet(sentiment_path)
            existing_sentiment.index = pd.to_datetime(existing_sentiment.index)
            # Remove any overlapping dates from existing data
            existing_sentiment = existing_sentiment[existing_sentiment.index < sentiment_df.index.min()]
            # Combine old and new data
            sentiment_df = pd.concat([existing_sentiment, sentiment_df])
            sentiment_df = sentiment_df[~sentiment_df.index.duplicated(keep='last')]
            sentiment_df = sentiment_df.sort_index()
        
        sentiment_df.to_parquet(sentiment_path)
        logger.info(f"Saved market sentiment to {sentiment_path}")
        
        # Generate volatility features
        logger.info("Starting volatility feature generation...")
        volatility_df = MarketVolatility(self.data_dir).generate_volatility_features(combined_df, start_date)
        
        return {
            "breadth": breadth_df,
            "sector": all_sectors_df,
            "sentiment": sentiment_df,
            "volatility": volatility_df
        }

    def get_ticker_list(self) -> List[str]:
        """Get list of tickers from the features directory.
        
        Returns:
            List of ticker symbols
        """
        return [f.stem.replace('_features', '') for f in (self.data_dir / "features").glob('*_features.parquet')] 