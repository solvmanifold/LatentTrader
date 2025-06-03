"""
Market features module.

This module handles the generation of various market features including:
- Market breadth
- Sector performance
- Market sentiment
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
from .sector_mapping import update_sector_mapping

logger = logging.getLogger(__name__)

def load_ticker_data(ticker_list: List[str], features_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load ticker data for market features.
    
    Args:
        ticker_list: List of tickers to load
        features_dir: Directory containing feature files
        
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
        """
        # Get list of tickers
        ticker_list = self.get_ticker_list()
        
        # Update sector mapping if requested
        if update_sector_mapping:
            logger.info("Updating sector mapping...")
            from .sector_mapping import update_sector_mapping as update_sector_mapping_fn
            update_sector_mapping_fn(ticker_list, str(self.market_features_dir))
        
        # Load ticker data once
        combined_df, ticker_df = load_ticker_data(ticker_list, str(self.data_dir / "features"))
        if combined_df.empty:
            return {}
        
        # Generate market breadth
        logger.info("Starting market breadth generation...")
        breadth_df = calculate_market_breadth(combined_df)
        breadth_path = self.market_features_dir / "daily_breadth.parquet"
        breadth_df.to_parquet(breadth_path)
        logger.info(f"Saved market breadth to {breadth_path}")
        
        # Generate sector performance
        logger.info("Starting sector performance generation...")
        sector_dfs = calculate_sector_performance(ticker_df, str(self.market_features_dir))
        
        # Save individual sector files
        sectors_dir = self.market_features_dir / "sectors"
        sectors_dir.mkdir(exist_ok=True)
        for sector, df in sector_dfs.items():
            sector_path = sectors_dir / f"{sector}.parquet"
            df.to_parquet(sector_path)
            logger.info(f"Saved {sector} performance to {sector_path}")
        
        # Save the wide-format combined table at the top level
        all_sectors_path = self.market_features_dir / "all_sectors.parquet"
        sector_dfs['all_sectors'].to_parquet(all_sectors_path)
        logger.info(f"Saved all sector performance to {all_sectors_path}")
        
        # Generate sentiment features
        logger.info("Starting sentiment feature generation...")
        sentiment_df = MarketSentiment(self.data_dir).generate_sentiment_features(start_date, days)
        sentiment_path = self.market_features_dir / "market_sentiment.parquet"
        sentiment_df.to_parquet(sentiment_path)
        logger.info(f"Saved market sentiment to {sentiment_path}")
        
        return {
            "breadth": breadth_df,
            "sector": sector_dfs['all_sectors'],
            "sentiment": sentiment_df
        }

    def get_ticker_list(self) -> List[str]:
        """Get list of tickers from the features directory.
        
        Returns:
            List of ticker symbols
        """
        return [f.stem.replace('_features', '') for f in (self.data_dir / "features").glob('*_features.parquet')] 