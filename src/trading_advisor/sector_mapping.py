"""
Sector mapping module.

This module handles the mapping of tickers to sectors and sector-related utilities.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
import yfinance as yf
from tqdm import tqdm
from trading_advisor.data import normalize_ticker

logger = logging.getLogger(__name__)

def get_sector_mapping(tickers: list[str]) -> pd.DataFrame:
    """
    Get sector mapping for a list of tickers.
    Returns a DataFrame with columns: [ticker, sector, subsector, last_updated]
    """
    sector_data = []
    
    for ticker in tqdm(tickers, desc="Getting sector info"):
        try:
            # Get sector info from yfinance
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            sector_data.append({
                'ticker': normalize_ticker(ticker),
                'sector': sector,
                'subsector': industry,
                'last_updated': pd.Timestamp.now().normalize()
            })
            
        except Exception as e:
            logger.warning(f"Failed to get sector info for {ticker}: {e}")
            sector_data.append({
                'ticker': normalize_ticker(ticker),
                'sector': 'Unknown',
                'subsector': 'Unknown',
                'last_updated': pd.Timestamp.now().normalize()
            })
    
    return pd.DataFrame(sector_data)

def load_sector_mapping(market_features_dir: str = "data/market_features") -> pd.DataFrame:
    """Load sector mapping from parquet file.
    
    Args:
        market_features_dir: Directory containing market feature files
        
    Returns:
        DataFrame with ticker to sector mapping
    """
    mapping_file = Path(market_features_dir) / "sector_mapping.parquet"
    if mapping_file.exists():
        return pd.read_parquet(mapping_file)
    return pd.DataFrame()

def update_sector_mapping(tickers: List[str], market_features_dir: str = "data/market_features") -> pd.DataFrame:
    """
    Update sector mapping for the given tickers.
    
    Args:
        tickers: List of ticker symbols
        market_features_dir: Directory to save market feature files
        
    Returns:
        DataFrame with updated sector mapping
    """
    # Get sector mapping
    sector_mapping = get_sector_mapping(tickers)
    
    # Save sector mapping
    save_sector_mapping(sector_mapping, market_features_dir)
    
    return sector_mapping

def save_sector_mapping(mapping: pd.DataFrame, market_features_dir: str = "data/market_features") -> None:
    """
    Save sector mapping to parquet file.
    
    Args:
        mapping: DataFrame with sector mapping
        market_features_dir: Directory to save market feature files
    """
    output_path = Path(market_features_dir) / "sector_mapping.parquet"
    output_path.parent.mkdir(exist_ok=True)
    mapping.to_parquet(output_path) 