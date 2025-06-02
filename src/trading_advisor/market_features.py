"""Market-wide feature calculation and management."""

import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

import pandas as pd
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

def save_sector_mapping(mapping: pd.DataFrame, market_features_dir: str = "market_features") -> None:
    """
    Save sector mapping to parquet file.
    
    Args:
        mapping: DataFrame with sector mapping
        market_features_dir: Directory to save market feature files
    """
    output_path = Path(market_features_dir) / "metadata" / "sector_mapping.parquet"
    output_path.parent.mkdir(exist_ok=True)
    
    # Add last_updated column with current date
    mapping['last_updated'] = datetime.now().date()
    
    mapping.to_parquet(output_path)
    logger.info(f"Saved sector mapping to {output_path}")

def load_sector_mapping(market_features_dir: str = "market_features") -> pd.DataFrame:
    """
    Load sector mapping from parquet file.
    
    Args:
        market_features_dir: Directory containing market feature files
        
    Returns:
        DataFrame with sector mapping, or empty DataFrame if file not found
    """
    mapping_path = Path(market_features_dir) / "metadata" / "sector_mapping.parquet"
    
    if not mapping_path.exists():
        logger.error(f"Sector mapping file not found: {mapping_path}")
        return pd.DataFrame()
        
    try:
        df = pd.read_parquet(mapping_path)
        return df
    except Exception as e:
        logger.error(f"Error loading sector mapping: {e}")
        return pd.DataFrame()

def update_sector_mapping(tickers: List[str], market_features_dir: str = "market_features") -> pd.DataFrame:
    """
    Update sector mapping for the given tickers.
    
    Args:
        tickers: List of ticker symbols
        market_features_dir: Directory to save market feature files
        
    Returns:
        DataFrame with updated sector mapping
    """
    # TODO: Implement sector mapping update
    # For now, just return empty DataFrame
    return pd.DataFrame() 