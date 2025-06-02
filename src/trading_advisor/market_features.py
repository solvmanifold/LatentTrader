"""Market-wide feature calculation and management."""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

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
                'ticker': ticker,
                'sector': sector,
                'subsector': industry,
                'last_updated': pd.Timestamp.now().normalize()
            })
            
        except Exception as e:
            logger.warning(f"Failed to get sector info for {ticker}: {e}")
            sector_data.append({
                'ticker': ticker,
                'sector': 'Unknown',
                'subsector': 'Unknown',
                'last_updated': pd.Timestamp.now().normalize()
            })
    
    return pd.DataFrame(sector_data)

def save_sector_mapping(df: pd.DataFrame, features_dir: str = "market_features") -> None:
    """Save sector mapping to parquet file."""
    features_dir = Path(features_dir)
    features_dir.mkdir(exist_ok=True)
    
    metadata_dir = features_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    
    output_path = metadata_dir / "sector_mapping.parquet"
    df.to_parquet(output_path)
    logger.info(f"Saved sector mapping to {output_path}")

def load_sector_mapping(features_dir: str = "market_features") -> Optional[pd.DataFrame]:
    """Load sector mapping from parquet file."""
    mapping_path = Path(features_dir) / "metadata" / "sector_mapping.parquet"
    
    if not mapping_path.exists():
        return None
    
    try:
        return pd.read_parquet(mapping_path)
    except Exception as e:
        logger.error(f"Error loading sector mapping: {e}")
        return None

def update_sector_mapping(tickers: list[str], features_dir: str = "market_features") -> None:
    """Update sector mapping for given tickers."""
    # Load existing mapping if available
    existing_mapping = load_sector_mapping(features_dir)
    
    # Get new mapping
    new_mapping = get_sector_mapping(tickers)
    
    if existing_mapping is not None:
        # Update existing entries and append new ones
        existing_mapping = existing_mapping[~existing_mapping['ticker'].isin(tickers)]
        updated_mapping = pd.concat([existing_mapping, new_mapping])
    else:
        updated_mapping = new_mapping
    
    # Save updated mapping
    save_sector_mapping(updated_mapping, features_dir) 