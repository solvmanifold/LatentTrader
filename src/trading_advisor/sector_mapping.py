"""
Sector mapping module.

This module handles the mapping of tickers to sectors and sector-related utilities.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging
import yfinance as yf
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import json
from trading_advisor.data import normalize_ticker

logger = logging.getLogger(__name__)

def get_sector_mapping(tickers: list[str]) -> Dict[str, str]:
    """
    Get sector mapping for a list of tickers.
    Returns a dictionary mapping tickers to their sectors.
    """
    sector_mapping = {}
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
        task = progress.add_task("Getting sector info...", total=len(tickers))
        for ticker in tickers:
            try:
                # Get sector info from yfinance
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                
                sector = info.get('sector', 'Unknown')
                sector_mapping[normalize_ticker(ticker)] = sector
                
            except Exception as e:
                logger.warning(f"Failed to get sector info for {ticker}: {e}")
                sector_mapping[normalize_ticker(ticker)] = 'Unknown'
            progress.update(task, advance=1)
    
    return sector_mapping

def load_sector_mapping(market_features_dir: str = "data/market_features") -> Dict[str, str]:
    """Load sector mapping from JSON file.
    
    Args:
        market_features_dir: Directory containing market feature files
        
    Returns:
        Dictionary mapping tickers to sectors
    """
    mapping_file = Path(market_features_dir) / "sector_mapping.json"
    logger.debug(f"Loading sector mapping from {mapping_file}")
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            raw_mapping = json.load(f)
            logger.debug(f"Raw mapping contents: {raw_mapping}")
            logger.debug(f"Raw mapping keys: {list(raw_mapping.keys())}")
            # Normalize all tickers in the mapping
            mapping = {normalize_ticker(ticker): sector for ticker, sector in raw_mapping.items()}
            logger.debug(f"Normalized mapping contents: {mapping}")
            logger.debug(f"Normalized mapping keys: {list(mapping.keys())}")
            logger.debug(f"Loaded sector mapping with {len(mapping)} tickers")
            logger.debug(f"First 5 tickers in mapping: {list(mapping.items())[:5]}")
            return mapping
    logger.warning(f"Sector mapping file not found at {mapping_file}")
    return {}

def update_sector_mapping(tickers: List[str], market_features_dir: str = "data/market_features") -> Dict[str, str]:
    """
    Update sector mapping for the given tickers.
    
    Args:
        tickers: List of ticker symbols
        market_features_dir: Directory to save market feature files
        
    Returns:
        Dictionary mapping tickers to sectors
    """
    # Get sector mapping
    sector_mapping = get_sector_mapping(tickers)
    
    # Save sector mapping
    save_sector_mapping(sector_mapping, market_features_dir)
    
    return sector_mapping

def save_sector_mapping(mapping: Dict[str, str], market_features_dir: str = "data/market_features") -> None:
    """
    Save sector mapping to JSON file.
    
    Args:
        mapping: Dictionary mapping tickers to sectors
        market_features_dir: Directory to save market feature files
    """
    output_path = Path(market_features_dir) / "sector_mapping.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2) 